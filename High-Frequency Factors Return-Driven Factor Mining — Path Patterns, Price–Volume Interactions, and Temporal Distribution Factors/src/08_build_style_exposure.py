# ------------------ 08_build_style_exposure.py ----------------
# 作用：构建 9 风格月度暴露面板（2016-07 ~ 2026-06，120 个形成月末），
#      补全研报局限第 1 条缺失的"完整九风格相关性定位与 Barra 式暴露分析"所需数据。
#
# 9 风格 = 规模 / 波动率 / 换手率 / 动量 / 反转 / BP / DP / 成长 / 盈利。
#
# 数据来源（三路，全部 point-in-time、只用形成月末 t 的 ≤t 信息）：
#   1) 本地自建（与 05 面板同源，保证与 S1/S2 回测完全一致）：
#        规模=ln流通市值、波动=21日收益std、换手=21日均换手 —— 直接复用 panel 现有列；
#        动量=P(t-21)/P(t-252)-1、反转=P(t)/P(t-21)-1 —— 由 F盘 close.hdf 计算（覆盖 2016-07 前置 252 日）。
#   2) 天软 get_factor_data（as-of t，天然 point-in-time，按形成月末批量）：
#        BP  = 1/PBLF（市净率 MRQ）；DP = 股息率TTM；盈利 = ROETTM。
#   3) 天软 get_table_data('主要财务指标')（全代码一次拉取，本地按 公布日≤t 对齐）：
#        成长 = 营业收入增长率(%)（最新已公布报告期）。
#
# 输出：
#   D:\hf_factor_cache\styles\style_exposure_monthly.parquet
#     长表：code, form_date, form_month, industry, size, vol, turn, mom, rev, bp, dp, growth, roe
#   D:\hf_factor_cache\styles\style_crossval.csv       （与 F盘 table_yuanshi.pkl 重叠期交叉验证）
#   D:\hf_factor_cache\styles\progress_factors.json    （天软因子拉取断点）
#   D:\hf_factor_cache\styles\progress_finance.json    （天软财务表拉取断点）
#
# 运行：python 08_build_style_exposure.py
# 依赖：日频数据挂载（动量前置 252 日）、cjpy 已持久化 token 或 CJ_API_TOKEN。

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib_common import DAILY_CACHE_DIR, F_BASE_DIR, PANEL_CACHE_DIR, ensure_dir
from lib_fdrive import read_base_wide

STYLES_DIR = DAILY_CACHE_DIR.parent / "styles"

# 优先使用 cjpy 已持久化的 token；也可通过环境变量 CJ_API_TOKEN 提供。
TINYSOFT_KEY = os.environ.get("CJ_API_TOKEN")


def ensure_cjpy_client(cjpy) -> None:
    """使用本机持久化凭证，或从环境变量初始化；不在源码中保存密钥。"""
    try:
        cjpy.get_default_client()
    except Exception as exc:
        if not TINYSOFT_KEY:
            raise RuntimeError(
                "cjpy 尚未配置凭证；请先持久化 token，或设置 CJ_API_TOKEN。"
            ) from exc
        cjpy.set_token(TINYSOFT_KEY, persist=True)

# 天软代码转换：面板后缀式 '000001.SZ' <-> 天软前缀式 'SZ000001'。
def to_ts(code: str) -> str:
    num, exch = code.split(".")
    return exch + num

def from_ts(code: str) -> str:
    return code[2:] + "." + code[:2]

# 天软因子（get_factor_data，as-of t）
FACTOR_FIELDS = ["PBLF", "股息率TTM", "ROETTM"]   # BP=1/PBLF、DP、盈利

# 财务表（get_table_data，point-in-time 对齐）
FIN_TABLE = "主要财务指标"
FIN_FIELDS = ["营业收入增长率(%)", "公布日"]

# 天软批量：get_factor_data 按形成月末（120 次，每次该月全部池内代码）；
#           get_table_data 按代码分批（每批 ~100，全池 ~54 批）。均在 30s 读超时内。
FIN_BATCH = 100

# 重试参数（天软 SSL/超时偶发，取较长退避；单次失败不中断整体，记入 failed 集最后补拉）。
RETRIES = 5
BACKOFF = [10, 20, 40, 80, 120]
CALL_SLEEP = 0.8   # 每次成功调用后的小间隔，降低连接抖动


def retry_call(fn, *args, **kwargs):
    # 天软调用重试（ReadTimeout 等网络错误），带退避。
    import cjpy
    last = None
    for i in range(RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [retry {i + 1}/{RETRIES}] {type(e).__name__}: {str(e)[:120]}")
            if i < RETRIES - 1:
                time.sleep(BACKOFF[i])
    raise last


def load_progress(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ============ 1. 量价类风格（本地自建，与 05 面板同源） ============

def build_local_styles() -> pd.DataFrame:
    # 以面板为骨架：复用 log_mktcap/turnover_21d/volatility_21d/industry，
    # 补充动量/反转（由 F盘 close.hdf 计算，覆盖形成月 t 前置 252 日）。
    panel = pd.read_parquet(PANEL_CACHE_DIR / "monthly_panel.parquet",
                            columns=["code", "form_date", "form_month", "industry",
                                     "log_mktcap", "turnover_21d", "volatility_21d"])
    panel = panel.rename(columns={
        "log_mktcap": "size",
        "turnover_21d": "turn",
        "volatility_21d": "vol",
    })
    # 动量/反转：读 F盘 close.hdf（2002 起，后复权，与本地缓存同基准）。
    close = read_base_wide("close")
    close = close.sort_index()
    mom = close.shift(21) / close.shift(252) - 1.0     # 12-1 动量（剔除最近 1 个月）
    rev = close / close.shift(21) - 1.0                # 1 个月反转
    form_dates = pd.DatetimeIndex(sorted(panel["form_date"].unique()))
    mom_m = mom.loc[form_dates].stack().reset_index()
    mom_m.columns = ["form_date", "code", "mom"]
    rev_m = rev.loc[form_dates].stack().reset_index()
    rev_m.columns = ["form_date", "code", "rev"]
    # 面板代码为 '000001.SZ'，close 列亦为 '000001.SZ'（read_base_wide 后缀式）——直接对齐。
    panel = panel.merge(mom_m, on=["code", "form_date"], how="left")
    panel = panel.merge(rev_m, on=["code", "form_date"], how="left")
    return panel


# ============ 2. 天软因子拉取（as-of 形成月末） ============

def pull_factor_styles(local: pd.DataFrame) -> pd.DataFrame:
    import cjpy
    ensure_cjpy_client(cjpy)

    prog = load_progress(STYLES_DIR / "progress_factors.json")
    done = set(prog.get("done", []))
    failed = set(prog.get("failed", []))
    raw_path = STYLES_DIR / "factor_styles_raw.parquet"
    out = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()
    form_dates = sorted(local["form_date"].unique())
    for i, t in enumerate(form_dates):
        ymd = t.strftime("%Y%m%d")
        if ymd in done:
            continue
        codes_ts = local.loc[local["form_date"] == t, "code"].map(to_ts).unique().tolist()
        print(f"[{i + 1}/{len(form_dates)}] {ymd} 代码数 {len(codes_ts)}", flush=True)
        try:
            df = retry_call(cjpy.get_factor_data, code=codes_ts, date=[ymd], factors=FACTOR_FIELDS)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {ymd}: {str(e)[:100]}（记入 failed，最后补拉）", flush=True)
            failed.add(ymd)
            save_progress(STYLES_DIR / "progress_factors.json", {"done": sorted(done), "failed": sorted(failed)})
            continue
        if df is None or len(df) == 0:
            done.add(ymd); save_progress(STYLES_DIR / "progress_factors.json", {"done": sorted(done), "failed": sorted(failed)})
            continue
        df["form_date"] = pd.Timestamp(ymd)
        df["code"] = df["代码"].map(from_ts)
        out = pd.concat([out, df], ignore_index=True)
        done.add(ymd)
        save_progress(STYLES_DIR / "progress_factors.json", {"done": sorted(done), "failed": sorted(failed)})
        out.to_parquet(raw_path, index=False)   # 逐日落盘，断点续跑不丢数据
        time.sleep(CALL_SLEEP)

    # 补拉 failed 日期（重试一轮）。
    for ymd in sorted(failed - done):
        if ymd in done:
            continue
        codes_ts = local.loc[local["form_date"] == pd.Timestamp(ymd), "code"].map(to_ts).unique().tolist()
        print(f"[补拉] {ymd} 代码数 {len(codes_ts)}", flush=True)
        try:
            df = retry_call(cjpy.get_factor_data, code=codes_ts, date=[ymd], factors=FACTOR_FIELDS)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL-补拉] {ymd}: {str(e)[:100]}", flush=True)
            continue
        if df is not None and len(df):
            df["form_date"] = pd.Timestamp(ymd)
            df["code"] = df["代码"].map(from_ts)
            out = pd.concat([out, df], ignore_index=True)
            done.add(ymd)
            save_progress(STYLES_DIR / "progress_factors.json", {"done": sorted(done), "failed": sorted(failed - done)})
            out.to_parquet(raw_path, index=False)
            time.sleep(CALL_SLEEP)
    return out


# ============ 3. 天软财务表拉取 + point-in-time 对齐（成长） ============

def pull_growth_styles(local: pd.DataFrame) -> pd.DataFrame:
    import cjpy
    ensure_cjpy_client(cjpy)

    all_codes = sorted(local["code"].unique())
    prog = load_progress(STYLES_DIR / "progress_finance.json")
    done = set(prog.get("done", []))
    failed = set(prog.get("failed", []))
    raw_path = STYLES_DIR / "finance_growth_raw.parquet"
    fin = pd.read_parquet(raw_path) if raw_path.exists() else pd.DataFrame()
    for i in range(0, len(all_codes), FIN_BATCH):
        chunk = all_codes[i:i + FIN_BATCH]
        key = chunk[0] + "_" + chunk[-1]
        if key in done:
            continue
        codes_ts = [to_ts(c) for c in chunk]
        print(f"[财务 {i // FIN_BATCH + 1}/{(len(all_codes) + FIN_BATCH - 1) // FIN_BATCH}] {key}", flush=True)
        try:
            df = retry_call(cjpy.get_table_data, code=codes_ts, table_name=FIN_TABLE, fields=FIN_FIELDS)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {key}: {str(e)[:100]}（记入 failed，最后补拉）", flush=True)
            failed.add(key)
            save_progress(STYLES_DIR / "progress_finance.json", {"done": sorted(done), "failed": sorted(failed)})
            continue
        if df is None or len(df) == 0:
            done.add(key); save_progress(STYLES_DIR / "progress_finance.json", {"done": sorted(done), "failed": sorted(failed)})
            continue
        df["code"] = df["CODE"].map(from_ts)
        df["公布日"] = pd.to_datetime(df["公布日"], format="%Y%m%d", errors="coerce")
        fin = pd.concat([fin, df], ignore_index=True)
        done.add(key)
        save_progress(STYLES_DIR / "progress_finance.json", {"done": sorted(done), "failed": sorted(failed)})
        fin.to_parquet(raw_path, index=False)   # 逐批落盘，断点续跑不丢数据
        time.sleep(CALL_SLEEP)

    # 补拉 failed 批次。
    for key in sorted(failed - done):
        i0, i1 = key.split("_")
        chunk = [c for c in all_codes if i0 <= c <= i1]
        codes_ts = [to_ts(c) for c in chunk]
        print(f"[财务补拉] {key}", flush=True)
        try:
            df = retry_call(cjpy.get_table_data, code=codes_ts, table_name=FIN_TABLE, fields=FIN_FIELDS)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL-补拉] {key}: {str(e)[:100]}", flush=True)
            continue
        if df is not None and len(df):
            df["code"] = df["CODE"].map(from_ts)
            df["公布日"] = pd.to_datetime(df["公布日"], format="%Y%m%d", errors="coerce")
            fin = pd.concat([fin, df], ignore_index=True)
            done.add(key)
            save_progress(STYLES_DIR / "progress_finance.json", {"done": sorted(done), "failed": sorted(failed - done)})
            fin.to_parquet(raw_path, index=False)
            time.sleep(CALL_SLEEP)

    if len(fin) == 0:
        return pd.DataFrame()

    # point-in-time 对齐：公布日→form_date 阶梯填充（最新已公布报告期的营收增速）。
    # 说明：不采用 merge_asof（其对 by+on 的排序要求在新版 pandas 易报错），
    #       改为 pivot 成"公布日×code"宽表后 reindex 全日期 + ffill——在 form_date 取到的即"公布日≤t 的最新值"。
    fin_s = fin.dropna(subset=["公布日", "营业收入增长率(%)"])
    wide = fin_s.pivot_table(index="公布日", columns="code",
                             values="营业收入增长率(%)", aggfunc="last")
    all_dates = sorted(set(wide.index) | set(local["form_date"]))
    wide = wide.reindex(all_dates).ffill()
    form_dates = sorted(local["form_date"].unique())
    sel = wide.loc[form_dates].stack().reset_index()
    sel.columns = ["form_date", "code", "growth"]
    return sel


# ============ 4. 主流程 ============

def main() -> None:
    t0 = time.time()
    ensure_dir(STYLES_DIR)

    print("步骤 1/4：自建量价类风格（规模/波动/换手/动量/反转） ...")
    local = build_local_styles()
    print(f"  骨架 {local.shape}，动量非空率 {local['mom'].notna().mean():.1%}，反转 {local['rev'].notna().mean():.1%}")

    print("步骤 2/4：天软拉取 as-of 因子（PBLF/股息率TTM/ROETTM） ...")
    fac = pull_factor_styles(local)
    if len(fac):
        fac = fac[["code", "form_date", "PBLF", "股息率TTM", "ROETTM"]]
        fac["bp"] = 1.0 / fac["PBLF"].where(fac["PBLF"] > 0)
        fac["dp"] = fac["股息率TTM"]
        fac["roe"] = fac["ROETTM"]
        local = local.merge(fac[["code", "form_date", "bp", "dp", "roe"]], on=["code", "form_date"], how="left")
        print(f"  合并后 bp/dp/roe 非空率 {local['bp'].notna().mean():.1%}/{local['dp'].notna().mean():.1%}/{local['roe'].notna().mean():.1%}")

    print("步骤 3/4：天软拉取财务表并 point-in-time 对齐（成长） ...")
    growth = pull_growth_styles(local)
    if len(growth):
        local = local.merge(growth, on=["code", "form_date"], how="left")
        print(f"  合并后 growth 非空率 {local['growth'].notna().mean():.1%}")

    cols = ["code", "form_date", "form_month", "industry",
            "size", "vol", "turn", "mom", "rev", "bp", "dp", "growth", "roe"]
    local = local[cols]
    local.to_parquet(STYLES_DIR / "style_exposure_monthly.parquet", index=False)
    print(f"写出 {STYLES_DIR / 'style_exposure_monthly.parquet'} shape={local.shape}")

    print("步骤 4/4：与 F盘 table_yuanshi.pkl 交叉验证 ...")
    crossval(local)

    print(f"完成，用时 {(time.time() - t0) / 60:.1f} 分钟")


def crossval(local: pd.DataFrame) -> None:
    # 与 F盘 table_yuanshi.pkl（2016-07~2023-11）重叠期比较 9 风格 Pearson 相关。
    f_ = F_BASE_DIR / "table_yuanshi.pkl"
    if not f_.exists():
        print("  [skip] F盘 table_yuanshi.pkl 未挂载，跳过交叉验证")
        return
    fy = pd.read_pickle(f_).reset_index()
    fy["code"] = fy["code"].astype(str)
    fy["form_date"] = pd.to_datetime(fy["date"])
    # 重叠期：本地 2016-07 起 & F盘 ≤2023-11。
    ov = local.merge(
        fy[["code", "form_date", "Return_1m", "Return_12m", "Std_Res_1m", "Turn_1m",
            "BP", "DP", "ROE_TTM", "Profit_Growth", "Ln_free_size"]],
        on=["code", "form_date"], how="inner")
    print(f"  重叠样本 {len(ov)} 行")
    pairs = [
        ("size", "Ln_free_size", "规模"),
        ("vol", "Std_Res_1m", "波动"),
        ("turn", "Turn_1m", "换手"),
        ("rev", "Return_1m", "反转"),
        ("mom", "Return_12m", "动量"),
        ("bp", "BP", "BP"),
        ("dp", "DP", "DP"),
        ("roe", "ROE_TTM", "盈利"),
        ("growth", "Profit_Growth", "成长"),
    ]
    out = []
    for lcol, fcol, cn in pairs:
        a = ov[lcol].to_numpy(dtype=float)
        b = ov[fcol].to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        r = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else np.nan
        out.append({"风格": cn, "本地列": lcol, "F盘列": fcol, "Pearson相关": round(r, 4),
                    "样本数": int(m.sum())})
    cv = pd.DataFrame(out)
    cv.to_csv(STYLES_DIR / "style_crossval.csv", index=False, encoding="utf-8-sig")
    print(cv.to_string(index=False))
    print(f"  交叉验证结果 -> {STYLES_DIR / 'style_crossval.csv'}")


if __name__ == "__main__":
    sys.exit(main())
