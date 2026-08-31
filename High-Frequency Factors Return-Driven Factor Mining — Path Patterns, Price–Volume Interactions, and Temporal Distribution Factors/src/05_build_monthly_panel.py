# ------------------ 05_build_monthly_panel.py ----------------
# 作用：构建月末回测面板（形成月 t 月末截面 + 持有期收益）。
#
# 口径（P1 修复，无前视）：
#   - 因子与风格用形成日 t（每月最后交易日）的 ≤t 信息；
#   - 建仓日 t1 = t 的次一个交易日（月末后首个交易日），按 t1 收盘建仓；
#   - next_ret_1m = close_adj(下月末) / close_adj(t1) − 1。
#
# 过滤（空值一律直接剔除，不填充）：
#   1) 股票池：index_weight 时点权重 > 0（含历史调入调出，调出股自然退出）；
#   2) 可交易（建仓日 t1，用户指定 0-1 乘积法）：(1−涨停)×(1−跌停)×tingpai×(1−ST) == 1 才保留；
#   3) 上市不足 60 个交易日（按 t 的日历位置推导）剔除；
#   4) 形成月内停牌 > 5 个交易日剔除；
#   5) 基础字段/下期收益任一为空 → 整行剔除（因子列不做统一 dropna，
#      各因子样本由 06 逐因子独立过滤，避免 P2 全因子交集污染）。
#   P3：t_next 无收盘价（退市/长停牌）→ 用 [t1, t_next] 窗口内最后有效收盘价结算，settle_flag=last_valid。
#
# 输出：D:\hf_factor_cache\panels\monthly_panel.parquet（长表）+ panel_summary.csv
# 运行：python 05_build_monthly_panel.py

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from lib_common import (
    DAILY_CACHE_DIR,
    FACTOR_COLS,
    FIRST_FORM_MONTH,
    LAST_FORM_MONTH,
    LIST_MIN_DAYS,
    PANEL_CACHE_DIR,
    SUSP_MAX_DAYS,
    UNIVERSES,
    ensure_dir,
    load_trade_calendar,
)

FACTORS_DAILY_DIR = PANEL_CACHE_DIR / "factors_daily"


def load_wide(name: str) -> pd.DataFrame:
    return pd.read_parquet(DAILY_CACHE_DIR / f"{name}.parquet")


def main() -> None:
    t0 = time.time()
    ensure_dir(PANEL_CACHE_DIR)

    calendar = load_trade_calendar()
    # 每月最后一个交易日。
    cal_s = pd.Series(calendar, index=calendar)
    month_last = cal_s.groupby(cal_s.index.strftime("%Y-%m")).max()
    form_months = [m for m in month_last.index if FIRST_FORM_MONTH <= m <= LAST_FORM_MONTH]
    form_dates = pd.DatetimeIndex([month_last[m] for m in form_months])
    # 持有期月末 = 下一个形成月末（形成月序列的下一期）。
    all_last = month_last.sort_index()
    next_dates = all_last.shift(-1)
    print(f"形成月 {form_months[0]} ~ {form_months[-1]}，共 {len(form_months)} 期")

    # ---- 日频宽表 ----
    print("读取日频缓存 ...")
    close = load_wide("close")
    cap = load_wide("cap")
    turn = load_wide("turn")
    zt = load_wide("zhangting")
    dt_ = load_wide("dieting")
    tp = load_wide("tingpai")
    st = load_wide("stpanduan")
    ind = load_wide("industry")
    bounds = pd.read_parquet(DAILY_CACHE_DIR / "stock_bounds.parquet").set_index("code")
    weights = {u: load_wide(f"index_weight_{u}") for u in UNIVERSES}

    # ---- 预计算 21 日风格（只用 ≤t 数据，滚动窗口右端点即 t，无前视） ----
    ret1 = close.pct_change()
    vol21 = ret1.rolling(21, min_periods=15).std()   # 21 日收益标准差（与 04 的 min_periods 口径一致）
    turn21 = turn.rolling(21, min_periods=15).mean() # 21 日平均换手率
    log_cap = np.log(cap.where(cap > 0))      # log 流通市值（万元，仅取正市值）

    # ---- 上市日推导：交易日历位置差 ----
    first_valid = pd.to_datetime(bounds["first_valid"])
    cal_pos = pd.Series(np.arange(len(calendar)), index=calendar)

    # ---- 因子宽表 ----
    factors = {f: pd.read_parquet(FACTORS_DAILY_DIR / f"{f}.parquet") for f in FACTOR_COLS}

    rows: list[pd.DataFrame] = []
    stat_rows = []
    for i, t in enumerate(form_dates):
        fm = form_months[i]
        t_next = next_dates.loc[fm]
        if pd.isna(t_next):
            # 最后一个形成月没有下一期（本例数据到 2026-07，2026-06 形成月仍有下一期）。
            continue
        # 建仓日 t1 = t 的次一个交易日（月末后首个交易日，P1 修复）。
        pos_t = calendar.get_loc(t)
        t1 = calendar[pos_t + 1] if pos_t + 1 < len(calendar) else pd.NaT
        if pd.isna(t1):
            continue
        # 各指数时点成分（权重>0；NaN=非成分）。
        mem = {}
        missing_w = []
        for u in UNIVERSES:
            if t in weights[u].index:
                w_row = weights[u].loc[t].dropna()
                mem[u] = set(w_row[w_row > 0].index)
            else:
                mem[u] = set()
                missing_w.append(u)
        if missing_w:
            print(f"[WARN] {fm}: 权重表缺 t 日，池 {missing_w} 为空")
        pool = set().union(*mem.values())
        if not pool:
            stat_rows.append({"form_month": fm, "pool": 0, "kept": 0,
                              "drop_flag": 0, "drop_listed": 0, "drop_susp": 0,
                              "note": "no_pool"})
            print(f"[WARN] {fm}: 全部股票池为空，整月跳过")
            continue
        codes = sorted(pool)

        def row_of_at(frame: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
            # 取 day 日截面并按池内代码对齐（缺失代码 → NaN）。
            r = frame.loc[day] if day in frame.index else pd.Series(dtype=float)
            return r.reindex(codes)

        # 可交易性（建仓日 t1）：0-1 乘积法（任一标记缺失 → NaN → 乘积 NaN → 剔除）。
        prod = ((1 - row_of_at(zt, t1)) * (1 - row_of_at(dt_, t1))
                * row_of_at(tp, t1) * (1 - row_of_at(st, t1)))
        tradable = prod == 1

        # 上市满 60 交易日。注意：用 .values 取日历位置再重建代码索引，
        # 避免 reindex 把索引变成上市日期时间戳、与 tradable 掩码错位。
        fv = first_valid.reindex(codes)
        if t in cal_pos.index:
            fv_pos = cal_pos.reindex(pd.DatetimeIndex(fv.values)).to_numpy()
            age = pd.Series(cal_pos.loc[t] - fv_pos, index=codes)
        else:
            age = pd.Series(np.nan, index=codes)
        listed_ok = age >= LIST_MIN_DAYS

        # 形成月内停牌天数 ≤ 5。
        month_rows = tp.loc[tp.index.strftime("%Y-%m") == fm]
        susp_days = (month_rows == 0).sum(axis=0).reindex(codes)
        susp_ok = susp_days <= SUSP_MAX_DAYS

        keep = tradable & listed_ok & susp_ok.fillna(False)
        kept = [c for c, k in zip(codes, keep) if k]
        stat_rows.append({
            "form_month": fm,
            "pool": len(pool),
            "kept": len(kept),
            "drop_flag": int((~tradable.fillna(False)).sum()),
            "drop_listed": int((tradable.fillna(False) & ~listed_ok.fillna(False)).sum()),
            "drop_susp": int((tradable.fillna(False) & listed_ok.fillna(False) & ~susp_ok.fillna(False)).sum()),
        })
        if not kept:
            continue

        rec = pd.DataFrame({"code": kept, "form_date": t, "form_month": fm})
        for u in UNIVERSES:
            rec[f"in_{u}"] = [c in mem[u] for c in kept]
        # t 日风格与行业（因子与风格都用形成日 t 信息）。
        rec["industry"] = row_of_at(ind, t).reindex(kept).values
        rec["log_mktcap"] = row_of_at(log_cap, t).reindex(kept).values
        rec["turnover_21d"] = row_of_at(turn21, t).reindex(kept).values
        rec["volatility_21d"] = row_of_at(vol21, t).reindex(kept).values
        # 下期收益（P1 修复）：t1 收盘建仓，持有至下月末收盘。
        #   next_ret_1m = close[t_next] / close[t1] − 1
        #   P3：t_next 无收盘价（退市/长停牌）→ 用 [t1, t_next] 窗口内最后有效收盘价结算，
        #   settle_flag 标记 normal / last_valid。
        c_t1 = row_of_at(close, t1).reindex(kept)
        c_next = close.loc[t_next].reindex(kept) if t_next in close.index else pd.Series(np.nan, index=kept)
        window_days = calendar[(calendar > t) & (calendar <= t_next)]
        if len(window_days) > 0:
            c_last = close.loc[window_days].ffill().iloc[-1].reindex(kept)
        else:
            c_last = c_t1
        rec["next_ret_1m"] = (c_last / c_t1 - 1).values
        rec["settle_flag"] = np.where(c_next.notna().to_numpy(), "normal", "last_valid")
        # 18 个因子在 t 日的取值（允许部分因子缺失；06 逐因子独立过滤，避免全因子交集污染）。
        for f in FACTOR_COLS:
            rec[f] = row_of_at(factors[f], t).reindex(kept).values
        # 基础字段与下期收益任一缺失 → 整行删（因子列不参与统一 dropna）。
        req = ["industry", "log_mktcap", "turnover_21d", "volatility_21d", "next_ret_1m"]
        rec = rec.dropna(subset=req)
        rows.append(rec)

    panel = pd.concat(rows, ignore_index=True)
    out = PANEL_CACHE_DIR / "monthly_panel.parquet"
    panel.to_parquet(out, index=False)
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(PANEL_CACHE_DIR / "panel_summary.csv", index=False, encoding="utf-8-sig")
    print(f"面板 shape={panel.shape}，写出 {out}")
    print(stats.describe().round(1).to_string())
    # 自检：各池月份数应=120；last_valid 结算占比打印。
    print(f"settle_flag: last_valid 占比 {panel['settle_flag'].eq('last_valid').mean():.3%}")
    for u in UNIVERSES:
        sub = panel[panel[f"in_{u}"]]
        print(f"  池 {u}: {sub['form_month'].nunique()} 个形成月，{len(sub)} 行")
    print(f"用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
