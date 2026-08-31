# ------------------ 02_build_base_cache.py ----------------
# 作用：把移动硬盘 F:\base 的日频宽表切片后缓存成本机 D 盘 parquet（阶段 1）。
# 原则："就地计算、只带结果"——只读 F 盘、把压缩结果写 D 盘，之后日频层不再碰硬盘。
#
# 产出（D:\hf_factor_cache\daily\）：
#   calendar.csv                 交易日历（close.hdf 的日期轴）
#   close.parquet                后复权收盘价（窗口内，宽表 日期×代码）
#   turn.parquet                 日换手率（百分数）
#   cap.parquet                  流通市值（万元）
#   zhangting/dieting/tingpai/stpanduan.parquet   交易状态 0/1 面板
#   industry.parquet             申万一级行业数值代码（indus_1）
#   stock_bounds.parquet         每只股票首个/末个有效收盘日（用于上市日推导，用全历史）
#   index_weight_{hs300,zz500,zz1000}.parquet     时点成分权重（宽表 日期×成分股）
#   index_price.parquet          三指数+中证全指 日线点位
#
# 运行：python 02_build_base_cache.py [--force]

from __future__ import annotations

import argparse
#时间标准库，记录程序开始/结束时间，计算耗时，暂停几秒，做简单性能统计
import time

import pandas as pd

from lib_common import (
    CACHE_START_DAILY,#日频缓存的起始日期常量
    DAILY_CACHE_DIR,#日频缓存输出目录
    UNIVERSE_IDXCODE,#股票池名称到指数代码的映射表
    ensure_dir,#工具函数：目录不存在就创建目录
)
#导入三大读取模块
from lib_fdrive import read_base_wide, read_index_price, read_index_weight

# 需要从 price_index.hdf Equity 组带走的指数列（三基准 + 中证全指留作参考）。
INDEX_PRICE_COLS = ["000300.SH", "000905.SH", "000852.SH", "000985.CSI"]


def _save(df: pd.DataFrame, name: str, force: bool) -> None:
    # 统一落盘：已存在且不强制覆盖就跳过（支持断点续跑）。
    path = DAILY_CACHE_DIR / name
    if path.exists() and not force:
        print(f"[skip] {name} 已存在")
        return
    df.to_parquet(path)
    print(f"[ok]   {name}  shape={df.shape}")


def build(force: bool = False) -> None:
    ensure_dir(DAILY_CACHE_DIR)
    t0 = time.time()

    # ---- 交易日历 + 收盘价（先读全历史，顺便推导上市/退市边界） ----
    print("读取 close.hdf（全历史）...")
    close_full = read_base_wide("close")
    # 交易日历 = close 的日期轴，一行一个日期，供全流程对齐。
    cal = pd.Series(close_full.index.strftime("%Y-%m-%d"))
    cal_path = DAILY_CACHE_DIR / "calendar.csv"
    if cal_path.exists() and not force:
        print("[skip] calendar.csv 已存在")
    else:
        cal.to_csv(cal_path, index=False, header=False)
        print(f"[ok]   calendar.csv  n={len(cal)}")

    # 上市/退市边界：首个/末个非 NaN 收盘日。用全历史算，避免窗口截断误判新股。
    notna = close_full.notna()
    first_idx = notna.values.argmax(axis=0)           # 第一个非 NaN 的行号
    has_any = notna.values.any(axis=0)
    first_dates = close_full.index[first_idx]
    # 末个有效日：反向 argmax。
    last_idx = (notna.values.shape[0] - 1) - notna.values[::-1].argmax(axis=0)
    last_dates = close_full.index[last_idx]
    bounds = pd.DataFrame(
        {
            "code": close_full.columns,
            "first_valid": first_dates,
            "last_valid": last_dates,
            "has_data": has_any,
        }
    )
    bounds = bounds[bounds["has_data"]].drop(columns=["has_data"])
    _save(bounds, "stock_bounds.parquet", force)

    # 收盘价只保留回测需要的窗口（减小体积），上市边界已单独保存。
    close_win = close_full.loc[close_full.index >= CACHE_START_DAILY]
    _save(close_win, "close.parquet", force)
    del close_full, close_win, notna

    # ---- 其余日频宽表（都从窗口起点切） ----
    for name in ["turn", "cap", "zhangting", "dieting", "tingpai", "stpanduan"]:
        print(f"读取 {name}.hdf ...")
        df = read_base_wide(name, start=CACHE_START_DAILY)
        _save(df, f"{name}.parquet", force)
        del df

    # ---- 申万一级行业（indus_1 数值代码） ----
    print("读取 industry_sw.hdf / indus_1 ...")
    ind = read_base_wide("industry_sw", group="indus_1", start=CACHE_START_DAILY)
    _save(ind, "industry.parquet", force)
    del ind

    # ---- 指数时点成分权重（三个回测池） ----
    for uni, idxcode in UNIVERSE_IDXCODE.items():
        print(f"读取 index_weight.hdf / {idxcode} ...")
        w = read_index_weight(idxcode, start=CACHE_START_DAILY)
        _save(w, f"index_weight_{uni}.parquet", force)
        del w

    # ---- 指数行情（基准） ----
    print("读取 price_index.hdf / Equity ...")
    px = read_index_price(columns=INDEX_PRICE_COLS, start=CACHE_START_DAILY)
    _save(px, "index_price.parquet", force)
    del px

    print(f"完成，用时 {time.time() - t0:.1f}s，输出目录 {DAILY_CACHE_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 F:\\base 日频宽表缓存为 D 盘 parquet。")
    parser.add_argument("--force", action="store_true", help="强制重建已存在的缓存文件。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(force=args.force)
