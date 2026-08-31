# ------------------ 04_build_daily_factor.py ----------------
# 作用：把 03 产出的日度原子因子做 20 日滚动，得到 18 个正式因子的日频宽表面板。
# 设计：原子长表 -> 逐因子 pivot 成（交易日×股票）宽表 -> rolling(20, min_periods=15)
#       沿日期轴向量滚动（比长表 groupby.rolling 快一到两个量级）-> 每个因子落一个宽表 parquet。
# 合成：
#   A2d_atom = A2c_atom × 当日换手率(turn.hdf，百分数)，再滚动均值 -> A2d_high_tau_turnover_20d
#   C3_entropy_std_20d = C1_entropy_atom 的 20 日滚动标准差
#   其余 16 个因子 = 对应原子的 20 日滚动均值
#
# 运行：python 04_build_daily_factor.py

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from lib_common import (
    ATOMIC_CACHE_DIR,
    ATOM_TO_FACTOR_MEAN,
    C3_SOURCE_ATOM,
    DAILY_CACHE_DIR,
    FACTOR_COLS,
    MIN_PERIODS,
    PANEL_CACHE_DIR,
    ROLL_WIN,
    ensure_dir,
)

FACTORS_DAILY_DIR = PANEL_CACHE_DIR / "factors_daily"


def load_atoms_long() -> pd.DataFrame:
    # 读 03 的全部 part 并拼接；按 (code, trade_date) 去重兜底（重复运行同一区间不会出错）。
    parts = sorted(ATOMIC_CACHE_DIR.glob("parts/atomic_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"{ATOMIC_CACHE_DIR / 'parts'} 下没有原子因子 part，先运行 03。")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["code", "trade_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def main() -> None:
    t0 = time.time()
    ensure_dir(FACTORS_DAILY_DIR)

    print("读取原子因子 part ...")
    atoms = load_atoms_long()
    print(f"  原子面板 shape={atoms.shape}，日期 {atoms.trade_date.min().date()} ~ "
          f"{atoms.trade_date.max().date()}")

    # ---- 换手率宽表（turn.hdf 缓存，百分数），用于 A2d 合成 ----
    turn = pd.read_parquet(DAILY_CACHE_DIR / "turn.parquet")
    # A2d_atom = A2c_atom × 当日换手率：stack 成长表后按原子表的索引对齐，
    # 用 reindex 而不是直接相乘，避免并集索引把原子表撑大。
    atoms = atoms.set_index(["trade_date", "code"])
    turn_long = turn.stack().reindex(atoms.index)
    # A2d 对齐健全性断言：turn 长表与原子索引应逐格匹配（防日期轴 dtype/格式错位静默全 NaN）。
    turn_share = float(turn_long.notna().mean())
    if turn_share < 0.99:
        raise RuntimeError(f"A2d 对齐异常：turn_long 非空率 {turn_share:.3%}，"
                           f"请检查 02 换手率缓存的日期轴与原子表 trade_date 是否一致")
    print(f"  A2d 对齐检查：turn_long 非空率 {turn_share:.3%}")
    atoms["A2d_atom"] = atoms["A2c_high_tau_vol_share_atom"] * turn_long

    # ---- 逐因子滚动（宽表沿日期轴） ----
    # 均值型因子：原子（含 A2d_atom）-> rolling mean。
    mean_map = dict(ATOM_TO_FACTOR_MEAN)          # 原子 -> 因子
    mean_map["A2d_atom"] = "A2d_high_tau_turnover_20d"

    for atom_col, factor_col in mean_map.items():
        # pivot：行=交易日，列=股票；缺失（停牌/未上市）为 NaN，滚动自动跳过。
        wide = atoms[atom_col].unstack(level="code")
        roll = wide.rolling(ROLL_WIN, min_periods=MIN_PERIODS).mean()
        out = FACTORS_DAILY_DIR / f"{factor_col}.parquet"
        roll.to_parquet(out)
        print(f"[ok] {factor_col}  shape={roll.shape}")
        del wide, roll

    # C3：C1 原子的 20 日滚动标准差（不是均值）。
    wide = atoms[C3_SOURCE_ATOM].unstack(level="code")
    roll = wide.rolling(ROLL_WIN, min_periods=MIN_PERIODS).std()
    roll.to_parquet(FACTORS_DAILY_DIR / "C3_entropy_std_20d.parquet")
    print(f"[ok] C3_entropy_std_20d  shape={roll.shape}")
    del wide, roll

    # 校验：18 个因子文件齐全。
    missing = [f for f in FACTOR_COLS if not (FACTORS_DAILY_DIR / f"{f}.parquet").exists()]
    if missing:
        raise RuntimeError(f"缺少因子文件: {missing}")
    print(f"完成：18 个日频因子宽表已写入 {FACTORS_DAILY_DIR}，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
