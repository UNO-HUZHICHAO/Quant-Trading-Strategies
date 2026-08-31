# ------------------ 03_build_daily_atomic.py ----------------
# 作用：流式读取 F:\highfreqnew 每个交易日的分钟 hdf，向量化计算日度原子因子，
#       每攒够 FLUSH_DAYS 个交易日写一个 part 文件到 D:\hf_factor_cache\factors_atomic\parts\
#       （04 脚本 glob 全部 part 读取；不再按年合并单表）。
# 原则：只读 F 盘、顺序整块读；断点续跑（progress.json 记录已落盘日期与失败日期）；空值/无效日直接剔除。
#
# 每日处理逻辑：
#   1) 读取当日 hdf，散布成 (股票×240) 的 OHLCV+成交额矩阵；
#   2) 仅保留沪深 A 股（.SH/.SZ，剔除北交所）；
#   3) 过滤无效日：日成交量=0（停牌冻结行）或有效分钟 < MIN_VALID_MINUTES；
#   4) prepare_day + factors_a/b/c 一次算出全部原子（含 C1_H/C1_N 诊断列）；
#   5) 汇总为长表 (code, trade_date, 原子列)，按批次落盘（part 文件覆盖式幂等）。
#
# 运行：
#   python 03_build_daily_atomic.py                 # 全量（断点续跑）
#   python 03_build_daily_atomic.py --limit 3       # 只跑前 3 天（小样验证）
#   python 03_build_daily_atomic.py --start 20160501 --end 20160510

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factors_a import compute_atoms_a
from factors_b import compute_atoms_b
from factors_c import compute_atoms_c
from factors_common import prepare_day
from lib_common import (
    ATOMIC_CACHE_DIR,
    ATOM_COLS,
    MIN_VALID_MINUTES,
    ensure_dir,
    list_minute_files,
    load_progress,
    save_progress,
)
from lib_fdrive import list_day_stocks, read_minute_day

# 攒够多少个交易日就落一次盘（控制内存，也决定断点粒度）。
FLUSH_DAYS = 42
# 需要的分钟字段名 -> 输出列下标由 MINUTE_FIELDS 决定，这里用名字取。
_FIELDS = ["开盘价", "最高价", "最低价", "收盘价", "成交量", "成交额"]


def _progress_path() -> Path:
    return ATOMIC_CACHE_DIR / "progress.json"


def _parts_dir() -> Path:
    # 每个落盘批次写一个 part 文件（04 脚本 glob 全部 part 读取），
    # 避免"读旧表+拼接+重写"的反复 I/O，也天然支持断点续跑。
    return ATOMIC_CACHE_DIR / "parts"


def process_day(day_file: Path) -> pd.DataFrame | None:
    # 处理单个交易日文件，返回长表 DataFrame（code, trade_date, 16 原子）；无有效股票返回 None。
    date_str = day_file.stem  # YYYYMMDD
    trade_date = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    day = read_minute_day(day_file)

    # 只保留沪深 A 股的股票下标。
    keep_idx = list_day_stocks(day)
    if len(keep_idx) == 0:
        return None
    codes = [day.codes[i] for i in keep_idx]

    # 散布 6 个字段成 (全部股票, 240)，再取沪深子集。
    grids = {}
    for f in _FIELDS:
        grids[f] = day.scatter(day.field_col(f))[keep_idx]
    close = grids["收盘价"]
    vol = grids["成交量"]

    # 快速过滤无效日（在跑因子前先砍掉，省计算）：
    #   收盘价有效分钟数 >= MIN_VALID_MINUTES，且日成交量 > 0（剔除全天停牌冻结行）。
    valid_min = (np.isfinite(close) & (close > 0)).sum(axis=1)
    vol_safe = np.where(np.isfinite(vol), vol, 0.0)
    volsum = vol_safe.sum(axis=1)
    good = (valid_min >= MIN_VALID_MINUTES) & (volsum > 0)
    if not good.any():
        return None

    codes_good = [c for c, g in zip(codes, good) if g]
    if len(codes_good) != len(set(codes_good)):
        print(f"[WARN] {date_str}: 单日文件存在重复代码，按首次出现去重")
        seen: set[str] = set()
        codes_good = [c for c in codes_good if not (c in seen or seen.add(c))]
    pack = prepare_day(
        close[good], grids["最高价"][good], grids["最低价"][good],
        vol[good], grids["成交额"][good],
    )

    atoms: dict[str, np.ndarray] = {}
    atoms.update(compute_atoms_a(pack))
    atoms.update(compute_atoms_b(pack))
    atoms.update(compute_atoms_c(pack))

    df = pd.DataFrame({"code": codes_good, "trade_date": trade_date})
    for col in ATOM_COLS:
        df[col] = atoms[col]
    return df


def flush(buffer: list[pd.DataFrame], done: set[str], failed: set[str] | None = None) -> None:
    # 把缓冲区写成一个 part 文件（不再回读旧表合并，I/O 最省、天然幂等可续跑）。
    # 落盘成功后才把对应日期写入 progress.json。
    if not buffer:
        return
    buf = pd.concat(buffer, ignore_index=True)
    buf = buf.sort_values(["trade_date", "code"]).reset_index(drop=True)
    dmin = buf["trade_date"].min().strftime("%Y%m%d")
    dmax = buf["trade_date"].max().strftime("%Y%m%d")
    parts = _parts_dir()
    ensure_dir(parts)
    # part 文件名带日期区间，重复运行同一区间会覆盖同名文件，不会产生重复数据。
    path = parts / f"atomic_{dmin}_{dmax}.parquet"
    buf.to_parquet(path, index=False)
    # 进度 = 本次落盘覆盖到的全部日期。
    done.update(buf["trade_date"].dt.strftime("%Y%m%d").unique())
    save_progress(_progress_path(), done, failed)


def run(start: str, end: str, limit: int | None, flush_days: int) -> None:
    ensure_dir(ATOMIC_CACHE_DIR)
    done = load_progress(_progress_path())
    files = list_minute_files(start=start, end=end)
    if limit is not None:
        files = files[:limit]

    buffer: list[pd.DataFrame] = []
    processed_since_flush = 0
    failed: set[str] = set()
    t0 = time.time()
    n_done = 0
    for i, day_file in enumerate(files):
        date_str = day_file.stem
        if date_str in done:
            continue
        try:
            df = process_day(day_file)
        except Exception as e:  # 单日失败不中断全流程，记录后继续（重跑会重试）。
            print(f"[ERROR] {date_str}: {e}")
            failed.add(date_str)
            processed_since_flush += 1
            continue
        if df is not None:
            buffer.append(df)
        processed_since_flush += 1
        n_done += 1
        if processed_since_flush >= flush_days:
            flush(buffer, done, failed)
            buffer = []
            processed_since_flush = 0
            el = time.time() - t0
            print(f"[flush] 已处理 {n_done}/{len(files)} 天，累计用时 {el:.0f}s，"
                  f"平均 {el / max(n_done,1):.2f}s/天")
    # 收尾落盘。
    flush(buffer, done, failed)
    el = time.time() - t0
    print(f"完成：本次处理 {n_done} 天，用时 {el:.0f}s；progress 共 {len(done)} 天"
          f"（失败 {len(failed)} 天）。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="流式计算日度原子因子（阶段 2）。")
    parser.add_argument("--start", default=None, help="起始日 YYYYMMDD，默认用 lib_common.MINUTE_START。")
    parser.add_argument("--end", default=None, help="结束日 YYYYMMDD，默认用 lib_common.DATA_END。")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件（小样验证用）。")
    parser.add_argument("--flush-days", type=int, default=FLUSH_DAYS, help="每处理多少天落盘一次。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from lib_common import DATA_END, MINUTE_START

    run(
        start=args.start or MINUTE_START,
        end=args.end or DATA_END,
        limit=args.limit,
        flush_days=args.flush_days,
    )
