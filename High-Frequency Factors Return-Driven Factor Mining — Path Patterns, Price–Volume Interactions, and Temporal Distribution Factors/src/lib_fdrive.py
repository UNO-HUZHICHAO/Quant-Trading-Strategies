# ------------------ lib_fdrive.py ----------------
# 作用：移动硬盘 F: 的只读数据读取器（全部用 h5py 直读，无需 PyTables）。
# 覆盖四类数据：
#   1. base 日频宽表（行=交易日，列=股票）
#   2. highfreqnew 单日分钟表（行=股票×240 分钟，股票主序）
#   3. index_weight 指数成分权重（组名=指数代码）
#   4. price_index 指数行情（Equity 组）
# 结构细节均已在 因子研究/docs/移动硬盘数据说明.md 附录实测记录。

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lib_common import F_BASE_DIR, F_MINUTE_DIR

# base 宽表统一布局：组名（industry_sw.hdf 例外，用 indus_1/2/3）。
_DEFAULT_GROUP = "data"

# highfreqnew 的 28 个字段（列顺序与 axis0 一致，这里显式列出便于按名取列）。
MINUTE_FIELDS = [
    "开盘价", "收盘价", "最高价", "最低价",
    "买一", "买二", "买三", "买四", "买五",
    "卖一", "卖二", "卖三", "卖四", "卖五",
    "成交量",
    "买一量", "买二量", "买三量", "买四量", "买五量",
    "卖一量", "卖二量", "卖三量", "卖四量", "卖五量",
    "成交笔数", "成交额", "复权因子",
]


def _decode(arr) -> list[str]:
    # h5py 读出的字符串数组是 bytes，统一解码成 str。
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


# ============ 1. base 日频宽表 ============


def read_base_wide(
    name: str,
    group: str = _DEFAULT_GROUP,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    # 读取 F:\base\{name}.hdf 的单个组，返回 DataFrame（行=交易日，列=股票代码）。
    # name  ：文件名（不含 .hdf），如 "close"、"turn"、"zhangting"。
    # group ：组名，默认 "data"；industry_sw.hdf 传 "indus_1"。
    # start ：可选，"YYYY-MM-DD"，只保留该日及以后的行（减小内存）。
    # end   ：可选，"YYYY-MM-DD"，只保留该日及以前的行。
    path = F_BASE_DIR / f"{name}.hdf"
    with h5py.File(path, "r") as f:
        g = f[group]
        # axis0 = 列（股票代码），axis1 = 行（交易日字符串 YYYY-MM-DD）。
        codes = _decode(g["axis0"][:])
        dates = _decode(g["axis1"][:])
        # block0_values：shape=(天数, 股票数) 的 float64 矩阵。
        mat = g["block0_values"][:]

    # 按日期切片：先定位行范围再切矩阵，避免先建大 DataFrame 再筛。
    if start is not None or end is not None:
        keep = np.ones(len(dates), dtype=bool)
        if start is not None:
            keep &= np.array(dates) >= start
        if end is not None:
            keep &= np.array(dates) <= end
        dates = [d for d, k in zip(dates, keep) if k]
        mat = mat[keep]

    df = pd.DataFrame(mat, index=pd.DatetimeIndex(pd.to_datetime(dates)), columns=codes)
    df.index.name = "trade_date"
    return df


# ============ 2. highfreqnew 单日分钟表 ============


class MinuteDay:
    # 单个交易日分钟文件的内存表示。
    # 行 = 股票×240 分钟，股票为主序但顺序与代码列表不一致，
    # 因此定位某只股票必须用 label 数组（lab1=每行的股票下标）。
    def __init__(self, codes, minutes, lab0, lab1, values):
        self.codes = codes        # list[str]：当日全部股票代码（axis1_level1）
        self.minutes = minutes    # list[str]：240 个分钟时间戳（axis1_level0）
        self.lab0 = lab0          # ndarray(int)：每行对应的分钟下标
        self.lab1 = lab1          # ndarray(int)：每行对应的股票下标
        self.values = values      # ndarray(float64)：shape=(行数, 28)

    @property
    def n_stocks(self) -> int:
        return len(self.codes)

    def field_col(self, field: str) -> int:
        # 字段名 -> 列号。
        return MINUTE_FIELDS.index(field)

    def scatter(self, col_idx: int, fill: float = np.nan) -> np.ndarray:
        # 把某一列的值散布成 (股票数, 240) 的二维矩阵。
        # 用 lab1/lab0 显式定位，天然兼容"行序≠股票列表序"的存储方式。
        # 缺失位置（理论上不应出现）保持 fill。
        grid = np.full((self.n_stocks, len(self.minutes)), fill, dtype=np.float64)
        grid[self.lab1, self.lab0] = self.values[:, col_idx]
        return grid


def read_minute_day(path: Path) -> MinuteDay:
    # 读取单个 YYYYMMDD.hdf，返回 MinuteDay。
    # 整块顺序读取（block0_values 一次读入），对 USB 硬盘最友好。
    with h5py.File(path, "r") as f:
        g = f["data"]
        minutes = _decode(g["axis1_level0"][:])
        codes = _decode(g["axis1_level1"][:])
        lab0 = g["axis1_label0"][:]
        lab1 = g["axis1_label1"][:]
        values = g["block0_values"][:]
    return MinuteDay(codes, minutes, lab0, lab1, values)


def list_day_stocks(day: MinuteDay) -> np.ndarray:
    # 返回当日"沪深 A 股"股票代码对应的行位置（股票下标数组）。
    # 剔除北交所（.BJ）与其它非沪深后缀（数据池中只保留 .SH/.SZ）。
    keep = []
    for i, c in enumerate(day.codes):
        if c.endswith(".SH") or c.endswith(".SZ"):
            keep.append(i)
    return np.array(keep, dtype=np.int64)


# ============ 3. index_weight 指数成分权重 ============


def prefix_to_suffix(code: str) -> str:
    # 前缀式代码 -> 后缀式：SH600000 -> 600000.SH，SZ000001 -> 000001.SZ，BJ920000 -> 920000.BJ。
    # index_weight.hdf 的列是前缀式，其余 base 表均为后缀式，合并前必须统一。
    if len(code) == 8 and code[2] != "." and "." not in code:
        return code[2:] + "." + code[:2]
    return code


def read_index_weight(idx_code: str, start: str | None = None) -> pd.DataFrame:
    # 读取某指数的日频成分权重矩阵（行=交易日，列=历史上出现过的全部成分股）。
    # 值=权重百分比（每期有值成分和=100），NaN/0 = 当日不是成分。
    # idx_code 形如 "SH000300"（组名即指数代码）。
    # 列名统一转成后缀式（与其余日频表一致），避免合并时错位。
    df = read_base_wide("index_weight", group=idx_code, start=start)
    df.columns = [prefix_to_suffix(c) for c in df.columns]
    return df


# ============ 4. price_index 指数行情 ============


def read_index_price(columns: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    # 读取 price_index.hdf 的 Equity 组（30 个股票指数日行情，行=日期，列=指数代码）。
    # columns：只保留指定指数列，如 ["000300.SH","000905.SH","000852.SH","000985.CSI"]。
    df = read_base_wide("price_index", group="Equity", start=start)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df
