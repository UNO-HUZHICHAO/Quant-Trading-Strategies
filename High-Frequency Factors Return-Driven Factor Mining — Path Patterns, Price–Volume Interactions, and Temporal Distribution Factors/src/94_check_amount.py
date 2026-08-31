# ------------------ 94_check_amount.py ----------------
# 用"成交额"勾稽确认分钟数据完好：成交额处处为真实元、无复权歧义，
# 分钟 Σ成交额 应与 amount.hdf 日成交额精确一致。若一致，则 2016 年成交量比例差异
# 纯属复权基准口径不同（对只用日内相对量的 V2 因子无影响）。

from __future__ import annotations

import numpy as np

from lib_common import F_MINUTE_DIR
from lib_fdrive import read_base_wide, read_minute_day


def check(date: str) -> None:
    day = read_minute_day(F_MINUTE_DIR / f"{date}.hdf")
    amt_col = day.field_col("成交额")
    amt_daily = read_base_wide("amount", start=f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                               end=f"{date[:4]}-{date[4:6]}-{date[6:8]}").iloc[0]
    diffs = []
    n = 0
    for i, code in enumerate(day.codes):
        if not (code.endswith(".SH") or code.endswith(".SZ")):
            continue
        rows = np.where(day.lab1 == i)[0]
        msum = day.values[rows, amt_col].sum()
        dval = amt_daily.get(code, float("nan"))
        if msum > 1e7 and np.isfinite(dval) and dval > 0:
            diffs.append(abs(msum - dval) / dval)
            n += 1
    diffs = np.array(diffs)
    print(f"{date}: 比对 {n} 只 | 相对误差 max={diffs.max():.2e} "
          f"median={np.median(diffs):.2e} | >1e-3 的个数={int((diffs > 1e-3).sum())}")


if __name__ == "__main__":
    check("20160503")
    check("20260731")
