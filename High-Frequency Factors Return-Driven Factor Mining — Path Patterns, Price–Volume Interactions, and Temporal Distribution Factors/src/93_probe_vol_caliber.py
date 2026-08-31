# ------------------ 93_probe_vol_caliber.py ----------------
# 排查 2016 年分钟成交量合计与 vol.hdf 日成交量的系统性比例差异来源。
# 猜想：分钟量与 vol.hdf 的复权因子基准不一致（常数缩放），验证"比例≈常数"即可放心
#       （因为 V2 因子只用日内相对量 w_t=V_t/ΣV，对常数缩放不敏感）。

from __future__ import annotations

import numpy as np
import pandas as pd

from lib_common import F_MINUTE_DIR
from lib_fdrive import read_base_wide, read_minute_day

DATE = "20160503"


def main() -> None:
    day = read_minute_day(F_MINUTE_DIR / f"{DATE}.hdf")
    vol_col = day.field_col("成交量")
    fac_col = day.field_col("复权因子")

    vol_daily = read_base_wide("vol", start=f"{DATE[:4]}-{DATE[4:6]}-{DATE[6:8]}",
                               end=f"{DATE[:4]}-{DATE[4:6]}-{DATE[6:8]}")
    vd = vol_daily.iloc[0]

    rows_all = []
    for i, code in enumerate(day.codes):
        if not (code.endswith(".SH") or code.endswith(".SZ")):
            continue
        rows = np.where(day.lab1 == i)[0]
        msum = day.values[rows, vol_col].sum()
        fac = day.values[rows[0], fac_col]
        dval = vd.get(code, float("nan"))
        if msum > 1e6 and np.isfinite(dval) and dval > 0:
            rows_all.append((code, msum, dval, msum / dval, fac))

    df = pd.DataFrame(rows_all, columns=["code", "minute_sum", "daily", "ratio", "factor"])
    df = df.sort_values("minute_sum", ascending=False)
    print(f"有效股票数: {len(df)}")
    print("ratio 分布 (minute_sum/daily):")
    print(df["ratio"].describe().round(6).to_string())
    print("\n前 10 大成交量股票:")
    print(df.head(10).to_string(index=False))
    # ratio 与 factor 的相关性：若 ratio≈1/factor 或 ≈factor 的某个倍数，说明是复权基准差异
    print("\nratio 与 复权因子 的相关系数:", round(df["ratio"].corr(df["factor"]), 4))
    print("(ratio*factor) 分布:")
    print((df["ratio"] * df["factor"]).describe().round(6).to_string())


if __name__ == "__main__":
    main()
