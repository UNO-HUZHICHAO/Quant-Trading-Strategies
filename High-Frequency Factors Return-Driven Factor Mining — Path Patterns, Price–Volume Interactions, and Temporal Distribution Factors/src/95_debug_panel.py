# ------------------ 95_debug_panel.py ----------------
# 诊断 05：对单个形成月逐步复现过滤链，打印每步存活样本数，定位清零环节。

from __future__ import annotations

import numpy as np
import pandas as pd

from lib_common import (
    DAILY_CACHE_DIR, FACTOR_COLS, LIST_MIN_DAYS, PANEL_CACHE_DIR,
    SUSP_MAX_DAYS, UNIVERSES, load_trade_calendar,
)

calendar = load_trade_calendar()
cal_s = pd.Series(calendar, index=calendar)
month_last = cal_s.groupby(cal_s.index.strftime("%Y-%m")).max()
fm = "2017-12"
t = month_last[fm]
print("formation date:", t)

w = pd.read_parquet(DAILY_CACHE_DIR / "index_weight_hs300.parquet")
w_row = w.loc[t].dropna()
mem = set(w_row[w_row > 0].index)
codes = sorted(mem)
print("members:", len(codes), "sample:", codes[:3])

zt = pd.read_parquet(DAILY_CACHE_DIR / "zhangting.parquet")
dt_ = pd.read_parquet(DAILY_CACHE_DIR / "dieting.parquet")
tp = pd.read_parquet(DAILY_CACHE_DIR / "tingpai.parquet")
st = pd.read_parquet(DAILY_CACHE_DIR / "stpanduan.parquet")


def row_of(frame):
    r = frame.loc[t] if t in frame.index else pd.Series(dtype=float)
    return r.reindex(codes)


prod = (1 - row_of(zt)) * (1 - row_of(dt_)) * row_of(tp) * (1 - row_of(st))
tradable = prod == 1
print("prod 非NaN个数:", int(prod.notna().sum()), "tradable 个数:", int(tradable.sum()))
print("prod 样例:", prod.head(3).tolist())

bounds = pd.read_parquet(DAILY_CACHE_DIR / "stock_bounds.parquet").set_index("code")
first_valid = pd.to_datetime(bounds["first_valid"])
fv = first_valid.reindex(codes)
print("fv 非NaN个数:", int(fv.notna().sum()), "样例:", fv.head(3).tolist())
cal_pos = pd.Series(np.arange(len(calendar)), index=calendar)
age = cal_pos.loc[t] - cal_pos.reindex(fv)
listed_ok = age >= LIST_MIN_DAYS
print("listed_ok 个数:", int(listed_ok.sum()))

month_rows = tp.loc[tp.index.strftime("%Y-%m") == fm]
susp_days = (month_rows == 0).sum(axis=0).reindex(codes)
susp_ok = susp_days <= SUSP_MAX_DAYS
print("susp_ok 个数:", int(susp_ok.sum()))

keep = tradable & listed_ok & susp_ok.fillna(False)
kept = [c for c, k in zip(codes, keep) if k]
print("keep 后存活:", len(kept))

# 因子 / 风格 / 下期收益可用性
fdir = PANEL_CACHE_DIR / "factors_daily"
f = pd.read_parquet(fdir / "A1_tau_20d.parquet")
fv2 = f.loc[t].reindex(kept) if t in f.index else None
print("kept 中 A1 非NaN:", int(fv2.notna().sum()) if fv2 is not None else "no t")
close = pd.read_parquet(DAILY_CACHE_DIR / "close.parquet")
ind = pd.read_parquet(DAILY_CACHE_DIR / "industry.parquet")
cap = pd.read_parquet(DAILY_CACHE_DIR / "cap.parquet")
print("close@t 非NaN(kept):", int(close.loc[t].reindex(kept).notna().sum()))
print("industry@t 非NaN(kept):", int(ind.loc[t].reindex(kept).notna().sum()))
print("cap@t 非NaN(kept):", int(cap.loc[t].reindex(kept).notna().sum()))
all_last = month_last.sort_index()
t_next = all_last.shift(-1).loc[fm]
print("t_next:", t_next, "close@t_next 非NaN(kept):",
      int(close.loc[t_next].reindex(kept).notna().sum()) if t_next in close.index else "missing")
