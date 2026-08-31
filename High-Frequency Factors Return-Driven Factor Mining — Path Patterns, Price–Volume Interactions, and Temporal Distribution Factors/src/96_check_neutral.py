# ------------------ 96_check_neutral.py ----------------
# 验证施密特中性化效果：
#   1) 中性化后因子与 log市值/换手率/波动率 的月截面 Pearson 相关应≈0（施密特保证线性正交）；
#   2) 各行业内部因子均值应≈0（06 全行业哑变量 drop_first=False 后，行业效应应被完整投影掉；
#      若仍用 drop_first=True 缺一维，最小行业代码那组均值会明显偏离 0）。
# 抽 3 个形成月、对 S1/S2、2 个代表因子各算一次。

from __future__ import annotations

import numpy as np
import pandas as pd

from lib_common import PANEL_CACHE_DIR, UNIVERSES, VARIANTS

FACTORS_CHECK = ["A1_tau_20d", "A3_tort_vol_joint_20d"]
STYLE_COLS = ["log_mktcap", "turnover_21d", "volatility_21d"]

panel = pd.read_parquet(PANEL_CACHE_DIR / "monthly_panel.parquet")
panel["form_month"] = panel["form_month"].astype(str)
months = sorted(panel["form_month"].unique())
pick = [months[len(months)//4], months[len(months)//2], months[-1]]
print("抽查月份:", pick)

for uni in UNIVERSES:
    sub = panel[panel[f"in_{uni}"]]
    for v in VARIANTS:
        df = pd.read_parquet(PANEL_CACHE_DIR / "processed" / f"{uni}_{v}.parquet")
        df["form_month"] = df["form_month"].astype(str)
        corrs = {c: [] for c in STYLE_COLS}
        ind_means: list[tuple] = []
        for f in FACTORS_CHECK:
            for m in pick:
                g = df[df["form_month"] == m][["code", f]].dropna()
                s = sub[sub["form_month"] == m][["code"] + STYLE_COLS + ["industry"]]
                j = g.merge(s, on="code").dropna()
                if len(j) < 30:
                    continue
                for c in STYLE_COLS:
                    # 施密特正交保证的是线性正交，用 Pearson 校验（秩相关不作为判据）。
                    fx, fy = j[f], j[c]
                    corrs[c].append(np.corrcoef(fx, fy)[0, 1])
                im = j.groupby("industry")[f].mean()
                if len(im):
                    worst_sec = int(im.abs().idxmax())
                    ind_means.append((f, m, float(im.abs().max()), worst_sec))
        line = "  ".join(f"{c}={np.mean(v_):+.4f}" for c, v_ in corrs.items() if v_)
        if ind_means:
            worst = max(ind_means, key=lambda x: x[2])
            print(f"{uni}/{v}: 风格相关 {line} | 行业均值|max|={worst[2]:.2e}"
                  f"（{worst[0]}@{worst[1]}，行业{worst[3]}）")
        else:
            print(f"{uni}/{v}: {line}")
