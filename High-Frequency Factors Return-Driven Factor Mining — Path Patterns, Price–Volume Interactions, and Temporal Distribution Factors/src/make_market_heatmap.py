# -*- coding: utf-8 -*-
"""生成 S1/S2 跨四市场 RankICIR 热力图：18 因子 × 2(变体) × 4(市场)。
   y 轴 = 因子（S1/S2 两行相邻），x 轴 = 四市场；红正蓝负，0 白。"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

csv = PROJECT_ROOT / 'result' / 'style_test' / 'rankicir_s1s2s3.csv'
# CSV 是 pivot 展平：首两行为 (universe, variant) 标签，第 3 行是 'factor' 空行，其后 factor 名+数值。
raw = pd.read_csv(csv, header=None)
unis = raw.iloc[0, 1:].astype(str).tolist()
vars_ = raw.iloc[1, 1:].astype(str).tolist()
data = raw.iloc[3:].reset_index(drop=True)
FACTORS = data[0].astype(str).tolist()
VAL = data.iloc[:, 1:].to_numpy(dtype=float)   # (18, 12)，列序对应 (unis, vars_)

POOLS = ["hs300", "zz500", "zz1000", "zzall"]
POOL_CN = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000", "zzall": "中证全指"}
MOD_BOUNDS = [7, 15, 18]  # A|B|C 边界（×2 因为每因子两行）

# 建立 (pool, variant) -> 列号
col_of = {}
for j, (u, v) in enumerate(zip(unis, vars_)):
    col_of[(u, v)] = j

M = np.full((len(FACTORS) * 2, len(POOLS)), np.nan)
for i, f in enumerate(FACTORS):
    for j, p in enumerate(POOLS):
        if (p, "S1") in col_of:
            M[i * 2, j] = VAL[i, col_of[(p, "S1")]]
        if (p, "S2") in col_of:
            M[i * 2 + 1, j] = VAL[i, col_of[(p, "S2")]]

vmax = max(np.nanmax(np.abs(M)), 0.5)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vmax=vmax, vcenter=0.0)
cmap = plt.get_cmap("RdBu_r")

fig, ax = plt.subplots(figsize=(9, 12))
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(range(len(POOLS)))
ax.set_xticklabels([POOL_CN[p] for p in POOLS], fontsize=11)
ax.set_yticks([i * 2 + 0.5 for i in range(len(FACTORS))])
ax.set_yticklabels(FACTORS, fontsize=10, family="DejaVu Sans")

# 单元格标注
for i in range(len(FACTORS)):
    for j in range(len(POOLS)):
        for k, v in [(0, M[i * 2, j]), (1, M[i * 2 + 1, j])]:
            if not np.isfinite(v):
                continue
            color = "black" if abs(v) < 0.55 * vmax else "white"
            ax.text(j, i * 2 + k, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=color, family="DejaVu Sans")

# S1/S2 行间色带标签
for i in range(len(FACTORS)):
    ax.text(-0.06, i * 2 + 0.72, "S1", ha="right", va="center", fontsize=7, color="gray",
            transform=ax.get_yaxis_transform())
    ax.text(-0.06, i * 2 + 0.28, "S2", ha="right", va="center", fontsize=7, color="gray",
            transform=ax.get_yaxis_transform())

# 模块分隔线
for b in MOD_BOUNDS[:-1]:
    ax.axhline(b * 2 - 0.5, color="black", linewidth=1.1)

ax.set_title("各因子 S1/S2 四市场 RankICIR 对比（红正蓝负；同因子上下两行=市值+行业中性 S1 / 再剥离量价风格 S2）",
             fontsize=12, pad=10)
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("RankICIR", fontsize=10)
fig.tight_layout()

out = PROJECT_ROOT / 'src' / 'report' / 'figs_v21' / 'fig_5_4_market_s1s2.png'
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
