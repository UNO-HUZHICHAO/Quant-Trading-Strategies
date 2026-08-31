# -*- coding: utf-8 -*-
"""生成 18 因子 × 4 股票池 RankICIR（S1）热力图。"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 中文字体（Windows）
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

with open(PROJECT_ROOT / "result" / "bt_summary.json", encoding="utf-8") as f:
    data = json.load(f)

POOLS = ["hs300", "zz500", "zz1000", "zzall"]
POOL_CN = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000", "zzall": "中证全指"}
FACTORS = ["A1", "A1v", "A2a", "A2b", "A2c", "A2d", "A3",
           "B1", "B2", "B2a", "B3", "B4", "B5a", "B5b", "B5c",
           "C1", "C2", "C3"]
MOD_BOUNDS = [7, 15, 18]  # A | B | C 分组边界

M = np.zeros((len(FACTORS), len(POOLS)))
for i, f in enumerate(FACTORS):
    for j, p in enumerate(POOLS):
        v = float(data[p][f]["S1"]["rankicir"])
        M[i, j] = v

vmax = max(abs(M.min()), abs(M.max()))  # 对称色标，0 为白
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vmax=vmax, vcenter=0.0)
cmap = plt.get_cmap("RdBu_r")  # 负=蓝（弱），正=红（强）

fig, ax = plt.subplots(figsize=(8.6, 7.6))
im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

# 网格线（白色细线）
ax.set_xticks(range(len(POOLS)))
ax.set_xticklabels([POOL_CN[p] for p in POOLS], fontsize=11)
ax.set_yticks(range(len(FACTORS)))
ax.set_yticklabels(FACTORS, fontsize=10, family="DejaVu Sans")

# 单元格标注
for i in range(len(FACTORS)):
    for j in range(len(POOLS)):
        v = M[i, j]
        color = "black" if abs(v) < 0.55 * vmax or vmax < 0.5 else "white"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=9.5, color=color, family="DejaVu Sans")

# 模块分隔线
for b in MOD_BOUNDS[:-1]:
    ax.axhline(b - 0.5, color="black", linewidth=1.1)

# 模块标签
ax.text(-0.62, 3, "A 微观结构", rotation=90, ha="center", va="center", fontsize=12,
        transform=ax.get_yaxis_transform(), fontweight="bold")
ax.text(-0.62, 11, "B 时序特征", rotation=90, ha="center", va="center", fontsize=12,
        transform=ax.get_yaxis_transform(), fontweight="bold")
ax.text(-0.62, 16.5, "C 成交集中度", rotation=90, ha="center", va="center", fontsize=12,
        transform=ax.get_yaxis_transform(), fontweight="bold")

ax.set_title("18 因子 RankICIR 全景（S1 市值+行业中性，2016.07–2026.06）", fontsize=13, pad=12)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("RankICIR", fontsize=10)

fig.tight_layout()
out = PROJECT_ROOT / "src" / "report" / "figs_v21" / "fig_5_1_rankicir_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
