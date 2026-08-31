"""研报图表的统一视觉规范。

本模块只负责展示，不改变任何回测数据或统计口径。
"""
from __future__ import annotations

from collections.abc import Sequence

RED = "#A52A32"
BLUE = "#315A7D"
NAVY = "#23384D"
GOLD = "#B08A47"
CHARCOAL = "#343A40"
GRAY = "#6C757D"
LIGHT_GRAY = "#D9DEE3"
GRID = "#E7EAED"
POSITIVE = RED
NEGATIVE = BLUE

# 分组回测使用离散高对比色，而不是连续渐变色。连续渐变在十组曲线
# 密集重叠时难以分辨；这里同时配合不同线型，兼顾投影和灰度打印。
GROUP_COLORS = (
    "#D7191C",  # G1  红
    "#F28E2B",  # G2  橙
    "#EDC948",  # G3  金黄
    "#59A14F",  # G4  绿
    "#008E8E",  # G5  青绿
    "#56B4E9",  # G6  天蓝
    "#0072B2",  # G7  蓝
    "#4B4BA3",  # G8  靛蓝
    "#9467BD",  # G9  紫
    "#CC79A7",  # G10 品红
)

GROUP_LINESTYLES = ("-", "--", "-.", ":", "-", "--", "-.", ":", "--", "-")


def apply_report_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#AEB5BC",
        "axes.labelcolor": CHARCOAL,
        "axes.titlecolor": "#20252A",
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "xtick.color": "#50565C",
        "ytick.color": "#50565C",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.7,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })


def group_palette(n: int) -> Sequence[str]:
    """返回高对比的固定分组色。"""
    if n <= 0:
        return ()
    if n == 1:
        return (CHARCOAL,)
    if n <= len(GROUP_COLORS):
        idx = [round(i * (len(GROUP_COLORS) - 1) / (n - 1)) for i in range(n)]
        return tuple(GROUP_COLORS[i] for i in idx)
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("RdBu_r")
    return tuple(cmap(i / (n - 1)) for i in range(n))


def group_linestyles(n: int) -> Sequence[str]:
    """返回与分组颜色配套的线型，增强曲线重叠时的辨识度。"""
    if n <= 0:
        return ()
    if n <= len(GROUP_LINESTYLES):
        idx = [round(i * (len(GROUP_LINESTYLES) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        return tuple(GROUP_LINESTYLES[i] for i in idx)
    return tuple(GROUP_LINESTYLES[i % len(GROUP_LINESTYLES)] for i in range(n))


def clean_axes(ax, *, zero_line: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if zero_line:
        ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--", zorder=0)
