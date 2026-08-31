"""由既有结果表生成修订版研报图，不重新计算因子或回测。"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib_common import FACTOR_COLS, FACTOR_SHORT, MOD_BOUNDS, UNIVERSE_CN, setup_matplotlib
from report_style import CHARCOAL, GRID, clean_axes

POOLS = ["hs300", "zz500", "zz1000", "zzall"]


def _matrix(summary: pd.DataFrame, value: str, variant: str = "S1") -> np.ndarray:
    sub = summary[summary["variant"].eq(variant)].set_index(["factor", "universe"])
    return np.array([[sub.loc[(f, u), value] for u in POOLS] for f in FACTOR_COLS], dtype=float)


def _heatmap(matrix: np.ndarray, rows: list[str], title: str, cbar_label: str,
             out: Path, *, percent: bool = False, dpi: int = 300) -> None:
    vmax = float(np.nanmax(np.abs(matrix)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig_h = max(6.8, len(rows) * 0.38)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(POOLS)))
    ax.set_xticklabels([UNIVERSE_CN[u] for u in POOLS], fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9, family="DejaVu Sans")
    ax.set_xticks(np.arange(len(POOLS)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            label = f"{v * 100:.1f}%" if percent else f"{v:.2f}"
            color = "white" if abs(v) > 0.58 * vmax else "#252A2E"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.2, color=color)
    ax.set_title(title, fontsize=12, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_rankicir(summary: pd.DataFrame, out_dir: Path) -> None:
    m = _matrix(summary, "rankicir", "S1")
    _heatmap(m, [FACTOR_SHORT[f] for f in FACTOR_COLS],
             "18 个因子在四个股票池的 RankICIR（S1）", "RankICIR",
             out_dir / "fig_1_rankicir_s1.png")

    rows, vals = [], []
    for i, f in enumerate(FACTOR_COLS):
        rows.extend([f"{FACTOR_SHORT[f]}  S1", f"{FACTOR_SHORT[f]}  S2"])
        one = summary[summary.factor.eq(f)].set_index(["variant", "universe"])
        vals.extend([[one.loc[(v, u), "rankicir"] for u in POOLS] for v in ("S1", "S2")])
    _heatmap(np.asarray(vals), rows, "S1 与 S2 的 RankICIR 对照", "RankICIR",
             out_dir / "fig_2_rankicir_s1s2.png")


def plot_effective_return(summary: pd.DataFrame, out_dir: Path) -> None:
    m = _matrix(summary, "ls_eff_ann", "S1")
    _heatmap(m, [FACTOR_SHORT[f] for f in FACTOR_COLS],
             "有效方向多空组合的净年化收益（S1，单边千三）", "净年化收益",
             out_dir / "fig_3_effective_net_return.png", percent=True)


def plot_turnover(summary: pd.DataFrame, out_dir: Path) -> None:
    sub = summary[summary.variant.eq("S1")]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(FACTOR_COLS))
    offsets = np.linspace(-0.27, 0.27, len(POOLS))
    for off, u in zip(offsets, POOLS):
        s = sub[sub.universe.eq(u)].set_index("factor").reindex(FACTOR_COLS)
        ax.scatter(x + off, s["turnover_mean"], s=24, label=UNIVERSE_CN[u], alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels([FACTOR_SHORT[f] for f in FACTOR_COLS], rotation=45, ha="right")
    ax.set_ylabel("单组平均月换手")
    ax.set_title("各因子的单组平均月换手（S1）")
    ax.legend(ncol=4, loc="upper center")
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_4_turnover.png", dpi=220)
    plt.close(fig)


def plot_existing_heatmap(csv_path: Path, title: str, out: Path,
                          vmin: float, vmax: float, fmt: str) -> None:
    mat = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xticks(np.arange(mat.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = float(mat.iloc[i, j])
            if np.isfinite(v):
                color = "white" if abs(v) > 0.58 * max(abs(vmin), abs(vmax)) else "#252A2E"
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=12, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def copy_group_figures(bt_root: Path, out_dir: Path) -> None:
    selected = {
        "fig_5_a3_group_nav.png": ("zz1000", "A3_tort_vol_joint_20d", "S1"),
        "fig_6_b5c_group_nav.png": ("zz1000", "B5c_high_vol_ratio_20d", "S1"),
        "fig_7_c2_group_nav.png": ("zz1000", "C2_entropy_diff_20d", "S1"),
        "fig_8_a2c_group_nav.png": ("zzall", "A2c_high_tau_vol_share_20d", "S1"),
    }
    for name, parts in selected.items():
        src = bt_root.joinpath(*parts, "group_nav.png")
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, out_dir / name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backtest-root", required=True)
    p.add_argument("--style-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    setup_matplotlib()
    bt_root = Path(args.backtest_root).resolve()
    style_root = Path(args.style_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(bt_root / "master_summary.csv")
    if len(summary) != 144:
        raise RuntimeError(f"master_summary 应为 144 行，实际 {len(summary)}")
    plot_rankicir(summary, out_dir)
    plot_effective_return(summary, out_dir)
    plot_turnover(summary, out_dir)
    plot_existing_heatmap(style_root / "style_corr_matrix.csv",
                          "因子与九类风格的月度截面相关（中证全指，S1）",
                          out_dir / "fig_9_style_corr.png", -0.6, 0.6, ".2f")
    plot_existing_heatmap(style_root / "exposure_tvals.csv",
                          "因子风格暴露回归的 t 值",
                          out_dir / "fig_10_exposure_tvals.png", -5.0, 5.0, ".1f")
    copy_group_figures(bt_root, out_dir)
    print(f"研报图已写入 {out_dir}")


if __name__ == "__main__":
    main()
