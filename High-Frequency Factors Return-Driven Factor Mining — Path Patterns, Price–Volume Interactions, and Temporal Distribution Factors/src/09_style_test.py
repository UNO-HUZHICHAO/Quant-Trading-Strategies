# ------------------ 09_style_test.py ----------------
# 作用：风格暴露检验（依赖 08 产出的 style_exposure_monthly.parquet）。
#
# Phase 1 风格相关性定位：18 因子 × 9 风格月度截面 Pearson 相关矩阵与热力图。
# Phase 2 S3 九风格中性化：S1 / S2 / S3 三方案 RankICIR 对比，量化借自价值/动量/成长/盈利 vs 低波/换手。
# Phase 3 Barra 式暴露回归：每月截面回归 因子 ~ 9风格 + 行业，输出因子×风格暴露 t 值热力图与稳定性。
# Phase 4 S3 残差纯增量：S3 中性化后残差的 IC/分组（即 Phase 2 的 S3 列）——证明哪些因子扣除全部风格后仍有效。
# Phase 5 合成因子风格体检：A2c/A3、C2+B5c 等权合成，暴露中和与 ICIR 对比。
#
# 输出：result\style_test\ 下相关矩阵 / 热力图 / 对比表。
# 运行：python 09_style_test.py
# 依赖：08 完成（style_exposure_monthly.parquet 存在）。

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib_common import (
    FACTOR_COLS,
    N_GROUPS,
    OUTPUTS_ROOT,
    PANEL_CACHE_DIR,
    UNIVERSES,
    UNIVERSE_CN,
    ensure_dir,
    setup_matplotlib,
)

STYLES_DIR = PANEL_CACHE_DIR.parent / "styles"
STYLE_TEST_DIR = OUTPUTS_ROOT.parent / "style_test"

# 9 风格列（本地自建 5 + 天软 4）。
STYLE_COLS = ["size", "vol", "turn", "mom", "rev", "bp", "dp", "growth", "roe"]
STYLE_CN = {"size": "规模", "vol": "波动率", "turn": "换手率", "mom": "动量",
            "rev": "反转", "bp": "BP", "dp": "DP", "growth": "成长", "roe": "盈利"}

# 变体：S1=市值+行业；S2=+换手+波动；S3=+动量+反转+BP+DP+成长+盈利。
VARIANTS = ("S1", "S2", "S3")
CONT_STYLES = ["size", "turn", "vol"]
S3_EXTRA = ["mom", "rev", "bp", "dp", "growth", "roe"]

# 复用 06 / 07 的核心函数（数字开头模块名用 importlib）。
_06 = importlib.import_module("06_process_factors")
_07 = importlib.import_module("07_backtest_ic_groups")
zscore_series = _06.zscore_series
neutralize_schmidt = _06.neutralize_schmidt
ic_series = _07.ic_series
expanding_icir = _07.expanding_icir


def load_merged_panel() -> pd.DataFrame:
    # 月末面板 + 9 风格暴露，按 (code, form_date) 对齐。
    # 注意：style 的 industry 与 panel 重复（同源），舍弃 style 侧，用 panel 侧（与 S1/S2 一致）。
    panel = pd.read_parquet(PANEL_CACHE_DIR / "monthly_panel.parquet")
    style = pd.read_parquet(STYLES_DIR / "style_exposure_monthly.parquet")
    panel["form_date"] = pd.to_datetime(panel["form_date"])
    style["form_date"] = pd.to_datetime(style["form_date"])
    style = style.drop(columns=["form_month", "industry"])
    merged = panel.merge(style, on=["code", "form_date"], how="left")
    return merged


def neutralize_variant(f: np.ndarray, g: pd.DataFrame, variant: str) -> tuple[np.ndarray, int]:
    # 构造设计矩阵并按 variant 中性化（列序：市值→行业→(S2)换手/波动→(S3)其余 6 风格）。
    z_cap = zscore_series(g["size"]).to_numpy()
    dummies = pd.get_dummies(g["industry"].astype(int), drop_first=False).astype(float)
    cols = [z_cap] + [dummies[c].to_numpy() for c in dummies.columns]
    if variant in ("S2", "S3"):
        cols += [zscore_series(g["turn"]).to_numpy(), zscore_series(g["vol"]).to_numpy()]
    if variant == "S3":
        cols += [zscore_series(g[c]).to_numpy() for c in S3_EXTRA]
    X = np.column_stack(cols)
    resid, n_zero = neutralize_schmidt(f, X)
    sd = resid.std()
    return (resid - resid.mean()) / sd if sd > 0 else resid * 0.0, n_zero


def build_variant_panels(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # 对 4 池 × 18 因子 × 3 方案做中性化，输出 {universe: {variant: 长表}}。
    out = {u: {v: [] for v in VARIANTS} for u in UNIVERSES}
    for u in UNIVERSES:
        sub = merged.loc[merged[f"in_{u}"]].copy()
        for fm, grp in sub.groupby("form_month", sort=True):
            for f in FACTOR_COLS:
                x = grp[f].astype(float)
                okf = x.notna()
                if int(okf.sum()) < 10:
                    continue
                grp_f = grp.loc[okf]
                x = x.loc[okf].to_numpy()
                # MAD 剔除（与 06 一致）。
                med = np.median(x); mad = np.median(np.abs(x - med))
                if np.isfinite(mad) and mad > 0:
                    half = 5 * 1.4826 * mad
                    keep = (x >= med - half) & (x <= med + half)
                else:
                    keep = np.ones(len(x), dtype=bool)
                grp_m = grp_f.loc[keep]; x_m = x[keep]
                if len(grp_m) < 10:
                    continue
                for v in VARIANTS:
                    need = ["size", "industry"] + (["turn", "vol"] if v != "S1" else []) + (S3_EXTRA if v == "S3" else [])
                    style_ok = grp_m[need].notna().all(axis=1)
                    if int(style_ok.sum()) < 10:
                        continue
                    g2 = grp_m.loc[style_ok]; f2 = x_m[style_ok.to_numpy()]
                    z, _ = neutralize_variant(f2, g2, v)
                    out[u][v].append(pd.DataFrame({
                        "code": g2["code"].to_numpy(), "form_date": g2["form_date"].to_numpy(),
                        "form_month": fm, f: z}))
        for v in VARIANTS:
            if out[u][v]:
                out[u][v] = pd.concat(out[u][v], ignore_index=True)
            else:
                out[u][v] = pd.DataFrame(columns=["code", "form_date", "form_month"] + FACTOR_COLS)
    return out


def _with_ret(df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    # 把 next_ret_1m 合并回中性化长表（ic_series 需要该列）。
    return df.merge(merged[["code", "form_date", "next_ret_1m"]], on=["code", "form_date"], how="left")


def ic_summary(vp: dict[str, pd.DataFrame], merged: pd.DataFrame) -> pd.DataFrame:
    # 返回 RankICIR 对比表（factor × universe × variant）。
    rows = []
    for u in UNIVERSES:
        for v in VARIANTS:
            df = _with_ret(vp[u][v], merged)
            for f in FACTOR_COLS:
                if df[f].notna().sum() < 10:
                    continue
                ics = ic_series(df[["form_month", f, "code", "form_date", "next_ret_1m"]].rename(
                    columns={f: "factor"}), "factor")
                if len(ics) == 0:
                    continue
                ric = ics["rank_ic"]
                rows.append({"universe": u, "variant": v, "factor": f,
                             "rankic": ric.mean(), "rankicir": ric.mean() / ric.std() * np.sqrt(12)
                             if ric.std() > 0 else np.nan,
                             "n_months": len(ics)})
    return pd.DataFrame(rows)


def style_corr_matrix(merged: pd.DataFrame, vp: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Phase 1：以 S1 残差因子 vs 9 风格原始值，逐月截面 Pearson 相关，取时序均值（中证全指池）。
    u = "zzall"
    df = vp[u]["S1"].copy()
    df = df.merge(merged[["code", "form_date"] + STYLE_COLS], on=["code", "form_date"], how="left")
    corrs = {f: {sc: [] for sc in STYLE_COLS} for f in FACTOR_COLS}
    for fm, grp in df.groupby("form_month", sort=True):
        for f in FACTOR_COLS:
            a = grp[f].astype(float)
            for sc in STYLE_COLS:
                b = grp[sc].astype(float)
                m = a.notna() & b.notna()
                if m.sum() > 30:
                    corrs[f][sc].append(np.corrcoef(a[m], b[m])[0, 1])
    mat = pd.DataFrame({f: {sc: float(np.mean(corrs[f][sc])) for sc in STYLE_COLS} for f in FACTOR_COLS}).T
    mat.columns = [STYLE_CN[c] for c in mat.columns]
    return mat


def exposure_regression(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Phase 3：每月截面回归 因子(z) ~ 9风格(z) + 行业；输出因子×风格 β 均值表 与 t 值表（中证全指池）。
    u = "zzall"
    sub = merged.loc[merged[f"in_{u}"]].copy()
    betas = {f: {sc: [] for sc in STYLE_COLS} for f in FACTOR_COLS}
    for fm, grp in sub.groupby("form_month", sort=True):
        need = STYLE_COLS + ["industry"]
        g2 = grp.dropna(subset=need).copy()
        if len(g2) < 60:
            continue
        for f in FACTOR_COLS:
            g3 = g2.dropna(subset=[f])
            if len(g3) < 60:
                continue
            y = zscore_series(g3[f]).to_numpy()
            X = np.column_stack([zscore_series(g3[sc]).to_numpy() for sc in STYLE_COLS]
                                + [pd.get_dummies(g3["industry"].astype(int)).to_numpy()])
            X1 = np.column_stack([np.ones(len(X)), X])
            try:
                b = np.linalg.lstsq(X1, y, rcond=None)[0]
            except Exception:
                continue
            for j, sc in enumerate(STYLE_COLS):
                betas[f][sc].append(b[j + 1])
    beta_mean = pd.DataFrame({f: {STYLE_CN[sc]: float(np.nanmean(betas[f][sc])) for sc in STYLE_COLS} for f in FACTOR_COLS}).T
    tvals = pd.DataFrame({f: {STYLE_CN[sc]: (float(np.nanmean(betas[f][sc])) /
                                            (float(np.nanstd(betas[f][sc])) / np.sqrt(len(betas[f][sc]))) if len(betas[f][sc]) > 2 else np.nan)
                              for sc in STYLE_COLS} for f in FACTOR_COLS}).T
    return beta_mean, tvals


def composite_test(merged: pd.DataFrame, vp: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Phase 5：A2c/A3、C2+B5c 等权合成（S1 残差口径），对比单因子与合成的 RankICIR。
    pairs = [("A2c_high_tau_vol_share_20d", "A3_tort_vol_joint_20d", "A2c+A3"),
             ("C2_entropy_diff_20d", "B5c_high_vol_ratio_20d", "C2+B5c")]
    rows = []
    u = "zzall"
    # 长表（每行只填一个因子）转宽表（code×form_date 一行、18 因子各一列）：
    # 同一 (code,form_date) 的 18 行中每因子恰有一个非空，groupby.max 取到该值。
    long = vp[u]["S1"].copy()
    df = long.groupby(["code", "form_date", "form_month"], as_index=False)[FACTOR_COLS].max()
    df = _with_ret(df, merged)
    df["composite_A"] = (df[pairs[0][0]] + df[pairs[0][1]]) / 2
    df["composite_C"] = (df[pairs[1][0]] + df[pairs[1][1]]) / 2
    for f in [pairs[0][0], pairs[0][1], "composite_A", pairs[1][0], pairs[1][1], "composite_C"]:
        ics = ic_series(df[["form_month", f, "code", "form_date", "next_ret_1m"]].rename(
            columns={f: "factor"}), "factor")
        if len(ics) == 0:
            continue
        ric = ics["rank_ic"]
        rows.append({"组合": f, "rankic": ric.mean(),
                     "rankicir": ric.mean() / ric.std() * np.sqrt(12) if ric.std() > 0 else np.nan,
                     "n_months": len(ics)})
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    ensure_dir(STYLE_TEST_DIR)
    setup_matplotlib()
    import matplotlib.pyplot as plt

    print("读取月末面板 + 9 风格暴露 ...")
    merged = load_merged_panel()
    print(f"  merged shape={merged.shape}，风格非空率：", {c: round(float(merged[c].notna().mean()), 3) for c in STYLE_COLS})

    print("构建 S1/S2/S3 中性化面板 ...")
    vp = build_variant_panels(merged)
    ic = ic_summary(vp, merged)
    pivot = ic.pivot_table(index=["factor"], columns=["universe", "variant"], values="rankicir")
    pivot.to_csv(STYLE_TEST_DIR / "rankicir_s1s2s3.csv", encoding="utf-8-sig")
    print("  RankICIR 表 -> result/style_test/rankicir_s1s2s3.csv")
    if "zz1000" in pivot.columns.get_level_values(0):
        print(pivot.xs("zz1000", axis=1, level="universe").round(2).to_string())

    print("Phase 1：风格相关性矩阵（S1 残差 vs 9 风格，中证全指） ...")
    mat = style_corr_matrix(merged, vp)
    mat.to_csv(STYLE_TEST_DIR / "style_corr_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.to_numpy()[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("18 因子 × 9 风格月度截面相关（S1 残差，2016.07–2026.06）")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(STYLE_TEST_DIR / "style_corr_matrix.png", dpi=140)
    plt.close(fig)
    print("  相关矩阵热力图 -> result/style_test/style_corr_matrix.png")

    print("Phase 3：Barra 式暴露回归 ...")
    beta_mean, tvals = exposure_regression(merged)
    beta_mean.to_csv(STYLE_TEST_DIR / "exposure_beta.csv", encoding="utf-8-sig")
    tvals.to_csv(STYLE_TEST_DIR / "exposure_tvals.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(tvals.to_numpy(), cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
    ax.set_xticks(range(tvals.shape[1])); ax.set_xticklabels(tvals.columns, rotation=45, ha="right")
    ax.set_yticks(range(tvals.shape[0])); ax.set_yticklabels(tvals.index, fontsize=8)
    for i in range(tvals.shape[0]):
        for j in range(tvals.shape[1]):
            v = tvals.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("因子×风格暴露 t 值（|t|>2 显著，2016.07–2026.06）")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(STYLE_TEST_DIR / "exposure_tvals_heatmap.png", dpi=140)
    plt.close(fig)
    print("  暴露 t 值热力图 -> result/style_test/exposure_tvals_heatmap.png")

    print("Phase 5：合成因子体检 ...")
    comp = composite_test(merged, vp)
    comp.to_csv(STYLE_TEST_DIR / "composite_test.csv", index=False, encoding="utf-8-sig")
    print(comp.round(3).to_string(index=False))
    print(f"完成，用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    sys.exit(main())
