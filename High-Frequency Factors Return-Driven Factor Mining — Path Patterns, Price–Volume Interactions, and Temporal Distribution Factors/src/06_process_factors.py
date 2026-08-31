# ------------------ 06_process_factors.py ----------------
# 作用：对月末面板逐「股票池×形成月×因子」做因子处理（各因子样本彼此独立）：
#   0) 该因子自己的样本：因子值非空（P2 修复后 05 不再统一 dropna，各因子样本独立）；
#   1) 5×MAD 极值剔除：med ± 5×1.4826×MAD 越界 → 直接删除该样本（不缩尾、不填充）；
#   2) 施密特正交化（修正 Gram-Schmidt）中性化，两个方案对比：
#        S1 = z(log流通市值) + 申万一级行业哑变量（全哑变量，秩亏列由 Gram-Schmidt 跳过）
#        S2 = S1 + z(21日换手率) + z(21日波动率)
#      列序固定为 市值 → 行业 →(换手 → 波动)，顺序施密特；
#      正交基 U 由修正 Gram-Schmidt 得到，残差 = f − U(Uᵀf)（与 QR 投影等价，秩亏列自动跳过）；
#   3) 残差截面 zscore。
#
# 输出（D:\hf_factor_cache\panels\processed\）：
#   {universe}_{variant}.parquet：长表（code, form_date, form_month + 18 因子列），
#     每行只填该样本存活的那个因子值、其余因子列为 NaN（同一 (code, form_month) 按因子重复多次）。
#     07 读取时按因子 dropna，故长表是"每因子独立样本"的有意设计。
#   neutralize_stats.csv：各截面样本量与秩亏列数记录
#
# 运行：python 06_process_factors.py

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from lib_common import (
    FACTOR_COLS,
    MAD_K,
    MAD_SCALE,
    PANEL_CACHE_DIR,
    UNIVERSES,
    VARIANTS,
    ensure_dir,
)

PROCESSED_DIR = PANEL_CACHE_DIR / "processed"

# 连续风格暴露列（进入设计矩阵前做截面 z 化，仅为数值稳定）。
CONT_STYLES = ["log_mktcap", "turnover_21d", "volatility_21d"]


def zscore_series(s: pd.Series) -> pd.Series:
    # 截面 z 化；std=0 时返回全 0（常数暴露不携带信息）。
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - s.mean()) / sd


def gram_schmidt_basis(X: np.ndarray) -> np.ndarray:
    # 修正 Gram-Schmidt：把 X 的列按顺序正交化、单位化，返回正交基 U（n×p）。
    # 秩亏列（与前面列线性相关）范数≈0，置零列跳过——等价于只对列空间做投影。
    U = np.zeros_like(X)
    for j in range(X.shape[1]):
        v = X[:, j].astype(np.float64).copy()
        for k in range(j):
            if U[:, k].any():
                v -= np.dot(U[:, k], v) * U[:, k]
        nrm = np.linalg.norm(v)
        if nrm > 1e-10:
            U[:, j] = v / nrm
    return U


def neutralize_schmidt(f: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, int]:
    # 施密特正交化中性化：f 对 X 列空间投影取残差。
    # 返回 (残差, 秩亏被跳过的列数)。
    U = gram_schmidt_basis(X)
    n_zero = int((np.abs(U).sum(axis=0) == 0).sum())
    resid = f - U @ (U.T @ f)
    return resid, n_zero


def process_one_universe(sub: pd.DataFrame, universe: str, stat_rows: list) -> dict[str, pd.DataFrame]:
    # sub：某股票池的全部月末行（含 form_month、风格列、18 因子列）。
    # 逐「形成月 × 因子」独立处理：每因子的存活样本互不影响（P2 修复后各因子样本独立）。
    # 返回 {variant: DataFrame(code, form_date, form_month + 18 因子，长表)}。
    out = {v: [] for v in VARIANTS}
    for fm, grp in sub.groupby("form_month", sort=True):
        for f in FACTOR_COLS:
            # ---- 0) 该因子自己的样本：因子值非空（收益非空已由 05 的 req 保证） ----
            x = grp[f].astype(float)
            okf = x.notna()
            n_nan = int((~okf).sum())
            grp_f = grp.loc[okf]
            x = x.loc[okf].to_numpy()
            n_before = len(grp_f)
            if n_before == 0:
                continue
            # ---- 1) MAD 剔除（越界样本直接删，不缩尾） ----
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            if np.isfinite(mad) and mad > 0:
                half = MAD_K * MAD_SCALE * mad
                keep_mad = (x >= med - half) & (x <= med + half)
            else:
                keep_mad = np.ones(n_before, dtype=bool)
            x_m = x[keep_mad]
            grp_m = grp_f.loc[keep_mad]
            n_after = int(keep_mad.sum())
            if n_after < 10:
                # 样本太少无法可靠中性化：该截面该因子记空。
                stat_rows.append({"universe": universe, "form_month": fm, "factor": f,
                                  "n_before": n_before, "n_nan": n_nan,
                                  "n_mad_drop": n_before - n_after,
                                  "n_final": 0, "note": "too few after MAD"})
                continue
            # ---- 2) 各变体独立构建设计矩阵并中性化 ----
            for v in VARIANTS:
                need_cols = ["log_mktcap", "industry"]
                if v == "S2":
                    need_cols += ["turnover_21d", "volatility_21d"]
                style_ok = grp_m[need_cols].notna().all(axis=1)
                if int(style_ok.sum()) < 10:
                    stat_rows.append({"universe": universe, "form_month": fm, "factor": f, "variant": v,
                                      "n_before": n_before, "n_nan": n_nan,
                                      "n_mad_drop": n_before - n_after,
                                      "n_final": 0, "note": "too few after style dropna"})
                    continue
                g2 = grp_m.loc[style_ok]
                f2 = x_m[style_ok.to_numpy()]
                # 市值 + 行业（全哑变量，drop_first=False；线性相关的秩亏列由 Gram-Schmidt 跳过）。
                z_cap = zscore_series(g2["log_mktcap"]).to_numpy()
                dummies = pd.get_dummies(g2["industry"].astype(int), drop_first=False).astype(float)
                cols = [z_cap] + [dummies[c].to_numpy() for c in dummies.columns]
                if v == "S2":
                    cols += [zscore_series(g2["turnover_21d"]).to_numpy(),
                             zscore_series(g2["volatility_21d"]).to_numpy()]
                X = np.column_stack(cols)
                resid, n_zero = neutralize_schmidt(f2, X)
                # ---- 3) 残差 zscore ----
                sd = resid.std()
                z = (resid - resid.mean()) / sd if sd > 0 else resid * 0.0
                rec = pd.DataFrame({
                    "code": g2["code"].to_numpy(),
                    "form_date": g2["form_date"].to_numpy(),
                    "form_month": fm,
                    f: z,
                })
                out[v].append(rec)
                stat_rows.append({"universe": universe, "form_month": fm, "factor": f, "variant": v,
                                  "n_before": n_before, "n_nan": n_nan,
                                  "n_mad_drop": n_before - n_after,
                                  "n_final": len(g2), "n_rankdef": n_zero})
    merged = {}
    for v in VARIANTS:
        if not out[v]:
            merged[v] = pd.DataFrame(columns=["code", "form_date", "form_month"] + FACTOR_COLS)
            continue
        # 长表：同一 (code, form_month) 按因子重复多次，每行只填该因子值（07 按因子 dropna，兼容）。
        m = pd.concat(out[v], ignore_index=True)
        merged[v] = m.reindex(columns=["code", "form_date", "form_month"] + FACTOR_COLS)
    return merged


def main() -> None:
    t0 = time.time()
    ensure_dir(PROCESSED_DIR)
    panel = pd.read_parquet(PANEL_CACHE_DIR / "monthly_panel.parquet")
    panel["form_month"] = panel["form_month"].astype(str)
    print(f"读入月末面板 shape={panel.shape}")

    stat_rows: list[dict] = []
    for u in UNIVERSES:
        sub = panel.loc[panel[f"in_{u}"]].copy()
        print(f"[{u}] 样本行数 {len(sub)}")
        merged = process_one_universe(sub, u, stat_rows)
        for v in VARIANTS:
            df = merged[v]
            if df.empty:
                continue
            path = PROCESSED_DIR / f"{u}_{v}.parquet"
            df.to_parquet(path, index=False)
            print(f"  [{u}/{v}] shape={df.shape} -> {path.name}")
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(PROCESSED_DIR / "neutralize_stats.csv", index=False, encoding="utf-8-sig")
    print(f"完成，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
