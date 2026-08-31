# ------------------ factors_c.py ----------------
# 模块 C：成交时序集中度模块（交易活动在时间轴上的分布形态）。
# 依据 研究计划/V2.md §四 与 cc代码plan 因子清单：
#   C1 基础成交集中度   H* = H/ln(240)，H = −Σ w_t·ln(w_t)，w_t = V_t/ΣV（V2 §4.2 原定义）
#   原始熵 H 与有量分钟数 N 一并落盘（C1_H_atom / C1_N_atom），便于未来切换归一化口径。
#   C2 价格条件化集中度 ΔH = H⁺ − H⁻（仅上涨分钟 / 仅下跌分钟各自的成交占比熵）
#   C3 = Std20d(H*)，在 04 脚本里用 C1 原子做 20 日滚动标准差，本文件只产出 C1/C2 原子。

from __future__ import annotations

import numpy as np

from factors_common import DayPack


def _entropy_from_weights(wgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 逐股票计算香农熵 H = −Σ w·ln(w)（只对 w>0 求和，约定 0·ln0=0）。
    # 输入 wgt：(S,T)，每行为一组权重（允许行和为 1 以外的值由调用方保证）。
    # 返回 (H, N)：H 为 (S,) 熵值；N 为 (S,) 正权重个数。
    pos = np.isfinite(wgt) & (wgt > 0)
    N = pos.sum(axis=1)
    term = np.where(pos, wgt * np.log(np.where(pos, wgt, 1.0)), 0.0)
    H = -term.sum(axis=1)
    return H, N


def compute_atoms_c(pack: DayPack) -> dict[str, np.ndarray]:
    # 输入：单交易日全市场 DayPack。输出：C1/C2 两个原子列，(S,) 数组。
    S = pack.r.shape[0]

    # ---------- C1：全天成交分布熵 ----------
    # w_t 已由 prepare_day 算好（ΣV<=0 时为 NaN）。N 取"有正成交量"的分钟数。
    H1, N1 = _entropy_from_weights(pack.w)
    # N>=2 才有"分布"可言；按 V2 原定义以 ln(240) 归一化（保留成交稀疏度信息）。
    C1 = np.where((pack.volsum > 0) & (N1 >= 2), H1 / np.log(240.0), np.nan)

    # ---------- C2：上涨分钟 vs 下跌分钟的集中度之差 ----------
    # 上涨分钟成交占比 w⁺_t = V_t·1{r_t>0} / Σ(V·1{r_t>0})，对下跌分钟同理。
    up = pack.up
    dn = pack.down
    vol_up = np.where(up, pack.vol, 0.0)
    vol_dn = np.where(dn, pack.vol, 0.0)
    sum_up = vol_up.sum(axis=1)
    sum_dn = vol_dn.sum(axis=1)
    w_up = vol_up / np.where(sum_up > 0, sum_up, 1.0)[:, None]
    w_dn = vol_dn / np.where(sum_dn > 0, sum_dn, 1.0)[:, None]
    H_up, N_up = _entropy_from_weights(w_up)
    H_dn, N_dn = _entropy_from_weights(w_dn)
    # 任一侧无量或分钟数不足 2 → 该日 ΔH 无定义。
    ok = (sum_up > 0) & (sum_dn > 0) & (N_up >= 2) & (N_dn >= 2)
    C2 = np.where(ok, H_up - H_dn, np.nan)

    return {
        "C1_entropy_atom": C1,
        "C2_entropy_diff_atom": C2,
        "C1_H_atom": H1,      # 原始熵（未归一化），口径切换用
        "C1_N_atom": N1.astype(float),  # 有正成交量的分钟数
    }
