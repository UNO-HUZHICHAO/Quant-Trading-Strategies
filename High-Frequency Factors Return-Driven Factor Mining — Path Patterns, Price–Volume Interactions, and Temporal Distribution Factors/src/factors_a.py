# ------------------ factors_a.py ----------------
# 模块 A：微观结构模块（价格路径的"纹理"与微观交易质量）。
# 依据 研究计划/V2.md §二 与 cc代码plan 因子清单：
#   A1  基础曲折度        τ = Σ|r_t| / (|Σr_t| + ε)
#   A1v 量加权曲折度变体  τ_v = Σ(|r_t|·V_t/V̄) / (|Σr_t| + ε)
#   A2a 高曲折时段收益波动 std({r_t : d_t=1})
#   A2b 低曲折时段累计收益 Σ(1−d_t)·r_t
#   A2c 高曲折时段成交占比 Σ d_t·w_t   （A2d 的乘子部分，换手率在 04 合成）
#   A3  曲折-放量联合占比 Σ g_t / T，g_t = d_t·e_t
# 切割定义：d_t = 1{τ_window(t) > Median(τ_window)}，τ_window 为 30 分钟滚动曲折度，
#           由 factors_common.prepare_day 预先算好（pack.tau_win / pack.d_high）。

from __future__ import annotations

import numpy as np

from factors_common import DayPack
from lib_common import EPSILON


def compute_atoms_a(pack: DayPack) -> dict[str, np.ndarray]:
    # 输入：单交易日全市场 DayPack（S 只股票 × 240 分钟）。
    # 输出：dict，键为原子列名，值为 (S,) 数组；定义不满足处为 NaN。
    r = pack.r
    vr = pack.valid_r
    S = r.shape[0]
    nan = np.full(S, np.nan)

    # ---------- A1 / A1v：总量曲折度 ----------
    # 路径长度 L = Σ|r_t|；净位移 D = |Σr_t|。只对有效收益分钟求和。
    n_r = vr.sum(axis=1)
    abs_r = np.where(vr, np.abs(r), 0.0)
    r_ = np.where(vr, r, 0.0)
    L = abs_r.sum(axis=1)
    D = np.abs(r_.sum(axis=1))
    # 至少要有 2 个有效分钟收益，否则路径无定义。
    ok_basic = n_r >= 2

    # A1v 的量加权：V̄ = 有效收益分钟上的平均成交量。
    vol_r = np.where(vr, pack.vol, 0.0)
    vbar = np.where(n_r > 0, vol_r.sum(axis=1) / np.maximum(n_r, 1), 0.0)
    # L_v = Σ |r_t|·(V_t/V̄)；V̄=0（全天无量）时 A1v 无定义。
    Lv = np.where(vbar > 0, (abs_r * vol_r / np.where(vbar > 0, vbar, 1.0)[:, None]).sum(axis=1), np.nan)

    A1 = np.where(ok_basic, L / (D + EPSILON), np.nan)
    A1v = np.where(ok_basic & (vbar > 0), Lv / (D + EPSILON), np.nan)

    # ---------- A2a：高曲折时段的收益波动 ----------
    # 注意：d_high 所在分钟必然有有效 r（滚动窗口完整要求其右端点 r 有效）。
    m_high = pack.d_high
    n_high = m_high.sum(axis=1)
    r_high = np.where(m_high, r, 0.0)
    mean_high = r_high.sum(axis=1) / np.maximum(n_high, 1)
    # 样本标准差（ddof=1）：至少 2 个高曲折分钟才有定义。
    var_high = np.where(m_high, (r - mean_high[:, None]) ** 2, 0.0).sum(axis=1) / np.maximum(n_high - 1, 1)
    A2a = np.where(n_high >= 2, np.sqrt(var_high), np.nan)

    # ---------- A2b：低曲折时段的累计收益 ----------
    # 在切割有定义的分钟（win_ok）内取低曲折部分；全天无有效窗口 → NaN。
    m_low = pack.win_ok & ~pack.d_high
    n_low = m_low.sum(axis=1)
    A2b = np.where(n_low > 0, np.where(m_low, r, 0.0).sum(axis=1), np.nan)

    # ---------- A2c：高曲折时段占全天成交的比重 ----------
    # w_t = V_t/ΣV 为全日内量占比；Σ d_t·w_t 即"高曲折时段集中了多少成交"。
    sum_dw = np.where(m_high, pack.w, 0.0).sum(axis=1)
    A2c = np.where(pack.volsum > 0, sum_dw, np.nan)

    # ---------- A3：高曲折且放量的分钟占比 ----------
    # e_t = 1{V_t > μ_V + σ_V}（μ、σ 为全天 240 分钟成交量的均值/标准差）；
    # g_t = d_t·e_t；分母 T = 切割有定义的分钟数（与 d_t 的定义域一致）。
    mu_v = pack.vol.mean(axis=1)
    sd_v = pack.vol.std(axis=1)
    e_vol = pack.vol > (mu_v + sd_v)[:, None]
    T_win = pack.win_ok.sum(axis=1)
    g = (pack.d_high & e_vol).sum(axis=1)
    A3 = np.where(T_win > 0, g / np.maximum(T_win, 1), np.nan)

    return {
        "A1_tau_atom": A1,
        "A1v_tau_vol_atom": A1v,
        "A2a_high_tau_ret_std_atom": A2a,
        "A2b_low_tau_cum_ret_atom": A2b,
        "A2c_high_tau_vol_share_atom": A2c,
        "A3_tort_vol_joint_atom": A3,
    }
