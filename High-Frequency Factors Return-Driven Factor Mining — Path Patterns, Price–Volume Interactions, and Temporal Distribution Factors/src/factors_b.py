# ------------------ factors_b.py ----------------
# 模块 B：时序特征模块（量价交互的时序方向与力量对比）。
# 依据 研究计划/V2.md §三 与 cc代码plan 因子清单：
#   B1  量价领先滞后   LL = Σ_{k=1..3} (1/k)·[Corr(Δv_{t−k}, r_t) − Corr(r_{t−k}, Δv_t)] / Σ(1/k)
#   B2  路径不对称比   PA = A⁺/(A⁻+ε)，A± = Σ d_t±·a_t·V_t（量版）
#   B2a 同上成交额版   用 Amount_t 替代 V_t（V2 B2 扩展变体）
#   B3  条件化领先滞后 DLL = LL⁺ − LL⁻（上/下行分钟子集各自的滞后1期领先滞后差）
#   B4  力量-动量交叉  Cross = (PA_day − 1) × LL_day
#   B5a 确认度衰减     Decay = ρ_late − ρ_early（滚动60分钟量价相关，10:30 与收盘）
#   B5b 趋势斜率背离   Div = −β̂_p·β̂_v（累计价路径 vs 累计量路径的标准化 OLS 斜率之积）
#   B5c 新高量能不足   VR = Mean(V|累计收益新高分钟) / Mean(V|非高点分钟)

from __future__ import annotations

import numpy as np

from factors_common import DayPack, lag_corr, masked_corr, rolling_corr_endpoints
from lib_common import B1_MAX_LAG, B1_MIN_PAIRS, B3_MIN_OBS, B5A_EARLY_BAR, B5A_ROLL, EPSILON

# B5b：路径回归的最短长度与允许的最大无效分钟占比。
_B5B_MIN_LEN = 20
_B5B_MAX_GAP = 0.10


def _lag_pair_count(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    # (x_{t-k}, y_t) 配对中两边都有效的个数，用于 B1 的最小样本对门槛。
    m = np.isfinite(x[:, :-k]) & np.isfinite(y[:, k:])
    return m.sum(axis=1)


def compute_atoms_b(pack: DayPack) -> dict[str, np.ndarray]:
    # 输入：单交易日全市场 DayPack。输出：B 模块 8 个原子列，(S,) 数组。
    r = pack.r
    S, T = r.shape
    nan = np.full(S, np.nan)

    # ---------- B1：多阶滞后交叉相关 ----------
    # ρ_{+k} = Corr(Δv_{t−k}, r_t)：量领先价；ρ_{−k} = Corr(r_{t−k}, Δv_t)：价领先量。
    # 权重 1/k 按信息衰减；某阶有效对不足 B1_MIN_PAIRS 时该阶不参与加权。
    dv = pack.dvol_pct
    weights = np.array([1.0 / k for k in range(1, B1_MAX_LAG + 1)])
    ll_sum = np.zeros(S)
    w_sum = np.zeros(S)
    for i, k in enumerate(range(1, B1_MAX_LAG + 1)):
        n_k = _lag_pair_count(dv, r, k)
        rho_vp = lag_corr(dv, r, k)          # Corr(Δv_{t-k}, r_t)
        rho_pv = lag_corr(r, dv, k)          # Corr(r_{t-k}, Δv_t)
        term = rho_vp - rho_pv
        ok_k = (n_k >= B1_MIN_PAIRS) & np.isfinite(term)
        ll_sum += np.where(ok_k, weights[i] * term, 0.0)
        w_sum += np.where(ok_k, weights[i], 0.0)
    B1 = np.where(w_sum > 0, ll_sum / np.where(w_sum > 0, w_sum, 1.0), np.nan)

    # ---------- B2 / B2a：路径不对称比 ----------
    # a_t = (H_t−L_t)/C_{t-1}（factors_common 已算）；d± 由 r 的符号给出。
    a_ok = np.isfinite(pack.amp)
    m_up = pack.up & a_ok
    m_dn = pack.down & a_ok
    # A⁺ = Σ d⁺·a·V；A⁻ = Σ d⁻·a·V。量版与成交额版只差一个"量"列。
    A_up_v = np.where(m_up, pack.amp * pack.vol, 0.0).sum(axis=1)
    A_dn_v = np.where(m_dn, pack.amp * pack.vol, 0.0).sum(axis=1)
    A_up_a = np.where(m_up, pack.amp * pack.amount, 0.0).sum(axis=1)
    A_dn_a = np.where(m_dn, pack.amp * pack.amount, 0.0).sum(axis=1)
    # 全天没有任何上/下行有效分钟（价格纹丝不动）→ 无定义。
    has_dir = (m_up.sum(axis=1) + m_dn.sum(axis=1)) > 0
    B2 = np.where(has_dir, A_up_v / (A_dn_v + EPSILON), np.nan)
    B2a = np.where(has_dir, A_up_a / (A_dn_a + EPSILON), np.nan)

    # ---------- B3：条件化领先滞后（分上/下行子集，滞后 1 期） ----------
    # 上行子集：只取 r_t>0 的分钟对；下行子集：只取 r_t<0 的分钟对。
    # 子集内相关仍用滞后 1 期的两个方向之差。
    n_up = pack.up.sum(axis=1)
    n_dn = pack.down.sum(axis=1)
    # corr(Δv_{t-1}, r_t | r_t>0)：mask 作用在配对时间轴上，条件为 y 端 r_t>0。
    up_mask_t = pack.up[:, 1:]     # 对齐到 t=1..T-1
    dn_mask_t = pack.down[:, 1:]
    r1_up = lag_corr(dv, r, 1, mask=up_mask_t)
    r2_up = lag_corr(r, dv, 1, mask=up_mask_t)
    r1_dn = lag_corr(dv, r, 1, mask=dn_mask_t)
    r2_dn = lag_corr(r, dv, 1, mask=dn_mask_t)
    LL_up = r1_up - r2_up
    LL_dn = r1_dn - r2_dn
    ok_b3 = (n_up >= B3_MIN_OBS) & (n_dn >= B3_MIN_OBS)
    B3 = np.where(ok_b3, LL_up - LL_dn, np.nan)

    # ---------- B4：力量-动量交互 ----------
    # Cross_day = (PA_day − 1) × LL_day；PA 或 LL 缺失时自然为 NaN。
    B4 = (B2 - 1.0) * B1

    # ---------- B5a：量价确认度日内衰减 ----------
    # ρ_t = Corr(r_s, ΔV_s) 在 60 分钟滚动窗口上；early=第 60 根 bar（10:30），late=收盘。
    # 注：V2 原文"ρ_90 与 10:30"表述矛盾，按 cc代码plan 决议以 10:30（第 60 bar）落地。
    T_last = T - 1
    rho_early, rho_late = rolling_corr_endpoints(pack.r, pack.dvol, B5A_ROLL, B5A_EARLY_BAR, T_last)
    B5a = rho_late - rho_early

    # ---------- B5b：趋势段量价斜率背离 ----------
    # P_t = cumsum(r)（累计对数收益路径），Q_t = cumsum(V_t − V̄)（累计量偏离路径）。
    # 在 [首个有效 r, 末个有效 r] 区间内做 OLS 斜率，再各自除以路径标准差消量纲。
    idx = np.arange(T, dtype=np.float64)[None, :]
    first = np.argmax(pack.valid_r, axis=1)                    # valid_r 全 False 时 argmax=0，下面用 n_r 过滤
    last = T - 1 - np.argmax(pack.valid_r[:, ::-1], axis=1)
    n_r = pack.valid_r.sum(axis=1)
    in_range = (idx >= first[:, None]) & (idx <= last[:, None])
    rng_len = (last - first + 1).astype(np.float64)
    # 区间内无效分钟占比过高（数据破碎）→ 不给值。
    gap_frac = 1.0 - n_r / np.maximum(rng_len, 1.0)
    ok_range = (n_r > 0) & (rng_len >= _B5B_MIN_LEN) & (gap_frac <= _B5B_MAX_GAP)

    r0 = np.where(pack.valid_r, r, 0.0)
    P = np.cumsum(r0, axis=1)
    vbar = pack.vol.mean(axis=1)
    Q = np.cumsum(pack.vol - vbar[:, None], axis=1)
    # 只保留区间内的点做回归；区间外置 NaN 便于 nanstd/掩码求和。
    P_r = np.where(in_range, P, np.nan)
    Q_r = np.where(in_range, Q, np.nan)

    def _slope_beta(vals: np.ndarray) -> np.ndarray:
        # 对 (t, vals) 的 OLS 斜率，仅用 in_range 内的点。
        x = np.where(in_range, idx, 0.0)
        y = np.where(in_range, vals, 0.0)
        n = in_range.sum(axis=1).astype(np.float64)
        sx = x.sum(axis=1)
        sy = y.sum(axis=1)
        sxy = (x * y).sum(axis=1)
        sx2 = (x * x).sum(axis=1)
        den = n * sx2 - sx * sx
        return np.where(den > 0, (n * sxy - sx * sy) / np.where(den > 0, den, 1.0), np.nan)

    beta_p = _slope_beta(P_r)
    beta_v = _slope_beta(Q_r)
    # 路径标准差（ddof=0）为 0 → 标准化无定义。
    with np.errstate(invalid="ignore"):
        std_p = np.nanstd(np.where(in_range, P_r, np.nan), axis=1)
        std_v = np.nanstd(np.where(in_range, Q_r, np.nan), axis=1)
    ok_std = (std_p > 0) & (std_v > 0)
    bph = beta_p / np.where(std_p > 0, std_p, 1.0)
    bvh = beta_v / np.where(std_v > 0, std_v, 1.0)
    B5b = np.where(ok_range & ok_std, -bph * bvh, np.nan)

    # ---------- B5c：新高量能不足（顶背离） ----------
    # h_t = 1{P_t 创累计收益新高}；VR = 高点分钟平均量 / 非高点分钟平均量。
    P_mask = np.where(pack.valid_r, P, -np.inf)
    run_max = np.maximum.accumulate(P_mask, axis=1)
    h_new = pack.valid_r & (P_mask >= run_max)     # 首个有效 r 分钟也算新高
    n_h = h_new.sum(axis=1)
    n_o = (pack.valid_r & ~h_new).sum(axis=1)
    v_h = np.where(h_new, pack.vol, 0.0).sum(axis=1) / np.maximum(n_h, 1)
    v_o = np.where(pack.valid_r & ~h_new, pack.vol, 0.0).sum(axis=1) / np.maximum(n_o, 1)
    B5c = np.where((n_h > 0) & (n_o > 0) & (v_o > 0), v_h / np.where(v_o > 0, v_o, 1.0), np.nan)

    return {
        "B1_lead_lag_atom": B1,
        "B2_path_asym_atom": B2,
        "B2a_path_asym_amt_atom": B2a,
        "B3_cond_lead_lag_atom": B3,
        "B4_cross_atom": B4,
        "B5a_corr_decay_atom": B5a,
        "B5b_slope_div_atom": B5b,
        "B5c_high_vol_ratio_atom": B5c,
    }
