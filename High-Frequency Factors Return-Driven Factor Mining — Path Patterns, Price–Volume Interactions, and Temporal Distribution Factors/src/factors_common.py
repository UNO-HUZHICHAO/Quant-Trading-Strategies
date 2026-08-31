# ------------------ factors_common.py ----------------
# 作用：三模块原子因子共享的"日内派生量"预处理与向量化数值工具。
# 输入：单交易日全市场矩阵（股票 S × 240 分钟 T），由 03 脚本从 hdf 散布得到。
# 输出：DayPack（一组 S×T 数组），供 factors_a/b/c.py 直接取用。
# 设计原则：全部 numpy 沿 axis=1 向量化；无效分钟（收盘价缺失/非正）以 NaN 传播，
#           让每个因子按自己的定义决定如何处理，不在这里偷偷填充。

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib_common import EPSILON, TAU_WINDOW


@dataclass
class DayPack:
    # 单股单日派生量容器（每个字段都是 shape=(S, T) 的数组，除非注明）。
    close: np.ndarray        # 收盘价（后复权）
    high: np.ndarray         # 最高价
    low: np.ndarray          # 最低价
    vol: np.ndarray          # 成交量（数据口径=真实股数/复权因子；日内份额不受影响）
    amount: np.ndarray       # 成交额（真实元）
    valid: np.ndarray        # bool：有效分钟（收盘价有限且 >0）
    n_valid: np.ndarray      # (S,) 每只股票有效分钟数
    r: np.ndarray            # 分钟对数收益 ln(C_t/C_{t-1})；首列与相邻无效处为 NaN
    valid_r: np.ndarray      # bool：r 有效的位置
    amp: np.ndarray          # 分钟振幅 a_t=(H_t-L_t)/C_{t-1}
    volsum: np.ndarray       # (S,) 当日成交量合计
    w: np.ndarray            # 分钟量占比 w_t=V_t/ΣV（ΣV<=0 处为 NaN）
    up: np.ndarray           # bool：r>0
    down: np.ndarray         # bool：r<0
    dvol: np.ndarray         # ΔV_t = V_t − V_{t-1}（首列 NaN）
    dvol_pct: np.ndarray      # Δv_t = (V_t−V_{t-1})/V_{t-1}（V_{t-1}<=0 处 NaN）
    tau_win: np.ndarray      # 30 分钟滚动曲折度 τ_window(t)，窗口不满/含无效 r 处为 NaN
    tau_med: np.ndarray      # (S,) 当日 τ_window 的中位数（切割阈值）
    win_ok: np.ndarray       # bool：τ_window 有定义的位置（即切割 d_t 有定义的位置）
    d_high: np.ndarray       # bool：高曲折分钟（τ_window > 当日中位数）


def _rolling_sum(arr: np.ndarray, win: int) -> np.ndarray:
    # 沿 axis=1 的滚动求和：输出 (S, T-win+1)，out[:, t] = arr[:, t : t+win] 之和。
    # 调用前需自行把 NaN 处理成 0（并配合有效计数判断窗口完整性）。
    cs = np.concatenate([np.zeros((arr.shape[0], 1)), np.cumsum(arr, axis=1)], axis=1)
    return cs[:, win:] - cs[:, :-win]


def prepare_day(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vol: np.ndarray,
    amount: np.ndarray,
) -> DayPack:
    # 由 OHLCV+成交额六个 (S,T) 原始矩阵构造 DayPack。
    # 无效分钟定义：收盘价不是有限正数（停牌冻结行的价格虽是常数但量全零，
    # 会在 03 层以"日成交量=0"整日剔除；这里只负责逐分钟的有效性）。

    # ---- 有效分钟与对数收益 ----
    valid = np.isfinite(close) & (close > 0)
    n_valid = valid.sum(axis=1)
    logc = np.where(valid, np.log(np.where(valid, close, 1.0)), np.nan)
    # r_t = ln(C_t/C_{t-1})：要求相邻两分钟都有效，否则 NaN。
    r = np.full_like(close, np.nan)
    both = valid[:, 1:] & valid[:, :-1]
    r[:, 1:] = np.where(both, logc[:, 1:] - logc[:, :-1], np.nan)
    valid_r = np.isfinite(r)

    # ---- 振幅 a_t = (H_t − L_t) / C_{t-1}（分母用前一分钟收盘） ----
    prev_close = np.full_like(close, np.nan)
    prev_close[:, 1:] = close[:, :-1]
    prev_ok = np.isfinite(prev_close) & (prev_close > 0)
    hl_ok = np.isfinite(high) & np.isfinite(low)
    amp = np.where(prev_ok & hl_ok, (high - low) / np.where(prev_ok, prev_close, 1.0), np.nan)

    # ---- 量占比与方向 ----
    vol_safe = np.where(np.isfinite(vol), vol, 0.0)
    volsum = vol_safe.sum(axis=1)
    w = np.where((volsum > 0)[:, None], vol_safe / np.where(volsum > 0, volsum, 1.0)[:, None], np.nan)
    up = valid_r & (r > 0)
    down = valid_r & (r < 0)

    # ---- 成交量变化（B 模块用） ----
    dvol = np.full_like(vol, np.nan)
    dvol[:, 1:] = vol_safe[:, 1:] - vol_safe[:, :-1]
    prev_vol = np.full_like(vol, np.nan)
    prev_vol[:, 1:] = vol_safe[:, :-1]
    # 前一分钟量为 0 时变化率无定义（NaN），避免 inf 污染相关系数。
    dvol_pct = np.where(prev_vol > 0, dvol / np.where(prev_vol > 0, prev_vol, 1.0), np.nan)

    # ---- 30 分钟滚动曲折度（A2/A3 的切割依据） ----
    # τ_window(t) 用窗口 [t-29, t] 内的 r 计算：L30/(|D30|+ε)。
    # 要求窗口内 29 个 r（窗口跨 30 根 bar，产生 29 个相邻收益）全部有效，否则 NaN。
    T = close.shape[1]
    n_win_ret = TAU_WINDOW - 1  # 一个窗口内的收益个数 = 29
    tau_win = np.full_like(close, np.nan)
    if T >= TAU_WINDOW:
        r_abs0 = np.where(valid_r, np.abs(r), 0.0)
        r_0 = np.where(valid_r, r, 0.0)
        cnt = valid_r.astype(np.float64)
        # 滚动和输出下标 t 对应窗口 [t, t+29]（左端点记法），长度 T-29。
        L30 = _rolling_sum(r_abs0, n_win_ret)      # (S, T-29)
        D30 = _rolling_sum(r_0, n_win_ret)
        C30 = _rolling_sum(cnt, n_win_ret)
        tw = np.where(C30 >= n_win_ret, L30 / (np.abs(D30) + EPSILON), np.nan)
        # 把窗口结果放回"窗口右端点"分钟上：
        # tw[j] 对应 bar 窗口 [j, j+28] 内的 29 个收益，右端点 = j + 28。
        tau_win[:, n_win_ret - 1:] = tw
    win_ok = np.isfinite(tau_win)
    # 当日切割阈值 = τ_window 的中位数；全天无有效窗口时为 NaN（该股 A2/A3 记 NaN）。
    tau_med = np.where(win_ok.any(axis=1), np.nanmedian(tau_win, axis=1), np.nan)
    d_high = win_ok & (tau_win > tau_med[:, None])

    return DayPack(
        close=close, high=high, low=low, vol=vol_safe, amount=np.where(np.isfinite(amount), amount, 0.0),
        valid=valid, n_valid=n_valid, r=r, valid_r=valid_r, amp=amp,
        volsum=volsum, w=w, up=up, down=down, dvol=dvol, dvol_pct=dvol_pct,
        tau_win=tau_win, tau_med=tau_med, win_ok=win_ok, d_high=d_high,
    )


# ============ 数值工具（沿 axis=1 向量化） ============


def masked_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    # 逐股票计算 corr(x_t, y_t)（沿 axis=1），仅使用 mask 内且两列均有限的位置。
    # 返回 (S,)；有效对数 <2 或方差为 0 → NaN。
    m = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        m &= mask
    n = m.sum(axis=1).astype(np.float64)
    xs = np.where(m, x, 0.0)
    ys = np.where(m, y, 0.0)
    sx = xs.sum(axis=1)
    sy = ys.sum(axis=1)
    sxy = (xs * ys).sum(axis=1)
    sx2 = (xs * xs).sum(axis=1)
    sy2 = (ys * ys).sum(axis=1)
    cov = n * sxy - sx * sy
    vx = n * sx2 - sx * sx
    vy = n * sy2 - sy * sy
    den = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
    ok = (n >= 2) & (den > 0)
    return np.where(ok, cov / np.where(ok, den, 1.0), np.nan)


def lag_corr(x: np.ndarray, y: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    # 逐股票 corr(x_{t-k}, y_t)（t=k..T-1），用于领先滞后结构。
    # 配对后时间轴长度为 T-k（xs=x[:, :-k]，ys=y[:, k:]）。
    # 传入的 mask 必须已与该配对轴对齐（长度 T-k），由调用方负责切好。
    xs = x[:, :-k]
    ys = y[:, k:]
    return masked_corr(xs, ys, mask)


def rolling_corr_endpoints(x: np.ndarray, y: np.ndarray, win: int, early: int, late: int) -> tuple[np.ndarray, np.ndarray]:
    # 滚动窗口相关系数在两个时点的取值（B5a 专用，避免算整条滚动序列）。
    # ρ_t = corr(x_s, y_s)_{s∈[t-win+1, t]}，仅在 pair 全部有效的窗口上有定义。
    # 返回 (ρ_early, ρ_late)，各自 (S,)。
    S, T = x.shape
    m = np.isfinite(x) & np.isfinite(y)
    xs = np.where(m, x, 0.0)
    ys = np.where(m, y, 0.0)
    cnt = m.astype(np.float64)
    # 滚动和：输出下标 j 对应窗口 [j, j+win-1]（左端点记法），长度 T-win+1。
    Sc = _rolling_sum(cnt, win)
    Sx = _rolling_sum(xs, win)
    Sy = _rolling_sum(ys, win)
    Sxy = _rolling_sum(xs * ys, win)
    Sx2 = _rolling_sum(xs * xs, win)
    Sy2 = _rolling_sum(ys * ys, win)
    cov = Sc * Sxy - Sx * Sy
    vx = Sc * Sx2 - Sx * Sx
    vy = Sc * Sy2 - Sy * Sy
    den = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
    # 窗口至少一半有效才给值（缺几分钟不应让整个因子 NaN）。
    ok = (Sc >= win // 2) & (den > 0)
    rho = np.where(ok, cov / np.where(ok, den, 1.0), np.nan)
    # 右端点 t 对应左端点 j = t-win+1。
    j_early = early - win + 1
    j_late = late - win + 1
    if j_early < 0 or j_late < 0 or j_late >= rho.shape[1]:
        nan = np.full(S, np.nan)
        return nan, nan
    return rho[:, j_early], rho[:, j_late]
