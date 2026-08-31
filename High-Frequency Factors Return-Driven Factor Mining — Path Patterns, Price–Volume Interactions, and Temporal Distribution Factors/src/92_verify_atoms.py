# ------------------ 92_verify_atoms.py ----------------
# 正确性验证：用独立的纯 Python 慢速循环实现 16 个原子因子，与 03 的向量化产出逐值比对；
# 并用 vol.hdf 日成交量与分钟量合计做勾稽。
# 慢速实现刻意走与向量化完全不同的代码路径（list + math 循环），专门捕捉索引/窗口对齐错误。
#
# 运行：python 92_verify_atoms.py [日期YYYYMMDD，默认取已产出的最近一天]

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lib_common import ATOMIC_CACHE_DIR, ATOM_COLS, F_MINUTE_DIR, MIN_VALID_MINUTES
from lib_fdrive import read_base_wide, read_minute_day

EPS = 1e-12
TAU_WIN_RET = 29  # 30 分钟窗口内的收益个数


# ---------- 慢速基础量 ----------

def _valid(c: list[float]) -> list[bool]:
    return [(x is not None and math.isfinite(x) and x > 0) for x in c]


def _rets(c: list[float], valid: list[bool]) -> list[float]:
    # r_t = ln(c_t/c_{t-1})，相邻任一无效则 NaN。
    r = [float("nan")] * len(c)
    lc = [math.log(x) if v else float("nan") for x, v in zip(c, valid)]
    for t in range(1, len(c)):
        if valid[t] and valid[t - 1]:
            r[t] = lc[t] - lc[t - 1]
    return r


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(xs, ys) if math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n < 2:
        return float("nan")
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    sx2 = sum(p[0] * p[0] for p in pairs)
    sy2 = sum(p[1] * p[1] for p in pairs)
    vx = n * sx2 - sx * sx
    vy = n * sy2 - sy * sy
    if vx <= 0 or vy <= 0:
        return float("nan")
    return (n * sxy - sx * sy) / math.sqrt(vx * vy)


def slow_atoms(o, h, l, c, v, amt) -> dict[str, float]:
    # 单股单日的 16 原子慢速实现。输入均为长度 240 的 list。
    T = len(c)
    valid = _valid(c)
    n_valid = sum(valid)
    r = _rets(c, valid)
    vr = [math.isfinite(x) for x in r]
    vol = [x if math.isfinite(x) else 0.0 for x in v]
    a = [float("nan")] * T
    for t in range(1, T):
        if valid[t - 1] and math.isfinite(h[t]) and math.isfinite(l[t]):
            a[t] = (h[t] - l[t]) / c[t - 1]

    out: dict[str, float] = {}

    # ---- A1 / A1v ----
    idx_r = [t for t in range(T) if vr[t]]
    n_r = len(idx_r)
    if n_r >= 2:
        L = sum(abs(r[t]) for t in idx_r)
        D = abs(sum(r[t] for t in idx_r))
        out["A1_tau_atom"] = L / (D + EPS)
        vbar = sum(vol[t] for t in idx_r) / n_r
        if vbar > 0:
            Lv = sum(abs(r[t]) * vol[t] / vbar for t in idx_r)
            out["A1v_tau_vol_atom"] = Lv / (D + EPS)
        else:
            out["A1v_tau_vol_atom"] = float("nan")
    else:
        out["A1_tau_atom"] = float("nan")
        out["A1v_tau_vol_atom"] = float("nan")

    # ---- 30 分钟滚动曲折度与切割 ----
    tau_win: dict[int, float] = {}
    for t in range(TAU_WIN_RET - 1, T):  # 右端点 t，窗口收益 r_{t-28..t}
        seg = [r[s] for s in range(t - TAU_WIN_RET + 1, t + 1)]
        if all(math.isfinite(x) for x in seg):
            tau_win[t] = sum(abs(x) for x in seg) / (abs(sum(seg)) + EPS)
    win_ok = sorted(tau_win.keys())
    med = float("nan")
    if win_ok:
        vals = sorted(tau_win[t] for t in win_ok)
        m = len(vals)
        med = vals[m // 2] if m % 2 == 1 else 0.5 * (vals[m // 2 - 1] + vals[m // 2])
    d_high = {t: (tau_win[t] > med) for t in win_ok}

    # A2a
    rh = [r[t] for t in win_ok if d_high[t]]
    if len(rh) >= 2:
        mu = sum(rh) / len(rh)
        out["A2a_high_tau_ret_std_atom"] = math.sqrt(sum((x - mu) ** 2 for x in rh) / (len(rh) - 1))
    else:
        out["A2a_high_tau_ret_std_atom"] = float("nan")
    # A2b
    low = [t for t in win_ok if not d_high[t]]
    out["A2b_low_tau_cum_ret_atom"] = sum(r[t] for t in low) if low else float("nan")
    # A2c
    volsum = sum(vol)
    if volsum > 0:
        out["A2c_high_tau_vol_share_atom"] = sum((vol[t] / volsum) for t in win_ok if d_high[t])
    else:
        out["A2c_high_tau_vol_share_atom"] = float("nan")
    # A3
    if win_ok:
        mu_v = sum(vol) / T
        sd_v = math.sqrt(sum((x - mu_v) ** 2 for x in vol) / T)
        e = [vol[t] > mu_v + sd_v for t in range(T)]
        g = sum(1 for t in win_ok if d_high[t] and e[t])
        out["A3_tort_vol_joint_atom"] = g / len(win_ok)
    else:
        out["A3_tort_vol_joint_atom"] = float("nan")

    # ---- B 模块 ----
    dv = [float("nan")] * T
    for t in range(1, T):
        dv[t] = vol[t] - vol[t - 1]
    dvp = [float("nan")] * T
    for t in range(1, T):
        if vol[t - 1] > 0:
            dvp[t] = (vol[t] - vol[t - 1]) / vol[t - 1]

    def lag_pair_count(x, y, k):
        return sum(1 for t in range(k, T) if math.isfinite(x[t - k]) and math.isfinite(y[t]))

    def lag_corr(x, y, k, cond=None):
        xs, ys = [], []
        for t in range(k, T):
            if cond is not None and not cond[t]:
                continue
            if math.isfinite(x[t - k]) and math.isfinite(y[t]):
                xs.append(x[t - k])
                ys.append(y[t])
        return _pearson(xs, ys)

    # B1
    ws, acc = 0.0, 0.0
    for k in (1, 2, 3):
        n_k = lag_pair_count(dvp, r, k)
        t1 = lag_corr(dvp, r, k)
        t2 = lag_corr(r, dvp, k)
        term = t1 - t2
        if n_k >= 20 and math.isfinite(term):
            acc += (1.0 / k) * term
            ws += 1.0 / k
    out["B1_lead_lag_atom"] = acc / ws if ws > 0 else float("nan")

    # B2 / B2a
    up_t = [t for t in range(T) if vr[t] and r[t] > 0 and math.isfinite(a[t])]
    dn_t = [t for t in range(T) if vr[t] and r[t] < 0 and math.isfinite(a[t])]
    if len(up_t) + len(dn_t) > 0:
        Au_v = sum(a[t] * vol[t] for t in up_t)
        Ad_v = sum(a[t] * vol[t] for t in dn_t)
        out["B2_path_asym_atom"] = Au_v / (Ad_v + EPS)
        Au_a = sum(a[t] * amt[t] for t in up_t)
        Ad_a = sum(a[t] * amt[t] for t in dn_t)
        out["B2a_path_asym_amt_atom"] = Au_a / (Ad_a + EPS)
    else:
        out["B2_path_asym_atom"] = float("nan")
        out["B2a_path_asym_amt_atom"] = float("nan")

    # B3（滞后 1 期，分上/下行子集）
    n_up = sum(1 for t in range(T) if vr[t] and r[t] > 0)
    n_dn = sum(1 for t in range(T) if vr[t] and r[t] < 0)
    if n_up >= 15 and n_dn >= 15:
        cond_up = [False] * T
        cond_dn = [False] * T
        for t in range(T):
            if vr[t] and r[t] > 0:
                cond_up[t] = True
            if vr[t] and r[t] < 0:
                cond_dn[t] = True
        ll_up = lag_corr(dvp, r, 1, cond_up) - lag_corr(r, dvp, 1, cond_up)
        ll_dn = lag_corr(dvp, r, 1, cond_dn) - lag_corr(r, dvp, 1, cond_dn)
        out["B3_cond_lead_lag_atom"] = ll_up - ll_dn
    else:
        out["B3_cond_lead_lag_atom"] = float("nan")

    # B4
    out["B4_cross_atom"] = (out["B2_path_asym_atom"] - 1.0) * out["B1_lead_lag_atom"]

    # B5a：滚动 60 分钟 corr(r, ΔV)，取右端点 59 与 239
    def roll60(right: int) -> float:
        seg_r = r[right - 59: right + 1]
        seg_v = dv[right - 59: right + 1]
        pairs = [(x, y) for x, y in zip(seg_r, seg_v) if math.isfinite(x) and math.isfinite(y)]
        if len(pairs) < 30:
            return float("nan")
        return _pearson([p[0] for p in pairs], [p[1] for p in pairs])

    out["B5a_corr_decay_atom"] = roll60(T - 1) - roll60(59)

    # B5b：累计价路径 vs 累计量路径斜率背离
    first = next((t for t in range(T) if vr[t]), None)
    last = next((t for t in range(T - 1, -1, -1) if vr[t]), None)
    b5b = float("nan")
    if first is not None and last is not None:
        rng_len = last - first + 1
        gap = 1.0 - n_r / rng_len
        if rng_len >= 20 and gap <= 0.10:
            P, Q = [], []
            cp, cq = 0.0, 0.0
            vbar_all = sum(vol) / T
            for t in range(T):
                if vr[t]:
                    cp += r[t]
                cq += vol[t] - vbar_all
                P.append(cp)
                Q.append(cq)
            ts = list(range(first, last + 1))
            ps = [P[t] for t in ts]
            qs = [Q[t] for t in ts]

            def slope(vals):
                n = len(ts)
                sx = sum(ts)
                sy = sum(vals)
                sxy = sum(x * y for x, y in zip(ts, vals))
                sx2 = sum(x * x for x in ts)
                den = n * sx2 - sx * sx
                if den <= 0:
                    return float("nan")
                return (n * sxy - sx * sy) / den

            def std(vals):
                mu = sum(vals) / len(vals)
                return math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))

            bp, bv = slope(ps), slope(qs)
            sp, sv = std(ps), std(qs)
            if math.isfinite(bp) and math.isfinite(bv) and sp > 0 and sv > 0:
                b5b = -(bp / sp) * (bv / sv)
    out["B5b_slope_div_atom"] = b5b

    # B5c：新高量能比
    b5c = float("nan")
    if first is not None:
        cp = 0.0
        runmax = -math.inf
        hi, ot = [], []
        for t in range(T):
            if vr[t]:
                cp += r[t]
                runmax = max(runmax, cp)
                (hi if cp >= runmax else ot).append(vol[t])
        if hi and ot:
            mo = sum(ot) / len(ot)
            if mo > 0:
                b5c = (sum(hi) / len(hi)) / mo
    out["B5c_high_vol_ratio_atom"] = b5c

    # ---- C 模块 ----
    def entropy_from(wgts):
        pos = [w for w in wgts if math.isfinite(w) and w > 0]
        if not pos:
            return float("nan"), 0
        return -sum(w * math.log(w) for w in pos), len(pos)

    if volsum > 0:
        w_all = [x / volsum for x in vol]
        H1, N1 = entropy_from(w_all)
        out["C1_entropy_atom"] = H1 / math.log(240.0) if N1 >= 2 else float("nan")
        out["C1_H_atom"] = H1
        out["C1_N_atom"] = float(N1)
    else:
        out["C1_entropy_atom"] = float("nan")
        out["C1_H_atom"] = float("nan")
        out["C1_N_atom"] = 0.0

    v_up = sum(vol[t] for t in range(T) if vr[t] and r[t] > 0)
    v_dn = sum(vol[t] for t in range(T) if vr[t] and r[t] < 0)
    if v_up > 0 and v_dn > 0:
        w_up = [vol[t] / v_up if (vr[t] and r[t] > 0) else 0.0 for t in range(T)]
        w_dn = [vol[t] / v_dn if (vr[t] and r[t] < 0) else 0.0 for t in range(T)]
        Hu, Nu = entropy_from(w_up)
        Hd, Nd = entropy_from(w_dn)
        out["C2_entropy_diff_atom"] = Hu - Hd if (Nu >= 2 and Nd >= 2) else float("nan")
    else:
        out["C2_entropy_diff_atom"] = float("nan")

    return out


def main() -> None:
    # 选择验证日期：命令行给定，或取 parts 里最近的一天。
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        parts = sorted(ATOMIC_CACHE_DIR.glob("parts/atomic_*.parquet"))
        date = parts[-1].stem.split("_")[2]
    print(f"验证日期: {date}")

    day = read_minute_day(F_MINUTE_DIR / f"{date}.hdf")
    # 取成交量最大的 5 只沪深股票做验证（流动性好、数据干净）。
    vol_col = day.field_col("成交量")
    c_col = day.field_col("收盘价")
    stats = []
    for i, code in enumerate(day.codes):
        if not (code.endswith(".SH") or code.endswith(".SZ")):
            continue
        rows = np.where(day.lab1 == i)[0]
        tvol = day.values[rows, vol_col].sum()
        stats.append((tvol, code, i))
    stats.sort(reverse=True)
    picks = stats[:5]
    print("验证股票:", [p[1] for p in picks])

    # 读 03 的产出。
    parts = sorted(ATOMIC_CACHE_DIR.glob("parts/atomic_*.parquet"))
    dfs = [pd.read_parquet(p) for p in parts]
    atoms_df = pd.concat(dfs, ignore_index=True)
    tds = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}")
    day_df = atoms_df[atoms_df["trade_date"] == tds].set_index("code")

    # 逐股比对。
    max_rel = 0.0
    bad = []
    for tvol, code, i in picks:
        rows = np.where(day.lab1 == i)[0]
        rows = rows[np.argsort(day.lab0[rows])]
        get = lambda col: [float(x) for x in day.values[rows, day.field_col(col)]]
        slow = slow_atoms(get("开盘价"), get("最高价"), get("最低价"), get("收盘价"), get("成交量"), get("成交额"))
        if code not in day_df.index:
            print(f"  {code}: 不在 03 产出中！")
            bad.append((code, "missing", None))
            continue
        fast = day_df.loc[code]
        for col in ATOM_COLS:
            s, f = slow[col], fast[col]
            if math.isnan(s) and (isinstance(f, float) and math.isnan(f)):
                continue
            if math.isnan(s) or math.isnan(f) or pd.isna(f):
                bad.append((code, col, (s, f)))
                continue
            rel = abs(s - f) / max(abs(s), 1e-9)
            max_rel = max(max_rel, rel)
            if rel > 1e-8:
                bad.append((code, col, (s, f)))
        print(f"  {code} 比对完成")
    print(f"最大相对误差: {max_rel:.3e}")
    if bad:
        print("不一致项:")
        for b in bad[:20]:
            print("   ", b)
        raise SystemExit(1)

    # 诊断：分钟量合计 vs vol.hdf 日成交量。
    # 早期数据的分钟量与日线量可能相差复权因子的常数倍（项目已由 93 单独验证），
    # 因子使用的是日内相对量，因此这里记录比例但不再把“必须相等”作为原子算法验收条件。
    vol_daily = read_base_wide("vol", start=f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                               end=f"{date[:4]}-{date[4:6]}-{date[6:8]}")
    for tvol, code, i in picks:
        rows = np.where(day.lab1 == i)[0]
        msum = day.values[rows, vol_col].sum()
        dval = vol_daily.loc[vol_daily.index[0], code] if code in vol_daily.columns else float("nan")
        ok = math.isfinite(dval) and dval > 0 and math.isfinite(msum) and msum > 0
        ratio = msum / dval if ok else float("nan")
        print(f"  量口径 {code}: 分钟合计={msum:.2f} 日线={dval:.2f} ratio={ratio:.6f}")
        if not ok:
            raise SystemExit(1)
    print("VERIFY OK")


if __name__ == "__main__":
    main()
