# ------------------ 07_backtest_ic_groups.py ----------------
# 作用：指数增强口径的分组回测 + IC 检验 + 图表（V2 第三步，风格暴露暂不做）。
# 遍历 universe(4) × factor(18) × variant(S1/S2) = 144 组合，每组产出：
#   - ic_stats.csv   ：IC/RankIC 均值与标准差、ICIR、RankICIR、IC 胜率、t 值、分年 IC
#   - group_stats.csv：各组年化收益/波动/夏普/最大回撤 + 多空（毛/净）+ 基准
#   - yearly_stats.csv：分年年化超额(vs 指数基准)、超额最大回撤、IR、月度胜率、盈亏比、多空年化
#   - cum_ic_rankic.png   ：累计 IC + 累计 RankIC（同图两条曲线）
#   - cum_icir_rankicir.png：expanding ICIR + expanding RankICIR（同图两条曲线）
#   - group_nav.png       ：各组净值 + 基准 + 多空毛/净双曲线
# 回测口径：形成月 t 末按因子值升序分组（300/500 五组、1000/全指 十组），
#           建仓日 t1 = 次月首个交易日，按 t1 收盘建仓、持有至下月末（与 05 标签一致，P1 修复）；
#           等权持有，单边千三成本按换手 Σ|Δw| 扣减；
#           基准 = 对应指数 px[下月末]/px[t1] − 1（price_index）；
#           多空方向按两端组合的全样本净值高低确定：做多收益较高的一端、做空收益较低的一端；
#           毛/净收益使用同一方向，净收益再扣除两腿成本。
#
# 运行：python 07_backtest_ic_groups.py [--universes hs300] [--factors A1_tau_20d] [--variants S1]

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lib_common import (
    COST,
    DAILY_CACHE_DIR,
    FACTOR_COLS,
    FACTOR_SHORT,
    N_GROUPS,
    OUTPUTS_ROOT,
    PANEL_CACHE_DIR,
    UNIVERSES,
    UNIVERSE_BENCH,
    UNIVERSE_CN,
    VARIANTS,
    ensure_dir,
    setup_matplotlib,
)
from report_style import BLUE, CHARCOAL, GOLD, RED, clean_axes, group_linestyles, group_palette

PROCESSED_DIR = PANEL_CACHE_DIR / "processed"
ANNUALIZE = 12  # 月度频率年化乘子（√12 用于 IR/ICIR，12 用于收益简单年化口径）


# ============ IC 序列 ============

def ic_series(fm: pd.DataFrame, factor: str, min_n: int = 10) -> pd.DataFrame:
    # 输入：某 universe×variant 的处理后因子表（code, form_date, form_month, factor）。
    # 输出：index=form_month 的 DataFrame，列 ic / rank_ic。min_n 与分组回测门槛一致。
    rows = []
    for fmonth, grp in fm.groupby("form_month", sort=True):
        f = grp[factor].astype(float)
        r = grp["next_ret_1m"].astype(float)
        ok = f.notna() & r.notna()
        if int(ok.sum()) < min_n:
            continue
        x, y = f[ok].to_numpy(), r[ok].to_numpy()
        # Pearson IC 与 Spearman RankIC（秩化后的 Pearson）。
        ic = np.corrcoef(x, y)[0, 1]
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        rank_ic = np.corrcoef(rx, ry)[0, 1]
        rows.append({"form_month": fmonth, "ic": ic, "rank_ic": rank_ic, "n": int(ok.sum())})
    return pd.DataFrame(rows).set_index("form_month")


def expanding_icir(ic: pd.Series) -> pd.Series:
    # expanding ICIR_t = mean(IC≤t)/std(IC≤t)×√12；样本不足 12 期记 NaN。
    m = ic.expanding().mean()
    s = ic.expanding().std()
    out = m / s * np.sqrt(ANNUALIZE)
    out[s.isna() | (ic.expanding().count() < 12)] = np.nan
    return out


# ============ 分组回测 ============

def group_backtest(fm: pd.DataFrame, factor: str, n_groups: int) -> dict:
    # 月末按因子升序分 n_groups 组，等权持有 1 月（t1 建仓口径，收益标签已在 05 对齐）。
    # 返回 dict：months(index), group_rets DataFrame(G1..GN), costs, n_groups。
    months = []
    g_rets: list[list[float]] = []
    costs: list[list[float]] = []
    # 上一期末的漂移权重。换手按目标权重与漂移权重的 L1 距离计算；
    # 这仍是原有“等权分组、月度调仓”框架，只修正原先按退出只数近似的误差。
    prev_post_weights: dict[int, dict[str, float]] = {}
    for fmonth, grp in fm.groupby("form_month", sort=True):
        sub = grp.dropna(subset=[factor, "next_ret_1m"])
        if len(sub) < n_groups * 2:
            continue
        sub = sub.sort_values([factor, "code"], kind="mergesort")
        ret_by_code = sub.set_index("code")["next_ret_1m"].astype(float)
        splits = np.array_split(sub["code"].to_numpy(), n_groups)
        ret_row, cost_row = [], []
        for gi, codes in enumerate(splits):
            code_list = [str(c) for c in codes]
            k = max(len(code_list), 1)
            target = {c: 1.0 / k for c in code_list}
            r_i = ret_by_code.reindex(code_list)
            ret_row.append(float(r_i.mean()))
            prev = prev_post_weights.get(gi, {})
            turnover = sum(abs(target.get(c, 0.0) - prev.get(c, 0.0))
                           for c in set(target) | set(prev))
            cost_row.append(COST * turnover)
            end_value = {c: target[c] * (1.0 + float(r_i.loc[c])) for c in code_list}
            gross = sum(end_value.values())
            prev_post_weights[gi] = ({c: value / gross for c, value in end_value.items()}
                                     if gross > 0 else target)
        months.append(fmonth)
        g_rets.append(ret_row)
        costs.append(cost_row)
    idx = pd.Index(months, name="form_month")
    gr = pd.DataFrame(g_rets, index=idx, columns=[f"G{i+1}" for i in range(n_groups)])
    cr = pd.DataFrame(costs, index=idx, columns=gr.columns)
    return {"group_rets": gr, "costs": cr, "n_groups": n_groups}


def nav_from(rets: pd.Series, costs: pd.Series | None = None) -> pd.Series:
    # 由月收益（与成本）累计净值。
    net = rets if costs is None else rets - costs
    return (1.0 + net).cumprod()


def perf_stats(nav: pd.Series, rets: pd.Series) -> dict:
    # 年化收益/波动/夏普/最大回撤（月频）。
    n = len(rets)
    total = nav.iloc[-1] - 1.0 if n > 0 else 0.0
    ann_ret = (1.0 + total) ** (ANNUALIZE / max(n, 1)) - 1.0
    vol = rets.std() * np.sqrt(ANNUALIZE)
    # 沿用研报 V2 原口径：几何年化收益除以年化波动率。
    sharpe = ann_ret / vol if vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1.0).min()
    return {"ann_ret": ann_ret, "vol": vol, "sharpe": sharpe, "max_dd": dd}


# ============ 单组合完整流程 ============

def run_combo(uni: str, factor: str, variant: str, fm: pd.DataFrame,
              bench_month: pd.Series, output_root: Path = OUTPUTS_ROOT,
              make_plots: bool = True) -> dict:
    out_dir = ensure_dir(output_root / uni / factor / variant)
    n_groups = N_GROUPS[uni]

    # ---- IC ----
    ics = ic_series(fm, factor, min_n=n_groups * 2)
    ic, ric = ics["ic"], ics["rank_ic"]
    icir = ic.mean() / ic.std() * np.sqrt(ANNUALIZE) if ic.std() > 0 else np.nan
    ricir = ric.mean() / ric.std() * np.sqrt(ANNUALIZE) if ric.std() > 0 else np.nan
    t_stat = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else np.nan
    # 分年 IC。
    yr = pd.Series(ic.index).str[:4]
    yearly_ic = ic.groupby(yr.values).agg(["mean", "std", "count"])
    yearly_ic["icir"] = yearly_ic["mean"] / yearly_ic["std"] * np.sqrt(ANNUALIZE)
    ic_stats = pd.DataFrame({
        "ic_mean": [ic.mean()], "ic_std": [ic.std()], "icir": [icir],
        "rankic_mean": [ric.mean()], "rankic_std": [ric.std()], "rankicir": [ricir],
        "ic_win_rate": [(ic > 0).mean()], "t_stat": [t_stat], "n_months": [len(ic)],
    })
    ic_stats.to_csv(out_dir / "ic_stats.csv", index=False, encoding="utf-8-sig")
    yearly_ic.to_csv(out_dir / "ic_yearly.csv", encoding="utf-8-sig")

    # ---- 分组回测 ----
    bt = group_backtest(fm, factor, n_groups)
    gr, cr = bt["group_rets"], bt["costs"]
    bench = bench_month.reindex(gr.index)
    if bench.isna().any():
        missing = bench.index[bench.isna()].tolist()
        raise ValueError(f"基准收益缺失，拒绝按 0 收益处理: {missing[:5]}")
    navs = pd.DataFrame({g: nav_from(gr[g], cr[g]) for g in gr.columns})
    # 多空方向不能机械固定为 GN−G1，也不能只看 RankIC 符号。
    # 直接比较报告中两端组合的全样本净值：终值较高的一端做多，较低的一端做空。
    # 确定方向后，毛收益与净收益必须保持同向；净收益只比毛收益多扣两腿成本。
    ls_raw_rets = gr.iloc[:, -1] - gr.iloc[:, 0]                    # 固定排序方向 GN−G1，仅供审计
    effective_sign = 1.0 if navs.iloc[-1, -1] >= navs.iloc[-1, 0] else -1.0
    ls_gross_rets = effective_sign * ls_raw_rets
    ls_costs = cr.iloc[:, 0] + cr.iloc[:, -1]
    ls_net_rets = ls_gross_rets - ls_costs
    if not (ls_net_rets <= ls_gross_rets + 1e-12).all():
        raise AssertionError("多空净收益不得高于同方向毛收益")
    nav_ls_gross = nav_from(ls_gross_rets)
    nav_ls_net = nav_from(ls_net_rets)
    nav_ls_effective = nav_ls_net  # 兼容既有字段名：有效方向净收益即同方向扣成本收益
    nav_bench = nav_from(bench)
    turnover_mean = float((cr.mean(axis=1) / COST).mean()) if not cr.empty else np.nan
    ls_turnover_mean = float(((cr.iloc[:, 0] + cr.iloc[:, -1]) / COST).mean()) if not cr.empty else np.nan

    gs_rows = {}
    for g in gr.columns:
        gs_rows[g] = perf_stats(navs[g], gr[g] - cr[g])
    gs_rows["LS"] = perf_stats(nav_ls_net, ls_net_rets)
    gs_rows["LS_gross"] = perf_stats(nav_ls_gross, ls_gross_rets)
    gs_rows["LS_effective"] = perf_stats(nav_ls_effective, ls_net_rets)
    gs_rows["LS_raw_gross_GN_minus_G1"] = perf_stats(nav_from(ls_raw_rets), ls_raw_rets)
    gs_rows["BENCH"] = perf_stats(nav_bench, bench)
    group_stats = pd.DataFrame(gs_rows).T
    group_stats.to_csv(out_dir / "group_stats.csv", encoding="utf-8-sig")

    # ---- 分年统计（首组 vs 基准 + 多空净收益） ----
    # 年化仅对完整 12 个月的年份计算；不完整年份（如 2016 下半年、2026 上半年）不 ^(12/n) 外推。
    def _ann_if_full(rets: pd.Series) -> float:
        if len(rets) == 12:
            return (1.0 + rets).prod() ** (ANNUALIZE / 12.0) - 1.0
        return np.nan

    # 多头腿为全样本收益较高的一端；分年统计沿用这一固定方向，避免每年事后翻转。
    years = sorted(set(gr.index.str[:4]))
    y_rows = []
    for y in years:
        m = gr.index.str[:4] == y
        b = bench[m]
        g1 = gr.iloc[:, 0][m] - cr.iloc[:, 0][m]
        gn = gr.iloc[:, -1][m] - cr.iloc[:, -1][m]
        long_leg = gn if effective_sign > 0 else g1
        short_leg = g1 if effective_sign > 0 else gn
        excess = long_leg - b                # 实际多头腿超额
        ls_net_y = ls_net_rets[m]            # 同方向毛收益减两腿成本
        ann_g1 = _ann_if_full(g1)
        ann_gn = _ann_if_full(gn)
        ann_b = _ann_if_full(b)
        ann_long = _ann_if_full(long_leg)
        ann_short = _ann_if_full(short_leg)
        ann_ls = _ann_if_full(ls_net_y)
        ir = excess.mean() / excess.std() * np.sqrt(ANNUALIZE) if excess.std() > 0 else np.nan
        nav_e = (1 + excess).cumprod()
        edd = (nav_e / nav_e.cummax() - 1).min()
        pos, neg = excess[excess > 0], excess[excess < 0]
        pl = pos.mean() / abs(neg.mean()) if len(neg) > 0 and neg.mean() != 0 else np.nan
        y_rows.append({"year": y, "n_months": int(len(g1)),
                       "g1_ann": ann_g1, "gN_ann": ann_gn,
                       "long_group": f"G{n_groups}" if effective_sign > 0 else "G1",
                       "short_group": "G1" if effective_sign > 0 else f"G{n_groups}",
                       "long_ann": ann_long, "short_ann": ann_short, "bench_ann": ann_b,
                       "ann_excess": ann_long - ann_b if np.isfinite(ann_long) else np.nan,
                       "excess_maxdd": edd, "ir": ir,
                       "win_rate": (excess > 0).mean(), "pl_ratio": pl, "ls_ann": ann_ls})
    pd.DataFrame(y_rows).to_csv(out_dir / "yearly_stats.csv", index=False, encoding="utf-8-sig")

    # ---- 图 ----
    if make_plots:
        _plot_combo(out_dir, uni, factor, variant, ic, ric, navs, nav_bench,
                    nav_ls_gross, nav_ls_effective, effective_sign)

    return {"universe": uni, "factor": factor, "variant": variant,
            "ic_mean": ic.mean(), "icir": icir, "rankic_mean": ric.mean(), "rankicir": ricir,
            "ic_win_rate": (ic > 0).mean(), "n_months": len(ic),
            "g1_ann": group_stats.loc["G1", "ann_ret"],
            "gN_ann": group_stats.loc[f"G{n_groups}", "ann_ret"],
            "ls_ann": group_stats.loc["LS", "ann_ret"],
            "ls_sharpe": group_stats.loc["LS", "sharpe"],
            "ls_ann_gross": group_stats.loc["LS_gross", "ann_ret"],
            "ls_sharpe_gross": group_stats.loc["LS_gross", "sharpe"],
            "effective_sign": int(effective_sign),
            "ls_eff_ann": group_stats.loc["LS_effective", "ann_ret"],
            "ls_eff_sharpe": group_stats.loc["LS_effective", "sharpe"],
            "long_group": f"G{n_groups}" if effective_sign > 0 else "G1",
            "short_group": "G1" if effective_sign > 0 else f"G{n_groups}",
            "turnover_mean": turnover_mean,
            "ls_turnover_mean": ls_turnover_mean,
            "bench_ann": group_stats.loc["BENCH", "ann_ret"]}


def _plot_combo(out_dir: Path, uni: str, factor: str, variant: str,
                ic: pd.Series, ric: pd.Series, navs: pd.DataFrame,
                nav_bench: pd.Series, nav_ls_gross: pd.Series,
                nav_ls_effective: pd.Series, effective_sign: float) -> None:
    import matplotlib.pyplot as plt

    label = f"{UNIVERSE_CN[uni]}｜{FACTOR_SHORT[factor]}｜{variant}"

    xs = range(len(ic))
    xl = [str(m)[:7] for m in ic.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, ic.cumsum(), label="累计 IC", color=BLUE)
    ax.plot(xs, ric.cumsum(), label="累计 RankIC", color=RED)
    clean_axes(ax, zero_line=True)
    ax.set_title(f"{label}：累计 IC 与累计 RankIC")
    ax.set_xlabel("形成月"); ax.set_ylabel("累计值"); ax.legend()
    _thin_xticks(ax, xs, xl)
    fig.tight_layout(); fig.savefig(out_dir / "cum_ic_rankic.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, expanding_icir(ic), label="累计 ICIR", color=BLUE)
    ax.plot(xs, expanding_icir(ric), label="累计 RankICIR", color=RED)
    clean_axes(ax, zero_line=True)
    ax.set_title(f"{label}：累计 ICIR 与累计 RankICIR")
    ax.set_xlabel("形成月"); ax.set_ylabel("expanding ICIR (年化)"); ax.legend()
    _thin_xticks(ax, xs, xl)
    fig.tight_layout(); fig.savefig(out_dir / "cum_icir_rankicir.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    palette = group_palette(len(navs.columns))
    linestyles = group_linestyles(len(navs.columns))
    for g, color, linestyle in zip(navs.columns, palette, linestyles):
        is_edge = g in {navs.columns[0], navs.columns[-1]}
        ax.plot(
            range(len(navs)), navs[g], label=g, color=color,
            linestyle=linestyle, linewidth=2.6 if is_edge else 1.45,
            alpha=1.0 if is_edge else 0.90, zorder=4 if is_edge else 2,
        )
    ax.plot(range(len(nav_bench)), nav_bench, label="基准指数", color="#111111",
            linewidth=2.4, linestyle="-", zorder=5)
    direction = "GN-G1" if effective_sign > 0 else "G1-GN"
    ax.plot(range(len(nav_ls_gross)), nav_ls_gross, label=f"多空毛收益 {direction}",
            linestyle="--", color="#7F7F7F", linewidth=2.2, alpha=1.0, zorder=6)
    ax.plot(range(len(nav_ls_effective)), nav_ls_effective, label=f"多空净收益 {direction} (扣成本)",
            linestyle="-", color="#8C564B", linewidth=2.8, zorder=7)
    clean_axes(ax)
    ax.grid(False)
    ax.yaxis.grid(True, color="#ECECEC", linewidth=0.65)
    ax.set_title(f"{label}：分组净值（等权月度调仓，单边千三）")
    ax.set_xlabel("形成月"); ax.set_ylabel("净值")
    ax.legend(fontsize=8.1, ncol=4, loc="upper left", frameon=False,
              columnspacing=1.25, handlelength=2.6)
    xl_nav = [str(m)[:7] for m in navs.index]
    _thin_xticks(ax, range(len(navs)), xl_nav)
    fig.tight_layout(); fig.savefig(out_dir / "group_nav.png", dpi=220); plt.close(fig)

def _thin_xticks(ax, xs, labels, target: int = 12) -> None:
    # x 轴标签抽稀，避免月份拥挤。
    xs = list(xs)
    if len(xs) <= target:
        idx = list(range(len(xs)))
    else:
        step = max(1, len(xs) // target)
        idx = list(range(0, len(xs), step))
    ax.set_xticks([xs[i] for i in idx])
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, fontsize=8)


def main() -> None:
    args = parse_args()
    setup_matplotlib()
    t0 = time.time()
    output_root = Path(args.output_root).resolve() if args.output_root else OUTPUTS_ROOT
    ensure_dir(output_root)

    # 基准月收益（P1 对齐）：形成月 t 末建仓日 t1 = 次月首个交易日，
    # 持有至下月末 → 收益 = P(下月末)/P(t1) − 1（与 05 的 next_ret_1m 同一持有窗）。
    idx_price = pd.read_parquet(DAILY_CACHE_DIR / "index_price.parquet")
    cal = pd.read_csv(DAILY_CACHE_DIR / "calendar.csv", header=None, names=["date"])
    cal_dt = pd.DatetimeIndex(pd.to_datetime(cal["date"]))
    pos_map = {d: i for i, d in enumerate(cal_dt)}
    month_last = pd.Series(cal_dt, index=cal_dt).groupby(cal_dt.strftime("%Y-%m")).max()
    ml = month_last.sort_index()
    bench_month_ret: dict[str, pd.Series] = {}
    for uni in UNIVERSES:
        col = UNIVERSE_BENCH[uni]
        px = idx_price[col]
        rows = {}
        for m in ml.index:
            i = ml.index.get_loc(m)
            if i + 1 >= len(ml):
                continue
            t_last = ml.iloc[i]
            t1 = cal_dt[pos_map[t_last] + 1] if pos_map[t_last] + 1 < len(cal_dt) else None
            t_next = ml.iloc[i + 1]
            if t1 is None or t1 not in px.index or t_next not in px.index:
                continue
            rows[m] = px.loc[t_next] / px.loc[t1] - 1.0
        bench_month_ret[uni] = pd.Series(rows, name="bench")

    panel = pd.read_parquet(PANEL_CACHE_DIR / "monthly_panel.parquet")
    summary_rows = []
    unis = args.universes or list(UNIVERSES)
    facs = args.factors or list(FACTOR_COLS)
    vars_ = args.variants or list(VARIANTS)
    total = len(unis) * len(facs) * len(vars_)
    done = 0
    for uni in unis:
        for variant in vars_:
            path = PROCESSED_DIR / f"{uni}_{variant}.parquet"
            if not path.exists():
                print(f"[skip] {path} 不存在")
                continue
            fm = pd.read_parquet(path)
            fm["form_month"] = fm["form_month"].astype(str)
            fm = fm.merge(panel[["code", "form_date", "next_ret_1m"]],
                          on=["code", "form_date"], how="left")
            for factor in facs:
                if factor not in fm.columns:
                    continue
                try:
                    summary_rows.append(run_combo(uni, factor, variant, fm, bench_month_ret[uni],
                                                  output_root, make_plots=not args.no_plots))
                except Exception as e:
                    print(f"[ERROR] {uni}/{factor}/{variant}: {e}")
                done += 1
                print(f"[{done}/{total}] {uni}/{factor}/{variant} 完成")
    if not args.no_summary:
        pd.DataFrame(summary_rows).to_csv(output_root / "master_summary.csv",
                                          index=False, encoding="utf-8-sig")
    print(f"全部完成，用时 {time.time()-t0:.0f}s，总表 {output_root / 'master_summary.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="分组回测 + IC 检验 + 图表。")
    p.add_argument("--universes", nargs="*", default=None)
    p.add_argument("--factors", nargs="*", default=None)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--output-root", default=None, help="另存回测结果的目录；默认写入 result/backtest_v4。")
    p.add_argument("--no-plots", action="store_true", help="只重算统计表，不重复生成组合图。")
    p.add_argument("--no-summary", action="store_true", help="局部重画时不覆盖 master_summary.csv。")
    return p.parse_args()


if __name__ == "__main__":
    main()
