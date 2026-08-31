# ------------------ 11_factor_decay.py ----------------
# 作用：因子效果时间序列衰减分析 + 可视化（07 主回测的扩展阶段）。
# 遍历 universe(4) × factor(18) × variant(S1/S2) = 144 组合，每组产出：
#   - decay_monthly.csv ：月度 IC/RankIC + 滚动(12/24月)均值 + 累计 RankIC + 月净多空 + 滚动12月累计净多空
#   - decay_yearly.csv  ：分年 RankIC 均值/标准差/ICIR/胜率（并列 ic_mean 供与 07 交叉核对）
#   - decay_stats.csv   ：量化衰减指标（全样本/前后半/2025前vs近期/滚动趋势OLS/累计RankIC回撤）
#   - decay_overview.png：三面板（月度RankIC+滚动线 / 分年RankIC红蓝柱 / 滚动12月累计净多空）
# 聚合层（每 universe×variant）：
#   - decay_master.csv       ：该池×变体下 18 因子衰减指标汇总表
#   - heatmap_year_rankic.png：18 因子 × 年份 RankIC 均值热力图（make_heatmap 风格）
# 全量跑完另写总表 result/decay_v4/decay_master.csv（144 行）并打印衰减排行。
# 口径：与 07 完全一致——形成月截面 Pearson IC / Spearman RankIC（min_n = 分组数×2），
#       分组等权月度调仓；多空做多全样本两端组合中收益较高的一端、做空较低的一端并扣两腿成本；
#       衰减维度为日历时间（分年/滚动/近期子样本），
#       不是持有期衰减。IC/分组逻辑按相同口径本地重写（07 文件名以数字开头无法 import）。
# 运行：python 11_factor_decay.py [--universes hs300] [--factors A1_tau_20d] [--variants S1]

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from lib_common import (
    COST,
    DECAY_RECENT_START,
    DECAY_ROLL_WINDOWS,
    DECAY_ROOT,
    FACTOR_COLS,
    FACTOR_SHORT,
    MOD_BOUNDS,
    N_GROUPS,
    PANEL_CACHE_DIR,
    UNIVERSES,
    UNIVERSE_CN,
    VARIANTS,
    ensure_dir,
    setup_matplotlib,
)
from report_style import BLUE, CHARCOAL, GOLD, LIGHT_GRAY, RED, clean_axes

PROCESSED_DIR = PANEL_CACHE_DIR / "processed"
ANNUALIZE = 12  # 月度频率年化乘子，与 07 一致

POS_RED = RED
NEG_BLUE = BLUE


# ============ IC 序列与分组回测（口径复制自 07） ============

def ic_series(fm: pd.DataFrame, factor: str, min_n: int = 10) -> pd.DataFrame:
    # 逐形成月截面 Pearson IC + Spearman RankIC，index=form_month。
    rows = []
    for fmonth, grp in fm.groupby("form_month", sort=True):
        f = grp[factor].astype(float)
        r = grp["next_ret_1m"].astype(float)
        ok = f.notna() & r.notna()
        if int(ok.sum()) < min_n:
            continue
        x, y = f[ok].to_numpy(), r[ok].to_numpy()
        ic = np.corrcoef(x, y)[0, 1]
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        rank_ic = np.corrcoef(rx, ry)[0, 1]
        rows.append({"form_month": fmonth, "ic": ic, "rank_ic": rank_ic, "n": int(ok.sum())})
    return pd.DataFrame(rows).set_index("form_month")


def group_backtest(fm: pd.DataFrame, factor: str, n_groups: int) -> dict:
    # 月末按因子升序分 n_groups 组，等权持有 1 月；返回月度组收益与成本（同 07）。
    months = []
    g_rets: list[list[float]] = []
    costs: list[list[float]] = []
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


def _thin_xticks(ax, xs, labels, target: int = 12) -> None:
    # x 轴标签抽稀（同 07）。
    xs = list(xs)
    if len(xs) <= target:
        idx = list(range(len(xs)))
    else:
        step = max(1, len(xs) // target)
        idx = list(range(0, len(xs), step))
    ax.set_xticks([xs[i] for i in idx])
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, fontsize=8)


def _icir(s: pd.Series) -> float:
    # 子样本 ICIR = mean/std×√12；样本不足或零方差记 NaN。
    if len(s) < 2:
        return np.nan
    sd = s.std()
    return float(s.mean() / sd * np.sqrt(ANNUALIZE)) if sd > 0 else np.nan


# ============ 衰减指标 ============

def decay_metrics(ic: pd.Series, ric: pd.Series) -> dict:
    # 时间维度衰减指标：全样本 / 前后半 / 2025前vs近期 / 滚动趋势 / 累计回撤。
    n = len(ric)
    row = {"n_months": n,
           "rankic_mean": float(ric.mean()), "rankicir": _icir(ric),
           "win_rate": float((ric > 0).mean())}
    # 前后半切分（按中位形成月），主衰减指标 half_diff = h2 − h1。
    h = n // 2
    row["rankicir_h1"] = _icir(ric.iloc[:h]) if h >= 6 else np.nan
    row["rankicir_h2"] = _icir(ric.iloc[h:]) if n - h >= 6 else np.nan
    row["half_diff"] = row["rankicir_h2"] - row["rankicir_h1"]
    # 幅值口径衰减（主指标）：负 ICIR 的反转类因子用带符号差会反向，
    # |ICIR| 之差才度量"预测强度"变化；负=衰减。
    row["half_mag_diff"] = abs(row["rankicir_h2"]) - abs(row["rankicir_h1"])
    # 近期子样本（≥ DECAY_RECENT_START，字符串月 "YYYY-MM" 可直接比较）。
    recent = ric.index >= DECAY_RECENT_START
    pre = ~recent
    row["rankic_mean_pre2025"] = float(ric[pre].mean()) if pre.sum() > 0 else np.nan
    row["rankicir_pre2025"] = _icir(ric[pre])
    row["rankic_mean_recent"] = float(ric[recent].mean()) if recent.sum() > 0 else np.nan
    row["rankicir_recent"] = _icir(ric[recent])
    row["win_rate_recent"] = float((ric[recent] > 0).mean()) if recent.sum() > 0 else np.nan
    # 滚动 RankIC 对时间 OLS：slope×12 年化 + p 值。
    w = DECAY_ROLL_WINDOWS[0]
    r12 = ric.rolling(w).mean().dropna()
    if len(r12) >= 3:
        res = sps.linregress(np.arange(len(r12)), r12.to_numpy())
        row["trend_slope_ann"] = float(res.slope * ANNUALIZE)
        row["trend_pval"] = float(res.pvalue)
    else:
        row["trend_slope_ann"] = np.nan
        row["trend_pval"] = np.nan
    # 累计 RankIC 曲线最大回撤（衰减深度）。
    cum = ric.cumsum()
    row["cum_rankic_dd"] = float((cum - cum.cummax()).min())
    return row


# ============ 单组合流程 ============

def run_combo(uni: str, factor: str, variant: str, fm: pd.DataFrame,
              output_root: Path) -> tuple[dict, pd.DataFrame]:
    import matplotlib.pyplot as plt

    out_dir = ensure_dir(output_root / uni / factor / variant)
    n_groups = N_GROUPS[uni]

    ics = ic_series(fm, factor, min_n=n_groups * 2)
    ic, ric = ics["ic"], ics["rank_ic"]

    bt = group_backtest(fm, factor, n_groups)
    gr, cr = bt["group_rets"], bt["costs"]
    group_terminal_nav = (1.0 + gr - cr).prod(axis=0)
    effective_sign = 1.0 if group_terminal_nav.iloc[-1] >= group_terminal_nav.iloc[0] else -1.0
    ls_gross = effective_sign * (gr.iloc[:, -1] - gr.iloc[:, 0])
    ls_net = ls_gross - cr.iloc[:, 0] - cr.iloc[:, -1]
    if not (ls_net <= ls_gross + 1e-12).all():
        raise AssertionError("多空净收益不得高于同方向毛收益")
    ls_trail12 = (1.0 + ls_net).rolling(12).apply(np.prod, raw=True) - 1.0

    # ---- 月度序列表 ----
    w1, w2 = DECAY_ROLL_WINDOWS
    mdf = pd.DataFrame({"ic": ic, "rank_ic": ric}, index=ric.index)
    mdf[f"roll{w1}_ic"] = ic.rolling(w1).mean()
    mdf[f"roll{w1}_rankic"] = ric.rolling(w1).mean()
    mdf[f"roll{w2}_rankic"] = ric.rolling(w2).mean()
    mdf["cum_rankic"] = ric.cumsum()
    mdf["ls_net"] = ls_net
    mdf["ls_trail12"] = ls_trail12
    mdf.index.name = "form_month"
    mdf.to_csv(out_dir / "decay_monthly.csv", encoding="utf-8-sig")

    # ---- 分年表（补 07 只有 Pearson IC 分年的缺口） ----
    yr = pd.Series(ric.index, index=ric.index).str[:4]
    yearly = pd.DataFrame({
        "n_months": ric.groupby(yr).count(),
        "rankic_mean": ric.groupby(yr).mean(),
        "rankic_std": ric.groupby(yr).std(),
        "win_rate": (ric > 0).astype(float).groupby(yr).mean(),
        "ic_mean": ic.groupby(yr).mean(),
    })
    yearly["rankicir"] = yearly["rankic_mean"] / yearly["rankic_std"] * np.sqrt(ANNUALIZE)
    yearly.index.name = "year"
    yearly = yearly[["n_months", "rankic_mean", "rankic_std", "rankicir", "win_rate", "ic_mean"]]
    yearly.to_csv(out_dir / "decay_yearly.csv", encoding="utf-8-sig")

    # ---- 衰减指标表 ----
    stats_row = {"universe": uni, "factor": factor,
                 "factor_short": FACTOR_SHORT[factor], "variant": variant,
                 "effective_sign": int(effective_sign),
                 "long_group": f"G{n_groups}" if effective_sign > 0 else "G1",
                 "short_group": "G1" if effective_sign > 0 else f"G{n_groups}"}
    stats_row.update(decay_metrics(ic, ric))
    pd.DataFrame([stats_row]).to_csv(out_dir / "decay_stats.csv", index=False, encoding="utf-8-sig")

    # ---- 三面板总览图 ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    xs = list(range(len(mdf)))
    xl = [str(m)[:7] for m in mdf.index]

    ax = axes[0]
    ax.bar(xs, mdf["rank_ic"], color=LIGHT_GRAY, alpha=0.9, width=0.8, label="月度 RankIC")
    ax.plot(xs, mdf[f"roll{w1}_rankic"], color=POS_RED, lw=1.6, label=f"滚动 {w1} 月均值")
    ax.plot(xs, mdf[f"roll{w2}_rankic"], color=NEG_BLUE, lw=1.6, label=f"滚动 {w2} 月均值")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("RankIC")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    clean_axes(ax)
    _thin_xticks(ax, xs, xl)

    ax = axes[1]
    yrs = list(yearly.index)
    vals = yearly["rankic_mean"].to_numpy()
    ax.bar(range(len(yrs)), vals, color=[POS_RED if v >= 0 else NEG_BLUE for v in vals])
    ax.axhline(float(ric.mean()), color="black", lw=1.0, ls="--",
               label=f"全样本均值 {float(ric.mean()):.4f}")
    for i, nm in enumerate(yearly["n_months"]):
        if nm != 12:  # 不完整年（2016/2026 各 6 个月）标注样本数
            ax.text(i, vals[i], f"n={int(nm)}", ha="center",
                    va="bottom" if vals[i] >= 0 else "top", fontsize=7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(range(len(yrs)))
    ax.set_xticklabels(yrs, rotation=45, fontsize=8)
    ax.set_ylabel("分年均值 RankIC")
    ax.legend(fontsize=8)
    clean_axes(ax)

    ax = axes[2]
    if len(ls_trail12) > 0:
        ax.plot(range(len(ls_trail12)), ls_trail12.to_numpy(), color=CHARCOAL, lw=1.6,
                label="滚动 12 月累计净多空 (扣成本)")
        _thin_xticks(ax, range(len(ls_trail12)), [str(m)[:7] for m in ls_trail12.index])
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("累计净多空")
    ax.set_xlabel("形成月")
    ax.legend(fontsize=8)
    clean_axes(ax)

    fig.suptitle(f"{UNIVERSE_CN[uni]}｜{FACTOR_SHORT[factor]}｜{variant}：因子效果时间序列", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "decay_overview.png", dpi=180)
    plt.close(fig)

    return stats_row, yearly


# ============ 聚合：master 表 + 分年 RankIC 热力图 ============

def plot_year_heatmap(uni: str, variant: str, yearly_map: dict[str, pd.DataFrame],
                      out_path: Path) -> None:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    facs = [f for f in FACTOR_COLS if f in yearly_map]
    if not facs:
        return
    years = sorted({y for f in facs for y in yearly_map[f].index})
    m = np.full((len(facs), len(years)), np.nan)
    for i, f in enumerate(facs):
        for j, y in enumerate(years):
            if y in yearly_map[f].index:
                m[i, j] = yearly_map[f].loc[y, "rankic_mean"]
    m100 = m * 100.0
    vmax = float(np.nanmax(np.abs(m100)))
    if not np.isfinite(vmax) or vmax <= 0:
        return
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vmax=vmax, vcenter=0.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#eeeeee")

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(m100, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, fontsize=9)
    ax.set_yticks(range(len(facs)))
    ax.set_yticklabels([FACTOR_SHORT[f] for f in facs], fontsize=10, family="DejaVu Sans")
    # 白色网格线
    ax.set_xticks(np.arange(len(years)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(facs)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    # 单元格标注（×100 后 1 位小数）
    for i in range(len(facs)):
        for j in range(len(years)):
            v = m100[i, j]
            if not np.isfinite(v):
                continue
            color = "black" if abs(v) < 0.55 * vmax or vmax < 0.5 else "white"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color=color, family="DejaVu Sans")
    # A|B|C 模块分隔线与标签（仅全因子集时画）
    if len(facs) == len(FACTOR_COLS):
        for b in MOD_BOUNDS[:-1]:
            ax.axhline(b - 0.5, color="black", lw=1.1)
        for (lo, hi), label in zip([(0, 6), (7, 14), (15, 17)],
                                   ["A 微观结构", "B 时序特征", "C 成交集中度"]):
            ax.text(-0.62, (lo + hi) / 2, label, rotation=90, ha="center", va="center",
                    fontsize=11, transform=ax.get_yaxis_transform(), fontweight="bold")
    ax.set_title(f"{UNIVERSE_CN[uni]} / {variant}：分年均值 RankIC（×100，"
                 f"{years[0]}–{years[-1]}）", fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("分年均值 RankIC (×100)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_decay_ranking(master: pd.DataFrame) -> None:
    # 按幅值差升序（越负=预测强度衰减越重；对负 ICIR 反转因子同样成立）打印全量排行。
    m = master.dropna(subset=["half_mag_diff"]).sort_values("half_mag_diff").copy()
    cols = ["universe", "variant", "factor_short", "rankicir", "rankicir_h1",
            "rankicir_h2", "half_mag_diff", "trend_slope_ann", "rankic_mean_recent"]
    print("\n==== 衰减排行（half_mag_diff = |后半RankICIR| - |前半RankICIR|，升序=衰减最重在前） ====")
    with pd.option_context("display.width", 200, "display.max_rows", None,
                           "display.float_format", lambda v: f"{v: .3f}"):
        print(m[cols].to_string(index=False))


# ============ 主流程 ============

def main() -> None:
    args = parse_args()
    setup_matplotlib()
    t0 = time.time()
    output_root = args.output_root or DECAY_ROOT
    ensure_dir(output_root)

    panel = pd.read_parquet(PANEL_CACHE_DIR / "monthly_panel.parquet")
    ret_cols = panel[["code", "form_date", "next_ret_1m"]]

    unis = args.universes or list(UNIVERSES)
    facs = args.factors or list(FACTOR_COLS)
    vars_ = args.variants or list(VARIANTS)
    total = len(unis) * len(facs) * len(vars_)
    done = 0
    all_stats: list[dict] = []
    for uni in unis:
        for variant in vars_:
            path = PROCESSED_DIR / f"{uni}_{variant}.parquet"
            if not path.exists():
                print(f"[skip] {path} 不存在")
                continue
            fm = pd.read_parquet(path)
            fm["form_month"] = fm["form_month"].astype(str)
            fm = fm.merge(ret_cols, on=["code", "form_date"], how="left")
            combo_stats: list[dict] = []
            yearly_map: dict[str, pd.DataFrame] = {}
            for factor in facs:
                if factor not in fm.columns:
                    continue
                try:
                    row, yearly = run_combo(uni, factor, variant, fm, output_root)
                except Exception as e:
                    print(f"[ERROR] {uni}/{factor}/{variant}: {e}")
                    continue
                combo_stats.append(row)
                yearly_map[factor] = yearly
                done += 1
                print(f"[{done}/{total}] {uni}/{factor}/{variant} 完成")
            if combo_stats:
                agg_dir = ensure_dir(output_root / uni / variant)
                pd.DataFrame(combo_stats).to_csv(agg_dir / "decay_master.csv",
                                                 index=False, encoding="utf-8-sig")
                plot_year_heatmap(uni, variant, yearly_map,
                                  agg_dir / "heatmap_year_rankic.png")
            all_stats.extend(combo_stats)

    if all_stats:
        master = pd.DataFrame(all_stats)
        master.to_csv(output_root / "decay_master.csv", index=False, encoding="utf-8-sig")
        print_decay_ranking(master)
    print(f"全部完成，用时 {time.time()-t0:.0f}s，总表 {output_root / 'decay_master.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="因子效果时间序列衰减分析 + 可视化。")
    p.add_argument("--universes", nargs="*", default=None)
    p.add_argument("--factors", nargs="*", default=None)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--output-root", type=Path, default=None,
                   help="另存衰减结果根目录；相对路径按当前工作目录解析")
    return p.parse_args()


if __name__ == "__main__":
    main()
