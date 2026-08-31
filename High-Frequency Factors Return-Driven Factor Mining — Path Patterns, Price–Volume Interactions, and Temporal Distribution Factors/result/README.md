# 结果目录说明

本目录保存回测统计表和图形，不包含原始行情或中间因子面板。

## 正式结果

- `backtest_v4/`：报告采用的正式回测口径。修正了权重漂移下的换手计算，并显式记录有效多空方向；
- `decay_v4/`：与 V4 回测口径一致的滚动 RankIC、分年结果、前后半样本与近期子样本分析；
- `style_test/`：九风格相关矩阵、暴露回归、S1/S2/S3 RankICIR 对照与合成检验；
- `bt_summary.json`：部分报告绘图脚本使用的聚合结果。

`backtest_v3/` 和 `decay/` 是历史口径，只用于审计差异，不应与 `report.pdf` 的正式结论混用。

## 回测目录层级

```text
backtest_v4/
├── master_summary.csv
└── {universe}/{factor}/{variant}/
    ├── ic_stats.csv
    ├── ic_yearly.csv
    ├── group_stats.csv
    ├── yearly_stats.csv
    ├── cum_ic_rankic.png
    ├── cum_icir_rankicir.png
    └── group_nav.png
```

其中：

- `universe` 为 `hs300`、`zz500`、`zz1000` 或 `zzall`；
- `factor` 为 18 个正式因子的英文列名；
- `variant` 为 S1 或 S2；
- S1 控制市值和行业；
- S2 进一步控制换手率和波动率。

## master_summary 主要字段

| 字段 | 含义 |
|---|---|
| `ic_mean` / `icir` | Pearson IC 月均值及年化信息比率 |
| `rankic_mean` / `rankicir` | Spearman RankIC 月均值及年化信息比率 |
| `ic_win_rate` | 月度 IC 与全样本方向一致的比例 |
| `g1_ann` / `gN_ann` | 首尾组净年化收益 |
| `ls_ann` | 固定 `GN-G1` 方向的净多空年化 |
| `ls_ann_gross` | 不扣成本的固定方向多空年化 |
| `effective_sign` | 根据全样本两端组合确定的有效方向 |
| `ls_eff_ann` | 有效方向的净多空年化 |
| `long_group` / `short_group` | 有效方向对应的多头组与空头组 |
| `turnover_mean` | 单组平均月换手 |
| `ls_turnover_mean` | 多空双腿平均月换手 |
| `bench_ann` | 基准年化收益 |

回测统计使用 2016-07 至 2026-06 的 120 个形成月，月末构建组合并持有下一个月。交易成本为单边 0.3%，按实际权重变化扣减。

