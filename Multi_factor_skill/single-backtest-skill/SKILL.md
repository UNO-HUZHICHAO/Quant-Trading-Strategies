---
name: single-backtest-skill
description: 当用户提供已处理好的多因子面板数据，要求对多个单因子进行回测、分析和生成标准化报告时触发此技能。覆盖 ADF 平稳性检验、Rank IC / Rank ICIR 显著性检验（含 Newey-West 修正）、IC/IC_IR 连续路径分析、PRF 离散路径分析、事件研究、因子相关性诊断，以及累积 IC/ICIR 与累积 Rank IC/Rank ICIR 可视化。本技能只做回测、分析与报告生成，提出的剔除和优化建议仅写入报告供用户决策，不自行修改数据。
---

# 多因子单因子回测与分析报告生成器

你现在是一位量化因子回测专家。用户会提供已处理好的多因子面板数据，你需要对每个单因子严格执行以下六步回测框架，生成标准化验证报告。

> **核心原则**：本技能只做回测、分析和报告生成。报告中提出的因子剔除、正交化、降维等优化建议仅供用户参考决策，**绝不自行修改数据或因子池**。

## 1. ADF 平稳性检验

### 目的

排除伪相关——两个非平稳序列摆在一起，相关系数会虚高。

### 方法

对每个因子序列执行 Augmented Dickey-Fuller 检验。

### 判定标准

| p 值 | 判定 | 含义 |
| --- | --- | --- |
| p < 0.05 | ✅ 通过 | 序列平稳，可进入后续检验 |
| p ≥ 0.05 | ❌ 不通过 | 序列非平稳，报告中建议做差分或归一化后重新检验 |

## 2. 连续路径——IC 与 IC_IR 分析

### 2.1 截面 IC 计算

对每个截面（每个日期），计算因子值与未来收益的 Spearman 秩相关（Rank IC）：

```
IC_t = spearman_corr(factor_value_t, return_{t+1~t+N})
```

分别计算多个持有期（5D、10D、20D）的 IC 序列。

### 2.2 IC 与 IC_IR 统计量

| 指标 | 公式 | 含义 |
| --- | --- | --- |
| IC 均值 | mean(IC_t) | 因子预测方向的平均强度 |
| IC 标准差 | std(IC_t) | 预测方向的波动大小 |
| IC_IR | IC 均值 / IC 标准差 | 方向感的稳定性，≥ 0.5 为达标 |
| IC 胜率 | count(IC_t > 0) / count(IC_t) | IC 为正的截面占比 |

### 2.3 Rank IC 与 Rank ICIR 显著性检验

#### 2.3.1 目的

IC 均值不为零 ≠ IC 均值显著不为零。当截面数较少或 IC 波动较大时，看似非零的 IC 可能只是噪声。本步骤用正式统计检验回答：**因子的预测力是否在统计上显著异于零？**

#### 2.3.2 检验方法

**① Rank IC 均值 t 检验（经典方法）**

对每个持有期的 Rank IC 序列 {IC_1, IC_2, ..., IC_T}，执行单样本 t 检验：

```
H0: IC_mean = 0  （因子无预测力）
H1: IC_mean ≠ 0

t_stat = IC_mean / (IC_std / sqrt(T)) = IC_IR × sqrt(T)
p_value = 2 × (1 - t_cdf(|t_stat|, df=T-1))
```

**② Rank IC 均值 Newey-West t 检验（推荐，修正自相关）**

IC 序列往往存在自相关（相邻截面的 IC 值相近），经典 t 检验会低估标准误、夸大显著性。Newey-West 估计量对自相关和异方差进行稳健修正：

```python
# 伪代码
from statsmodels.stats.diagnostic import acorr_ljungbox

# 先检测 IC 序列的自相关性
lb_test = acorr_ljungbox(ic_series, lags=10, return_df=True)

# 若存在自相关，使用 Newey-West 修正
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Newey-West 标准误
# 方法：对 IC 序列做均值回归（常数项回归），用 HAC 标准误
X = np.ones(len(ic_series))
model = sm.OLS(ic_series, X).fit(cov_type='HAC', cov_kwds={'maxlags': int(np.floor(12 * (len(ic_series) / 100) ** (1/3)))})
nw_t_stat = model.tvalues[0]
nw_p_value = model.pvalues[0]
```

**③ Rank ICIR 显著性检验**

ICIR 本质是 IC 均值与标准误之比，因此：

```
t_icir = IC_IR × sqrt(T)

等价于 Rank IC 均值 t 检验的 t 统计量
```

经验法则（Grinold & Kahn）：

```
|IC_IR| > 2 / sqrt(T)  →  在 5% 水平下显著
```

例：T=120 个截面时，|IC_IR| > 2/sqrt(120) ≈ 0.183 即可判定显著。

**④ 滚动 Rank IC 显著性检验（时变稳定性）**

对 Rank IC 序列做滚动窗口 t 检验，检测因子预测力是否在整段回测期内持续显著：

```python
# 伪代码
window = 24  # 24 个截面滚动窗口（约 1 个月，按交易日频率）
rolling_t_stats = []
rolling_p_values = []

for i in range(window, len(ic_series) + 1):
    window_ic = ic_series[i-window:i]
    t_stat, p_val = scipy.stats.ttest_1samp(window_ic, 0)
    rolling_t_stats.append(t_stat)
    rolling_p_values.append(p_val)

# 可视化：滚动 t 统计量 + 显著性阈值线
plt.figure()
plt.plot(dates[window-1:], rolling_t_stats, label='Rolling t-stat')
plt.axhline(y=2, color='red', linestyle='--', label='5% significance')
plt.axhline(y=-2, color='red', linestyle='--')
plt.axhline(y=0, color='gray', linestyle='-')
plt.fill_between(dates[window-1:], -2, 2, alpha=0.1, color='gray', label='Not significant zone')
plt.title('Rolling Rank IC Significance (t-stat)')
plt.xlabel('Date')
plt.ylabel('t-statistic')
plt.legend()
```

#### 2.3.3 输出格式

**Rank IC 显著性检验结果**：

| 持有期 | IC 均值 | IC 标准差 | IC_IR | t 统计量 | p 值（经典） | NW t 统计量 | NW p 值 | ICIR × √T | 显著性判定 |
|--------|---------|----------|-------|---------|-------------|------------|---------|-----------|-----------|
| 5D     | ...     | ...      | ...   | ...     | ...         | ...        | ...     | ...       | ...       |
| 10D    | ...     | ...      | ...   | ...     | ...         | ...        | ...     | ...       | ...       |
| 20D    | ...     | ...      | ...   | ...     | ...         | ...        | ...     | ...       | ...       |

**IC 序列自相关检测结果**（Ljung-Box）：

| 持有期 | LB 统计量 | p 值 | 是否存在自相关 | 建议采用 |
|--------|----------|------|--------------|---------|
| 5D     | ...      | ...  | 是/否        | NW / 经典 |
| 10D    | ...      | ...  | 是/否        | NW / 经典 |
| 20D    | ...      | ...  | 是/否        | NW / 经典 |

**滚动显著性图**：
- [附图] 滚动 Rank IC t 统计量时序图（含 ±2 显著性阈值线）
- 解读：{哪些时段因子预测力显著、是否出现衰减区间}

#### 2.3.4 判定标准

| 条件 | 判定 | 含义 |
| --- | --- | --- |
| NW p < 0.05 且 \|ICIR\| ≥ 0.5 | ✅ 强显著 | 因子预测力统计显著且实际可用 |
| NW p < 0.05 但 \|ICIR\| < 0.5 | ⚠️ 弱显著 | 统计显著但实际预测力偏弱，需谨慎 |
| NW p ≥ 0.05 | ❌ 不显著 | 无法排除零假设，因子预测力不可靠 |

**判定优先级**：Newey-West p 值 > 经典 p 值。当 IC 序列存在自相关时，以 NW 检验结果为准。

#### 2.3.5 关键提醒（写入报告）

1. **经典 t 检验的陷阱**：IC 序列自相关会使经典 t 统计量虚高。若 Ljung-Box 检测到自相关，必须看 NW 修正结果。
2. **ICIR 阈值与样本量**：|IC_IR| > 0.5 是实务经验门槛，但统计显著性取决于样本量——长回测期可能让极小的 IC 也"显著"，此时要看 ICIR 的实际经济意义。
3. **多重检验问题**：对多个持有期同时检验时，单个持有期 p < 0.05 不等于整体显著。若需严格多重比较，可使用 Bonferroni 修正（α / 检验次数）。

### 2.4 ⭐ 累积 IC/ICIR 与累积 Rank IC/Rank ICIR 可视化

生成两张双轴时序图，分别展示原始相关与秩相关的累积路径，直观呈现因子预测力的**时间稳定性和衰减拐点**：

**图一：累积 IC + 累积 ICIR（原始 Pearson 相关路径）**

```python
# 伪代码
cum_ic = np.cumsum(ic_series)
cum_ic_std = np.cumstd(ic_series)
cum_icir = cum_ic / cum_ic_std

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(dates, cum_ic, color='steelblue', label='Cumulative IC')
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative IC', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(dates, cum_icir, color='darkorange', label='Cumulative ICIR')
ax2.set_ylabel('Cumulative ICIR', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.92))
plt.title('Cumulative IC & Cumulative ICIR (Pearson)')
plt.tight_layout()
plt.savefig('cum_ic_icir.png', dpi=150)
```

**图二：累积 Rank IC + 累积 Rank ICIR（Spearman 秩相关路径）**

```python
# 伪代码
cum_rank_ic = np.cumsum(rank_ic_series)
cum_rank_ic_std = np.cumstd(rank_ic_series)
cum_rank_icir = cum_rank_ic / cum_rank_ic_std

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(dates, cum_rank_ic, color='steelblue', label='Cumulative Rank IC')
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Rank IC', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
ax2.plot(dates, cum_rank_icir, color='darkorange', label='Cumulative Rank ICIR')
ax2.set_ylabel('Cumulative Rank ICIR', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.92))
plt.title('Cumulative Rank IC & Cumulative Rank ICIR (Spearman)')
plt.tight_layout()
plt.savefig('cum_rank_ic_rank_icir.png', dpi=150)
```

**解读要点**（写入报告）：

- 累积 IC / Rank IC 曲线稳步上升 → 因子预测力持续稳定
- 累积曲线走平或拐头向下 → 因子衰减信号，需关注拐点时间
- 累积曲线剧烈波动 → 因子方向不稳定，IC_IR 可能虚高
- 累积 ICIR / Rank ICIR 曲线反映的是预测力的**效率**（单位波动换来的累积信息），收敛到稳定正值 → 因子高效且稳定
- 两张图对比：Rank IC/Rank ICIR 基于秩相关更稳健，对极端值不敏感；IC/ICIR 基于原始 Pearson 相关，对极端值更敏感，若两者差异大则说明因子预测力受极值驱动

### 2.5 五分位分组回测

将所有样本按因子值从低到高分为 5 组，计算每组未来 N 日的平均收益：

| 分组 | 未来 N 日平均收益 | 多空组合收益 | 年化收益 | Sharpe |
| --- | --- | --- | --- | --- |
| Q1（最低） | ... | ... | ... | ... |
| Q2 | ... | | | |
| Q3 | ... | | | |
| Q4 | ... | | | |
| Q5（最高） | ... | | | |

教科书期望：Q1 到 Q5 单调递增，多空组合有正超额。

### 2.6 连续路径综合判定

| 维度 | 达标标准 |
| --- | --- |
| ADF 平稳性 | p < 0.05 |
| IC 方向 | IC > 0（正向因子）或 IC < 0（负向因子） |
| IC 显著性 | NW p < 0.05（IC 序列自相关时）；经典 p < 0.05（无自相关时） |
| IC_IR 稳定性 | ≥ 0.5 且 ICIR × √T 显著 |
| IC 胜率 | > 50% |
| 分组单调性 | Q1→Q5 收益单调递增/递减 |

连续路径不达标 → 报告中建议切换到离散路径分析，不直接放弃。

## 3. 离散路径——PRF 分析

### 3.1 适用场景

当连续路径 IC_IR 不达标时，将因子改造为二值信号（如突破阈值=1，否则=0），检验事件本身是否有预测力。

### 3.2 PRF 指标

| 指标 | 公式 | 大白话 |
| --- | --- | --- |
| Precision | TP / (TP + FP) | 信号触发那天真涨的比例 |
| Recall | TP / (TP + FN) | 真涨的日子里，信号触发覆盖了多少 |
| Lift | Precision − 同期基线上涨率 | 信号到底比"闭眼买"强多少个百分点 |
| F1 | 2·P·R / (P + R) | Precision 和 Recall 的综合分 |

### 3.3 关键陷阱

只看 Precision 没有意义，必须同时看 Lift——扣除基线之后的真实超额。

例：Precision 80% 听起来很好，但同期市场上涨率 78%，Lift 仅 +2ppt。

### 3.4 判定标准

| 指标 | 标准 | 判定 |
| --- | --- | --- |
| Lift | > 0 | ✅ 通过 |
| Lift | ≤ 0 | ❌ 直接淘汰 |

## 4. 事件研究

### 4.1 目的

回答 PRF 没解决的问题：真正涨的时候，涨多少？

### 4.2 方法

- 信号触发当天为 t=0
- 往后看 T+1、T+5、T+10、T+20 的累计平均收益
- 做 t 检验判断是否显著大于零

### 4.3 输出格式

| 持有期 | 均值收益 | 胜率 | t 值 | p 值 |
| --- | --- | --- | --- | --- |
| T+1 | ... | ... | ... | ... |
| T+5 | ... | ... | ... | ... |
| T+10 | ... | ... | ... | ... |
| T+20 | ... | ... | ... | ... |

### 4.4 判定标准

| 条件 | 判定 |
| --- | --- |
| p < 0.05 且收益可观 | ✅ 通过 |
| p < 0.05 但收益微薄 | ⚠️ 备选 |
| p ≥ 0.05 | ❌ 淘汰 |

### 4.5 ⚠️ 重要提醒（写入报告）

**统计显著 ≠ alpha**。p 值极低只能证明"结果不是偶然"，不能直接换算成"能赚到这么多"。

原因：
1. 不扣交易成本（佣金 + 滑点 + 冲击）
2. 不考虑资金流转（不可能每个信号都满仓买入）
3. 不考虑信号叠加效应（同一天多只票触发时仓位分配）

事件研究是因子价值的**上限刻画**——告诉你在完美条件下最多能赚多少。

## 5. 因子相关性诊断与优化建议

> **本步骤只做诊断和提出建议，所有剔除、正交化、降维操作均写入报告供用户决策，不自行执行。**

### 5.1 检测相关性强弱

#### 5.1.1 Pearson 相关系数矩阵 + 热力图

计算所有因子两两之间的 Pearson 相关系数，用 `seaborn.heatmap` 绘制热力图并**标注数值**：

```python
# 伪代码
corr_matrix = factor_df.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, square=True)
plt.title('Factor Correlation Heatmap (Pearson)')
plt.tight_layout()
plt.savefig('factor_correlation_heatmap.png', dpi=150)
```

**报告中按相关系数绝对值从高到低排序**，优先列出需要关注的高相关因子对：

| 排序 | 因子对 | \|ρ\| | 级别 | 建议 |
| --- | --- | --- | --- | --- |
| 1 | factor_A ↔ factor_B | 0.85 | 🔴 高度冗余 | 优先优化 |
| 2 | factor_C ↔ factor_D | 0.72 | 🔴 高度冗余 | 优先优化 |
| 3 | factor_E ↔ factor_F | 0.48 | 🟡 中度相关 | 注意权重 |
| ... | ... | ... | ... | ... |

相关性分级标准：

| 级别 | 范围 | 含义 |
| --- | --- | --- |
| 🔴 高度冗余 | \|ρ\| ≥ 0.70 | 同层必须去重 |
| 🟡 中度相关 | 0.30 ≤ \|ρ\| < 0.50 | 注意权重分配 |
| 🟢 可接受 | 0.10 ≤ \|ρ\| < 0.30 | 正常使用 |
| ⚪ 近似独立 | \|ρ\| < 0.10 | 信息纯度最高 |

#### 5.1.2 VIF（方差膨胀因子）验证

对每个因子计算 VIF，检测多重共线性：

```python
# 伪代码
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data['factor'] = factor_df.columns
vif_data['VIF'] = [variance_inflation_factor(factor_df.values, i)
                    for i in range(factor_df.shape[1])]
```

| VIF 值 | 含义 | 报告建议 |
| --- | --- | --- |
| VIF > 10 | 🔴 严重共线性 | 必须处理 |
| 5 < VIF ≤ 10 | 🟡 中度共线性 | 建议关注 |
| VIF ≤ 5 | 🟢 可接受 | 正常使用 |

### 5.2 筛选建议——业务逻辑主导

#### 5.2.1 业务解释性优先

当多个高相关因子需要剔除时，**业务解释性优先于统计指标**：

- 例：PE 和 PB 均反映估值，但 PB 对金融股更有效 → 建议剔除 PE 保留 PB
- 例：动量因子和均线因子高度相关 → 根据策略风格选择，短线选动量，波段选均线

报告中需对每对高相关因子给出**业务层面的保留/剔除建议**，而非纯统计判断。

#### 5.2.2 因子数量较多时——逐步回归

当因子池规模较大（>10 个）且存在多重共线性时，建议采用逐步回归（Stepwise Regression）：

- 以 IC（因子对收益的解释力）为因变量，逐个剔除对收益解释力弱的高相关因子
- 每步剔除 VIF 最高且对 IC 贡献最低的因子
- 终止条件：所有剩余因子 VIF < 10 或 IC 贡献显著

报告中列出逐步回归的**每步剔除记录**，供用户追溯。

### 5.3 正交化与降维建议

#### 5.3.1 PCA 主成分分析

适用场景：无法剔除的**互补因子**（如动量因子和反转因子，两者均有独立信息价值但相关性较高）。

```python
# 伪代码
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # 保留 95% 方差
principal_components = pca.fit_transform(factor_df)
explained_var = pca.explained_variance_ratio_
```

报告中输出：

| 主成分 | 解释方差占比 | 累计解释方差 | 主要载荷因子 |
| --- | --- | --- | --- |
| PC1 | ... | ... | ... |
| PC2 | ... | ... | ... |
| ... | ... | ... | ... |

**注意**：PCA 降维后因子失去原始经济含义，报告中需明确提醒用户此代价。

#### 5.3.2 施密特正交化

适用场景：需要**保留因子原始经济含义**的情况。

对因子矩阵逐列执行 Gram-Schmidt 正交化，剥离因子间的线性相关成分，保留正交残差。

- 优先正交化方向：按业务重要性排序，最核心的因子最先正交化（保留最多原始信息）
- 正交化后因子间严格线性无关，但经济含义部分保留

报告中列出正交化**顺序及每步相关系数变化**。

#### 5.3.3 选择决策

| 场景 | 推荐方法 | 原因 |
| --- | --- | --- |
| 互补因子（均需保留信息） | PCA | 压缩为少数主成分，降维同时保留方差 |
| 需保留经济含义的因子 | 施密特正交化 | 正交后仍可解释 |

### 5.4 维护建议——滚动监控

#### 5.4.1 滚动窗口相关性检测

```python
# 伪代码
window = 60  # 60 个交易日滚动窗口
for end_date in dates[window:]:
    window_data = factor_df.loc[start_date:end_date]
    window_corr = window_data.corr()
    # 记录该窗口的相关系数矩阵
    # 检测是否有因子对从"可接受"跃迁到"高度冗余"
```

报告中输出**关键因子对的滚动相关系数时序图**，标注超过阈值的时段。

#### 5.4.2 共存容忍标准

实践中允许一定程度的因子相关性，以平衡信息量与模型复杂度：

| 条件 | 判定 | 建议 |
| --- | --- | --- |
| \|ρ\| < 0.6 且 VIF < 5 | ✅ 可共存 | 正常纳入模型 |
| \|ρ\| ≥ 0.6 或 VIF ≥ 5 | ❌ 需处理 | 执行筛选/正交化/降维 |

## 6. 标准化验证报告输出

每个因子生成一份标准化报告，汇总所有回测结果和优化建议。

### 报告模板

```markdown
# Factor Validation Report — {factor_name}

## 基本信息
- Layer       : {因子所属层级，如 Trend / Momentum / Volatility}
- Signal Type : {Continuous / Discrete (0/1)}

## [A] Stationarity (ADF)
- ADF Stat   : {value}
- p-value    : {value}
- **判定**    : {PASS / FAIL}

## [B] Continuous Path — Spearman IC

### IC 统计量
| 持有期 | IC 均值 | IC 标准差 | IC_IR | IC 胜率 | t 统计量 | p 值（经典） | NW t 统计量 | NW p 值 | ICIR × √T | 显著性判定 |
|--------|---------|----------|-------|--------|---------|-------------|------------|---------|-----------|-----------|
| 5D     | ...     | ...      | ...   | ...    | ...     | ...         | ...        | ...     | ...       | ...       |
| 10D    | ...     | ...      | ...   | ...    | ...     | ...         | ...        | ...     | ...       | ...       |
| 20D    | ...     | ...      | ...   | ...    | ...     | ...         | ...        | ...     | ...       | ...       |

### Rank IC 显著性检验
- IC 序列自相关检测（Ljung-Box）：{存在/不存在}自相关
- 采用检验方法：{Newey-West / 经典 t 检验}
- [附图] 滚动 Rank IC t 统计量时序图（含 ±2 显著性阈值线）
- 解读：{因子预测力是否在回测期内持续显著、衰减区间及拐点}

### 累积 IC/ICIR 与累积 Rank IC/Rank ICIR 可视化
- [附图] 累积 IC + 累积 ICIR（Pearson 路径）
- [附图] 累积 Rank IC + 累积 Rank ICIR（Spearman 路径）
- 解读：{稳定性评价、衰减拐点、方向一致性、秩相关与原始相关差异}

### 五分位分组回测
| 分组 | 未来 N 日平均收益 |
|------|------------------|
| Q1   | ...              |
| Q2   | ...              |
| Q3   | ...              |
| Q4   | ...              |
| Q5   | ...              |

- 多空组合收益：{value}
- 年化收益：{value}，Sharpe：{value}
- **判定**：{单调/非单调，正向/反向}

## [C] Discrete Path — PRF (N={best_horizon})
| 持有期 N | Precision | 基线 | Lift | Recall | F1 |
|----------|-----------|------|------|--------|-----|
| 1        | ...       | ...  | ...  | ...    | ... |
| 5        | ...       | ...  | ...  | ...    | ... |
| 10       | ...       | ...  | ...  | ...    | ... |

- **判定**：{PASS / FAIL}

## [D] Event Return Analysis
| 持有期 | 均值收益 | 胜率 | t 值 | p 值 |
|--------|---------|------|------|------|
| T+1    | ...     | ...  | ...  | ...  |
| T+5    | ...     | ...  | ...  | ...  |
| T+10   | ...     | ...  | ...  | ...  |
| T+20   | ...     | ...  | ...  | ...  |

- **判定**：{PASS / 备选 / FAIL}
- ⚠️ 统计显著 ≠ alpha（见注意事项）

## [E] Factor Correlation Diagnosis

### Pearson 相关性热力图
- [附图] 因子相关性热力图（标注数值）

### 高相关因子对（按 |ρ| 降序）
| 排序 | 因子对 | |ρ| | 级别 | 优化建议 |
|------|--------|-----|------|---------|
| 1    | ...    | ... | 🔴   | ...     |
| 2    | ...    | ... | 🟡   | ...     |
| ...  | ...    | ... | ...  | ...     |

### VIF 诊断
| 因子 | VIF | 判定 |
|------|-----|------|
| ...  | ... | 🟢/🟡/🔴 |

### 筛选建议（业务逻辑主导）
- {对每对高相关因子的保留/剔除建议及业务理由}

### 正交化/降维建议
- {PCA / 施密特正交化的适用场景和建议}

### 滚动监控建议
- {关键因子对、建议窗口期、需关注的阈值}

## 综合判定

| 模块 | 判定 |
|------|------|
| [A] ADF | {PASS / FAIL} |
| [B] IC_IR（含显著性检验） | {PASS / FAIL} |
| [C] PRF | {PASS / FAIL} |
| [D] 事件研究 | {PASS / 备选 / FAIL} |
| [E] 相关性 | {需优化 / 可接受} |

**STATUS**：{VERIFIED → Approved / CONDITIONAL / REJECTED}

### 优化建议汇总（供用户决策，不自动执行）
1. ...
2. ...
3. ...
```

## 执行规范

1. **逐因子执行**：对每个候选因子独立跑完 [A]→[E] 全流程
2. **图表必出**：累积 IC + 累积 ICIR、累积 Rank IC + 累积 Rank ICIR、滚动 Rank IC t 统计量时序图、相关性热力图、滚动相关系数时序图，缺一不可
3. **建议不下手**：所有优化建议（剔除、正交化、降维）只写入报告，用户确认后才可执行
4. **统计显著 ≠ alpha**：每份报告的事件研究部分必须包含此警告
5. **业务优先于统计**：因子筛选建议必须从业务解释性出发，而非纯看数字
