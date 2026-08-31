# 日内量价路径形态因子研究

本项目研究 1 分钟 K 线中未被日频统计量充分描述的路径形态信息，构建了微观结构、量价时序与成交时序集中度三个模块、18 个日频因子，并在沪深300、中证500、中证1000和中证全指四个股票池中完成月度截面检验。

研究区间为 2016-07 至 2026-06，共 120 个形成月。正式结果采用修正权重漂移与多空方向后的 V4 口径：4 个股票池 × 18 个因子 × 2 种中性化方案，共 144 组基础回测；九风格中性化与因子衰减分析作为扩展检验。

完整研究结论见根目录的 [`report.pdf`](report.pdf)。本 README 侧重技术框架、算法与复现方式。

## 项目结构

```text
.
├── data/
│   └── README.md               # 数据来源、字段、数据形状、质量问题与读取示例
├── src/
│   ├── 02_...09_*.py           # 从原始数据到风格检验的主流水线
│   ├── 11_factor_decay.py      # 时序衰减分析
│   ├── 12_build_report_figures.py
│   ├── factors_*.py            # 三模块因子定义
│   ├── lib_common.py           # 路径、参数、因子注册表和公共函数
│   ├── lib_fdrive.py           # HDF5 数据访问层
│   ├── report/                 # 报告 Markdown、TeX 与图形资产
│   ├── requirements.txt
│   ├── run_pipeline.ps1
│   └── build_report.ps1
├── result/
│   ├── backtest_v4/            # 正式回测结果
│   ├── decay_v4/               # 与 V4 一致的衰减分析
│   ├── style_test/             # 九风格检验
│   ├── backtest_v3/            # 历史口径，仅供审计对照
│   └── decay/                  # 历史衰减结果，仅供审计对照
├── README.md
└── report.pdf
```

原始分钟行情、日频宽表与中间 Parquet 缓存不进入仓库。原因包括数据授权限制、原始分钟库体量约 819 GB，以及缓存可以由源码重新生成。所需数据的精确格式见 [`data/README.md`](data/README.md)。

## 方法框架

### 1. 统一算子语法

每个高频因子写成两类算子的嵌套链：

- 数据变换：向量到向量，例如分钟收益、成交占比、滞后差分、条件标记；
- K 线聚合：向量到标量，例如求和、标准差、熵、相关系数、条件均值和滚动聚合。

因子最后必须变为“股票-交易日”粒度的标量。本项目先由 1 分钟数据计算日度原子量，再以 20 个交易日窗口生成正式因子，其中最少要求 15 个有效日。

### 2. 时间段切割与时间点划分

日内信息定位分为三个层次：

1. 整体聚合：对全日分钟序列做无差别统计；
2. 时间段切割：用曲折度、成交放量或涨跌方向筛选具有不同经济含义的分钟区间；
3. 时间点划分：先定位累计收益新高等稀疏事件，再比较锚点与非锚点状态。

实证表明，切割本身并不必然产生增量。只有当条件能够分离经济含义相反的状态时，结构因子才显著优于总量因子。

### 3. 三模块与 18 个因子

| 模块 | 研究问题 | 因子 |
|---|---|---|
| A 微观结构 | 价格路径如何演化，噪音集中在哪些时段 | A1、A1v、A2a、A2b、A2c、A2d、A3 |
| B 时序特征 | 量与价孰先孰后，趋势是否得到成交确认 | B1、B2、B2a、B3、B4、B5a、B5b、B5c |
| C 成交时序集中度 | 成交何时发生，买卖方向的时间分布是否不同 | C1、C2、C3 |

核心定义包括：

- A1：日内路径长度与净位移之比；
- A2：按 30 分钟滚动曲折度切分高、低曲折时段；
- A3：高曲折且放量分钟占比；
- B1：成交量变化与收益之间的多阶领先滞后差；
- B5c：累计收益创新高时的量能相对非新高时段的比例；
- C1：标准化分钟成交占比熵；
- C2：上涨与下跌分钟的成交集中度差；
- C3：C1 日度原子的 20 日标准差。

完整公式、定义域和经济含义见报告附录及 `src/factors_a.py`、`src/factors_b.py`、`src/factors_c.py`。

## 数据处理与回测流水线

```text
分钟 HDF5 ──> 日度原子 ──> 20 日滚动因子 ──┐
                                              ├─> 月末面板 ─> 去极值/中性化 ─> IC 与分组回测
日频宽表 ──> 交易日历、收益、状态、行业 ──┘                         │
                                                                    ├─> 九风格检验
                                                                    └─> 时序衰减分析
```

主流程的关键实现约束：

- 单日分钟数据整块顺序读取，并散布为“股票 × 240 分钟”矩阵；
- 全部原子因子用 NumPy 沿分钟轴向量化；
- 每 42 个交易日分片落盘，使用进度文件支持断点续跑；
- 无效分钟和无定义因子值保持 NaN，不做填充；
- 月末面板按因子独立取有效样本，避免全因子交集造成样本选择偏差；
- 截面先做 5 倍 MAD 去极值，再进行施密特正交化；
- S1 控制市值与行业，S2 进一步控制换手率与波动率，S3 扩展至九类风格；
- 沪深300和中证500分 5 组，中证1000和中证全指分 10 组；
- 月度调仓，交易成本按单边 0.3% 与实际权重变化扣减。

## 环境

测试环境：Windows 11、Python 3.12、MiKTeX/XeLaTeX。

主要 Python 依赖为 NumPy、pandas、SciPy、Matplotlib、h5py 与 PyArrow；构建九风格暴露时额外需要 `cjpy` 及有权限的数据账户。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\src\requirements.txt
```

报告编译需要 XeLaTeX 和 Windows 中文字体。已有 MiKTeX 时可运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\src\build_report.ps1
```

## 数据与路径配置

默认路径兼容原研究环境，但所有关键路径都可通过环境变量覆盖：

```powershell
$env:HF_MINUTE_DIR = "E:\market_data\minute_1m"
$env:HF_BASE_DIR = "E:\market_data\daily_base"
$env:HF_CALENDAR_FILE = "E:\market_data\trading_calendar.npy"
$env:HF_CACHE_ROOT = "E:\hf_factor_cache"
$env:PYTHON_EXE = ".\.venv\Scripts\python.exe"
```

`08_build_style_exposure.py` 优先使用 `cjpy` 已持久化的凭证，也可从 `CJ_API_TOKEN` 环境变量读取。仓库不包含任何 API token、账户口令或 `.env` 文件。

## 复现

### 快速验证因子实现

该步骤只使用合成数据，不需要市场数据：

```powershell
& $env:PYTHON_EXE .\src\91_smoke_atoms.py
```

独立慢循环对照验证位于 `92_verify_atoms.py`，数据口径勾稽与中性化诊断位于 `93_` 至 `96_` 脚本。

### 完整复现

确认数据目录与缓存目录后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\src\run_pipeline.ps1
```

流水线按以下顺序运行：日频缓存、日度原子、滚动因子、月末面板、因子处理、V4 回测、九风格暴露、风格检验、V4 衰减分析和报告图生成。分钟数据阶段耗时和磁盘读写量较大，脚本支持断点续跑。

也可以逐个执行编号脚本。所有研究参数集中在 `src/lib_common.py`，复现实验时应优先修改环境变量和该文件中的参数，避免在下游脚本中重复改口径。

## 结果口径

正式结果入口：

- `result/backtest_v4/master_summary.csv`：144 组回测的 IC、RankIC、ICIR、有效方向收益、换手与基准统计；
- `result/style_test/`：九风格相关、暴露回归和 S1/S2/S3 对照；
- `result/decay_v4/decay_master.csv`：前后半样本、2025 年以来子样本、趋势和回撤等衰减指标。

`backtest_v3` 与 `decay` 保留用于口径审计，不应与正式报告数字混用。结果目录的字段说明见 [`result/README.md`](result/README.md)。

## 复现边界与风险提示

- 仓库可以复现算法与结果生成过程，但不再分发受授权限制的原始行情；
- 财务类风格因子依赖外部 point-in-time 数据权限；若无该权限，可以完成 S1/S2 主回测，但不能完整复现 S3；
- 研究使用后复权价格和特定成交量口径，替换数据源时必须重新验证量价单位、停牌行与复权规则；
- 历史回测不代表未来表现，本项目仅用于研究与方法展示，不构成投资建议。

