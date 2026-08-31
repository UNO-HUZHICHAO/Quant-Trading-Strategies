# ------------------ lib_common.py ----------------
# 作用：全流水线共享的常量、路径、参数与工具函数。
# 所有编号脚本（02~07）都从这里取路径和口径参数，保证全流程口径一致。
# 口径依据：研究计划/V2.md + 因子研究/docs/移动硬盘数据说明.md + 批准的实施计划。

#未来特性开关导入：（1）写类型注解更方便；2.减少向前引用、循环引用的问题
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ============ 一、路径常量 ============

# 当前脚本自己的绝对路径（lib_common.py）。
_LIB_PATH = Path(__file__).resolve()
# GitHub 项目根目录（src 的上一级）。
RESEARCH_ROOT = _LIB_PATH.parents[1]
# 脚本目录本身。
SCRIPTS_DIR = RESEARCH_ROOT / "src"
# 研究成果输出目录（图表、统计表等小文件放项目内，方便查看）。
RESULTS_ROOT = RESEARCH_ROOT / "result"
OUTPUTS_ROOT = RESULTS_ROOT / "backtest_v4"

# 移动硬盘（只读数据源，原始数据一律不拷贝）。
F_MINUTE_DIR = Path(os.environ.get("HF_MINUTE_DIR", "F:/highfreqnew"))
F_BASE_DIR = Path(os.environ.get("HF_BASE_DIR", "F:/base"))
F_CALENDAR_NPY = Path(os.environ.get("HF_CALENDAR_FILE", "F:/highfreq/minite_jihe.npy"))

# 本机 D 盘缓存根目录（"就地计算、只带结果"，全部中间件落这里）。
CACHE_ROOT = Path(os.environ.get("HF_CACHE_ROOT", "D:/hf_factor_cache"))
DAILY_CACHE_DIR = CACHE_ROOT / "daily"         # 阶段1：日频缓存 parquet
ATOMIC_CACHE_DIR = CACHE_ROOT / "factors_atomic"  # 阶段2：日度原子因子（按日分片）
PANEL_CACHE_DIR = CACHE_ROOT / "panels"        # 日频因子面板 / 月末面板 / 处理后因子

# Python 解释器（仅用于提示，脚本自身直接用该环境运行即可）。
PYTHON_EXE = os.environ.get("PYTHON_EXE", "python")

# ============ 二、回测区间 ============

# 分钟数据处理起点：首个形成月 2016-07 需要 20 日滚动预热，再多留约 1 个月缓冲。
MINUTE_START = "20160501"
# 日频缓存窗口起点：早于分钟起点，保证 21 日换手/波动风格与月末采样有足够前置数据。
CACHE_START_DAILY = "2016-01-01"
# 数据终点（硬盘数据止于 2026-07-31）。
DATA_END = "20260731"
# 首个形成月与末个形成月（形成月末 t 用 ≤t 数据算因子，持有 t+1 月）。
FIRST_FORM_MONTH = "2016-07"
LAST_FORM_MONTH = "2026-06"

# ============ 三、计算参数 ============

EPSILON = 1e-12          # 防除零小量
MIN_VALID_MINUTES = 120  # 单股单日有效分钟数下限，不足整日剔除
ROLL_WIN = 20            # 20 日滚动窗口（V2：Mean20d / Std20d）
MIN_PERIODS = 15         # 滚动窗口最少有效天数
MAD_K = 5                # 5 倍 MAD 极值剔除
MAD_SCALE = 1.4826       # MAD→标准差换算系数（正态下一致）
COST = 0.003             # 单边交易成本 千三（V2 §6.1）
LIST_MIN_DAYS = 60       # 上市不足 60 交易日剔除（V2 §6.1）
SUSP_MAX_DAYS = 5        # 形成月内停牌超过 5 个交易日剔除（V2 §6.1）
TAU_WINDOW = 30          # A2 切割用的日内滚动曲折度窗口（30 分钟，V2 §2.2）
B1_MAX_LAG = 3           # B1 多阶滞后 k=1..3（V2 §3.2）
B1_MIN_PAIRS = 20        # B1 交叉相关最少有效样本对
B3_MIN_OBS = 15          # B3 上/下行子集各自最少分钟数
B5A_ROLL = 60            # B5a 滚动量价相关窗口（60 分钟）
B5A_EARLY_BAR = 59       # B5a 早盘确认度取第 60 根 bar（0-based 59，即 10:30）

# 分组数：沪深300/中证500 五组，中证1000/中证全指 十组（研报口径）。
N_GROUPS = {"hs300": 5, "zz500": 5, "zz1000": 10, "zzall": 10}

# ============ 四、股票池与基准 ============

# 四个回测股票池（内部名 -> index_weight.hdf 组名 / price_index.hdf Equity 列名）。
UNIVERSES = ("hs300", "zz500", "zz1000", "zzall")
UNIVERSE_CN = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000", "zzall": "中证全指"}
UNIVERSE_IDXCODE = {"hs300": "SH000300", "zz500": "SH000905", "zz1000": "SH000852", "zzall": "SH000985"}
UNIVERSE_BENCH = {"hs300": "000300.SH", "zz500": "000905.SH", "zz1000": "000852.SH", "zzall": "000985.CSI"}

# ============ 五、因子注册表 ============

# 日度原子因子（03 由分钟数据直接算出，每只股票每个交易日一行）。
# A2d 的原子量是高曲折时段成交占比 A2c_atom（乘日换手率在 04 合成）；
# C3 的原子量是 C1_entropy_atom（20 日标准差在 04 合成）。
ATOM_COLS = [
    "A1_tau_atom",            # 日度曲折度 τ = Σ|r|/(|Σr|+ε)
    "A1v_tau_vol_atom",       # 量加权曲折度（V2 A1 变体）
    "A2a_high_tau_ret_std_atom",   # 高曲折时段收益 std
    "A2b_low_tau_cum_ret_atom",    # 低曲折时段累计收益
    "A2c_high_tau_vol_share_atom", # 高曲折时段成交占比 Σd·w（A2d 的成分）
    "A3_tort_vol_joint_atom",      # 高曲折且放量分钟占比
    "B1_lead_lag_atom",       # 量价领先滞后差 LL（1/2/3 阶 1/k 加权）
    "B2_path_asym_atom",      # 路径不对称比 PA（量版）
    "B2a_path_asym_amt_atom", # 路径不对称比 PA（成交额版，V2 B2 变体）
    "B3_cond_lead_lag_atom",  # 条件化领先滞后差 DLL = LL+ − LL−
    "B4_cross_atom",          # (PA−1)×LL 交互项
    "B5a_corr_decay_atom",    # 量价确认度衰减 ρ_late − ρ_early
    "B5b_slope_div_atom",     # 趋势斜率背离 −β̂p·β̂v
    "B5c_high_vol_ratio_atom",# 新高量能不足比 VR
    "C1_entropy_atom",        # 标准化成交熵 H/ln(240)（C3 的成分，V2 原定义）
    "C2_entropy_diff_atom",   # 条件熵差 H+ − H−
    "C1_H_atom",              # 原始熵 H（诊断/口径切换用，不参与因子合成）
    "C1_N_atom",              # 有正成交量分钟数 N（同上）
]

# 最终 18 个因子（输出列名，命名与 cc代码plan 一致）。
FACTOR_COLS = [
    "A1_tau_20d",
    "A1v_tau_vol_20d",
    "A2a_high_tau_ret_std_20d",
    "A2b_low_tau_cum_ret_20d",
    "A2c_high_tau_vol_share_20d",
    "A2d_high_tau_turnover_20d",
    "A3_tort_vol_joint_20d",
    "B1_lead_lag_20d",
    "B2_path_asym_20d",
    "B2a_path_asym_amt_20d",
    "B3_cond_lead_lag_20d",
    "B4_cross_20d",
    "B5a_corr_decay_20d",
    "B5b_slope_div_20d",
    "B5c_high_vol_ratio_20d",
    "C1_entropy_20d",
    "C2_entropy_diff_20d",
    "C3_entropy_std_20d",
]

# 原子列 -> 因子列 的滚动均值映射（04 用）。
# 其中 A2c 原子对应两个因子：自身均值（A2c）与乘换手率后的均值（A2d）。
ATOM_TO_FACTOR_MEAN = {
    "A1_tau_atom": "A1_tau_20d",
    "A1v_tau_vol_atom": "A1v_tau_vol_20d",
    "A2a_high_tau_ret_std_atom": "A2a_high_tau_ret_std_20d",
    "A2b_low_tau_cum_ret_atom": "A2b_low_tau_cum_ret_20d",
    "A2c_high_tau_vol_share_atom": "A2c_high_tau_vol_share_20d",
    "A3_tort_vol_joint_atom": "A3_tort_vol_joint_20d",
    "B1_lead_lag_atom": "B1_lead_lag_20d",
    "B2_path_asym_atom": "B2_path_asym_20d",
    "B2a_path_asym_amt_atom": "B2a_path_asym_amt_20d",
    "B3_cond_lead_lag_atom": "B3_cond_lead_lag_20d",
    "B4_cross_atom": "B4_cross_20d",
    "B5a_corr_decay_atom": "B5a_corr_decay_20d",
    "B5b_slope_div_atom": "B5b_slope_div_20d",
    "B5c_high_vol_ratio_atom": "B5c_high_vol_ratio_20d",
    "C1_entropy_atom": "C1_entropy_20d",
    "C2_entropy_diff_atom": "C2_entropy_diff_20d",
}
# C3 = C1 原子的 20 日标准差（不是均值）。
C3_SOURCE_ATOM = "C1_entropy_atom"

# 中性化变体：S1=市值+行业；S2=市值+行业+换手率+波动率。算法统一为施密特正交化。
VARIANTS = ("S1", "S2")
VARIANT_CN = {"S1": "市值+行业", "S2": "市值+行业+换手率+波动率"}

# ============ 五·二、因子时间序列衰减分析（11_factor_decay） ============

# 衰减分析输出根目录（与正式版回测口径一致）。
DECAY_ROOT = RESULTS_ROOT / "decay_v4"
# 月度 RankIC 滚动窗口（月）。
DECAY_ROLL_WINDOWS = (12, 24)
# 近期子样本起点（呼应研报 V2 §5.6 "2025 年以来 IC 普遍下降"）。
DECAY_RECENT_START = "2025-01"

# 因子短码（目录名 A1_tau_20d -> A1，热力图与汇总表用）。
FACTOR_SHORT = {
    "A1_tau_20d": "A1",
    "A1v_tau_vol_20d": "A1v",
    "A2a_high_tau_ret_std_20d": "A2a",
    "A2b_low_tau_cum_ret_20d": "A2b",
    "A2c_high_tau_vol_share_20d": "A2c",
    "A2d_high_tau_turnover_20d": "A2d",
    "A3_tort_vol_joint_20d": "A3",
    "B1_lead_lag_20d": "B1",
    "B2_path_asym_20d": "B2",
    "B2a_path_asym_amt_20d": "B2a",
    "B3_cond_lead_lag_20d": "B3",
    "B4_cross_20d": "B4",
    "B5a_corr_decay_20d": "B5a",
    "B5b_slope_div_20d": "B5b",
    "B5c_high_vol_ratio_20d": "B5c",
    "C1_entropy_20d": "C1",
    "C2_entropy_diff_20d": "C2",
    "C3_entropy_std_20d": "C3",
}
# A|B|C 模块边界（热力图分隔线用，原散落在 make_heatmap.py）。
MOD_BOUNDS = [7, 15, 18]

# ============ 六、工具函数 ============


def ensure_dir(path: Path) -> Path:
    # 目录不存在就递归创建，已存在不报错。
    path.mkdir(parents=True, exist_ok=True)
    return path


def yyyymmdd_to_ts(s: str) -> pd.Timestamp:
    # "20160501" -> Timestamp("2016-05-01")。
    return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}")


def ts_to_yyyymmdd(ts: pd.Timestamp) -> str:
    # Timestamp -> "20160501"。
    return ts.strftime("%Y%m%d")


def list_minute_files(start: str = MINUTE_START, end: str = DATA_END) -> list[Path]:
    # 列出 [start, end] 闭区间内的全部分钟数据文件（按文件名即日期排序）。
    # 文件名形如 20160503.hdf，直接按字符串比较即可保证日期顺序。
    files = []
    for p in sorted(F_MINUTE_DIR.glob("*.hdf")):
        name = p.stem
        if start <= name <= end:
            files.append(p)
    return files


def load_trade_calendar() -> pd.DatetimeIndex:
    # 交易日历：02 运行时从 close.hdf 日期轴导出 calendar.csv（一行一个日期）。
    # 日频层的一切对齐都以它为准。
    cal_path = DAILY_CACHE_DIR / "calendar.csv"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"未找到 {cal_path}，请先运行 02_build_base_cache.py 生成日频缓存。"
        )
    df = pd.read_csv(cal_path, header=None, names=["date"], dtype=str)
    return pd.DatetimeIndex(pd.to_datetime(df["date"]))


def load_progress(progress_path: Path) -> set[str]:
    # 读断点续跑进度文件（已完成的日期集合，元素为 YYYYMMDD）。
    if not progress_path.exists():
        return set()
    with open(progress_path, "r", encoding="utf-8") as f:
        return set(json.load(f).get("done", []))


def save_progress(progress_path: Path, done: set[str], failed: set[str] | None = None) -> None:
    # 保存断点续跑进度。sorted 让文件内容稳定、可 diff。
    # failed：曾处理失败的日期（重跑会重试，但留痕供验收与告警）。
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done), "failed": sorted(failed or set())}, f)


def setup_matplotlib() -> None:
    # 图表全局设置集中在 report_style，避免各脚本各自定义颜色和字体。
    from report_style import apply_report_style

    apply_report_style()


def month_last_trading_days(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    # 由交易日历推导每月最后一个交易日。
    # 返回 DataFrame：index=月份字符串(YYYY-MM)，列 last_day=该月最后交易日 Timestamp。
    s = pd.Series(calendar, index=calendar)
    last = s.groupby(s.index.strftime("%Y-%m")).max()
    return pd.DataFrame({"last_day": last})
