# -*- coding: utf-8 -*-
"""聚合正式版 backtest_v4 结果：IC 统计、多空收益与成本口径。"""
import csv, os, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "result" / "backtest_v4"
POOLS = ['hs300', 'zz500', 'zz1000', 'zzall']
SPS = ['S1', 'S2']
# 目录名 -> 因子代码
DIR2CODE = {
    'A1_tau_20d': 'A1', 'A1v_tau_vol_20d': 'A1v', 'A2a_high_tau_ret_std_20d': 'A2a',
    'A2b_low_tau_cum_ret_20d': 'A2b', 'A2c_high_tau_vol_share_20d': 'A2c',
    'A2d_high_tau_turnover_20d': 'A2d', 'A3_tort_vol_joint_20d': 'A3',
    'B1_lead_lag_20d': 'B1', 'B2_path_asym_20d': 'B2', 'B2a_path_asym_amt_20d': 'B2a',
    'B3_cond_lead_lag_20d': 'B3', 'B4_cross_20d': 'B4', 'B5a_corr_decay_20d': 'B5a',
    'B5b_slope_div_20d': 'B5b', 'B5c_high_vol_ratio_20d': 'B5c',
    'C1_entropy_20d': 'C1', 'C2_entropy_diff_20d': 'C2', 'C3_entropy_std_20d': 'C3',
}

def read_csv(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return rows

def get(d, k, default=''):
    if d is None:
        return default
    v = d.get(k)
    return '' if v is None else v

out = {}
for pool in POOLS:
    out[pool] = {}
    for d in sorted(os.listdir(os.path.join(ROOT, pool))):
        code = DIR2CODE.get(d, d)
        out[pool][code] = {}
        for sp in SPS:
            ic = read_csv(os.path.join(ROOT, pool, d, sp, 'ic_stats.csv'))
            gs = read_csv(os.path.join(ROOT, pool, d, sp, 'group_stats.csv'))
            rec = {}
            if ic:
                rec['rankic'] = float(get(ic[0], 'rankic_mean', 'nan') or 'nan') * 100
                rec['rankicir'] = float(get(ic[0], 'rankicir', 'nan') or 'nan')
                rec['ic'] = float(get(ic[0], 'ic_mean', 'nan') or 'nan') * 100
                rec['icir'] = float(get(ic[0], 'icir', 'nan') or 'nan')
                rec['win'] = float(get(ic[0], 'ic_win_rate', 'nan') or 'nan')
            if gs:
                rec['ls_gross'] = float(get(gs[-1] if gs[-1].get('') or True else gs[-1], 'ann_ret', 'nan') or 'nan') * 100 if 'ls_gross' in str(gs[-1]) else None
                for row in gs:
                    key = row.get('') or row.get('Unnamed: 0') or ''
                    if key == 'LS_gross':
                        rec['ls_gross'] = float(row.get('ann_ret', 'nan')) * 100
                    elif key == 'LS':
                        rec['ls_net'] = float(row.get('ann_ret', 'nan')) * 100
                        rec['ls_sharpe'] = float(row.get('sharpe', 'nan'))
                        rec['ls_maxdd'] = float(row.get('max_dd', 'nan'))
            out[pool][code][sp] = rec

# 打印 RankICIR 全景表
print('=== RankICIR (S1 / S2) ===')
hdr = '因子  | ' + ' | '.join(f'{p}:S1/S2' for p in POOLS)
print(hdr)
for code in ['A1','A1v','A2a','A2b','A2c','A2d','A3','B1','B2','B2a','B3','B4','B5a','B5b','B5c','C1','C2','C3']:
    cells = []
    for p in POOLS:
        s1 = out[p][code]['S1'].get('rankicir')
        s2 = out[p][code]['S2'].get('rankicir')
        cells.append(f"{s1:.2f}/{s2:.2f}" if s1 is not None and s2 is not None else '--')
    print(code.ljust(4) + ' | ' + ' | '.join(cells))

print()
print('=== RankIC(%) S1 ===')
for code in ['A1','A2a','A2b','A2c','A2d','A3','B1','B2','B2a','B3','B4','B5a','B5b','B5c','C1','C2','C3']:
    cells = []
    for p in POOLS:
        v = out[p][code]['S1'].get('rankic')
        cells.append(f"{v:.2f}" if v is not None else '--')
    print(code.ljust(4) + ' | ' + ' | '.join(cells))

print()
print('=== 毛多空 LS_gross(%) / 净多空 LS_net(%) S1 ===')
for code in ['A1','A2a','A2b','A2c','A2d','A3','B1','B2','B2a','B3','B4','B5a','B5b','B5c','C1','C2','C3']:
    cells = []
    for p in POOLS:
        g = out[p][code]['S1'].get('ls_gross')
        n = out[p][code]['S1'].get('ls_net')
        cells.append(f"{g:.1f}/{n:.1f}" if g is not None and n is not None else '--')
    print(code.ljust(4) + ' | ' + ' | '.join(cells))

# 保存汇总供后续使用（make_heatmap.py 读取）
OUT_JSON = PROJECT_ROOT / "result" / "bt_summary.json"
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, default=str)
print()
print('saved', OUT_JSON)
