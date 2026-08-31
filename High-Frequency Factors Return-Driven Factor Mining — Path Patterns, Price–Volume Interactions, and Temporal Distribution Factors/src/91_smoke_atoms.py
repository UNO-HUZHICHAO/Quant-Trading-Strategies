# ------------------ 91_smoke_atoms.py ----------------
# 冒烟测试：用合成随机游走数据跑通 prepare_day + 三模块 16 个原子因子
# （+ C1_H/C1_N 两个诊断列），检查输出形状、取值范围（τ≥1、熵∈[0,1]、占比≤1）。
# 运行：python 91_smoke_atoms.py

from __future__ import annotations

import numpy as np

from factors_common import prepare_day
from factors_a import compute_atoms_a
from factors_b import compute_atoms_b
from factors_c import compute_atoms_c


def main() -> None:
    rng = np.random.default_rng(7)
    S, T = 5, 240
    # 合成价格路径：几何随机游走，保证 OHLC 关系自洽。
    rets = rng.normal(0, 0.001, size=(S, T))
    close = 100 * np.exp(np.cumsum(rets, axis=1))
    open_ = np.roll(close, 1, axis=1)
    open_[:, 0] = 100.0
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 3e-4, (S, T))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 3e-4, (S, T))))
    vol = rng.integers(100, 100000, size=(S, T)).astype(float)
    amount = vol * close

    pack = prepare_day(close, high, low, vol, amount)
    atoms: dict = {}
    atoms.update(compute_atoms_a(pack))
    atoms.update(compute_atoms_b(pack))
    atoms.update(compute_atoms_c(pack))

    print(f"atom cols = {len(atoms)}")
    for k, v in atoms.items():
        print(f"{k:34s}", np.array2string(np.asarray(v), precision=4, suppress_small=True))

    assert len(atoms) == 18, "应当恰好 16 个原子列 + 2 个诊断列（C1_H/C1_N）"
    assert np.all(np.isfinite(atoms["A1_tau_atom"])), "合成数据 A1 不应有 NaN"
    assert np.all(atoms["A1_tau_atom"] >= 1.0), "曲折度必须 >= 1"
    assert np.all((atoms["C1_entropy_atom"] >= 0) & (atoms["C1_entropy_atom"] <= 1.000001)), "熵应在 [0,1]"
    assert np.all(atoms["A2c_high_tau_vol_share_atom"] <= 1.000001), "占比不应超过 1"
    assert np.all(atoms["A3_tort_vol_joint_atom"] <= 1.000001), "占比不应超过 1"
    print("SMOKE OK")


if __name__ == "__main__":
    main()
