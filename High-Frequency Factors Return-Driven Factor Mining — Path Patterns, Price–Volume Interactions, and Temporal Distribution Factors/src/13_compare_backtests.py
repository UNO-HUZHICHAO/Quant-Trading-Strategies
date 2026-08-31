"""比较旧版与修订版回测，生成可供研报复核的差异表。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    old = pd.read_csv(Path(args.old) / "master_summary.csv")
    new = pd.read_csv(Path(args.new) / "master_summary.csv")
    keys = ["universe", "factor", "variant"]
    merged = old.merge(new, on=keys, suffixes=("_v3", "_v4"), validate="one_to_one")
    for col in ["ic_mean", "icir", "rankic_mean", "rankicir", "ls_ann", "ls_ann_gross", "ls_sharpe"]:
        merged[f"{col}_diff"] = merged[f"{col}_v4"] - merged[f"{col}_v3"]
    merged["conclusion_sign_changed"] = (
        np.sign(merged["rankic_mean_v3"]) != np.sign(merged["rankic_mean_v4"])
    )
    keep = keys + [
        "ic_mean_v3", "ic_mean_v4", "ic_mean_diff",
        "rankicir_v3", "rankicir_v4", "rankicir_diff",
        "ls_ann_v3", "ls_ann_v4", "ls_ann_diff",
        "ls_eff_ann", "turnover_mean_v4", "ls_turnover_mean",
        "ls_sharpe_v3", "ls_sharpe_v4", "ls_sharpe_diff",
        "conclusion_sign_changed",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged[keep].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"比较表 {len(merged)} 行 -> {out}")
    print(f"RankIC 方向变化: {int(merged['conclusion_sign_changed'].sum())}")
    print(f"净多空年化变化中位数: {merged['ls_ann_diff'].median():.4%}")


if __name__ == "__main__":
    main()
