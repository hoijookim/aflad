#!/usr/bin/env python3
"""PUAD-M 의 L+S(논리/구조 분리)를 시드 {42,43,44} 로 재채점.

표 5 는 L+S 를 보고하는데 `puad_5seed_all_cats.json` 에는 **pooled AUROC 만** 있었다.
시드별 원시 점수(`puad_m_multiseed_scores/{cat}_seed{S}.npz`: puad/ead/labels)가 남아 있고,
그 `labels` 가 정본(`algml_v6_5/aupr_bootstrap/scores`)의 이진 라벨과 5범주 전부 일치하므로
정본의 `label_type` 을 붙여 논리·구조를 분리할 수 있다(정렬 검증 후 진행).

test 라벨은 이미 발표된 베이스라인 수치를 시드 집합만 바꿔 재집계하는 데만 쓴다.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
SC = R / "reports/path_y/puad_m_multiseed_scores"
V4 = R / "algml_v6_5/aupr_bootstrap/scores"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
TARGET = (42, 43, 44)


def main():
    out = {}
    for key in ("puad", "ead"):
        per = {}
        for sd in TARGET:
            lg, st = [], []
            for c in CATS:
                z = np.load(SC / f"{c}_seed{sd}.npz", allow_pickle=True)
                k = np.load(V4 / f"scores_{c}_seed42.npz", allow_pickle=True)
                lab, lt = k["label"].astype(int), k["label_type"].astype(str)
                assert (z["labels"].astype(int) == lab).all(), f"{c}/seed{sd} 라벨 불일치"
                s = z[key].astype(float)
                for acc, kind in ((lg, "logical"), (st, "structural")):
                    m = (lt == "good") | (lt == kind)
                    acc.append(roc_auc_score(lab[m], s[m]))
            per[sd] = {"LS": float(0.5 * (np.mean(lg) + np.mean(st))),
                       "logical": float(np.mean(lg)), "structural": float(np.mean(st))}
        a = np.array([per[s]["LS"] for s in TARGET])
        nm = "PUAD-M" if key == "puad" else "EAD-M(동일 하네스)"
        out[nm] = {"per_seed": {str(k): v for k, v in per.items()},
                   "LS_mean": float(a.mean()), "LS_std": float(a.std(ddof=1)),
                   "logical": float(np.mean([per[s]["logical"] for s in TARGET])),
                   "structural": float(np.mean([per[s]["structural"] for s in TARGET]))}
        print(f"  {nm:20s} L+S {a.mean():.4f} ± {a.std(ddof=1):.4f}  "
              f"(log {out[nm]['logical']:.4f} / str {out[nm]['structural']:.4f})  "
              + " ".join(f"s{s} {per[s]['LS']:.4f}" for s in TARGET))
    p = R / "reports/countgd/realign_puad_3seed.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
