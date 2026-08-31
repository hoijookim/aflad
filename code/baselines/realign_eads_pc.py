#!/usr/bin/env python3
"""EAD-S / DINOv3-L PatchCore 단독 성능을 시드 {42,43,44} 로 재집계.

표 5 의 두 방법은 시드별 원값이 남아 있어 재실행 없이 재집계만으로 정렬된다.
단일 분기 AUROC 는 단조변환 불변이므로 z-정규화 여부와 무관하다.
test 라벨은 이미 발표된 베이스라인 수치를 시드 집합만 바꿔 재집계하는 데만 쓴다.
"""
import json
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import prereg_testfree as PT  # noqa: E402
import fuse_testfree_v2 as F  # noqa: E402

CATS = F.CATS
TARGET = (42, 43, 44)


def axis(score, lab, lt, k):
    m = (lt == "good") | (lt == k)
    return roc_auc_score(lab[m], score[m])


def main():
    out = {}
    for nm in ("EAD-S", "DINOv3-L PC"):
        per = {}
        for sd in TARGET:
            lg, st = [], []
            for c in CATS:
                t, v, lab, lt = PT.ead_one(c, sd)
                s = t if nm == "EAD-S" else PT.pc_one(c, sd)[0]
                lg.append(axis(s, lab, lt, "logical"))
                st.append(axis(s, lab, lt, "structural"))
            per[sd] = {"LS": float(0.5 * (np.mean(lg) + np.mean(st))),
                       "logical": float(np.mean(lg)), "structural": float(np.mean(st))}
            print(f"  {nm:12s} seed{sd}: L+S {per[sd]['LS']:.4f}", flush=True)
        a = np.array([per[s]["LS"] for s in TARGET])
        out[nm] = {"per_seed": {str(k): v for k, v in per.items()},
                   "LS_mean": float(a.mean()), "LS_std": float(a.std(ddof=1)),
                   "logical": float(np.mean([per[s]["logical"] for s in TARGET])),
                   "structural": float(np.mean([per[s]["structural"] for s in TARGET]))}
        print(f"  => {nm}: {a.mean():.4f} ± {a.std(ddof=1):.4f} "
              f"(log {out[nm]['logical']:.4f} / str {out[nm]['structural']:.4f})\n", flush=True)
    p = R / "reports/countgd/realign_eads_pc_3seed.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
