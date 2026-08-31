#!/usr/bin/env python3
"""남은 채점 자유도를 **공식 특징 경로 위에서** 훑는다 — train 전용, test 미조회.

훑는 축 (각각 사전 동기가 있다):
  w_c  h:c 가중        현행 1.0. scale_type(max) 은 융합의 z-정규화에서 상수배로
                       상쇄되지만, h 와 c 를 **서로 다른** 상수로 나누므로 둘 사이의
                       비율은 상쇄되지 않는다. 그게 실질 자유도인데 미탐색이었다.
  k    이웃 수          현행 1. 메모리 항목 하나가 점수를 통째로 결정하는 취약성이 있다.
                       PatchCore 도 k-NN 을 쓴다.
  merge 혼동쌍 병합     train 혼동행렬로 자동 검출(screw_bag c4+c5). 채택 예정 설정.

기준: 연결성분 단위 캐시로 재추론 없이 **정확히** 모사한 remove / swap 의 LOO 1-NN
거리 AUROC. 성분별 (픽셀수, 특징합) 만 있으면 공식 규약
  ft_cls = (fts*mask_cls).sum() / (mask_cls.sum()+1)
을 오차 없이 재구성할 수 있다.

채택 규칙(선언, test 를 보기 전에 고정): 5범주 × 2유형 평균이 현행 대비 +0.005 이상
개선되고, 악화 범주의 낙폭이 그 범주 현행값의 1% 이내일 것.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import criterion_exact as CE  # noqa: E402

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
KINDS = ("remove", "swap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=",".join(CATS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", type=int, default=1, choices=[0, 1])
    a = ap.parse_args()
    cats = a.cats.split(",")
    grid = [(w, k) for w, k in itertools.product([0.25, 0.5, 1.0, 2.0, 4.0], [1, 3, 5])]

    res = {}
    for w, k in grid:
        row = {}
        for cat in cats:
            vals = []
            for kind in KINDS:
                m, _ = CE.evaluate(cat, 1, kind, a.seed, w_c=w, k=k, merge=bool(a.merge))
                if m is not None:
                    vals.append(m)
            row[cat] = float(np.mean(vals)) if vals else None
        res[f"w{w}_k{k}"] = row
        got = [v for v in row.values() if v is not None]
        print(f"w_c={w:<5} k={k}  평균 {np.mean(got):.4f}  " +
              " ".join(f"{c[:5]} {row[c]:.4f}" for c in cats if row[c] is not None), flush=True)

    base = res["w1.0_k1"]
    bm = np.mean([base[c] for c in cats])
    print("\n" + "=" * 78)
    print(f"현행 w_c=1.0 k=1 평균 {bm:.4f}")
    print("채택 규칙: 평균 +0.0050 이상 개선 & 악화 낙폭이 현행값의 1% 이내\n")
    ok = []
    for key, row in res.items():
        if key == "w1.0_k1":
            continue
        d = {c: row[c] - base[c] for c in cats}
        gain = float(np.mean(list(d.values())))
        worst = min(d, key=d.get)
        if gain >= 0.005 and d[worst] >= -0.01 * base[worst]:
            ok.append((gain, key, worst, d[worst]))
    ok.sort(reverse=True)
    if not ok:
        print("두 조건을 함께 만족하는 후보 없음 -> 현행(w_c=1, k=1) 유지")
    for gain, key, worst, dw in ok[:6]:
        print(f"  통과 {key:12s} 평균 {gain:+.4f}  최악 {worst[:5]} {dw:+.4f}")
    p = R / f"reports/countgd/sweep_exact_merge{a.merge}.json"
    json.dump({"grid": [f"w{w}_k{k}" for w, k in grid], "results": res,
               "accepted": [{"config": k, "gain": g} for g, k, _, _ in ok]},
              open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
