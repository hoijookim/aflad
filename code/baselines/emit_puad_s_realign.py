#!/usr/bin/env python3
"""재채점 결과를 논문 세션이 요청한 형식(`realign_puad_3seed.json` 구조)으로 내보낸다.

요청서: docs/260821_puad_s_3seed_request.md §원하는 산출 형식
재채점 산출물과 값은 같고 키 구조만 맞춘다 — 그쪽 도구가 그대로 읽게 하려는 것이다.
"""
import json
from pathlib import Path
import numpy as np

REPO = Path("/workspace/ai-vision-research")
SRC = REPO / "reports/countgd/puad_s_rescore_3seed.json"
OUT = REPO / "reports/countgd/realign_puad_s_3seed.json"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = ["42", "43", "44"]

d = json.load(open(SRC))["per_seed"]
K = ("LS", "logical", "structural")

per_cat = {}
for c in CATS:
    per_cat[c] = {}
    for k in K:
        v = [d[s]["per_cat"][c][k] for s in SEEDS]
        per_cat[c][k] = float(np.mean(v))
        per_cat[c][k + "_std"] = float(np.std(v, ddof=1))
    per_cat[c]["per_seed"] = {s: {k: d[s]["per_cat"][c][k] for k in K} for s in SEEDS}

agg = {}
for k in K:
    v = [d[s][k] for s in SEEDS]
    agg[k + "_mean" if k == "LS" else k] = float(np.mean(v))
    if k == "LS":
        agg["LS_std"] = float(np.std(v, ddof=1))

out = {"PUAD-S": {
    "per_seed": {s: {k: d[s][k] for k in K} for s in SEEDS},
    **agg,
    "per_cat": per_cat,
    "seeds": [42, 43, 44],
    "method": "PUAD-S = z(EAD-S 점수) + z(Mahalanobis), 둘 다 검증셋 통계로 정규화. "
              "체크포인트는 3시드 모두 reports/phase0/efficient_ad_official_small_seeds_std. 학습 없음.",
    "source": "puad_s_rescore_3seed.json",
    "provenance": "puad_s_provenance_audit.json",
    "supersedes": {"value": 0.9276, "seeds": [44],
                   "why": "ls_canonical_scores.json(260604 커밋)이 시드 0·42·1234 점수"
                          "(260612 생성)보다 먼저 굳어 글롭이 시드 하나만 잡았다."},
}}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
p = out["PUAD-S"]
print(f"PUAD-S  L+S {p['LS_mean']:.4f} ± {p['LS_std']:.4f}  "
      f"logical {p['logical']:.4f}  structural {p['structural']:.4f}")
print(f"[saved] {OUT}")
