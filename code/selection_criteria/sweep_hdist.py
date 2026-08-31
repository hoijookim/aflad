#!/usr/bin/env python3
"""h 항(면적 히스토그램)의 거리·표현을 훑는다 — train 전용, test 미조회.

동기: h 는 클래스 수 K-1 (≈6, screw_bag 은 병합 후 4) 짜리 저차원 벡터인데 성분들이
강하게 상관한다 — 전경 총면적이 대체로 보존되므로 한 클래스가 커지면 다른 쪽이 준다.
현행은 차원별 표준화 후 **유클리드**라 그 상관 구조를 버린다. 저차원이라 공분산을
train 360장으로 안정적으로 추정할 수 있으므로 마할라노비스가 자연스러운 대안이다.
(V3.2 에서도 GAP+마할라노비스가 가장 잘 들었다.)

축:
  dist   euclid(현행) / maha        — 표준화 유클리드 vs 공분산 반영
  rep    abs(현행) / frac           — 절대 면적 vs 전경 총면적 대비 비율
                                      (조명·분할 임계 변동으로 전 클래스가 함께
                                       흔들리는 성분을 제거한다)

c 항은 1792*(K-1) 차원이라 공분산 추정이 불가능하므로 건드리지 않는다.

기준: criterion_exact 와 동일 — 연결성분 캐시로 정확히 모사한 remove/swap 의
LOO 1-NN 거리 AUROC, probe 시드 5개.

채택 규칙(선언): 5범주 × 2유형 평균이 현행 대비 +0.005 이상이고, 악화 범주의 낙폭이
그 범주 현행값의 1% 이내일 것.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import criterion_exact as CE  # noqa: E402

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
KINDS = ("remove", "swap")
PROBES = [0, 1, 2, 3, 4]


def rows(cat, seed, merge):
    CLS, CNT, FSUM = CE.build_cache(cat, seed, merge)
    K = CE.NUM_CLS[cat]
    M = np.stack([CE.memory_row(CLS[i], CNT[i], FSUM[i], K) for i in range(len(CLS))])
    return CLS, CNT, FSUM, K, M


def prep_h(A, rep):
    """A: [N,K] 원 면적. 표현 변환 후 표준화 파라미터까지 반환."""
    if rep == "frac":
        tot = A[:, 1:].sum(1, keepdims=True) + 1.0
        A = np.concatenate([A[:, :1], A[:, 1:] / tot], axis=1)
    mu, sd = A.mean(0), A.std(0) + 1e-10
    return A, mu, sd


def evaluate(cat, dist, rep, seed=42, merge=True, n_probe=120):
    CLS, CNT, FSUM, K, M = rows(cat, seed, merge)
    A = M[:, :K]                                   # 면적 부분
    C = M[:, K:]                                   # 임베딩 부분
    Ah, mu_a, sd_a = prep_h(A, rep)
    Za = (Ah - mu_a) / sd_a
    mu_c, sd_c = C.mean(0), C.std(0) + 1e-10
    Zc = (C - mu_c) / sd_c

    keep = Za.std(0) > 1e-8                        # 상수 열(빈 클래스)은 제외
    Zak = Za[:, keep]
    if dist == "maha":
        cov = np.cov(Zak, rowvar=False) + 1e-3 * np.eye(Zak.shape[1])
        Wi = np.linalg.inv(cov)
    else:
        Wi = None

    def hdist(q, ex):
        d = Zak - q
        v = np.einsum("ij,jk,ik->i", d, Wi, d) if Wi is not None else (d ** 2).sum(1)
        v = np.sqrt(np.maximum(v, 0)); v[ex] = np.inf
        return v.min()

    def cdist(q, ex):
        v = np.sqrt(((Zc - q) ** 2).sum(1)); v[ex] = np.inf
        return v.min()

    hs = np.array([hdist(Zak[i], i) for i in range(len(Zak))])
    cs = np.array([cdist(Zc[i], i) for i in range(len(Zc))])
    hn, cn = max(hs.max(), 1e-10), max(cs.max(), 1e-10)
    normal = hs / hn + cs / cn

    out = {}
    for kind in KINDS:
        vals = []
        for ps in PROBES:
            rng = np.random.default_rng(ps)
            pert = []
            for i in rng.choice(len(M), min(n_probe, len(M)), replace=False):
                nc = len(CLS[i])
                if nc == 0:
                    continue
                j = int(rng.integers(nc))
                if kind == "remove":
                    q = CE.memory_row(CLS[i], CNT[i], FSUM[i], K, drop=j)
                else:
                    labs = [v for v in np.unique(CLS[i]) if v != CLS[i][j] and v != 0]
                    if not labs:
                        continue
                    q = CE.memory_row(CLS[i], CNT[i], FSUM[i], K,
                                      swap=(j, int(labs[rng.integers(len(labs))])))
                qa, qc = q[:K][None, :], q[K:]
                qa = prep_h(qa, rep)[0][0]
                qa = ((qa - mu_a) / sd_a)[keep]
                pert.append(hdist(qa, i) / hn + cdist((qc - mu_c) / sd_c, i) / cn)
            if len(pert) < 10:
                break
            vals.append(roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(pert))],
                                      np.r_[normal, pert]))
        out[kind] = float(np.mean(vals)) if vals else None
    got = [v for v in out.values() if v is not None]
    return (float(np.mean(got)) if got else None), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=",".join(CATS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", type=int, default=1)
    a = ap.parse_args()
    cats = a.cats.split(",")
    res = {}
    for dist, rep in itertools.product(["euclid", "maha"], ["abs", "frac"]):
        row = {}
        for cat in cats:
            m, _ = evaluate(cat, dist, rep, a.seed, bool(a.merge))
            row[cat] = m
        res[f"{dist}_{rep}"] = row
        got = [v for v in row.values() if v is not None]
        mark = "  <= 현행" if (dist, rep) == ("euclid", "abs") else ""
        print(f"{dist:7s} {rep:5s} 평균 {np.mean(got):.4f}  " +
              " ".join(f"{c[:5]} {row[c]:.4f}" for c in cats) + mark, flush=True)

    base = res["euclid_abs"]
    print("\n채택 규칙: 평균 +0.0050 이상 & 악화 낙폭이 현행값의 1% 이내")
    ok = []
    for key, row in res.items():
        if key == "euclid_abs":
            continue
        d = {c: row[c] - base[c] for c in cats}
        gain = float(np.mean(list(d.values())))
        worst = min(d, key=d.get)
        status = "통과" if (gain >= 0.005 and d[worst] >= -0.01 * base[worst]) else "미달"
        print(f"  {status} {key:14s} 평균 {gain:+.4f}  최악 {worst[:5]} {d[worst]:+.4f}")
        if status == "통과":
            ok.append((gain, key))
    if not ok:
        print("  -> 현행(euclid_abs) 유지")
    p = R / "reports/countgd/sweep_hdist.json"
    json.dump({"results": res, "accepted": [k for _, k in sorted(ok, reverse=True)]},
              open(p, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
