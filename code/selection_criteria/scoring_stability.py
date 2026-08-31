#!/usr/bin/env python3
"""채점 옵션 후보를 probe 시드 다중화로 재평가 — 단일 시드 우연을 배제한다.

psad_scoring_search.py 는 probe 시드 0 하나로 돌렸다. standardize=0 이 screw_bag
+2.54pp / breakfast +1.10pp 로 크게 이겼지만 pushpins 는 -0.31pp 로 졌다. 이 부호가
probe 표본의 우연인지 실재하는 차이인지 가르려면 시드를 늘려야 한다.

기준은 동일하다: train LOO 1-NN 거리에서 정상 vs 마스크 부품제거의 AUROC.
test 는 어느 단계에서도 쓰지 않는다.
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
import psad_scoring_search as S  # noqa: E402

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
PROBE_SEEDS = [0, 1, 2, 3, 4]
CANDS = [("현행 std=1", "f0f1f2", 1, "max"),
         ("standardize=0", "f0f1f2", 0, "max"),
         ("+f3", "f0f1f2f3", 1, "max")]


def score_seeded(area, feat, cols, standardize, scale_type, probe_seed, n_probe=120):
    """score_variant 와 동일하되 probe 표본 추출 시드를 바꾼다."""
    N, K = area.shape
    M = np.concatenate([area, feat[:, :, cols].reshape(N, -1)], axis=1)
    if standardize:
        mu, sd = M.mean(0), M.std(0) + 1e-10
        M = (M - mu) / sd
    else:
        mu, sd = np.zeros(M.shape[1]), np.ones(M.shape[1])

    def dists(q, exclude):
        d = (M - q) ** 2
        h = np.sqrt(d[:, :K].sum(1)); c = np.sqrt(d[:, K:].sum(1))
        if exclude is not None:
            h[exclude] = np.inf; c[exclude] = np.inf
        return h.min(), c.min()

    hs, cs = map(np.array, zip(*[dists(M[i], i) for i in range(N)]))
    if scale_type == "max":
        hn, cn = hs.max(), cs.max()
    elif scale_type == "std":
        hn, cn = hs.std() + 1e-10, cs.std() + 1e-10
    else:
        hn = cn = 1.0
    normal = hs / hn + cs / cn

    rng = np.random.default_rng(probe_seed)
    removed = []
    for i in rng.choice(N, min(n_probe, N), replace=False):
        cand = [k for k in range(1, K) if area[i, k] > 200]
        if not cand:
            continue
        k = cand[rng.integers(len(cand))]
        a2 = area[i].copy(); a2[k] = 0
        f2 = feat[i].copy(); f2[k] = 0
        q = np.concatenate([a2, f2[:, cols].reshape(-1)])
        if standardize:
            q = (q - mu) / sd
        h, c = dists(q, i)
        removed.append(h / hn + c / cn)
    from sklearn.metrics import roc_auc_score
    removed = np.array(removed)
    return roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(removed))],
                         np.r_[normal, removed])


def main():
    out = {}
    for cat in CATS:
        area, feat = S.build_cache(cat)
        out[cat] = {}
        for nm, ln, stdz, sc in CANDS:
            cols = S.layer_slice(ln)
            v = np.array([score_seeded(area, feat, cols, stdz, sc, ps) for ps in PROBE_SEEDS])
            out[cat][nm] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                            "per_probe": v.tolist()}
        print(f"{cat:22s} " + "  ".join(
            f"{nm} {out[cat][nm]['mean']:.4f}±{out[cat][nm]['std']:.4f}" for nm, *_ in CANDS),
            flush=True)

    print("\n" + "=" * 78)
    print(f"{'후보':16s} {'5범주 평균':>10s}   범주별 대현행 차이")
    base = "현행 std=1"
    for nm, *_ in CANDS:
        m = np.mean([out[c][nm]["mean"] for c in CATS])
        dd = " ".join(f"{c[:5]} {out[c][nm]['mean'] - out[c][base]['mean']:+.4f}" for c in CATS)
        print(f"{nm:16s} {m:10.4f}   {dd}")

    # 판정: 시드 간 변동을 넘어서는 개선인지
    print("\n판정 (현행 대비, probe 시드 5개 대응표본):")
    for nm, *_ in CANDS:
        if nm == base:
            continue
        diffs = np.array([np.mean(out[c][nm]["per_probe"]) - np.mean(out[c][base]["per_probe"])
                          for c in CATS])
        wins = int((diffs > 0).sum())
        print(f"  {nm:16s} 평균 {diffs.mean():+.4f}, 개선 범주 {wins}/5, "
              f"최대 악화 {diffs.min():+.4f}")

    p = R / "reports/countgd/psad_scoring_stability.json"
    json.dump({"probe_seeds": PROBE_SEEDS, "results": out}, open(p, "w"), indent=2,
              ensure_ascii=False)
    print(f"[saved] {p}")


if __name__ == "__main__":
    main()
