#!/usr/bin/env python3
"""의사레이블 표현을 train 만으로 평가 — 안정성과 변별력을 함께 잰다.

문제: 변동계수(CV)만 쓰면 퇴화한다. 전경 총면적 1차원이 CV 0.147 로 가장 "안정적"
이지만 부품이 빠져도 총면적은 거의 안 변하므로 논리 이상을 못 잡는다.

기준: 구성 분기가 해야 할 일 자체를 재현한다.
  잡음바닥 = 정상 이미지의 LOO 1-NN 거리 (자기 자신 제외)
  신호     = **마스크에서 부품 하나를 제거한** 이미지의 1-NN 거리
  판정     = 두 분포의 AUROC

부품 제거를 **이미지가 아니라 마스크에서** 하는 것이 핵심이다. 260412 FM-MEAD 에서
확인했듯 SALAD 는 이미지 Cut-Paste 를 쓰지 않고 composition map label 만 조작한다 —
이미지 합성은 경계 아티팩트로 국소 기법(EAD/PC)에 유리해지지만, 여기서는 구성 분기
표현 하나만 평가하므로 마스크 수준이 정확히 맞는 층위다.

test 는 어느 단계에서도 쓰지 않는다. train 의사레이블만 입력이다.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
D = R / "external/PSAD_official/LOCO_MVTec_AD/csad_pseudo_seg"
RNG = np.random.default_rng(0)


def load_masks(cat, subdir=None):
    d = D / (subdir or cat)
    return [np.array(Image.open(p)) for p in sorted(d.glob("pred_*.png"))]


# ---------- 표현 후보 ----------
def rep_classwise(m, K):
    return np.array([(m == k).sum() for k in range(1, K)], float)


def rep_class_plus_count(m, K):
    return np.array([(m == k).sum() for k in range(1, K)]
                    + [ndimage.label(m == k)[1] for k in range(1, K)], float)


def rep_total(m, K):
    return np.array([(m > 0).sum()], float)


def rep_merge_unstable(m, K, stable, unstable):
    return np.array([(m == k).sum() for k in stable]
                    + [sum((m == k).sum() for k in unstable)], float)


def rep_profile(m, K, n=6):
    comp, c = ndimage.label(m > 0)
    sz = sorted([s for s in ndimage.sum(m > 0, comp, range(1, c + 1)) if s > 200], reverse=True)
    return np.array((sz + [0] * n)[:n], float)


# ---------- 부품 제거 (마스크 수준) ----------
def remove_part(m, rng):
    """면적 200 이상인 성분 하나를 배경으로 지운다."""
    comp, n = ndimage.label(m > 0)
    if n == 0:
        return None
    sizes = ndimage.sum(m > 0, comp, range(1, n + 1))
    cand = [i + 1 for i, s in enumerate(sizes) if s > 200]
    if not cand:
        return None
    out = m.copy()
    out[comp == cand[rng.integers(len(cand))]] = 0
    return out


def evaluate(masks, rep_fn, K, n_probe=120):
    """정상 LOO 거리 vs 부품제거 거리의 AUROC (표준화 후 1-NN L2)."""
    X = np.stack([rep_fn(m, K) for m in masks])
    mu, sd = X.mean(0), X.std(0) + 1e-10
    Z = (X - mu) / sd

    def nn_dist(q, exclude=None):
        d = np.sqrt(((Z - q) ** 2).sum(1))
        if exclude is not None:
            d[exclude] = np.inf
        return d.min()

    normal = np.array([nn_dist(Z[i], exclude=i) for i in range(len(Z))])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(masks), min(n_probe, len(masks)), replace=False)
    removed = []
    for i in idx:
        mm = remove_part(masks[i], rng)
        if mm is None:
            continue
        q = (rep_fn(mm, K) - mu) / sd
        removed.append(nn_dist(q, exclude=i))     # 원본 자기 자신은 제외
    removed = np.array(removed)
    y = np.r_[np.zeros(len(normal)), np.ones(len(removed))]
    s = np.r_[normal, removed]
    return roc_auc_score(y, s), float(np.mean(normal)), float(np.mean(removed))


def main():
    cats = sys.argv[1:] or ["breakfast_box", "juice_bottle", "pushpins",
                            "screw_bag", "splicing_connectors"]
    for cat in cats:
        masks = load_masks(cat)
        K = max(m.max() for m in masks) + 1
        pres = {k: np.mean([(m == k).any() for m in masks]) for k in range(1, K)}
        unst = [k for k in range(1, K) if pres[k] < 0.95]
        st = [k for k in range(1, K) if pres[k] >= 0.95]
        cands = {
            "V0 현행 클래스별 면적": lambda m, K: rep_classwise(m, K),
            "V2 전경 총면적": lambda m, K: rep_total(m, K),
            "V3 면적 프로파일(6)": lambda m, K: rep_profile(m, K),
            "V4 클래스면적+성분수": lambda m, K: rep_class_plus_count(m, K),
        }
        if unst:
            cands[f"V1 불안정{unst} 병합"] = (
                lambda m, K, st=st, unst=unst: rep_merge_unstable(m, K, st, unst))
        print(f"\n=== {cat} (K={K-1}, {len(masks)}장) ===")
        print(f"{'표현':26s} {'제거탐지 AUROC':>14s} {'정상거리':>9s} {'제거거리':>9s}")
        best = None
        for nm, fn in cands.items():
            auc, dn, dr = evaluate(masks, fn, K)
            print(f"{nm:26s} {auc:14.4f} {dn:9.3f} {dr:9.3f}")
            if best is None or auc > best[1]:
                best = (nm, auc)
        print(f"  -> 최선: {best[0]} (AUROC {best[1]:.4f})")


if __name__ == "__main__":
    main()
