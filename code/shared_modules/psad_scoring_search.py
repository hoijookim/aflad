#!/usr/bin/env python3
"""PSAD 채점 옵션을 train 만으로 탐색 — 표현이 아니라 점수 산출 쪽.

앞선 label_criterion.py 는 마스크 히스토그램 표현만 평가했다(현행이 최선으로 판정).
여기서는 공식 psad.py 가 노출하지만 손대지 않았던 채점 옵션을 본다:
  standardize  1(현행)/0    — 메모리 차원별 (x-mean)/std. 저분산 차원이 폭발 원인이었다.
  scale_type   max(현행)/std/none — h,c 각 항의 정규화 방식
  layers       f0f1f2(현행)/+f3  — Encoder.forward 는 layer4(f3)를 계산해두고 쓰지 않는다
  avgpool      5(현행)/3/7

기준(검증됨): train LOO 1-NN 거리에서 **정상 vs 마스크 부품제거**의 AUROC.
pushpins 3x5(개선) 0.9937 vs 5x5(구) 0.7575 로 실제 개선을 맞힌 기준이다.
test 는 어느 단계에서도 쓰지 않는다.

캐시 전략: 이미지당 클래스별 (면적, 레이어별 마스크평균 특징)을 한 번만 뽑아둔다.
부품 제거는 해당 클래스의 면적·특징을 0 으로 만드는 것과 같으므로 재추론이 필요없다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
PSAD_DIR = R / "external/PSAD_official"
sys.path.insert(0, str(PSAD_DIR))
import psad as P  # noqa: E402

DATA = PSAD_DIR / "LOCO_MVTec_AD"
CACHE = R / "reports/countgd/psad_feat_cache"
NUM_CLS = {"breakfast_box": 7, "juice_bottle": 9, "pushpins": 26,
           "screw_bag": 7, "splicing_connectors": 10}
LAYER_DIMS = [256, 512, 1024, 2048]        # resnet101 layer1..4


class Args:
    input_size = 256
    avgpool_size = 5


@torch.no_grad()
def build_cache(cat, n_max=None):
    """클래스별 면적 + 레이어별 마스크평균 특징을 캐시."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{cat}.npz"
    if f.exists():
        z = np.load(f)
        return z["area"], z["feat"]        # [N,K], [N,K,sum(dims)]
    num_cls = NUM_CLS[cat]
    enc = P.Encoder().cuda().eval()
    imnet_mean = np.array([0.485, 0.456, 0.406]); imnet_std = np.array([0.229, 0.224, 0.225])
    seg_dir = DATA / "csad_pseudo_seg" / cat
    img_dir = DATA / "orig_512" / cat / "train" / "good"
    paths = sorted(img_dir.glob("*.png"))[:n_max]
    areas, feats = [], []
    for i, p in enumerate(paths):
        m = np.array(Image.open(seg_dir / f"pred_{p.name}"))
        mask = torch.zeros(num_cls, *m.shape, device="cuda")
        mask.scatter_(0, torch.tensor(m).long().unsqueeze(0).cuda(), 1)
        im = np.array(Image.open(p).convert("RGB")) / 255.
        im = (im - imnet_mean) / imnet_std
        x = torch.tensor(im).float().permute(2, 0, 1).unsqueeze(0).cuda()
        raw = enc.extract_ft(x)                       # layer1..4 (저해상도)
        per_layer = []
        # 브로드캐스트 [K,C,H,W] 는 layer4 에서 15GB 를 잡는다. 마스크를 특징 해상도로
        # 내려 행렬곱으로 클래스별 합을 구한다: [K,hw] x [hw,C].
        # 주의 — 이것은 공식 경로와 **수치가 다르다**. 공식 Encoder.forward 는
        # avg_pool2d(5) 평활화 후 특징을 512 로 올려 마스크와 곱한다. 여기서는 반대로
        # 마스크를 내리고 평활화도 없다. 즉 이 캐시는 채점 옵션 A/B 를 싸게 비교하기
        # 위한 **프록시**다. 채택 후보는 criterion_exact.py 에서 공식 경로로 재검증한다.
        for li in range(4):
            g = raw[li][0]                            # [C,h,w]
            C, h, w = g.shape
            mk = Fn.interpolate(mask.unsqueeze(0), size=(h, w), mode="nearest")[0]
            mf = mk.reshape(mask.shape[0], -1)        # [K,hw]
            gf = g.reshape(C, -1).t()                 # [hw,C]
            num = mf @ gf                             # [K,C]
            den = mf.sum(1, keepdim=True) + 1
            per_layer.append((num / den).cpu().numpy())
        areas.append(mask.sum((1, 2)).cpu().numpy())
        feats.append(np.concatenate(per_layer, axis=1))
        if i % 50 == 0:
            print(f"    {cat} {i}/{len(paths)}", flush=True)
    area = np.stack(areas); feat = np.stack(feats)
    np.savez(f, area=area, feat=feat)
    del enc
    torch.cuda.empty_cache()
    return area, feat


def layer_slice(names):
    """레이어 이름 -> 특징 열 인덱스."""
    off, out = 0, []
    for i, d in enumerate(LAYER_DIMS):
        if f"f{i}" in names:
            out.append(np.arange(off, off + d))
        off += d
    return np.concatenate(out)


def score_variant(area, feat, cols, standardize, scale_type, n_probe=120):
    """train LOO: 정상 vs 부품제거 의 h+c 거리 AUROC."""
    N, K = area.shape
    F_ = feat[:, :, cols].reshape(N, -1)
    M = np.concatenate([area, F_], axis=1)
    if standardize:
        mu, sd = M.mean(0), M.std(0) + 1e-10
        M = (M - mu) / sd
    else:
        mu = np.zeros(M.shape[1]); sd = np.ones(M.shape[1])
    ns = K                                     # 앞 K 열이 면적(h), 나머지가 임베딩(c)

    def dists(q, exclude):
        d = (M - q) ** 2
        h = np.sqrt(d[:, :ns].sum(1)); c = np.sqrt(d[:, ns:].sum(1))
        if exclude is not None:
            h[exclude] = np.inf; c[exclude] = np.inf
        return h.min(), c.min()

    # train LOO 정상 거리 + 스케일 기준
    hs, cs = zip(*[dists(M[i], i) for i in range(N)])
    hs, cs = np.array(hs), np.array(cs)
    if scale_type == "max":
        hn, cn = hs.max(), cs.max()
    elif scale_type == "std":
        hn, cn = hs.std() + 1e-10, cs.std() + 1e-10
    else:
        hn = cn = 1.0
    normal = hs / hn + cs / cn

    rng = np.random.default_rng(0)
    idx = rng.choice(N, min(n_probe, N), replace=False)
    removed = []
    for i in idx:
        cand = [k for k in range(1, K) if area[i, k] > 200]
        if not cand:
            continue
        k = cand[rng.integers(len(cand))]
        a2 = area[i].copy(); a2[k] = 0
        f2 = feat[i].copy(); f2[k] = 0
        m2 = np.concatenate([a2, f2[:, cols].reshape(-1)])
        m2 = (m2 - mu) / sd if standardize else m2
        h, c = dists(m2, i)
        removed.append(h / hn + c / cn)
    removed = np.array(removed)
    y = np.r_[np.zeros(len(normal)), np.ones(len(removed))]
    return roc_auc_score(y, np.r_[normal, removed])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="screw_bag,breakfast_box")
    a = ap.parse_args()
    for cat in a.cats.split(","):
        print(f"\n=== {cat} ===", flush=True)
        area, feat = build_cache(cat)
        base = ("f0f1f2", 1, "max")
        cands = [
            ("현행: f0f1f2 / std=1 / max", "f0f1f2", 1, "max"),
            ("+f3 (layer4 추가)", "f0f1f2f3", 1, "max"),
            ("f1f2 만", "f1f2", 1, "max"),
            ("f2 만", "f2", 1, "max"),
            ("standardize=0", "f0f1f2", 0, "max"),
            ("scale_type=std", "f0f1f2", 1, "std"),
            ("scale_type=none", "f0f1f2", 1, "none"),
        ]
        best = None
        for nm, ln, stdz, sc in cands:
            auc = score_variant(area, feat, layer_slice(ln), stdz, sc)
            mark = "  <= 현행" if (ln, stdz, sc) == base else ""
            print(f"  {nm:28s} 제거탐지 AUROC {auc:.4f}{mark}", flush=True)
            if best is None or auc > best[1]:
                best = (nm, auc)
        print(f"  -> 최선: {best[0]} ({best[1]:.4f})")


if __name__ == "__main__":
    main()
