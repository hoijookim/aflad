#!/usr/bin/env python3
"""splicing 커넥터 좌/우 분리의 실효성 — **쌍 불일치** 섭동으로 잰다. train 전용.

## 왜 새 섭동이 필요한가
기준 v2(remove/duplicate/swap/resize)는 splicing 에서 양쪽 표현 모두 1.0000 으로 포화해
판별하지 못한다. 전경이 한 덩어리라 "성분 제거"가 물체 전체 제거와 같아지는 **섭동 퇴화**다.

LOCO splicing 의 대표 논리 이상은 **클램프 개수가 다른 두 커넥터가 짝지어진 경우**다.
이 이상은 총면적을 거의 바꾸지 않고 **한쪽 커넥터의 외형만** 바꾼다. 그래서 섭동도
그렇게 만든다: 한쪽 커넥터의 픽셀을 **다른 학습 이미지의 커넥터**로 치환한다(마스크는 유지).

## 무엇을 가르는가
  기존 표현(클래스 2): 두 커넥터가 한 클래스 -> c 항이 둘의 **평균**이 되어 한쪽만 바뀐
                       변화가 절반으로 희석된다.
  분리 표현(클래스 3): 바뀐 커넥터의 c 항이 통째로 바뀐다.
분리가 실효라면 이 섭동에서 AUROC 차이가 나야 한다. 안 나면 UNet 재학습(2~3시간)을
할 이유가 없다.

test 는 어느 단계에서도 쓰지 않는다.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
PSAD_DIR = R / "external/PSAD_official"
sys.path.insert(0, str(PSAD_DIR))
import psad as P  # noqa: E402

DATA = PSAD_DIR / "LOCO_MVTec_AD"
CAT = "splicing_connectors"
PROBES = [0, 1, 2, 3, 4]


@torch.no_grad()
def row(enc, img, lab, K):
    im = (img / 255. - P.imagenet_mean) / P.imagenet_std
    x = torch.tensor(np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)).float().unsqueeze(0).cuda()
    fts = enc(x)[0]
    C = fts.shape[0]
    F2 = fts.reshape(C, -1).t()
    area = np.zeros(K); out = np.zeros((K, C))
    flat = lab.ravel()
    for k in range(1, K):
        idx = np.nonzero(flat == k)[0]
        area[k] = len(idx)
        if len(idx):
            out[k] = F2.index_select(0, torch.from_numpy(idx).cuda()).sum(0).cpu().numpy()
    del fts, F2
    return np.concatenate([area, (out[1:] / (area[1:, None] + 1)).ravel()])


def connector_components(lab, split):
    """커넥터 성분들의 (마스크, 클래스) 목록. split 이면 cls1/cls3, 아니면 cls1 의 두 성분."""
    if split:
        return [(lab == k, k) for k in (1, 3) if (lab == k).sum() > 500]
    comp, n = ndimage.label(lab == 1)
    if n == 0:
        return []
    sizes = ndimage.sum(lab == 1, comp, range(1, n + 1))
    return [(comp == i + 1, 1) for i, s in enumerate(sizes) if s > 500]


@torch.no_grad()
def evaluate(seg_dir, split, n_probe=60):
    enc = P.Encoder(avgpool_size=5).cuda().eval()
    img_dir = DATA / "orig_512" / CAT / "train/good"
    seg = DATA / seg_dir / CAT
    names = sorted(p.name for p in img_dir.glob("*.png"))
    imgs, labs = [], []
    for n in names:
        imgs.append(np.array(Image.open(img_dir / n).convert("RGB")))
        labs.append(np.array(Image.open(seg / f"pred_{n}")))
    K = int(max(l.max() for l in labs)) + 1
    M = np.stack([row(enc, imgs[i], labs[i], K) for i in range(len(names))])
    mu, sd = M.mean(0), M.std(0) + 1e-10
    Z = (M - mu) / sd

    def nn(q, ex):
        d = np.sqrt(((Z - q) ** 2).sum(1)); d[ex] = np.inf
        return d.min()

    normal = np.array([nn(Z[i], i) for i in range(len(Z))])
    vals = []
    for ps in PROBES:
        rng = np.random.default_rng(ps); pert = []
        for i in rng.choice(len(Z), min(n_probe, len(Z)), replace=False):
            cc = connector_components(labs[i], split)
            if not cc:
                continue
            m_i, _ = cc[int(rng.integers(len(cc)))]
            # 다른 이미지의 커넥터 픽셀로 치환 (마스크는 유지)
            for _ in range(10):
                j = int(rng.integers(len(Z)))
                if j == i:
                    continue
                dc = connector_components(labs[j], split)
                if dc:
                    break
            else:
                continue
            m_j, _ = dc[int(rng.integers(len(dc)))]
            src = imgs[j][m_j]
            pi = imgs[i].copy()
            pi[m_i] = src[rng.integers(0, len(src), size=int(m_i.sum()))]
            q = (row(enc, pi, labs[i], K) - mu) / sd
            pert.append(nn(q, i))
        if len(pert) < 10:
            break
        vals.append(roc_auc_score(np.r_[np.zeros(len(normal)), np.ones(len(pert))],
                                  np.r_[normal, pert]))
    del enc
    torch.cuda.empty_cache()
    v = np.array(vals)
    return (float(v.mean()), float(v.std(ddof=1))) if len(v) > 1 else (None, None)


if __name__ == "__main__":
    print("splicing 쌍 불일치 섭동 — 한쪽 커넥터만 다른 개체로 치환 (마스크 유지)")
    for seg, split, nm in [("csad_pseudo_seg", False, "기존 (클래스 2, 커넥터 합쳐짐)"),
                           ("csad_split_seg", True, "좌/우 분리 (클래스 3)")]:
        m, s = evaluate(seg, split)
        print(f"  {nm:32s} AUROC {m:.4f} ± {s:.4f}" if m is not None
              else f"  {nm:32s} 측정 실패")
