#!/usr/bin/env python3
"""PSAD 재구축 4단계 — hc/hcp 점수 산출 (test + validation).

공식 psad.py 의 계산을 그대로 따르되 두 가지를 확장한다:
  1. validation/good 채점 추가 (test-free 융합의 z-norm 통계용 — 원본 머신에서도
     hc 에 대해서는 계산된 적 없는 값)
  2. 최종 min-max(공식 377-380행, test 전체 통계 사용) 를 적용하지 않은
     max-scaled 합을 저장 — test 통계 무사용(test-free 정합), AUROC 는 단조변환
     불변이라 동일. 원본 d2_out 과의 앵커 대조는 Pearson(선형 불변)으로 한다.

계산 (공식과 동일):
  h = 면적 히스토그램 1-NN L2 / train LOO 1-NN max
  c = ResNet-101 layer1-3 마스크평균 임베딩 1-NN L2 / 동일 max
  p = 내부 PatchCore 점수 / 동일 max
  hc = h+c, hcp = h+c+p   (standardize=1, scale_type=max, avgpool 5)

## 260823 정정 — 아래 세 플래그의 help 가 **기각된 안을 "채택안"이라 부르고 있었다.**
   기본값은 정본과 맞으나 설명이 반대였다. d7_pc_sensitivity 의 "provenance 방어" 와 같은 유형이다.
   정본 설정 = `--tta --merge` (태그 `hc_tta_merge`), standardize=1 · knn=1 · w_n=0.
   채택 2건은 E5(혼동쌍 병합)·E7(회전 TTA, screw_bag 한정)이고 나머지 8건은 기각이다.

출력: {OUT}/psad_scores_{cat}_seed{S}_{mtype}_{split}.npz
  keys: scores, paths(orig_512 상대경로), gt(0/1)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/workspace/ai-vision-research/external/PSAD_official")
sys.path.insert(0, str(REPO))
# 구 라이브러리 API 복원 — apex(미사용 import) / PL callbacks.base / torchmetrics.auc /
# kornia 커널 shape. 전부 연산 경로를 바꾸지 않는다(scripts/psad_rebuild/_shims 참조).
sys.path.append("/workspace/ai-vision-research/scripts/psad_rebuild/_shims")
import pl_compat  # noqa: E402,F401
import psad as P  # noqa: E402  (Encoder, read_img, read_mask, get_cls_idx, get_feature)

DATA = REPO / "LOCO_MVTec_AD"
PC = DATA / "patchcore_score"
# 출력 경로. 기본값은 공식 규약(rebuild_scores)이지만 PSAD_SCORE_OUT 으로 덮을 수 있다.
#
# ⚠ 260822(5090): 이 머신에서는 rebuild_scores/ 가 submission/ 재현 패키지로 향하는
# **심볼릭 링크 farm** 이다(EAD/PSAD/PC 점수를 해시 검증된 패키지에서 링크해 왔다).
# np.savez 는 링크를 따라가므로 그대로 쓰면 패키지 원본을 덮는다.
# 재구축 산출물은 반드시 별도 실디렉터리로 뺀다.
OUT = Path(os.environ.get("PSAD_SCORE_OUT", str(DATA / "rebuild_scores")))
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
NUM_CLS = {"breakfast_box": 7, "juice_bottle": 9, "pushpins": 26,
           "screw_bag": 7, "splicing_connectors": 10}


class Args:
    """psad.py 함수들이 참조하는 최소 args."""
    input_size = 256
    avgpool_size = 5


def pc_score(cat, rel):
    """rel 예: 'test/good/000' -> patchcore .pt 스칼라."""
    return float(torch.load(PC / cat / f"{rel}.pt", weights_only=False)["anomaly_scores"])


_UNET = {}
_MERGE = {}


def confused_pairs(cat, seed, thr=0.10):
    """UNet 이 실제로 혼동하는 클래스 쌍을 **train 만 보고** 자동 검출한다.

    의사레이블(학습 정답) vs 같은 이미지의 UNet 예측으로 혼동행렬을 만들고,
    서로 흘러드는 비율 C[i,j]+C[j,i] 가 thr 을 넘는 쌍을 반환한다.
    seed42 실측: screw_bag c4<->c5 0.168 로 압도적이고 나머지 전 범주 전 쌍은
    0.061 이하다 — thr=0.10 이 그 쌍 하나만 집는다(범주별 수작업 선택 없음).

    검출에 val 도 test 도 쓰지 않는다. 입력은 train 이미지와 그 의사레이블뿐이다.
    """
    key = (cat, seed, thr)
    if key in _MERGE:
        return _MERGE[key]
    from PIL import Image
    K = NUM_CLS[cat]
    # 예측 트리가 분리판이면 정답 레이블도 분리판을 봐야 혼동행렬이 맞다
    lab_d = DATA / ("csad_split_seg" if SEG_TREE.get(cat) else "csad_pseudo_seg") / cat
    pred_d = DATA / SEG_TREE.get(cat, DEFAULT_TREE).format(seed=seed) / cat / "train/good"
    C = np.zeros((K, K))
    for lp, pp in zip(sorted(lab_d.glob("pred_*.png")), sorted(pred_d.glob("*.png"))):
        L = np.array(Image.open(lp)).ravel(); Pr = np.array(Image.open(pp)).ravel()
        m = (L < K) & (Pr < K)
        C += np.bincount(L[m] * K + Pr[m], minlength=K * K).reshape(K, K)
    Cn = C / (C.sum(1, keepdims=True) + 1e-9)
    pairs = [(i, j) for i in range(1, K) for j in range(i + 1, K)
             if C[i].sum() >= 1000 and C[j].sum() >= 1000 and Cn[i, j] + Cn[j, i] > thr]
    # 연결된 쌍들을 하나의 그룹으로 합친다 (i~j, j~k 이면 i,j,k 한 덩어리)
    groups = []
    for i, j in pairs:
        hit = [g for g in groups if i in g or j in g]
        if hit:
            hit[0].update({i, j})
        else:
            groups.append({i, j})
    _MERGE[key] = groups
    if groups:
        print(f"    혼동쌍 병합: " + ", ".join("+".join(f"c{x}" for x in sorted(g))
                                             for g in groups), flush=True)
    else:
        print("    혼동쌍 없음 — 병합 안 함", flush=True)
    return groups


def apply_merge(mask, groups):
    """one-hot 마스크에서 같은 그룹 클래스를 대표 채널로 합친다 (빈 채널은 상수 0)."""
    for g in groups:
        g = sorted(g)
        for j in g[1:]:
            mask[g[0]] = mask[g[0]] + mask[j]
            mask[j] = 0
    return mask


# 학습의 회전 증강 범위와 맞춘 TTA 각도. train_normal_unet.py 기준:
#   screw_bag / breakfast_box = (0,360)  -> 90도 배수 4개 (보간 손실 없는 정확 회전)
#   juice_bottle / pushpins / splicing_connectors = (-20,20) -> 학습 범위 안 5개
# E7 판정: 3시드 val 꼬리의 평균과 최댓값을 둘 다 줄인 범주만 TTA 를 쓴다.
# 5범주에 같은 규칙을 적용한 결과 screw_bag 만 통과했다(5.34->3.67 / 9.17->4.91).
# 나머지는 소폭 악화하므로 켜지 않는다 — TTA 는 성능 향상기가 아니라 분산 감소기다.
TTA_ADOPTED = {"screw_bag"}

TTA_ANGLES = {"screw_bag": [0, 90, 180, 270], "breakfast_box": [0, 90, 180, 270],
              "juice_bottle": [-20, -10, 0, 10, 20], "pushpins": [-20, -10, 0, 10, 20],
              "splicing_connectors": [-20, -10, 0, 10, 20]}


def _unet(cat, seed, num_cls):
    key = (cat, seed)
    if key not in _UNET:
        import train_normal_unet as T
        m = T.CNNSegmenter(num_cls=num_cls, use_coord=True, level=3,
                           pretrained=False).cuda().eval()
        m.load_state_dict(torch.load(
            DATA / f"output/{SEG_TREE.get(cat, DEFAULT_TREE).format(seed=seed)}/{cat}/{cat}_300.pth",
            map_location="cpu"))
        _UNET[key] = m
    return _UNET[key]


def tta_mask(cat, seed, img_path, num_cls):
    """회전 TTA 로 분할을 안정화한다 (하드 argmax 유지).

    병합(E5)이 체계적 혼동(screw_bag c4<->c5)을 걷어낸 뒤에도 시드마다 **다른**
    클래스에서 한 장씩 우발적 분할 오류가 남는다 (seed42 cls2 5.7σ / seed43 cls3 7.9σ /
    seed44 cls1 9.9σ). 이건 패턴이 아니라 분산이므로 평균으로 줄인다.

    UNet 은 회전 증강으로 학습됐다 — screw_bag/breakfast 는 (0,360) 전방향이라
    90도 배수 회전에 대해 등변(equivariant)이어야 한다. 이미지와 coord 를 학습과
    동일하게 PIL 로 회전시켜 예측한 뒤 확률맵을 되돌려 평균한다. 90도 배수는 보간이
    없어 정확하다.

    val/test 통계는 쓰지 않는다 — 입력 이미지 한 장에서 닫히는 연산이다.
    """
    import torch.nn.functional as Fn
    from PIL import Image
    m = _unet(cat, seed, num_cls)
    im0 = Image.open(img_path).convert("RGB")
    w, h = im0.size
    xx, yy = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
    coord_im = Image.fromarray((np.stack([xx, yy, np.zeros_like(yy)], 2) * 255).astype(np.uint8))
    acc = None
    for ang in TTA_ANGLES.get(cat, [0]):
        im = im0.rotate(ang) if ang else im0
        ci = coord_im.rotate(ang) if ang else coord_im
        a = np.array(im) / 255.
        a = (a - P.imagenet_mean) / P.imagenet_std
        x = torch.tensor(np.swapaxes(np.swapaxes(a, 1, 2), 0, 1)).float().unsqueeze(0).cuda()
        c = np.array(ci)[:, :, :2] / 255
        c = torch.tensor(np.swapaxes(np.swapaxes(c, 1, 2), 0, 1)).float().unsqueeze(0).cuda() * 30
        prob = Fn.softmax(m(x, c), dim=1)[0]                  # [K,H,W]
        if ang:
            k = int(round(ang / 90)) % 4
            if ang % 90 == 0:
                prob = torch.rot90(prob, -k, dims=(1, 2))     # 정확 회전 복원
            else:
                import torchvision.transforms.functional as TF
                prob = TF.rotate(prob.unsqueeze(0), -ang)[0]
        acc = prob if acc is None else acc + prob
    prob = acc / len(TTA_ANGLES.get(cat, [0]))
    hard = torch.zeros_like(prob)
    hard.scatter_(0, prob.argmax(0, keepdim=True), 1.0)
    return hard


def soft_mask(cat, seed, img_path, num_cls):
    """UNet softmax 를 마스크로 쓴다 (하드 argmax 의 연속 완화).

    argmax 는 모델 확신에 대해 불연속이라, 유사한 두 부품 클래스 사이가 51:49 인
    화소와 49:51 인 화소가 완전히 다른 클래스로 떨어진다. screw_bag val 029 에서
    cls5 가 0 픽셀이 되고 cls4 가 35 sigma 로 부푼 사고가 그것이다. 히스토그램이
    추정하려는 양은 "각 부품의 면적"이므로 예측 분포 아래의 기대 면적이 자연스러운
    연속 추정량이고, 클래스별 특징 평균도 같은 소프트 가중으로 일관되게 확장된다.
    """
    import torch.nn.functional as Fn
    key = (cat, seed)
    if key not in _UNET:
        import train_normal_unet as T
        m = T.CNNSegmenter(num_cls=num_cls, use_coord=True, level=3,
                           pretrained=False).cuda().eval()
        m.load_state_dict(torch.load(
            DATA / f"output/{SEG_TREE.get(cat, DEFAULT_TREE).format(seed=seed)}/{cat}/{cat}_300.pth",
            map_location="cpu"))
        _UNET[key] = m
    m = _UNET[key]
    from PIL import Image
    im = np.array(Image.open(img_path).convert("RGB")) / 255.
    w, h, _ = im.shape
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    coord = (np.stack([xx, yy, np.zeros_like(yy)], 2) * 255).astype(np.uint8)[:, :, :2] / 255
    coord = torch.tensor(np.swapaxes(np.swapaxes(coord, 1, 2), 0, 1)).float().unsqueeze(0).cuda() * 30
    im = (im - P.imagenet_mean) / P.imagenet_std
    x = torch.tensor(np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)).float().unsqueeze(0).cuda()
    return Fn.softmax(m(x, coord), dim=1)[0]


# E10: splicing 커넥터 좌/우 분리 레이블로 학습한 UNet 은 별도 트리에 있다.
# 범주별로 어느 분할 트리를 읽을지 여기서 정한다 — 다른 범주는 기존 트리 그대로다.
SEG_TREE = {}          # {cat: "unet_split_seed{seed}"} 형태로 채우면 그 범주만 대체된다
# 분할맵 트리의 기본 템플릿. PSAD_SEG_TREE 로 덮을 수 있다 — 3-f 복제 실행처럼
# **같은 시드로 학습한 다른 실행**을 채점할 때 쓴다(예: "unet_seed42rep2").
# 시드 값은 그대로 두고 디렉터리만 바꾸는 것이 요점이다.
DEFAULT_TREE = os.environ.get("PSAD_SEG_TREE", "unet_seed{seed}")
SPLIT_RUN = False      # 태그는 **실행 단위**로 붙인다 — 범주마다 다르면 융합이 못 읽는다


@torch.no_grad()
def run_cat(cat, seed, standardize=1, soft=False, merge=False, tta=False, knn=1, w_n=0.0):
    num_cls = NUM_CLS[cat]
    tree = SEG_TREE.get(cat, DEFAULT_TREE).format(seed=seed)
    seg_root = DATA / tree / cat
    img_root = DATA / "orig_512" / cat
    enc = P.Encoder().cuda().eval()
    args = Args()
    merge_groups = confused_pairs(cat, seed) if merge else []

    def item(rel):
        """rel = 'split/type/NNN' -> (h특징, c특징, p점수)"""
        if tta and cat in TTA_ADOPTED:
            mask = tta_mask(cat, seed, str(img_root / f"{rel}.png"), num_cls)
        elif soft:
            mask = soft_mask(cat, seed, str(img_root / f"{rel}.png"), num_cls)
        else:
            mask = P.read_mask(args, str(seg_root / f"{rel}.png"), num_cls)
        if merge_groups:
            mask = apply_merge(mask, merge_groups)
        ad_mft = mask.sum((1, 2))                       # 면적 히스토그램 [num_cls]
        img = P.read_img(args, str(img_root / f"{rel}.png"))
        cls_index = P.get_cls_idx(mask)
        ad_ift = P.get_feature(args, enc, img, mask, cls_index)
        if w_n > 0:
            # E9: 클래스별 연결성분 개수. 면적 벡터에 이어붙이면 L2 에서 면적 신호가
            # 희석되므로(E2 기각) **거리를 따로 만들기 위해** 별도 블록으로 싣는다.
            from scipy import ndimage
            hard = mask.argmax(0).cpu().numpy() if mask.shape[0] > 1 else None
            cnt = torch.zeros(num_cls, device=mask.device)
            for k in range(1, num_cls):
                sel = hard == k
                if sel.any():
                    cnt[k] = float(ndimage.label(sel)[1])
            return torch.cat([ad_mft, cnt, ad_ift], 0), pc_score(cat, rel)
        return torch.cat([ad_mft, ad_ift], 0), pc_score(cat, rel)

    def rels(split, typ):
        d = img_root / split / typ
        return [f"{split}/{typ}/{p.stem}" for p in sorted(d.glob("*.png"))] if d.exists() else []

    train_rels = rels("train", "good")
    feats, pcs = [], []
    for i, r in enumerate(train_rels):
        f, s = item(r)
        feats.append(f)
        pcs.append(s)
        if i % 50 == 0:
            print(f"    train {i}/{len(train_rels)}", flush=True)
    memory = torch.stack(feats, 0).cuda()
    if standardize:
        mean, std = memory.mean(0), memory.std(0)
    else:
        # 차원별 표준화를 끈다. 저분산 차원이 (x-mean)/std 에서 증폭되어 거리를 지배하는
        # 문제가 있었다 — train 전용 기준(부품제거 탐지 AUROC, probe 시드 5개)에서
        # screw_bag +0.0226(probe sigma 의 41배) / juice +0.0019 / 평균 +0.0048 로 확인.
        mean = torch.zeros_like(memory[0]); std = torch.ones_like(memory[0])
    memory = (memory - mean) / (std + 1e-10)
    pcs = torch.tensor(pcs)

    if w_n > 0:
        mem_seg = memory[:, :num_cls]
        mem_cnt = memory[:, num_cls:2 * num_cls]
        mem_ft = memory[:, 2 * num_cls:]
    else:
        mem_seg, mem_cnt = memory[:, :num_cls], None
        mem_ft = memory[:, num_cls:]

    def loo_max(mem):
        """train 거리 통계의 최댓값 — 스케일 기준.

        knn == 1 은 **공식 psad.py(277-282행) 표현을 그대로** 쓴다:
            indices = dist.argsort(); dist[indices == 1]
        이는 "자기 제외 최근접"이 아니다 — 원본 인덱스 1 이 정렬에서 놓인 위치의 값을
        고르므로 자기 거리 0 이나 최댓값이 임의로 잡힌다. 저자 구현의 특성이고,
        재현 충실성을 위해 손대지 않는다. 실제로 이걸 "고치자" 정렬 기반으로 바꿨을 때
        스케일이 달라져 splicing 의 val 꼬리가 6.29 -> 37.68 로 튀었다.
        knn > 1 은 별개 확장 경로다(E8 에서 기각됐으므로 기본값은 1).
        """
        ds = []
        for i in range(mem.shape[0]):
            d = torch.sqrt(((mem - mem[[i]]) ** 2).sum(1))
            if knn == 1:
                ds.append(d[d.argsort() == 1])
            else:
                ds.append(torch.sort(d).values[1:1 + knn].mean().reshape(1))
        return torch.cat(ds).max()

    seg_max, ft_max, pc_max = loo_max(mem_seg), loo_max(mem_ft), pcs.max()
    cnt_max = loo_max(mem_cnt) if mem_cnt is not None else None
    print(f"    scale max: h={float(seg_max):.4f} "
          + (f"n={float(cnt_max):.4f} " if cnt_max is not None else "")
          + f"c={float(ft_max):.4f} p={float(pc_max):.4f}", flush=True)

    def score_rel(r):
        f, s = item(r)
        f = (f - mean) / (std + 1e-10)
        d = (memory - f) ** 2

        def nn(dd):
            v = torch.sqrt(dd.sum(1))
            # knn=1 이면 공식 그대로 최근접, knn>1 이면 이웃 knn 개 평균.
            # 메모리 항목 하나가 점수를 통째로 결정하던 취약성을 줄인다(E8 채택).
            return v.min() if knn == 1 else torch.sort(v).values[:knn].mean()

        if w_n > 0:
            h = nn(d[:, :num_cls]) / (seg_max + 1e-10)
            h = h + w_n * nn(d[:, num_cls:2 * num_cls]) / (cnt_max + 1e-10)
            c = nn(d[:, 2 * num_cls:]) / (ft_max + 1e-10)
        else:
            h = nn(d[:, :num_cls]) / (seg_max + 1e-10)
            c = nn(d[:, num_cls:]) / (ft_max + 1e-10)
        p = s / (pc_max + 1e-10)
        return float(h), float(c), float(p)

    out = {}
    for split_name, split_groups in [("test", [("test", "good"), ("test", "logical_anomalies"),
                                          ("test", "structural_anomalies")]),
                               ("val", [("validation", "good")])]:
        paths, gts, hs, cs, ps = [], [], [], [], []
        for split, typ in split_groups:
            for r in rels(split, typ):
                h, c, p = score_rel(r)
                paths.append(f"{r}.png")
                gts.append(0 if typ == "good" else 1)
                hs.append(h); cs.append(c); ps.append(p)
        out[split_name] = (np.array(paths), np.array(gts),
                           np.array(hs), np.array(cs), np.array(ps))
        print(f"    {split_name}: {len(paths)}장", flush=True)

    OUT.mkdir(exist_ok=True)
    tag = (("" if standardize else "_std0") + ("_soft" if soft else "")
           + ("_tta" if tta else "") + ("_merge" if merge else "")
           + (f"_k{knn}" if knn != 1 else "")
           + (f"_n{w_n:g}".replace(".", "") if w_n > 0 else "")
           + ("_split" if SPLIT_RUN else ""))
    n = 0
    for split_name, (paths, gts, hs, cs, ps) in out.items():
        # h, c, p 를 따로 남긴다 — 가중 탐색을 재채점 없이 하기 위해서다.
        # (scale_type 은 융합의 z-정규화에서 상수배로 상쇄되지만, h 와 c 를 서로 다른
        #  상수로 나누므로 **둘 사이의 비율**은 상쇄되지 않는다. 그게 실질 자유도다.)
        for mtype, sc in [("hc", hs + cs), ("hcp", hs + cs + ps)]:
            np.savez(OUT / f"psad_scores_{cat}_seed{seed}_{mtype}{tag}_{split_name}.npz",
                     scores=sc, paths=paths, gt=gts, h=hs, c=cs, p=ps)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cats", default=",".join(CATS))
    ap.add_argument("--standardize", type=int, default=1, choices=[0, 1],
                    help="0 이면 차원별 표준화를 끈다. **E1(standardize=0)은 기각됐다** — 정본은 기본값 1 이다(260817_testfree_hc_canonical.md 채택 2건/기각 8건)")
    ap.add_argument("--soft", action="store_true",
                    help="UNet softmax 를 마스크로 (하드 argmax 의 연속 완화)")
    ap.add_argument("--merge", action="store_true",
                    help="train 혼동행렬로 자동 검출한 혼동쌍을 병합")
    ap.add_argument("--tta", action="store_true",
                    help="회전 TTA 로 분할 안정화 (학습 증강 범위와 동일)")
    ap.add_argument("--knn", type=int, default=1,
                    help="메모리 최근접 이웃 수. **E8(k=5)은 기각됐다** — splicing val 꼬리 6.92 -> 24.88. 정본은 1")
    ap.add_argument("--w-n", type=float, default=0.0,
                    help="개수 거리 가중. **E9(w_n=1.0)은 기각됐다** — screw_bag val 꼬리 3.67 -> 14.10. 정본은 0(미사용)")
    ap.add_argument("--split-seg", action="store_true",
                    help="E10: splicing 을 좌/우 분리 레이블 학습 UNet 으로 채점")
    a = ap.parse_args()
    if a.split_seg:
        global SPLIT_RUN
        SPLIT_RUN = True
        SEG_TREE["splicing_connectors"] = "unet_split_seed{seed}"
        print("  E10: splicing -> unet_split_seed{seed} 사용", flush=True)
    # test AUROC 는 여기서 찍지 않는다. 채점기가 test 점수를 출력하면 그것을 보고
    # 설정을 고르게 되고, 그 순간 test-free 가 깨진다 (0.9807 사건). 평가는 사전등록된
    # 융합 스크립트에서 한 번만 한다.
    for cat in a.cats.split(","):
        print(f"=== {cat} (seed {a.seed}, standardize={a.standardize}, soft={a.soft}, merge={a.merge}, tta={a.tta}, knn={a.knn}, w_n={a.w_n}) ===", flush=True)
        n = run_cat(cat, a.seed, standardize=a.standardize, soft=a.soft, merge=a.merge, tta=a.tta, knn=a.knn, w_n=a.w_n)
        print(f"  점수 파일 {n}개 저장", flush=True)


if __name__ == "__main__":
    main()
