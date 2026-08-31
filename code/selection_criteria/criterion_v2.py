#!/usr/bin/env python3
"""선택 기준 v2 — LOCO 논리 이상 유형을 마스크 수준에서 4가지로 모사한다.

v1(label_criterion.py)의 한계: **부품 제거** 한 가지만 봤다. 그래서 splicing 0.9996,
pushpins 0.9979 처럼 포화되어 후보를 가르지 못했다. 그런데 MVTec LOCO 의 실제 논리
이상은 제거만이 아니다 — 개수 초과, 종류 바뀜, 크기 이상이 함께 있다.

모사 유형 (전부 **마스크 위에서만** 조작한다):
  remove    성분 하나를 배경으로 지운다            (부품 누락)
  duplicate 성분 하나를 복사해 빈 곳에 붙인다       (개수 초과)
  swap      성분 하나의 클래스 번호를 다른 것으로   (종류 바뀜 / 오조립)
  resize    성분 하나를 0.6배 또는 1.5배로          (크기 이상)

이미지가 아니라 마스크를 건드리는 이유는 260412 FM-MEAD 에서 확인한 그대로다:
이미지 Cut-Paste 는 경계 아티팩트를 남겨 국소 기법에 부당하게 유리해진다. 여기서는
구성 분기 하나만 평가하므로 마스크 수준이 정확히 맞는 층위다.

test 는 어느 단계에서도 쓰지 않는다. 입력은 train 의사레이블뿐이다.
"""
import numpy as np
from scipy import ndimage


def _components(m, min_size=200):
    comp, n = ndimage.label(m > 0)
    if n == 0:
        return comp, []
    sizes = ndimage.sum(m > 0, comp, range(1, n + 1))
    return comp, [i + 1 for i, s in enumerate(sizes) if s >= min_size]


def perturb_remove(m, rng):
    comp, cand = _components(m)
    if not cand:
        return None
    out = m.copy()
    out[comp == cand[rng.integers(len(cand))]] = 0
    return out


def perturb_duplicate(m, rng):
    """성분 하나를 잘라 빈 배경 위치로 평행이동해 붙인다 (개수 +1)."""
    comp, cand = _components(m)
    if not cand:
        return None
    cid = cand[rng.integers(len(cand))]
    sel = comp == cid
    ys, xs = np.nonzero(sel)
    h, w = m.shape
    patch = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    pm = sel[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ph, pw = patch.shape
    if ph >= h or pw >= w:
        return None
    out = m.copy()
    for _ in range(30):                      # 배경이 충분히 비어있는 자리 찾기
        y = int(rng.integers(0, h - ph)); x = int(rng.integers(0, w - pw))
        win = out[y:y + ph, x:x + pw]
        if (win[pm] == 0).mean() > 0.9:      # 기존 부품을 덮지 않는 자리
            win[pm] = patch[pm]
            return out
    return None


def perturb_swap(m, rng):
    """성분 하나의 클래스 번호를 이미지 안의 다른 클래스로 바꾼다 (종류 바뀜)."""
    labs = [int(v) for v in np.unique(m) if v != 0]
    if len(labs) < 2:
        return None
    comp, cand = _components(m)
    if not cand:
        return None
    cid = cand[rng.integers(len(cand))]
    sel = comp == cid
    cur = int(np.bincount(m[sel]).argmax())
    others = [v for v in labs if v != cur]
    if not others:
        return None
    out = m.copy()
    out[sel] = others[rng.integers(len(others))]
    return out


def perturb_resize(m, rng):
    """성분 하나를 0.6배 축소 또는 1.5배 확대해 같은 중심에 다시 놓는다."""
    comp, cand = _components(m)
    if not cand:
        return None
    cid = cand[rng.integers(len(cand))]
    sel = comp == cid
    lab = int(np.bincount(m[sel]).argmax())
    f = 0.6 if rng.random() < 0.5 else 1.5
    ys, xs = np.nonzero(sel)
    sub = sel[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    zh = max(1, int(sub.shape[0] * f)); zw = max(1, int(sub.shape[1] * f))
    zoom = ndimage.zoom(sub, (zh / sub.shape[0], zw / sub.shape[1]), order=0) > 0
    out = m.copy()
    out[sel] = 0                             # 원래 자리 비우고
    cy = (ys.min() + ys.max()) // 2; cx = (xs.min() + xs.max()) // 2
    y0 = cy - zoom.shape[0] // 2; x0 = cx - zoom.shape[1] // 2
    h, w = m.shape
    ys0, xs0 = max(0, y0), max(0, x0)
    ye, xe = min(h, y0 + zoom.shape[0]), min(w, x0 + zoom.shape[1])
    if ye <= ys0 or xe <= xs0:
        return None
    z = zoom[ys0 - y0:ye - y0, xs0 - x0:xe - x0]
    win = out[ys0:ye, xs0:xe]
    win[z] = lab
    return out


PERTURBS = {"remove": perturb_remove, "duplicate": perturb_duplicate,
            "swap": perturb_swap, "resize": perturb_resize}
