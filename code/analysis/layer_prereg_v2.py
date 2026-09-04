#!/usr/bin/env python3
"""사전등록 v2 집행 — 계층 선택, **마지막 시도**.

규칙: `reports/countgd/PREREG_layer_selection_v2_260822.md` (실행 전 커밋 `899b01d`).

v1 대비 달라진 점 (전부 결과와 무관한 근거로 사전 선언):
  - `global_photo` 는 **채점에서 제외**하고 불변성 관찰로만 보고한다.
    부호를 뒤집지 않는다 — 뒤집으면 L17 의 패배가 승리로 바뀌고 그런 승리는 쓸 수 없다.
    지우지도 않는다. 계속 측정하고 표에 싣되 순위에서만 뺀다.
  - 강도 **3단(weak/med/strong)** 을 전부 채점한다. 난이도 보정 대신 곡선 전체를 보고해
    튜닝 여지를 없앤다.
  - `warp`(경계 없는 국소 탄성 변형) 추가 — v1 통제군이 하려던 일을 제대로 하는 계열.
  - 동점은 **평균 순위**, 포화 칸(세 계층 전부 >0.995)은 **자동 제외**.

**test 를 열지 않는다.** 캐시의 `test_*` 키를 읽지 않는다.

측정 경로는 v1 과 동일하며 캐시 규약(336 BICUBIC / ImageNet norm / `h[0,5:,:]`)을 따른다 —
정상 측은 캐시의 `train_L{n}[ntr:]`(= validation), 이상 측만 새로 인코딩한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

R = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(R / "scripts/direction_I"))
from I_dinov3_sl_cache import resolve_blocks  # noqa: E402

CACHE = R / "cache/dinov3_multilayer_vitl16"
LOCO = R / "datasets/MVTecLOCO"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
LAYERS = [8, 17, 23]
LN = {8: "L8", 17: "L17", 23: "L23"}
LEVELS = ["weak", "med", "strong"]
SCORED = ["paste", "fill", "blur", "warp"]          # 채점 대상
OBSERVE = ["global_photo"]                           # 관찰 전용 (순위 제외)
SAT = 0.995
BANK, RES = 50000, 336
MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = R / "reports/countgd/layer_prereg_v2.json"

# 강도표 (사전등록 D3/D4 그대로)
AREA = {"paste": {"weak": .01, "med": .025, "strong": .06},
        "fill": {"weak": .01, "med": .025, "strong": .06},
        "blur": {"weak": .04, "med": .04, "strong": .04},
        "warp": {"weak": .06, "med": .06, "strong": .06}}
KERNEL = {"weak": 5, "med": 11, "strong": 17}
DISP = {"weak": 2.0, "med": 4.0, "strong": 7.0}
PHOTO = {"weak": (0.95, 1.05, 8), "med": (0.85, 1.15, 18), "strong": (0.75, 1.30, 30)}

TF = transforms.Compose([
    transforms.Resize((RES, RES), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def _box(img, frac, rng):
    h, w = img.shape[:2]
    b = int((h * w * frac) ** 0.5)
    y = int(rng.integers(0, max(h - b, 1))); x = int(rng.integers(0, max(w - b, 1)))
    return y, x, b


def perturb(img, fam, lvl, rng, pool):
    out = img.copy()
    if fam == "global_photo":                         # 관찰 전용, 경계 없음
        lo, hi, bmax = PHOTO[lvl]
        a = rng.uniform(lo, 1.0) if rng.random() < 0.5 else rng.uniform(1.0, hi)
        return np.clip(out.astype(np.float32) * a + rng.uniform(-bmax, bmax), 0, 255).astype(np.uint8)

    y, x, b = _box(img, AREA[fam][lvl], rng)
    reg = out[y:y + b, x:x + b]

    if fam == "fill":
        out[y:y + b, x:x + b] = reg.reshape(-1, 3).mean(0).astype(np.uint8)
    elif fam == "paste":
        src = pool[int(rng.integers(0, len(pool)))]
        sy = int(rng.integers(0, max(src.shape[0] - b, 1)))
        sx = int(rng.integers(0, max(src.shape[1] - b, 1)))
        out[y:y + b, x:x + b] = src[sy:sy + b, sx:sx + b]
    elif fam == "blur":
        k = KERNEL[lvl]
        t = torch.from_numpy(reg.astype(np.float32)).permute(2, 0, 1)[None]
        t = F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, 1)
        out[y:y + b, x:x + b] = t[0].permute(1, 2, 0).numpy().astype(np.uint8)
    elif fam == "warp":
        # 국소 탄성 변형. 변위장을 Hann 창으로 감쇠시켜 **영역 경계에서 변위=0** 이 되게 한다
        # -> 붙이기도 없고 경계 불연속도 없다(사전등록의 "경계 없음").
        t = torch.from_numpy(reg.astype(np.float32)).permute(2, 0, 1)[None]
        g = 4
        d = torch.from_numpy(rng.normal(0, 1, (1, 2, g, g)).astype("f4"))
        d = F.interpolate(d, size=(b, b), mode="bicubic", align_corners=True)
        win = torch.hann_window(b).view(1, 1, -1, 1) * torch.hann_window(b).view(1, 1, 1, -1)
        d = d * win * (2.0 * DISP[lvl] / max(b, 1))      # 정규화 좌표계(-1..1) 단위
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, b), torch.linspace(-1, 1, b), indexing="ij")
        grid = torch.stack([xs + d[0, 0], ys + d[0, 1]], -1)[None]
        w = F.grid_sample(t, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        out[y:y + b, x:x + b] = w[0].permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
    return out


@torch.no_grad()
def encode(model, store, imgs):
    acc = {L: [] for L in LAYERS}
    for im in imgs:
        t = TF(Image.fromarray(im)).unsqueeze(0).to(DEV)
        store.clear(); model(t)
        for L in LAYERS:
            acc[L].append(store[L][0, 5:, :].float().cpu().numpy())
    return {L: np.stack(acc[L]) for L in LAYERS}


def knn_max(bank, X):
    b = torch.from_numpy(bank).to(DEV)
    q = torch.from_numpy(X.reshape(-1, X.shape[2]).astype("f4")).to(DEV)
    m = [torch.cdist(q[i:i + 2048], b).min(1).values for i in range(0, q.shape[0], 2048)]
    o = torch.cat(m).cpu().numpy().reshape(X.shape[0], X.shape[1]).max(1)
    del b, q; torch.cuda.empty_cache()
    return o


def avg_ranks(vals):
    """동점은 평균 순위 (사전등록 §4-1)."""
    order = sorted(vals.values(), reverse=True)
    return {k: float(np.mean([i + 1 for i, x in enumerate(order) if x == v])) for k, v in vals.items()}


def main():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
    blocks = resolve_blocks(model)
    store = {}
    for L in LAYERS:
        blocks[L].register_forward_hook(
            lambda m, i, o, L=L: store.__setitem__(L, o[0] if isinstance(o, tuple) else o))

    res = {}
    for cat in CATS:
        c = np.load(CACHE / f"{cat}.npz")
        ntr = len(list((LOCO / cat / "train/good").glob("*.png")))
        va_paths = sorted((LOCO / cat / "validation/good").glob("*.png"))
        va_imgs = [np.asarray(Image.open(p).convert("RGB")) for p in va_paths]
        pool = [np.asarray(Image.open(p).convert("RGB"))
                for p in sorted((LOCO / cat / "train/good").glob("*.png"))[:40]]
        trf = {L: c[f"train_L{L}"][:ntr] for L in LAYERS}
        vaf = {L: c[f"train_L{L}"][ntr:] for L in LAYERS}
        assert vaf[8].shape[0] == len(va_paths), f"{cat}: val 개수 불일치"
        banks = {}
        for seed in SEEDS:
            for L in LAYERS:
                fl = trf[L].reshape(-1, trf[L].shape[2]).astype("f4")
                if fl.shape[0] > BANK:
                    fl = fl[np.random.default_rng(seed).choice(fl.shape[0], BANK, replace=False)]
                banks[(seed, L)] = fl
        base = {L: {s: knn_max(banks[(s, L)], vaf[L].astype("f4")) for s in SEEDS} for L in LAYERS}

        for fam in SCORED + OBSERVE:
            for lvl in LEVELS:
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    pf = encode(model, store, [perturb(im, fam, lvl, rng, pool) for im in va_imgs])
                    for L in LAYERS:
                        sn, sp = base[L][seed], knn_max(banks[(seed, L)], pf[L])
                        au = float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sp))], np.r_[sn, sp]))
                        res.setdefault(fam, {}).setdefault(lvl, {}).setdefault(LN[L], {}).setdefault(cat, []).append(au)
                print(f"  [{cat} {fam}/{lvl}] " + "  ".join(
                    f"{LN[L]}={np.mean(res[fam][lvl][LN[L]][cat]):.4f}" for L in LAYERS), flush=True)

    # ---------- 판정 (사전등록 §4) ----------
    cell = {f: {lv: {LN[L]: float(np.mean([np.mean(res[f][lv][LN[L]][c]) for c in CATS]))
                     for L in LAYERS} for lv in LEVELS} for f in SCORED + OBSERVE}
    used, dropped, rk = [], [], {}
    for f in SCORED:
        for lv in LEVELS:
            v = cell[f][lv]
            if min(v.values()) > SAT:                       # 포화 칸 자동 제외
                dropped.append(f"{f}/{lv}"); continue
            used.append((f, lv)); rk[(f, lv)] = avg_ranks(v)
    ar = {LN[L]: float(np.mean([rk[k][LN[L]] for k in used])) for L in LAYERS}
    win = min(ar, key=ar.get)
    fam_rank = {f: {LN[L]: np.mean([rk[(f, lv)][LN[L]] for lv in LEVELS if (f, lv) in rk])
                    for L in LAYERS} for f in SCORED if any((f, lv) in rk for lv in LEVELS)}
    lastf = [f for f, r in fam_rank.items() if r[win] == max(r.values()) and len(set(r.values())) > 1]
    stable = not lastf
    verdict = win if stable else "NO_SINGLE_WINNER"

    print("\n=== 채점 계열 (5범주 × 3시드 평균 AUROC) ===")
    print(f"{'계열/강도':<18}" + "".join(f"{LN[L]:>10}" for L in LAYERS) + "   상태")
    for f in SCORED:
        for lv in LEVELS:
            v = cell[f][lv]; st = "제외(포화)" if f"{f}/{lv}" in dropped else ""
            print(f"{f+'/'+lv:<18}" + "".join(f"{v[LN[L]]:>10.4f}" for L in LAYERS) + f"   {st}")
    print(f"\n=== 관찰 전용 (순위 제외) — 낮을수록 교란에 강함 ===")
    for lv in LEVELS:
        v = cell["global_photo"][lv]
        print(f"{'global_photo/'+lv:<18}" + "".join(f"{v[LN[L]]:>10.4f}" for L in LAYERS))
    print(f"\n{'평균 순위':<18}" + "".join(f"{ar[LN[L]]:>10.3f}" for L in LAYERS))
    print(f"  사용한 칸 {len(used)}/12   제외 {dropped or '없음'}")
    print(f"  주 판정 : {win}")
    print(f"  안정성  : {'충족' if stable else '위반 — ' + ','.join(lastf)}")
    print(f"\n  ==> 사전등록 v2 판정: **{verdict}**")
    if verdict == "NO_SINGLE_WINNER":
        print("      §0 종결 조항 — 더 시도하지 않고 A(선언 규칙 + 민감도 보고)로 간다.")

    OUT.write_text(json.dumps({
        "prereg": "reports/countgd/PREREG_layer_selection_v2_260822.md (899b01d)",
        "test_opened": False, "cells": cell, "used_cells": [f"{f}/{lv}" for f, lv in used],
        "dropped_saturated": dropped, "avg_rank": ar, "family_rank": {k: {a: float(b) for a, b in v.items()} for k, v in fam_rank.items()},
        "winner": win, "stable": stable, "last_place_families": lastf,
        "verdict": verdict, "raw": res}, ensure_ascii=False, indent=2))
    print(f"  [saved] {OUT}")


if __name__ == "__main__":
    main()
