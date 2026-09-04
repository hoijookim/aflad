#!/usr/bin/env python3
"""사전등록 집행 — 패치 메모리 계층을 **test 없이** 고른다.

규칙: `reports/countgd/PREREG_layer_selection_260822.md` (실행 전 커밋, 7cb0f1b).

  뱅크    : train 정상 50,000 패치 무작위 표본 (주 설정과 동일)
  정상 측 : **val 정상** (캐시에서 읽는다)
  이상 측 : 같은 val 정상 이미지에 섭동 4계열을 가해 **새로 인코딩**
  점수    : 패치별 1-NN L2 의 공간 최댓값 (주 설정 readout)
  지표    : 정상 vs 섭동 AUROC

**test 를 열지 않는다** — 캐시의 `test_*` 키를 읽지 않는다.

## 캐시 규약 (I_dinov3_sl_cache.py 와 반드시 일치시켜야 한다)
  - `train_L{n}` 은 **train+validation 이 이 순서로 연결**돼 있다(breakfast_box 413=351+62).
  - 336x336 **BICUBIC** resize, ImageNet mean/std 정규화.
  - 블록 forward hook 출력에서 `h[0, 5:, :]` — **CLS + 레지스터 4개 제거**.
섭동본을 다르게 인코딩하면 정상/이상이 서로 다른 파이프라인에서 나와 비교가 무너진다.
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
from I_dinov3_sl_cache import resolve_blocks  # noqa: E402  (캐시와 동일한 블록 해석)

CACHE = R / "cache/dinov3_multilayer_vitl16"
LOCO = R / "datasets/MVTecLOCO"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
LAYERS = [8, 17, 23]
LN = {8: "L8", 17: "L17", 23: "L23"}
FAMILIES = ["paste", "blur", "fill", "global_photo"]
BANK, RES = 50000, 336
MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = R / "reports/countgd/layer_prereg_trainonly.json"

TF = transforms.Compose([
    transforms.Resize((RES, RES), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def perturb(img, family, rng, pool):
    """img: HxWx3 uint8 (원본 해상도). 결정적으로 섭동한 사본."""
    h, w = img.shape[:2]
    out = img.copy()
    if family == "global_photo":                       # 경계 없음 (통제군)
        a = rng.uniform(0.75, 0.9) if rng.random() < 0.5 else rng.uniform(1.1, 1.3)
        return np.clip(out.astype(np.float32) * a + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    bh = bw = int((h * w * rng.uniform(0.04, 0.10)) ** 0.5)
    y = int(rng.integers(0, max(h - bh, 1))); x = int(rng.integers(0, max(w - bw, 1)))
    reg = out[y:y + bh, x:x + bw]
    if family == "fill":                               # 경계 중간
        out[y:y + bh, x:x + bw] = reg.reshape(-1, 3).mean(0).astype(np.uint8)
    elif family == "blur":                             # 경계 부드러움
        t = torch.from_numpy(reg.astype(np.float32)).permute(2, 0, 1)[None]
        k = 2 * int(rng.integers(4, 9)) + 1
        t = F.avg_pool2d(F.pad(t, (k // 2,) * 4, mode="reflect"), k, 1)
        out[y:y + bh, x:x + bw] = t[0].permute(1, 2, 0).numpy().astype(np.uint8)
    elif family == "paste":                            # 경계 단단
        src = pool[int(rng.integers(0, len(pool)))]
        sy = int(rng.integers(0, max(src.shape[0] - bh, 1)))
        sx = int(rng.integers(0, max(src.shape[1] - bw, 1)))
        out[y:y + bh, x:x + bw] = src[sy:sy + bh, sx:sx + bw]
    return out


@torch.no_grad()
def encode(model, store, imgs):
    """섭동 이미지 리스트 -> {layer: (N, 441, D)}. 캐시와 동일 경로."""
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
        tr_feat = {L: c[f"train_L{L}"][:ntr] for L in LAYERS}
        va_feat = {L: c[f"train_L{L}"][ntr:] for L in LAYERS}
        assert va_feat[8].shape[0] == len(va_paths), f"{cat}: val 개수 불일치"

        for fam in FAMILIES:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                pf = encode(model, store, [perturb(im, fam, rng, pool) for im in va_imgs])
                for L in LAYERS:
                    tr = tr_feat[L]
                    fl = tr.reshape(-1, tr.shape[2]).astype("f4")
                    if fl.shape[0] > BANK:
                        fl = fl[np.random.default_rng(seed).choice(fl.shape[0], BANK, replace=False)]
                    sn, sp = knn_max(fl, va_feat[L].astype("f4")), knn_max(fl, pf[L])
                    au = float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sp))], np.r_[sn, sp]))
                    res.setdefault(fam, {}).setdefault(LN[L], {}).setdefault(cat, []).append(au)
            print(f"  [{cat} {fam}] " + "  ".join(
                f"{LN[L]}={np.mean(res[fam][LN[L]][cat]):.4f}" for L in LAYERS), flush=True)

    # ---------- 판정 (사전등록 §2) ----------
    fm = {f: {LN[L]: float(np.mean([np.mean(res[f][LN[L]][c]) for c in CATS])) for L in LAYERS}
          for f in FAMILIES}
    rk = {f: {l: i + 1 for i, (l, _) in enumerate(sorted(fm[f].items(), key=lambda kv: -kv[1]))}
          for f in FAMILIES}
    ar = {LN[L]: float(np.mean([rk[f][LN[L]] for f in FAMILIES])) for L in LAYERS}
    win = min(ar, key=ar.get)
    lastf = [f for f in FAMILIES if rk[f][win] == 3]
    stable = not lastf
    flip = rk["paste"] != rk["global_photo"]
    gwin = max(fm["global_photo"], key=fm["global_photo"].get)

    print("\n=== 계열별 평균 AUROC (5범주 × 3시드) ===")
    print(f"{'계열':<14}" + "".join(f"{LN[L]:>10}" for L in LAYERS) + "   1위")
    for f in FAMILIES:
        print(f"{f:<14}" + "".join(f"{fm[f][LN[L]]:>10.4f}" for L in LAYERS)
              + f"   {max(fm[f], key=fm[f].get)}")
    print(f"\n{'평균 순위':<14}" + "".join(f"{ar[LN[L]]:>10.2f}" for L in LAYERS))
    print(f"\n  주 판정(평균 순위 최고) : {win}")
    print(f"  안정성 요건(꼴찌 없음)   : {'충족' if stable else '위반 — ' + ','.join(lastf)}")
    print(f"  경계 점검(paste vs global_photo) : {'뒤집힘' if flip else '일치'}"
          + (f"  -> 통제군 우선: {gwin}" if flip else ""))
    verdict = win if stable else "NO_SINGLE_WINNER"
    print(f"\n  ==> 사전등록 판정: **{verdict}**")

    OUT.write_text(json.dumps({
        "prereg": "reports/countgd/PREREG_layer_selection_260822.md (7cb0f1b)",
        "test_opened": False, "per_family_mean": fm, "ranks": rk, "avg_rank": ar,
        "winner_by_avg_rank": win, "stable": stable, "last_place_families": lastf,
        "paste_vs_global_flip": flip, "global_photo_winner": gwin,
        "verdict": verdict, "raw": res}, ensure_ascii=False, indent=2))
    print(f"  [saved] {OUT}")


if __name__ == "__main__":
    main()
