#!/usr/bin/env python3
"""사전등록 집행 — 계층 **전수 스윕 L1~L24** (요청 ⑨).

규칙: `reports/countgd/PREREG_layer_full_sweep_260822.md` (실행 전 커밋 `07574d5`).

v2(`layer_prereg_v2.py`)의 채점 로직을 **한 글자도 바꾸지 않고** 후보 집합만
{L8, L17, L23} → {L0 … L23} 으로 넓힌다. 추가로 **계층별 융합 L+S**(민감도 곡선)를 낸다 —
요청서 판단대로 논문에 실릴 본체는 순위표가 아니라 이쪽이다.

## 캐시를 쓰지 않고 전부 인코딩하는 이유
`cache/dinov3_multilayer_vitl16` 에는 L8/L17/L23 만 있다. 전수를 하려면 train·val·test 를
전 계층으로 다시 인코딩해야 한다. 캐시 규약(336 BICUBIC / ImageNet norm / `h[0,5:,:]`)을
그대로 따르고, **L8/L17/L23 은 캐시와 일치하는지 자체 검증**한다(무결성 확인).

## 메모리
범주마다 train+val 인코딩 → 계층별 뱅크 구축 → train 특징 해제 → test 인코딩 → 채점 → 해제.
피크는 범주당 약 18 GB(여유 56 GB). **디스크 증가 0.**

## test 사용에 대하여
1단(선택)은 train/val 만 쓴다 — test 무접촉. test 는 **민감도 곡선 보고**에만 쓰며 이는
선택이 아니다. 교체 게이트(2단)는 test 를 참조하지만 보수적 문턱이다(사전등록 §2).
"""
import gc
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
sys.path.insert(0, str(R / "scripts/psad_rebuild"))
from I_dinov3_sl_cache import resolve_blocks           # noqa: E402
from layer_prereg_v2 import perturb, avg_ranks, TF, knn_max  # noqa: E402  (v2 로직 재사용)

CACHE = R / "cache/dinov3_multilayer_vitl16"
LOCO = R / "datasets/MVTecLOCO"
EAD_REPRO = R / "reports/phase0/ead_repro_npz"
EAD_STD = R / "reports/phase0/efficient_ad_official_small/npz"
EAD_VAL = R / "reports/phase0/efficient_ad_official_small_seeds_std_val"
PSAD = R / "external/PSAD_official/LOCO_MVTec_AD/rebuild_scores"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
LAYERS = list(range(24))                                # 전수
LN = {L: f"L{L}" for L in LAYERS}
LEVELS = ["weak", "med", "strong"]
SCORED = ["paste", "fill", "blur", "warp"]
OBSERVE = ["global_photo"]
SAT, BANK = 0.995, 50000
MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = R / "reports/countgd/layer_full_sweep.json"
DEFECTS = ["good", "logical_anomalies", "structural_anomalies"]
DLT = {"good": "good", "logical_anomalies": "logical", "structural_anomalies": "structural"}


@torch.no_grad()
def encode(model, store, imgs):
    acc = {L: [] for L in LAYERS}
    for im in imgs:
        t = TF(Image.fromarray(im)).unsqueeze(0).to(DEV)
        store.clear(); model(t)
        for L in LAYERS:
            acc[L].append(store[L][0, 5:, :].float().cpu().numpy())
    return {L: np.stack(acc[L]) for L in LAYERS}


def z(t, v):
    return (t - v.mean()) / (v.std() + 1e-12)


def ls_of(score, lt):
    g = lt == "good"
    au = lambda m: float(roc_auc_score(np.r_[np.zeros(g.sum()), np.ones(m.sum())],
                                       np.r_[score[g], score[m]]))
    return 0.5 * (au(lt == "logical") + au(lt == "structural"))


def main():
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).to(DEV).eval()
    blocks = resolve_blocks(model)
    store = {}
    for L in LAYERS:
        blocks[L].register_forward_hook(
            lambda m, i, o, L=L: store.__setitem__(L, o[0] if isinstance(o, tuple) else o))

    res, fusion, integrity = {}, {}, {}
    for cat in CATS:
        tr_paths = sorted((LOCO / cat / "train/good").glob("*.png"))
        va_paths = sorted((LOCO / cat / "validation/good").glob("*.png"))
        te_paths, lt = [], []
        for d in DEFECTS:
            for p in sorted((LOCO / cat / "test" / d).glob("*.png")):
                te_paths.append(p); lt.append(DLT[d])
        lt = np.array(lt)

        rd = lambda ps: [np.asarray(Image.open(p).convert("RGB")) for p in ps]
        va_imgs = rd(va_paths)
        pool = rd(tr_paths[:40])

        print(f"[{cat}] train {len(tr_paths)} · val {len(va_paths)} · test {len(te_paths)} 인코딩", flush=True)
        trf = encode(model, store, rd(tr_paths))
        vaf = encode(model, store, va_imgs)

        # 무결성 — 캐시 보유 계층(L8/L17/L23)과 일치하는가
        c = np.load(CACHE / f"{cat}.npz")
        ntr = len(tr_paths)
        for L in (8, 17, 23):
            d1 = float(np.abs(trf[L] - c[f"train_L{L}"][:ntr]).max())
            d2 = float(np.abs(vaf[L] - c[f"train_L{L}"][ntr:]).max())
            integrity.setdefault(cat, {})[LN[L]] = {"train_maxdiff": d1, "val_maxdiff": d2}
        print(f"  무결성(캐시 대조) " + "  ".join(
            f"{LN[L]} {integrity[cat][LN[L]]['train_maxdiff']:.2e}" for L in (8, 17, 23)), flush=True)

        banks = {}
        for seed in SEEDS:
            for L in LAYERS:
                fl = trf[L].reshape(-1, trf[L].shape[2]).astype("f4")
                if fl.shape[0] > BANK:
                    fl = fl[np.random.default_rng(seed).choice(fl.shape[0], BANK, replace=False)]
                banks[(seed, L)] = fl
        del trf; gc.collect()

        base = {L: {s: knn_max(banks[(s, L)], vaf[L].astype("f4")) for s in SEEDS} for L in LAYERS}

        # ---- 민감도 곡선 (보고용) ----
        tef = encode(model, store, rd(te_paths))
        for seed in SEEDS:
            e = np.load((EAD_REPRO if seed == 42 else EAD_STD) / f"scores_{cat}_seed{seed}.npz", allow_pickle=True)
            ev = np.load(EAD_VAL / f"val_good_{cat}_seed{seed}.npz", allow_pickle=True)
            b = f"{PSAD}/psad_scores_{cat}_seed{seed}_hc_tta_merge"
            ct = np.load(f"{b}_test.npz", allow_pickle=True)["scores"].astype(float)
            cv = np.load(f"{b}_val.npz", allow_pickle=True)["scores"].astype(float)
            assert (np.array([str(x) for x in e["label_type"]]) == lt).all(), f"{cat}: 정렬 규약 불일치"
            ze, zc = z(np.asarray(e["score"], float), np.asarray(ev[ev.files[0]], float)), z(ct, cv)
            for L in LAYERS:
                v = ls_of(ze + z(knn_max(banks[(seed, L)], tef[L].astype("f4")), base[L][seed]) + zc, lt)
                fusion.setdefault(LN[L], {}).setdefault(cat, []).append(v)
        del tef; gc.collect()
        print(f"  [융합] " + "  ".join(f"{LN[L]}={np.mean(fusion[LN[L]][cat]):.4f}"
                                       for L in (0, 8, 17, 23)), flush=True)

        # ---- 채점 (v2 규칙 그대로) ----
        for fam in SCORED + OBSERVE:
            for lvl in LEVELS:
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    pf = encode(model, store, [perturb(im, fam, lvl, rng, pool) for im in va_imgs])
                    for L in LAYERS:
                        sn, sp = base[L][seed], knn_max(banks[(seed, L)], pf[L])
                        au = float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sp))], np.r_[sn, sp]))
                        res.setdefault(fam, {}).setdefault(lvl, {}).setdefault(LN[L], {}).setdefault(cat, []).append(au)
                    del pf; gc.collect()
            print(f"  [{fam}] 완료", flush=True)
        del banks, base, vaf; gc.collect()

    # ---------- 판정 (사전등록 §2-1단) ----------
    cell = {f: {lv: {LN[L]: float(np.mean([np.mean(res[f][lv][LN[L]][c]) for c in CATS]))
                     for L in LAYERS} for lv in LEVELS} for f in SCORED + OBSERVE}
    used, dropped, rk = [], [], {}
    for f in SCORED:
        for lv in LEVELS:
            v = cell[f][lv]
            if min(v.values()) > SAT:
                dropped.append(f"{f}/{lv}"); continue
            used.append((f, lv)); rk[(f, lv)] = avg_ranks(v)
    ar = {LN[L]: float(np.mean([rk[k][LN[L]] for k in used])) for L in LAYERS}
    win = min(ar, key=ar.get)
    fam_rank = {f: {LN[L]: float(np.mean([rk[(f, lv)][LN[L]] for lv in LEVELS if (f, lv) in rk]))
                    for L in LAYERS} for f in SCORED if any((f, lv) in rk for lv in LEVELS)}
    lastf = [f for f, r in fam_rank.items() if r[win] == max(r.values()) and len(set(r.values())) > 1]
    stable = not lastf
    fu = {LN[L]: float(np.mean([np.mean(fusion[LN[L]][c]) for c in CATS])) for L in LAYERS}

    # ---------- 교체 게이트 (사전등록 §2-2단) ----------
    gate = {"winner": win, "winner_is_L17": win == "L17",
            "fusion_win": fu[win], "fusion_L17": fu["L17"],
            "delta": fu[win] - fu["L17"], "threshold": 0.0021}
    gate["replace_candidate"] = (win != "L17") and (gate["delta"] > gate["threshold"])

    print("\n=== 1단 선택 (train/val 만) ===")
    top = sorted(ar.items(), key=lambda kv: kv[1])[:6]
    print("  평균 순위 상위 6: " + "  ".join(f"{k} {v:.2f}" for k, v in top))
    print(f"  L17 평균 순위 {ar['L17']:.2f}   사용 칸 {len(used)}/12   제외 {dropped or '없음'}")
    print(f"  1위 {win}   안정성 {'충족' if stable else '위반 — ' + ','.join(lastf)}")
    print("\n=== 민감도 곡선 (융합 L+S, 보고용) ===")
    for i in range(0, 24, 6):
        print("  " + "  ".join(f"{LN[L]} {fu[LN[L]]:.4f}" for L in LAYERS[i:i + 6]))
    best_fu = max(fu, key=fu.get)
    print(f"  최고 {best_fu} {fu[best_fu]:.4f}   L17 {fu['L17']:.4f}   폭 {max(fu.values())-min(fu.values()):.4f}")
    print("\n=== 2단 교체 게이트 ===")
    print(f"  1위={win}  융합 {gate['fusion_win']:.4f} vs L17 {gate['fusion_L17']:.4f}  "
          f"Δ={gate['delta']:+.4f}  문턱 {gate['threshold']}")
    print(f"  ==> {'교체 후보 (저자 판단)' if gate['replace_candidate'] else 'L17 유지'}")

    OUT.write_text(json.dumps({
        "prereg": "reports/countgd/PREREG_layer_full_sweep_260822.md (07574d5)",
        "layers": [LN[L] for L in LAYERS], "cells": cell,
        "used_cells": [f"{f}/{lv}" for f, lv in used], "dropped_saturated": dropped,
        "avg_rank": ar, "family_rank": fam_rank, "winner": win, "stable": stable,
        "last_place_families": lastf, "fusion_ls_per_layer": fu,
        "fusion_ls_per_cat": {k: {c: float(np.mean(x)) for c, x in v.items()} for k, v in fusion.items()},
        "gate": gate, "cache_integrity": integrity, "raw": res}, ensure_ascii=False, indent=2))
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
