#!/usr/bin/env python3
"""요청 ③ — 표 C1 을 5 카테고리 × 5 backbone 으로 채운다.

프로토콜은 scripts/countgd/d2d3_multidraw_tables67.py 와 동일하다:
  frozen backbone 다층 캐시 → patch-memory readout(coreset 50000, 1-NN, patch max)
  → 뱅크 추첨 시드 42~46 5회 → mean±std → 구조 AUROC 최대인 계층 선택.

기존 표와 달라지는 점이 하나 있다 — **열 정의를 통일한다.**
  기존 screw_bag 블록은 'cardinality' 열을 `logical` 클래스 전체로 계산했고
  (검증: 표의 0.4852±0.0115 가 logical 값과 비트 단위로 같다),
  pushpins 블록만 픽셀값 필터로 진짜 개수축을 썼다. 한 표 안에 두 정의가 있었다.
  여기서는 **logical 과 cardinality 를 별도 열로** 내고, cardinality 는 부록 C 의 사전등록
  STRAND 로 정의한다. 그래야 25칸이 하나의 규칙을 따른다.

juice_bottle 은 개수축 하위유형이 0개라 cardinality 칸이 **원리적으로 빈다**(값이 낮은 게 아니다).
"""
import ast
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
CACHE = R / "cache"
LOCO = R / "datasets/MVTecLOCO"
CANON = R / "algml_v6_5/aupr_bootstrap/scores"
SEEDS = [42, 43, 44, 45, 46]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_N = 4
OUT = R / "reports/countgd/tableC1_full_grid.json"

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
BACKBONES = [("DINOv3-S/16", "dinov3_multilayer_vits16", 22),
             ("DINOv3-B/16", "dinov3_multilayer_vitb16", 86),
             ("DINOv3-L/16", "dinov3_multilayer_vitl16", 303),
             ("DINOv2-B/14", "dinov2_multilayer_vitb14", 87),
             ("DINOv2-L/14", "dinov2_multilayer_vitl14", 304)]   # 파라미터는 260821 실측값

_tree = ast.parse((R / "scripts/countgd/footprint_resolved_allcat.py").read_text())
STRAND = next(ast.literal_eval(n.value) for n in _tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "STRAND")


def gpu_nn(Xtr, Xte, seed, cs=50000):
    """d2d3_multidraw_tables67.gpu_nn_scores 와 동일."""
    npat, dd = Xtr.shape[1], Xtr.shape[2]
    fl = Xtr.reshape(-1, dd).astype("f4")
    if fl.shape[0] > cs:
        fl = fl[np.random.default_rng(seed).choice(fl.shape[0], cs, replace=False)]
    bank = torch.from_numpy(fl).to(DEV)
    q = torch.from_numpy(Xte.reshape(-1, dd).astype("f4")).to(DEV)
    mins = []
    for i in range(0, q.shape[0], 2048):
        mins.append(torch.cdist(q[i:i + 2048], bank).min(1).values)
    del bank, q
    torch.cuda.empty_cache()
    return torch.cat(mins).cpu().numpy().reshape(Xte.shape[0], npat).max(1)


def axis_index(cat, lt):
    """good / structural / logical / cardinality 인덱스. 축 라벨은 부록 C 와 같은 경로."""
    cfg = {d["pixel_value"]: d["defect_name"]
           for d in json.load(open(LOCO / cat / "defects_config.json"))}
    gt = LOCO / cat / "ground_truth/logical_anomalies"
    subs = []
    for p in sorted((LOCO / cat / "test/logical_anomalies").glob("*.png")):
        d = gt / p.stem
        pv = set()
        if d.is_dir():
            for m in d.glob("*.png"):
                pv |= set(np.unique(np.array(Image.open(m))).tolist())
        names = sorted(cfg[v] for v in pv if v in cfg)
        subs.append(names[0] if len(names) == 1 else "MIXED")
    lp = np.where(lt == "logical")[0]
    assert len(lp) == len(subs), f"{cat}: 논리 이미지 수 불일치 ({len(lp)} vs {len(subs)})"
    card = [lp[i] for i, n in enumerate(subs)
            if n != "MIXED" and STRAND[cat].get(n) == "cardinality"]
    return {"good": np.where(lt == "good")[0], "structural": np.where(lt == "structural")[0],
            "logical": lp, "cardinality": np.array(card, dtype=int)}


def main():
    ms = lambda v: [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)]
    grid, counts = {}, {}
    for cat in CATS:
        canon = np.load(CANON / f"scores_{cat}_seed42.npz", allow_pickle=True)
        lt_fb = np.array([str(x) for x in canon["label_type"]])
        grid[cat] = {}
        for name, cdir, pm in BACKBONES:
            path = CACHE / cdir / f"{cat}.npz"
            if not path.exists():
                print(f"  [missing] {name} / {cat} — 캐시 없음"); continue
            z = np.load(path)
            lt = np.array([str(x) for x in z["test_ltypes"]]) if "test_ltypes" in z.files else lt_fb
            idx = axis_index(cat, lt)
            if cat not in counts:
                counts[cat] = {k: int(len(v)) for k, v in idx.items()}
            layers = sorted([k[6:] for k in z.files if k.startswith("train_L")])
            per_layer = {}
            for L in layers:
                tr, te = z[f"train_{L}"], z[f"test_{L}"]
                acc = {k: [] for k in ("structural", "logical", "cardinality")}
                for s in SEEDS:
                    sc = gpu_nn(tr, te, s)
                    for k in acc:
                        pos = idx[k]
                        if len(pos) < MIN_N:
                            continue
                        acc[k].append(float(roc_auc_score(
                            np.r_[np.zeros(len(idx["good"])), np.ones(len(pos))],
                            np.r_[sc[idx["good"]], sc[pos]])))
                per_layer[L] = {k: (ms(v) if v else None) for k, v in acc.items()}
                print(f"  [{cat} {name} {L}] str {per_layer[L]['structural']} "
                      f"log {per_layer[L]['logical']} card {per_layer[L]['cardinality']}", flush=True)
            best = max(per_layer, key=lambda L: per_layer[L]["structural"][0])
            grid[cat][name] = {"params_M": pm, "best_struct_layer": best,
                               **{k: per_layer[best][k] for k in
                                  ("structural", "logical", "cardinality")},
                               "per_layer": per_layer}
            del z

    res = {"seeds": SEEDS, "grid": grid, "axis_image_counts": counts,
           "protocol": "patch-memory readout (coreset 50000, 1-NN, patch max), 뱅크 추첨 5회, "
                       "구조 AUROC 최대 계층 선택. d2d3_multidraw_tables67.py 와 동일.",
           "column_note": "기존 표 C1 은 screw_bag 블록에서 'cardinality' 열을 logical 로 계산했고 "
                          "pushpins 블록만 진짜 개수축이었다. 여기서는 두 열을 분리하고 "
                          "cardinality 를 부록 C 의 사전등록 STRAND 로 통일했다. "
                          "juice_bottle 은 개수축 하위유형이 0개라 그 칸이 원리적으로 빈다.",
           "params_note": "파라미터는 260821 실측값(param_counts_measured.json)."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)

    for metric in ("structural", "cardinality", "logical"):
        print(f"\n=== {metric} (구조 최적 계층, 뱅크 5회 mean)")
        print(f"  {'category':22s}" + "".join(f"{n.split('/')[0]:>13s}" for n, _, _ in BACKBONES))
        for cat in CATS:
            row = ""
            for n, _, _ in BACKBONES:
                v = grid[cat].get(n, {}).get(metric)
                row += f"{v[0]:>13.4f}" if v else f"{'—':>13s}"
            print(f"  {cat:22s}{row}")
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
