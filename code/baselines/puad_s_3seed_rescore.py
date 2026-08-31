#!/usr/bin/env python3
"""PUAD-S 를 시드 {42, 43, 44} 로 재채점한다 — 학습 없이 추론만.

배경
  표 3 의 PUAD-S 행(0.9276)은 **시드 44 단독**이다. 기존 per-image 점수
  (`reports/path_y/puad_scores/`)에 시드 0·42·44·1234 는 있는데 **43 이 없어서**
  {42,43,44} 재집계가 막혀 있었다. 그런데 EAD-S 체크포인트는 43 을 포함해 5범주 전부
  남아 있다 — 재학습이 아니라 **재채점**으로 풀리는 문제였다. PUAD-M 을 푼 것과 같다.

파이프라인은 `PY_real_puad_M_MULTISEED.py` 와 동일하고 체크포인트 경로만 -S 로 바꾼다.
PUAD 점수 = z(EAD 점수) + z(마할라노비스), 둘 다 **검증셋 통계로** 정규화한다.

검증
  시드 44 를 함께 돌려 기존 산출물(0.9276)이 재현되는지 먼저 본다. 재현되면 같은
  파이프라인이라는 뜻이고, 그때 시드 43 값을 믿을 수 있다.

사용: python3 scripts/psad_rebuild/puad_s_3seed_rescore.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path("/workspace/ai-vision-research")
sys.path.insert(0, str(REPO / "scripts/path_Y"))

# 원 스크립트는 에너지 측정에 pynvml 을 쓴다. 우리는 점수만 필요하고 이 환경엔 없으므로
# import 만 통과시킨다 (EnergyTracker 를 호출하지 않으니 동작에 영향이 없다).
if "pynvml" not in sys.modules:
    import types as _t
    sys.modules["pynvml"] = _t.ModuleType("pynvml")

import PY_real_puad_M_MULTISEED as M   # noqa: E402  파이프라인 함수 재사용

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
CKPT = REPO / "reports/phase0/efficient_ad_official_small_seeds_std"
OUT_SCORES = REPO / "reports/path_y/puad_s_multiseed_scores"
OUT_JSON = REPO / "reports/countgd/puad_s_rescore_3seed.json"
REF = {"seed44_LS": 0.9276}     # 기존 산출물. 재현되면 파이프라인 일치.


def ckpt_dir(seed, cat):
    return CKPT / f"{cat}_seed{seed}" / "trainings/mvtec_loco" / cat


def test_paths_with_type(cat):
    """list_split 과 같은 순서(하위 디렉터리 sorted)로 유형 라벨을 함께 만든다."""
    base = M.LOCO_DIR / cat / "test"
    out = []
    for sub in sorted(base.iterdir()):
        t = ("good" if sub.name == "good"
             else "logical" if "logical" in sub.name else "structural")
        for p in sorted(sub.iterdir()):
            out.append((p, 0 if t == "good" else 1, t))
    return out


@torch.no_grad()
def score_one(seed, cat):
    cd = ckpt_dir(seed, cat)
    teacher = torch.load(cd / "teacher_final.pth", map_location=M.DEVICE, weights_only=False)
    student = torch.load(cd / "student_final.pth", map_location=M.DEVICE, weights_only=False)
    ae = torch.load(cd / "autoencoder_final.pth", map_location=M.DEVICE, weights_only=False)
    teacher.eval(); student.eval(); ae.eval()

    train_paths = M.list_split(cat, "train")
    val_paths = M.list_split(cat, "validation")
    t_mean, t_std = M.teacher_normalization(teacher, train_paths)
    qs = M.map_normalization(val_paths, teacher, student, ae, t_mean, t_std)
    mean_v, cov_inv, _, _ = M.fit_puad_train(train_paths, teacher, student, ae, t_mean, t_std, *qs)
    ead_mu, ead_sig, maha_mu, maha_sig = M.fit_puad_val(
        val_paths, teacher, student, ae, t_mean, t_std, *qs, mean_v, cov_inv)

    ead_s, puad_s, labels, types = [], [], [], []
    for p, lbl, t in test_paths_with_type(cat):
        s, f = M.predict_full(M.load_image(p), teacher, student, ae, t_mean, t_std, *qs)
        c = f - mean_v
        maha = float(np.sqrt(max(0.0, c @ cov_inv @ c)))
        ead_s.append(s)
        puad_s.append((s - ead_mu) / max(ead_sig, 1e-6) + (maha - maha_mu) / max(maha_sig, 1e-6))
        labels.append(lbl); types.append(t)
    return (np.array(ead_s), np.array(puad_s), np.array(labels), np.array(types))


def axes(scores, types):
    g = types == "good"
    def au(m):
        return float(roc_auc_score(np.r_[np.zeros(g.sum()), np.ones(m.sum())],
                                   np.r_[scores[g], scores[m]]))
    lo, st = au(types == "logical"), au(types == "structural")
    return {"logical": lo, "structural": st, "LS": 0.5 * (lo + st)}


def main():
    OUT_SCORES.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for seed in SEEDS:
        rows = {}
        for cat in CATS:
            if not (ckpt_dir(seed, cat) / "teacher_final.pth").exists():
                print(f"  [skip] seed{seed}/{cat} 체크포인트 없음"); continue
            ead, puad, lab, typ = score_one(seed, cat)
            np.savez(OUT_SCORES / f"{cat}_seed{seed}.npz",
                     ead=ead, puad=puad, labels=lab, types=typ)
            rows[cat] = axes(puad, typ)
            print(f"  seed{seed} {cat:22s} L+S {rows[cat]['LS']:.4f}")
        if len(rows) == len(CATS):
            per_seed[seed] = {k: float(np.mean([rows[c][k] for c in CATS]))
                              for k in ("logical", "structural", "LS")}
            per_seed[seed]["per_cat"] = rows
            print(f"  == seed{seed} 5범주 평균 L+S {per_seed[seed]['LS']:.4f}\n")

    done = [s for s in SEEDS if s in per_seed]
    ls = [per_seed[s]["LS"] for s in done]
    agg = {"seeds": done, "LS": {"mean": float(np.mean(ls)),
                                 "std": float(np.std(ls, ddof=1)) if len(ls) > 1 else None},
           "logical": float(np.mean([per_seed[s]["logical"] for s in done])),
           "structural": float(np.mean([per_seed[s]["structural"] for s in done]))}
    print(f"\n  PUAD-S {done} L+S = {agg['LS']['mean']:.4f} ± {agg['LS']['std']:.4f}")

    if 44 in per_seed:
        d = abs(per_seed[44]["LS"] - REF["seed44_LS"])
        print(f"  파이프라인 검증: seed44 재채점 {per_seed[44]['LS']:.4f} vs "
              f"기존 {REF['seed44_LS']:.4f} · 차이 {d:.4f} "
              f"→ {'일치' if d < 1e-3 else '불일치 — 확인 필요'}")
        agg["pipeline_check"] = {"seed44_rescored": per_seed[44]["LS"],
                                 "seed44_existing": REF["seed44_LS"], "diff": d}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"per_seed": {str(k): v for k, v in per_seed.items()}, "aggregate": agg,
               "note": "EAD-S 체크포인트로 PUAD-S 재채점. 학습 없음. "
                       "PUAD 점수 = z(EAD) + z(Mahalanobis), 검증셋 통계로 정규화."},
              open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"  [saved] {OUT_JSON}")


if __name__ == "__main__":
    main()
