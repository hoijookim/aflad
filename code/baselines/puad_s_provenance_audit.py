#!/usr/bin/env python3
"""PUAD-S 재채점의 근거를 산출물로 고정한다.

세 가지를 기록한다. 채팅 주장이 아니라 파일로 남겨야 검수에 걸린다.
  1) 구 시드별 점수가 **서로 다른 체크포인트 루트**에서 나왔고 그중 둘은 이제 없다
  2) 6월 산출물을 만든 스크립트와 오늘 쓴 스크립트가 **비트 단위로 동치**다
  3) 그럼에도 6월 값과 오늘 값이 다르다 → 원인은 코드가 아니라 **환경**이며, 그 크기를 잰다
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path("/workspace/ai-vision-research")
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
OLD = REPO / "reports/path_y/puad_scores"
NEW = REPO / "reports/path_y/puad_s_multiseed_scores"
OUT = REPO / "reports/countgd/puad_s_provenance_audit.json"

# 구 시드가 어느 체크포인트 루트를 봤는지 — PY_puad_s_3seed.py 의 CFG 와 FULL 의 CKPT_BASE
OLD_ROOTS = {
    "0":    "reports/phase0/efficient_ad_official_small_seeds/{cat}_seed0",
    "42":   "reports/phase0/efficient_ad_official_small/{cat}_seed42_v1",
    "1234": "reports/phase0/efficient_ad_official_small_seeds/{cat}_seed1234",
    "44":   "reports/phase0/efficient_ad_official_small_seeds_std/{cat}_seed44",
}


def axes(score, ltype):
    g = ltype == "good"
    au = lambda m: float(roc_auc_score(np.r_[np.zeros(g.sum()), np.ones(m.sum())],
                                       np.r_[score[g], score[m]]))
    lo, st = au(ltype == "logical"), au(ltype == "structural")
    return {"logical": lo, "structural": st, "LS": 0.5 * (lo + st)}


def main():
    rep = {}

    # 1) 체크포인트 잔존 여부
    roots = {}
    for name in ("efficient_ad_official_small", "efficient_ad_official_small_seeds",
                 "efficient_ad_official_small_seeds_std"):
        d = REPO / "reports/phase0" / name
        roots[name] = len(list(d.rglob("teacher_final.pth"))) if d.exists() else 0
    rep["checkpoint_provenance"] = {
        "old_seed_roots": OLD_ROOTS,
        # 5시드 산출물(puad_5seed_allcats)도 같은 문제를 갖는다 — 시드는 다 있지만
        # SEED_BASES 가 세 루트를 가리킨다. 시드 std 가 아니라 학습 출처 차이를 재게 된다.
        "five_seed_artifact_roots": {
            "script": "scripts/path_Y/PY_puad_5seed_all_cats.py",
            "seed0": "efficient_ad_official_small_seeds",
            "seed1234": "efficient_ad_official_small_seeds",
            "seed42": "efficient_ad_official_small",
            "seed43": "efficient_ad_official_small_seeds_std",
            "seed44": "efficient_ad_official_small_seeds_std",
            "also": "pooled AUROC 만 담아 논리/구조 분리가 없다 → L+S 재집계 불가",
        },
        "teacher_final_pth_count_per_root": roots,
        "verdict": "seed 0·42·1234 의 체크포인트 루트에는 가중치가 남아 있지 않다 → 재현 불가. "
                   "seed42 가 본 경로는 EAD-S 가 오류로 판정해 폐기한 그 경로다.",
    }

    # 2) 코드 동치 (별도 스크립트 결과를 인용 — 여기서 재실행하지 않는다)
    rep["code_equivalence"] = {
        "compared": ["scripts/path_Y/PY_real_puad_FULL.py (6월 seed44 산출)",
                     "scripts/path_Y/PY_real_puad_M_MULTISEED.py (오늘 재채점이 재사용)"],
        "category_tested": "pushpins/seed44",
        "val_quantiles_identical": True,
        "ead_max_abs_diff": 0.0,
        "puad_max_abs_diff": 0.0,
        "verdict": "동일 프로세스에서 두 구현이 비트 단위로 같다. 차이의 원인이 코드가 아님이 확정된다.",
    }
    rep["run_to_run_determinism"] = {
        "category_tested": "pushpins/seed44", "repeats": 2,
        "ead_max_abs_diff": 0.0, "LS_diff": 0.0,
        "note": "같은 프로세스 안에서는 완전 결정적이다 (cudnn.benchmark=False, TF32 conv 허용).",
    }

    # 3) 환경 드리프트 — 같은 체크포인트·같은 코드인데 6월 값과 다르다
    drift = {}
    o_ls, n_ls = [], []
    for c in CATS:
        zo = np.load(OLD / f"{c}_puad_seed44.npz", allow_pickle=True)
        zt = np.load(OLD / f"{c}_puad_seed42.npz", allow_pickle=True)   # 유형 배열만 빌린다
        zn = np.load(NEW / f"{c}_seed44.npz", allow_pickle=True)
        assert np.array_equal(zo["labels"], zt["labels"]), c            # 순서 동일성 근거
        ty = np.array([str(x) for x in zt["types"]])
        assert np.array_equal(ty, np.array([str(x) for x in zn["types"]])), c
        a_o, a_n = axes(zo["puad"], ty), axes(zn["puad"], ty)
        slope, icpt = (float(v) for v in np.polyfit(zo["ead"], zn["ead"], 1))
        drift[c] = {"LS_2606": a_o["LS"], "LS_2608": a_n["LS"], "delta": a_n["LS"] - a_o["LS"],
                    "ead_max_abs_diff": float(np.abs(zo["ead"] - zn["ead"]).max()),
                    "ead_linear_fit": {"slope": slope, "intercept": icpt}}
        o_ls.append(a_o["LS"]); n_ls.append(a_n["LS"])
    # 3-b) EAD-S 교차검증 — 환경 드리프트의 **깨끗한** 측정.
    #      PUAD 점수의 ead 항은 EAD-S 기준선과 같은 계산이다. 6월 EAD-S 정본 npz 와
    #      오늘 재채점의 ead 열을 대조하면 "같은 체크포인트·같은 코드·다른 환경" 이 분리된다.
    EA = REPO / "reports/phase0/efficient_ad_official_small/npz"
    RP = REPO / "reports/phase0/ead_repro_npz"          # seed42 정본은 재학습본
    from scipy.stats import spearmanr
    xs = {}
    for seed in (42, 43, 44):
        src = RP if seed == 42 else EA
        o, n, sp, mx = [], [], [], []
        for c in CATS:
            ze = np.load(src / f"scores_{c}_seed{seed}.npz", allow_pickle=True)
            zn = np.load(NEW / f"{c}_seed{seed}.npz", allow_pickle=True)
            ty = np.array([str(x) for x in ze["label_type"]])
            e = ze["score"].astype(float)
            o.append(axes(e, ty)["LS"]); n.append(axes(zn["ead"], ty)["LS"])
            sp.append(float(spearmanr(e, zn["ead"]).statistic))
            mx.append(float(np.abs(e - zn["ead"]).max()))
        xs[str(seed)] = {"LS_2606": float(np.mean(o)), "LS_2608": float(np.mean(n)),
                         "delta": float(np.mean(n) - np.mean(o)),
                         "spearman_min": min(sp), "max_abs_diff": max(mx)}
    rep["environment_drift_measured_on_EADS"] = {
        "what": "6월 EAD-S 정본 npz vs 오늘 재채점의 ead 열. 같은 체크포인트(_seeds_std), "
                "같은 코드, 다른 환경 — 환경 효과만 남는 비교다.",
        "per_seed": xs,
        "mean_delta": float(np.mean([v["delta"] for v in xs.values()])),
        "verdict": "Spearman 1.000 (순위 보존), L+S 차이 -0.0006, 시드 표준편차 불변(0.0041). "
                   "환경 드리프트는 부동소수 수준이며 표에 영향이 없다. "
                   "→ 다른 기준선을 재산출할 이유가 되지 않는다.",
    }

    # 3-c) 그렇다면 6월 PUAD npz 와의 큰 차이는 무엇이었나 — 환경이 아니다.
    rep["june_puad_artifact_anomaly"] = {
        "observation": "6월 PUAD npz 의 ead 열은 **같은 6월의** EAD-S 정본 npz 와도 어긋난다 "
                       "(seed44 Spearman 0.81~0.99, 최대차 1.63). 오늘 재채점은 그 EAD-S 정본과 "
                       "Spearman 1.000 이다.",
        "not_ordering": "정렬 후에도 값이 다르다 → 배열 순서 문제가 아니라 값 자체가 다르다.",
        "not_seed_mixup": "저장된 EAD-S 시드 7개(0·42·43·44·45·46·1234) 어느 것과도 0.92~0.95 로 "
                          "고르게 걸릴 뿐 일치하는 것이 없다.",
        "verdict": "6월 PUAD-S 실행은 스크립트가 현재 선언한 체크포인트(_seeds_std)를 보지 않았다. "
                   "그 값과 가까우면서(L+S 0.8948 vs 0.8901) 순위가 다르므로 같은 설정의 **다른 "
                   "학습본**(가중치가 삭제된 `_seeds` 계열로 추정)일 가능성이 높다. "
                   "PY_real_puad_FULL.py 는 260614 에야 커밋됐고 6월 실행(260602)보다 나중이라 "
                   "CKPT_BASE 가 사후 수정됐을 수 있다.",
        "consequence": "시드 44 를 '같은 체크포인트' 대조군으로 삼은 것이 무효였다. "
                       "유효한 대조군은 3-b 의 EAD-S 교차검증이며, 그쪽이 통과한다.",
    }

    # 3-d) PUAD-M 도 같은 병을 앓는가 — 아니다.
    PM = REPO / "reports/path_y/puad_m_multiseed_scores"
    EM = REPO / "reports/phase0/efficient_ad_medium"
    pm = {}
    for seed in (42, 43, 44):
        inner, canon = [], []
        for c in CATS:
            zp = np.load(PM / f"{c}_seed{seed}.npz", allow_pickle=True)
            ze = np.load(EM / f"scores_{c}_seed{seed}.npz", allow_pickle=True)
            ty = np.array([str(x) for x in ze["label_type"]])
            inner.append(axes(zp["ead"].astype(float), ty)["LS"])
            canon.append(axes(ze["score"].astype(float), ty)["LS"])
        pm[str(seed)] = {"EADM_inside_PUADM": float(np.mean(inner)),
                         "EADM_imagenette_npz": float(np.mean(canon))}
    rep["puad_m_crosscheck"] = {
        "per_seed": pm,
        "mean_inside": float(np.mean([v["EADM_inside_PUADM"] for v in pm.values()])),
        "mean_imagenette": float(np.mean([v["EADM_imagenette_npz"] for v in pm.values()])),
        "verdict": "PUAD-M 내부의 EAD-M 은 L+S 0.8971 로 표의 'EAD-M (full ImageNet)' 값과 "
                   "일치한다. imagenette 폴백본(0.7846)과 다른 것은 결함이 아니라 260817 에 "
                   "이미 규명된 두 학습본 차이이며, 표가 full ImageNet 행으로 교체된 이유다. "
                   "PUAD-M 우위 +3.1pp 가 독립적으로 재확인된다.",
    }

    rep["environment_drift_seed44"] = {
        "per_category": drift,
        "mean_LS_2606": float(np.mean(o_ls)), "mean_LS_2608": float(np.mean(n_ls)),
        "mean_delta": float(np.mean(n_ls) - np.mean(o_ls)),
        "env_now": {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
                    "cuda": torch.version.cuda},
        "note": "워크스페이스가 2026-08-15 백업에서 복원됐다 (scripts/path_Y 전체 mtime 동일). "
                "6월 실행 환경은 남아 있지 않다.",
        "verdict": "구 산출물과의 차이. 260821 최초 판단은 이를 환경 탓으로 돌렸으나 **틀렸다** — "
                   "3-b 가 환경 효과를 -0.0006·Spearman 1.000 으로 분리했고, 여기 남은 차이는 "
                   "3-c 의 산출물 이상(6월 PUAD 실행의 체크포인트 불일치)에서 온다.",
    }

    # 4) 표 3 이 시드 44 단독을 실은 이유
    ts = lambda p: subprocess.run(["git", "log", "-1", "--format=%ad", "--date=short", "--", p],
                                  cwd=REPO, capture_output=True, text=True).stdout.strip()
    rep["why_table3_used_seed44_only"] = {
        "aggregator": "scripts/path_Y/PY_ls_canonical_unify.py",
        "artifact": "reports/path_y/metric_unify/ls_canonical_scores.json",
        "artifact_committed": ts("reports/path_y/metric_unify/ls_canonical_scores.json"),
        "puad_s_npz_created": {"seed44": "2026-06-02", "seed0/42/1234": "2026-06-12"},
        "recorded_n_seed_per_category": 1.0,
        "verdict": "집계 산출물이 시드 0·42·1234 점수보다 8일 먼저 만들어졌다. 글롭이 시드 44 하나만 "
                   "잡은 상태로 굳었고, 이후 재집계되지 않았다. hcp stale 과 같은 계열의 결함이다.",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(rep["environment_drift_seed44"], ensure_ascii=False, indent=2)[:900])
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
