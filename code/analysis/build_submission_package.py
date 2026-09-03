#!/usr/bin/env python3
"""MDPI 제출용 재현 패키지 조립 — 논문의 모든 수치를 뒷받침하는 코드·산출물을 한 폴더로.

원칙
  1. **논문에 실린 수치를 뒷받침하는 것만** 담는다. 탐색 과정의 폐기 실험은 제외하되,
     사전등록 문서에 기각 근거로 남은 것은 진단 산출물로 포함한다.
  2. **대용량은 제외하고 재생성 절차를 문서화한다** (DINOv3 캐시 18GB, UNet 체크포인트 9GB).
     per-image 점수(~17MB)는 포함한다 — 부록 B1 의 시드별 원값을 검증하려면 필요하다.
  3. **없는 파일은 조용히 건너뛰지 않는다.** 결측을 MANIFEST 에 명시한다.
  4. 파일마다 **원본 경로와 SHA-256** 을 남겨 출처를 추적할 수 있게 한다.

사용: python3 scripts/build_submission_package.py [--out submission]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

R = Path("/workspace/ai-vision-research")

# ---------------------------------------------------------------- 코드
CODE = {
    "branches/reconstruction_ead": [
        "scripts/phase0/train_ead_seeds.sh",
        "scripts/phase0/run_eads_seed.py",
        "scripts/phase0/ead_infer_seeds.py",
        "scripts/phase0/train_ead_m_seeds_43_44.sh",
        "scripts/phase0/run_ead_m_loco.py",
    ],
    "branches/patch_memory_pc": [
        "scripts/psad_rebuild/patchcore_family_trainonly.py",
    ],
    "branches/composition_psad": [
        "scripts/psad_rebuild/build_inputs.py",
        "scripts/psad_rebuild/build_patchcore_pt.py",
        "scripts/psad_rebuild/train_unet_seed.py",
        "scripts/psad_rebuild/train_unet_seeds.sh",
        "scripts/psad_rebuild/score_psad.py",
        "scripts/psad_rebuild/csad_regen.py",
    ],
    "fusion": [
        "scripts/psad_rebuild/prereg_testfree.py",
        "scripts/psad_rebuild/fuse_testfree_v2.py",
        "scripts/countgd/val_znorm_fusion.py",
    ],
    "selection_criteria": [
        "scripts/psad_rebuild/criterion_v2.py",
        "scripts/psad_rebuild/criterion_exact.py",
        "scripts/psad_rebuild/criterion_appearance.py",
        "scripts/psad_rebuild/eval_criterion_v2.py",
        "scripts/psad_rebuild/label_criterion.py",
        "scripts/psad_rebuild/scoring_stability.py",
        "scripts/psad_rebuild/sweep_exact.py",
        "scripts/psad_rebuild/sweep_hdist.py",
        "scripts/psad_rebuild/sweep_hsplit.py",
        "scripts/psad_rebuild/rep_search_v2.py",
        "scripts/psad_rebuild/soft_areas.py",
        "scripts/psad_rebuild/eval_splicing_split.py",
    ],
    "baselines": [
        "scripts/psad_rebuild/realign_3seed.py",
        "scripts/psad_rebuild/realign_eads_pc.py",
        "scripts/psad_rebuild/realign_puad.py",
        "scripts/psad_rebuild/realign_eadm_fullimagenet.py",
        "scripts/path_Y/PY_ls_canonical_unify.py",
        "scripts/path_Y/PY_ls_patchcore_family_5seed.py",
        "scripts/path_Y/PY_salad_multiseed_v2_FULL.py",
        "scripts/path_Y/PY_real_puad_M_MULTISEED.py",
        "scripts/psad_rebuild/puad_s_3seed_rescore.py",
        "scripts/psad_rebuild/puad_s_provenance_audit.py",
        "scripts/psad_rebuild/emit_puad_s_realign.py",
        "scripts/psad_rebuild/measure_param_counts.py",
        "scripts/psad_rebuild/branch_axis_decomposition.py",
        "scripts/psad_rebuild/homogeneous_ensemble_control.py",
        "scripts/psad_rebuild/tableC1_full_grid.py",
        "scripts/psad_rebuild/tableC2_true_cardinality.py",
        "scripts/psad_rebuild/pc_backbone_swap.py",
        "scripts/psad_rebuild/composition_provenance_audit.py",
        "scripts/psad_rebuild/layer_choice_testfree_check.py",
        "scripts/psad_rebuild/coreset_greedy_vs_random.py",
        "scripts/path_Y/PY_ead_m_standalone.py",
        "scripts/path_X/salad_multiseed_train.sh",
    ],
    # 패키지 코드가 import 하는 우리 모듈. 빠지면 해당 스크립트가 import 단계에서 죽는다.
    "shared_modules": [
        "scripts/phase0/eval_loco_unified.py",      # L+S 지표 정의 자체
        "scripts/psad_rebuild/psad_scoring_search.py",
    ],
    "analysis": [
        "scripts/psad_rebuild/reproduce_from_package.py",
        "scripts/build_submission_package.py",
        "scripts/audit_submission_package.py",
        "scripts/psad_rebuild/testfree_supplements.py",
        "scripts/psad_rebuild/diagnose_rebuild_gap.py",
        # 260901 커버리지 감사로 추가 — 원고가 인용하는 값을 만든 스크립트들.
        "scripts/countgd/deployment_operating_point.py",      # 표 8 배포 구성 비교
        "scripts/countgd/requirement_cost_curve.py",          # 표 9 요건 곡선
        "scripts/countgd/subtype_branch_auroc.py",            # 부록 D 하위유형 x 네 점수원
        "scripts/psad_rebuild/comad_axis_perseed.py",         # 표 2 ComAD 축별 std
        "scripts/psad_rebuild/measure_param_inference.py",    # §3.1 파라미터 3분할
        "scripts/psad_rebuild/profile_forward_modules.py",    # 위의 forward hook 실측
        "scripts/psad_rebuild/audit_package_coverage.py",     # 이 패키지의 커버리지 감사
    ],
    "audit": [
        "scripts/phase0/audit_ead_repro.py",
        "scripts/phase0/audit_splicing_outlier.py",
        "scripts/phase0/ead_5seed_verdict.py",
        "scripts/psad_rebuild/audit_seed_stage.py",
        "scripts/psad_rebuild/audit_rebuild.py",
        "scripts/psad_rebuild/audit_inductive.py",
    ],
}

# ---------------------------------------------------------------- 결과
RESULTS = {
    "main": [
        "reports/countgd/testfree_final_eadfix_3seed.json",
        "reports/countgd/testfree_final_eadfix_origproto_3seed.json",
        "reports/countgd/testfree_final_hcp_3seed.json",
    ],
    "tables": [
        "reports/countgd/testfree_supplements_3seed.json",
        "reports/countgd/table5_3seed_42_43_44.json",
        "reports/countgd/patchcore_family_trainonly_3seed.json",
        "reports/countgd/realign_eadm_fullimagenet_3seed.json",
        "reports/countgd/realign_salad_3seed.json",
        "reports/countgd/realign_puad_3seed.json",
        "reports/countgd/puad_s_rescore_3seed.json",
        "reports/countgd/realign_puad_s_3seed.json",
        "reports/countgd/param_counts_measured.json",
        "reports/countgd/branch_axis_decomposition.json",
        "reports/countgd/homogeneous_ensemble_control.json",
        "reports/countgd/tableC1_full_grid.json",
        "reports/countgd/tableC2_true_cardinality.json",
        "reports/countgd/pc_backbone_swap.json",
        "reports/countgd/composition_provenance_audit.json",
        "reports/countgd/layer_choice_testfree_check.json",
        # 260901 커버리지 감사로 발견 — 원고가 인용하는데 패키지에 없던 것들.
        # §4.5 배포 구성 비교(표 8)·요건 곡선(표 9)은 260828~30 신설이라 이전 빌드에 없었다.
        "reports/countgd/deployment_operating_point.json",
        "reports/countgd/requirement_cost_curve.json",
        # 부록 D 하위유형 x 네 점수원(요청 12). §4.3·§5.2 가 이 값을 인용한다.
        "reports/countgd/subtype_branch_auroc.json",
        # 표 2 ComAD 축별 표준편차(요청 11). README 가 "산출 불가"로 적고 있던 것.
        "reports/countgd/comad_axis_perseed.json",
        # 요청 10 추론 파라미터 3분할 + forward hook 실측
        "reports/countgd/param_counts_inference.json",
        "reports/countgd/forward_profile.json",
        # 위 README §1 "왜 점수를 배포하는가" 의 근거 (3-f 복제 실행)
        "reports/countgd/psad_variance_decomp.json",
        "reports/countgd/psad_3way_compare.json",
        "reports/countgd/coreset_greedy_vs_random.json",
        "reports/countgd/realign_eads_pc_3seed.json",
        "reports/countgd/realign_3seed_42_43_44.json",
        "reports/path_y/metric_unify/ls_canonical_scores.json",
        "reports/path_y/metric_unify/ls_patchcore_family_5seed.json",
        "reports/countgd/dsq_color_composition_full_table.json",
        "reports/countgd/d5_visa_principle.json",
        "reports/path_y/fps_comparison_260621.json",
        "reports/countgd/d2d3_multidraw_tables67.json",
        "reports/salad_free/p18_heldout_fullres.json",
        "reports/salad_free/p21_pixel_auroc.json",
        "reports/salad_free/p4_hybrid_baseline.json",
    ],
    "figures": [
        "reports/countgd/testfree_supplements2_3seed.json",
        "reports/countgd/testfree_pairwise_percat_3seed.json",
    ],
    "diagnostics": [
        "reports/countgd/PREREG_branch_improvement.md",
        "reports/countgd/criterion_appearance.json",
        "reports/countgd/psad_scoring_stability.json",
        "reports/countgd/criterion_v2_csad_pseudo_seg.json",
        "reports/countgd/rep_search_v2_csad_pseudo_seg.json",
        "reports/countgd/sweep_exact_merge1.json",
        "reports/countgd/sweep_hdist.json",
        "reports/countgd/sweep_hsplit.json",
        "reports/countgd/val_znorm_fusion.json",
        "_logs/ead_seed42_crosscheck.json",
    ],
    "reproduction": [
        "reports/phase0/repro_audit/ead_5seed_verdict.json",
        "reports/phase0/repro_audit/splicing_outlier_audit.json",
        # PUAD-S 재채점의 근거: 구 시드의 체크포인트 출처, 코드 동치 증명, 환경 드리프트 크기
        "reports/countgd/puad_s_provenance_audit.json",
    ],
}

# per-image 점수 디렉터리 (부록 B1 시드별 원값의 근거)
SCORE_DIRS = [
    ("reports/phase0/efficient_ad_official_small/npz", "per_image_scores/ead_s", "*.npz"),
    ("reports/phase0/ead_repro_npz", "per_image_scores/ead_s_seed42_repro", "*.npz"),
    # z-정규화의 기준값. 이게 없으면 융합을 재현할 수 없다.
    ("reports/phase0/efficient_ad_official_small_seeds_std_val",
     "per_image_scores/ead_s_val", "val_good_*_seed4[234].npz"),
    # PC 분기는 18GB 캐시에서 즉석 계산되므로 저장 파일이 없었다. 내보낸 것을 담는다.
    ("reports/countgd/pc_branch_scores", "per_image_scores/pc_branch", "*.npz"),
    ("reports/phase0/efficient_ad_medium", "per_image_scores/ead_m_imagenette", "scores_*.npz"),
    ("reports/path_y/salad_multiseed_v2", "per_image_scores/salad", "*.npz"),
    ("reports/path_y/puad_m_multiseed_scores", "per_image_scores/puad_m", "*.npz"),
    # 표 4 의 PUAD-S 행은 시드 44 단독이었다. {42,43,44} 재채점본을 담는다.
    ("reports/path_y/puad_s_multiseed_scores", "per_image_scores/puad_s", "*.npz"),
    # 인용되는 두 계열만. 기각된 탐색 변형(k5_n1, n1 등 14계열)은 담지 않는다 —
    # 기각 근거는 PREREG 와 sweep_*.json 에 있고, 점수 자체는 score_psad.py 로 재생성된다.
    ("external/PSAD_official/LOCO_MVTec_AD/rebuild_scores",
     "per_image_scores/psad_composition",
     ["*_hc_tta_merge_test.npz", "*_hc_tta_merge_val.npz",
      "*_hcp_tta_merge_k5_test.npz", "*_hcp_tta_merge_k5_val.npz"]),
    ("reports/countgd/val_scores", "per_image_scores/original_pipeline_reference", "*.npz"),
    # ComAD 시드별 per-image 점수. 이전 README 는 "이 머신에 없어 산출 불가"라 적었으나
    # 5090 에는 있었다(요청 11). 표 2 의 ComAD 축별 ± 가 여기서 나온다.
    ("reports/countgd/comad_per_image", "per_image_scores/comad", "*.npz"),
]

DOCS = [
    "docs/260820_paper_corrections.md",
    "docs/260821_puad_s_3seed.md",
    "docs/260821_param_counts.md",
    "docs/260821_ad2_no_logical_labels.md",
    "docs/260821_axis_decomposition_results.md",
    "docs/260822_branch_swap_results.md",
    "docs/260822_combination_history.md",
    "docs/260822_composition_provenance_results.md",
    "docs/260822_l17_coreset_provenance_results.md",
    "docs/260817_testfree_hc_canonical.md",
    "docs/260817_baseline_seed_audit.md",
    "docs/260817_testfree_pending_requests.md",
    "docs/260716_v25_numbers_canonical.md",
]


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission")
    a = ap.parse_args()
    OUT = R / a.out
    if OUT.exists():
        shutil.rmtree(OUT)

    rows, missing = [], []

    def put(src_rel, dst_rel):
        s = R / src_rel
        if not s.exists():
            missing.append(src_rel); return False
        d = OUT / dst_rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        rows.append((dst_rel, src_rel, s.stat().st_size, sha(s)))
        return True

    for grp, files in CODE.items():
        for f in files:
            put(f, f"code/{grp}/{Path(f).name}")
    for grp, files in RESULTS.items():
        for f in files:
            put(f, f"results/{grp}/{Path(f).name}")
    for f in DOCS:
        put(f, f"docs/{Path(f).name}")
    # README 는 손으로 쓰지 않는다 — 저장소 소스에서 복사한다. rmtree 후 재빌드에도 남는다.
    # 260904 요청 15: 배치를 빌드에서 확정한다. 이전에는 빌드 뒤 손으로 rename 해서
    # MANIFEST 의 README.md 해시가 한국어판 것이었고 README.en.md 는 유령 행이 됐다.
    put("submission_extra/README_en.md", "README.md")        # 공개 저장소 기본
    put("docs/submission_README.md", "README.ko.md")         # 한국어 원본 보존
    put("submission_extra/LICENSE", "LICENSE")
    put("submission_extra/.gitattributes", ".gitattributes")  # CRLF 로 해시가 깨지는 것을 막는다
    put("submission_extra/gitignore", ".gitignore")

    n_scores = 0
    for src, dst, pat in SCORE_DIRS:
        sd = R / src
        pats = [pat] if isinstance(pat, str) else pat
        if not sd.exists():
            missing.append(src + f" ({pats})"); continue
        for pt in pats:
            hit = sorted(sd.glob(pt))
            if not hit:
                missing.append(f"{src} ({pt}) — 일치 파일 없음"); continue
            for p in hit:
                put(p.relative_to(R).as_posix(), f"results/{dst}/{p.name}")
                n_scores += 1

    write_scores_readme(OUT)
    write_retarget(OUT)
    # 260904 요청 15: 빌드가 생성하는 파일은 put() 을 안 거쳐 MANIFEST 에서 빠져 있었다.
    # 원본이 저장소에 없으므로 출처를 "(생성)" 으로 적고 해시는 쓴 결과에서 계산한다.
    for rel in ("code/RETARGET.md", "results/per_image_scores/README.md"):
        f = OUT / rel
        if f.exists():
            rows.append((rel, "(빌드가 생성)", f.stat().st_size, sha(f)))
    total = sum(r[2] for r in rows)
    (OUT / "MANIFEST.md").write_text(manifest(rows, missing, total, n_scores))
    print(f"파일 {len(rows)}개 · {total/1024/1024:.1f}MB · per-image 점수 {n_scores}개")
    if missing:
        print(f"결측 {len(missing)}건 (MANIFEST 에 기재):")
        for m in missing[:12]:
            print(f"  - {m}")
    print(f"→ {OUT}")


# per-image 점수 디렉터리 → 무엇을 뒷받침하나. 시드 목록은 실제 파일에서 읽는다.
SCORE_ROLE = {
    "ead_s": "EAD-S 재구성 분기 test 점수. 표 4 의 EAD-S 행과 융합의 첫째 항.",
    "ead_s_seed42_repro": "seed42 재학습본 test 점수. 정본 seed42 는 원본 모델 산출물이라 "
                          "val(재학습본)과 출처가 섞였다. 융합은 이 파일을 쓴다.",
    "ead_s_val": "EAD-S 검증분할 정상 표본 점수. z-정규화의 기준값 — 없으면 융합이 재현되지 않는다.",
    "pc_branch": "DINOv3-L L17 PatchCore 분기 (coreset 50000 / k=1 / max / train-only 뱅크). "
                 "표 4 의 DINOv3-L 행과 융합의 둘째 항이 같은 값이다.",
    "psad_composition": "PSAD 구성분기. hc_tta_merge = 주 설정, hcp_tta_merge_k5 = 부록 비교.",
    "ead_m_imagenette": "EAD-M 의 imagenette 폴백 학습본. 표 4 에서 교체된 수치의 원본 증거.",
    "salad": "SALAD 재현 (자체 학습 3시드).",
    "puad_m": "PUAD-M 재현 (시드별 EAD-M + 공식 Mahalanobis). npz 에 유형 배열이 없다 — "
              "test 이미지 순서가 `ead_s` 와 동일함을 이진 라벨 배열 일치로 확인했으므로 "
              "`label_type` 을 거기서 빌려 쓴다(재현 스크립트가 그렇게 한다).",
    "puad_s": "PUAD-S 재현 (EAD-S 체크포인트 + 공식 Mahalanobis, 학습 없이 재채점). "
              "구 산출물은 시드마다 체크포인트 루트가 달라 셋 다 `_seeds_std` 에서 다시 매겼다.",
    "original_pipeline_reference": "원본 PSAD/PatchCore 파이프라인 참조 점수 (재구축 대조군).",
}


def write_scores_readme(OUT):
    import re
    D = OUT / "results/per_image_scores"
    L = ["# per-image 점수 — 무엇을 뒷받침하나", "",
         "각 npz 는 이미지 단위 이상 점수다. 이 파일들만으로 헤드라인이 재계산된다 "
         "(`code/analysis/reproduce_from_package.py`). 캐시·체크포인트는 필요 없다.", "",
         "| 디렉터리 | 파일 | 시드 | 용도 |", "|---|--:|---|---|"]
    for d in sorted(D.iterdir()):
        if not d.is_dir():
            continue
        names = [x.name for x in d.iterdir()]
        seeds = sorted({m.group(1) for n in names if (m := re.search(r"seed(\d+)", n))}, key=int)
        L.append(f"| `{d.name}` | {len(names)} | {', '.join(seeds) or '—'} | "
                 f"{SCORE_ROLE.get(d.name, '')} |")
    L += ["", "## 시드에 관해", "",
          "주 표는 **{42, 43, 44}** 다. `ead_s` 의 45·46 은 부록의 EAD-S 5시드 재현"
          "(`results/reproduction/ead_5seed_verdict.json`), 0·1234 는 원본 PSAD 프로토콜"
          "(0/42/1234)과 맞춘 대조 실행에 쓰인다.", "",
          "## 담지 않은 것", "",
          "PSAD 구성분기의 기각된 탐색 변형(k=5·개수거리·소프트마스크 등 14계열)은 "
          "제외했다. 기각 근거는 `results/diagnostics/PREREG_branch_improvement.md` 와 "
          "`sweep_*.json` 에 있고, 점수 자체는 `code/branches/composition_psad/score_psad.py` "
          "의 해당 플래그로 재생성된다.", ""]
    (D / "README.md").write_text("\n".join(L))


def write_retarget(OUT):
    C = OUT / "code"
    files = [f for f in C.rglob("*") if f.is_file() and f.suffix in (".py", ".sh")]
    hard = [f for f in files if "/workspace/ai-vision-research" in f.read_text(errors="replace")]
    (OUT / "code/RETARGET.md").write_text(f"""# 저장소 경로 재지정

여기 담긴 스크립트는 **실제로 실행된 그대로**다. 따라서 저장소 루트가
`/workspace/ai-vision-research` 로 하드코딩돼 있다({len(files)}개 중 {len(hard)}개). 실행 기록을
사후 편집하지 않으려고 그대로 뒀다.

다른 위치에서 돌리려면 한 줄로 바꾼다:

```bash
grep -rl /workspace/ai-vision-research . | xargs sed -i "s|/workspace/ai-vision-research|$PWD|g"
```

예외: `analysis/reproduce_from_package.py` 는 절대경로가 없다. 패키지만으로 헤드라인을
재계산하는 이 스크립트는 어디서든 그대로 돈다.

```bash
python3 code/analysis/reproduce_from_package.py --pkg .
```
""")


def manifest(rows, missing, total, n_scores):
    L = ["# MANIFEST — 파일 출처와 무결성",
         "",
         "> `MANIFEST.md` 자신은 목록에 없다 — 자기 해시를 담을 수 없기 때문이다.",
         "> **해시는 LF 줄바꿈 기준**이다. 저장소에 `.gitattributes`(`* text=auto eol=lf`)가",
         "> 있어 `core.autocrlf=true` 인 Windows clone 에서도 그대로 일치한다.",
         "",
         f"총 {len(rows)}개 파일 · {total/1024/1024:.1f} MB · per-image 점수 {n_scores}개",
         "",
         "각 행은 `패키지 경로 | 원본 저장소 경로 | 크기 | SHA-256(앞 16자리)` 이다.",
         "원본 저장소에서 같은 경로의 파일과 해시를 대조하면 출처를 확인할 수 있다.",
         "", "| 패키지 경로 | 원본 경로 | 크기 | SHA-256 |", "|---|---|--:|---|"]
    for dst, src, sz, h in sorted(rows):
        L.append(f"| `{dst}` | `{src}` | {sz:,} | `{h}` |")
    L += ["", "## 빌드가 생성한 파일 (원본 없음, 위 표에 미포함)", "",
          "- `README.md` — `docs/submission_README.md` 에서 복사",
          "- `results/per_image_scores/README.md` — 실제 파일 구성에서 생성",
          "- `code/RETARGET.md` — 경로 재지정 안내",
          "- `results/main/reproduced_from_package.json` — 재현 스크립트를 돌리면 생긴다"]
    if missing:
        L += ["", "## 결측 (원본 저장소에 없음)", ""]
        L += [f"- `{m}`" for m in missing]
        L += ["", "결측 사유는 README 의 '재현 불가 항목' 절 참조."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
