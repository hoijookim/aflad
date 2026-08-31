#!/usr/bin/env python3
"""EAD-S 재학습 적대적 검수 — "제대로 실험된 게 맞는가"를 반증하려 시도한다.

각 검사는 '통과하면 안심'이 아니라 '실패하면 실험 무효'인 반증 시도로 설계했다.

A. 시드가 실제로 먹었는가        — seed42 vs seed43 점수가 동일하면 시드 무효
B. 70k 스텝을 실제로 돌았는가    — 로그의 스텝 수·소요시간
C. ImageNet 페널티가 걸렸는가    — 경로 유효성 + 학습 로그
D. 정렬 규약이 정본과 같은가      — label/label_type 배열이 정본과 완전 일치
E. 편차가 시드 잡음 범위인가      — per-cat Δ 를 원본 5-seed std 로 정규화 (σ 단위)
F. 편차가 계통적인가 무작위인가   — 부호 일관성·순위상관 (계통적이면 환경 차이 의심)
G. test 누수는 없는가            — 정규화가 validation 만 쓰는지 코드 확인
H. 설정이 동일한가               — image_size/model_size/teacher 가중치 해시

사용: python3 scripts/phase0/audit_ead_repro.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
NPZ = R / "reports/phase0/efficient_ad_official_small/npz"
CKPT = R / "reports/phase0/efficient_ad_official_small_seeds_std"
LOCO = R / "datasets/MVTecLOCO"
LOG = R / "_logs/ead_train_seeds.log"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
DEFECTS = ["good", "logical_anomalies", "structural_anomalies"]

FAIL, WARN, OK = [], [], []


def rec(level, tag, msg):
    {"FAIL": FAIL, "WARN": WARN, "OK": OK}[level].append(f"[{tag}] {msg}")
    print(f"  {level:4s} [{tag}] {msg}", flush=True)


def ls_auc(s, lab, lt):
    return 0.5 * sum(roc_auc_score(lab[(lt == "good") | (lt == k)], s[(lt == "good") | (lt == k)])
                     for k in ["logical", "structural"])


def load(seed, cat):
    p = NPZ / f"scores_{cat}_seed{seed}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def main():
    print("=" * 78)
    print("EAD-S 재학습 적대적 검수")
    print("=" * 78)

    # ---------- A. 시드 유효성 ----------
    print("\nA. 시드가 실제로 반영되었는가 (반증: seed 간 점수 동일)")
    import subprocess
    infer = R / "scripts/phase0/ead_infer_seeds.py"
    seeds_avail = sorted({int(m.group(1)) for d in CKPT.glob("*_seed*")
                          if (m := re.search(r"_seed(\d+)$", d.name))
                          and (d / "trainings/mvtec_loco" / d.name.rsplit("_seed", 1)[0]
                               / "teacher_final.pth").exists()})
    print(f"       학습 완료 시드: {seeds_avail}")
    # 체크포인트 바이트 해시로 시드 차이 확인 (추론 없이 저렴)
    for cat in CATS[:2]:
        hs = {}
        for sd in seeds_avail[:3]:
            f = CKPT / f"{cat}_seed{sd}/trainings/mvtec_loco/{cat}/student_final.pth"
            if f.exists():
                hs[sd] = hashlib.md5(f.read_bytes()).hexdigest()[:12]
        if len(set(hs.values())) == len(hs) and len(hs) > 1:
            rec("OK", "A", f"{cat}: 시드별 student 가중치 전부 상이 {hs}")
        elif len(hs) > 1:
            rec("FAIL", "A", f"{cat}: 시드가 달라도 가중치 동일 → 시드 미적용 {hs}")

    # ---------- B. 70k 스텝 ----------
    print("\nB. 70,000 스텝을 실제로 돌았는가")
    full = LOG.read_text(errors="ignore") if LOG.exists() else ""
    # 로그는 append 되므로 마지막 '학습 시작' 마커 이후만 본다.
    # (앞선 실패 시도의 '완료 0분' 기록이 섞이면 오탐이 난다 — 실제로 발생했던 오탐)
    marks = [m.start() for m in re.finditer(r"=== EAD-S 학습 시작", full)]
    txt = full[marks[-1]:] if marks else full
    if len(marks) > 1:
        rec("OK", "B", f"로그에 학습 시도 {len(marks)}회 기록 — 마지막 시도만 채점")
    steps = re.findall(r"(\d+)/70000", txt)
    if steps:
        mx = max(int(s) for s in steps)
        rec("OK" if mx >= 69000 else "FAIL", "B",
            f"로그상 최대 도달 스텝 {mx}/70000")
    times = [int(m) for m in re.findall(r"완료 (\d+)분", txt)]
    if times:
        short = [t for t in times if t < 10]
        rec("OK" if not short else "FAIL", "B",
            f"범주별 소요 {min(times)}~{max(times)}분 (중앙 {int(np.median(times))}분), "
            f"10분 미만 {len(short)}건")

    # ---------- C. ImageNet ----------
    print("\nC. ImageNet 페널티 경로가 유효한가")
    inp = R / "datasets/imagenet1k/train"
    if inp.exists():
        ncls = sum(1 for _ in inp.iterdir())
        rec("OK" if ncls >= 900 else "FAIL", "C", f"클래스 디렉토리 {ncls}개")
    else:
        rec("FAIL", "C", "ImageNet 경로 없음 → 페널티 항 미적용 가능")

    # ---------- D. 정렬 규약 ----------
    print("\nD. 정렬 규약이 정본과 동일한가 (label/label_type 배열 일치)")
    for cat in CATS:
        ref = load(42, cat)
        if ref is None:
            continue
        counts = [len(sorted((LOCO / cat / "test" / d).glob("*.png"))) for d in DEFECTS]
        exp_lab = np.array([0] * counts[0] + [1] * (counts[1] + counts[2]))
        ok_lab = np.array_equal(ref["label"].astype(int), exp_lab)
        exp_lt = np.array(["good"] * counts[0] + ["logical"] * counts[1] + ["structural"] * counts[2])
        ok_lt = np.array_equal(ref["label_type"].astype(str), exp_lt)
        rec("OK" if (ok_lab and ok_lt) else "FAIL", "D",
            f"{cat}: n={sum(counts)} {counts} label={'일치' if ok_lab else '불일치'} "
            f"label_type={'일치' if ok_lt else '불일치'}")

    # ---------- E/F. 편차 분석 ----------
    print("\nE. 재현 편차가 원본 시드 산포 범위 내인가 (σ 단위)")
    pcf = R / "reports/countgd/percat_table_fill.json"
    std_ref = {}
    if pcf.exists():
        d = json.load(open(pcf))["per_cat"]
        std_ref = {c: d[c]["EAD"]["std"] for c in CATS}

    repro = json.load(open(R / "_logs/ead_seed42_crosscheck.json")) \
        if (R / "_logs/ead_seed42_crosscheck.json").exists() else None
    if repro is None:
        rec("WARN", "E", "크로스체크 결과 파일 없음 — --compare 를 먼저 저장하도록 실행 필요")
    else:
        sig = {}
        for cat in CATS:
            if cat not in repro:
                continue
            delta = abs(repro[cat]["repro"] - repro[cat]["canonical"])
            s = std_ref.get(cat, float("nan"))
            z = delta / s if s and s > 0 else float("inf")
            sig[cat] = z
            lvl = "OK" if z <= 2 else ("WARN" if z <= 3 else "FAIL")
            rec(lvl, "E", f"{cat}: Δ={delta:.4f}, 원본 seed std={s:.4f} → {z:.1f}σ")
        print("\nF. 편차가 계통적인가 (부호 일관성 — 계통적이면 환경 차이)")
        signs = [np.sign(repro[c]["repro"] - repro[c]["canonical"]) for c in CATS if c in repro]
        npos = sum(1 for s in signs if s > 0)
        rec("WARN" if npos in (0, len(signs)) else "OK", "F",
            f"재현>정본 {npos}/{len(signs)}범주 — "
            f"{'한쪽 쏠림(계통적 차이 시사)' if npos in (0,len(signs)) else '혼재(무작위 시드 잡음에 부합)'}")

    # ---------- G. test 누수 ----------
    print("\nG. 정규화에 test 가 쓰이지 않는가 (코드 대조)")
    src = (R / "scripts/phase0/ead_infer_seeds.py").read_text()
    uses_val = "validation" in src and "map_normalization(val_loader" in src
    uses_test_norm = re.search(r"map_normalization\([^)]*test", src) is not None
    rec("OK" if (uses_val and not uses_test_norm) else "FAIL", "G",
        f"map_normalization 입력 = {'validation' if uses_val else '?'}, "
        f"test 사용 {'있음' if uses_test_norm else '없음'}")

    # ---------- H. 설정 ----------
    print("\nH. 학습 설정이 원본과 동일한가")
    ead = (R / "external/efficient_ad_official/efficientad.py").read_text()
    m_img = re.search(r"image_size\s*=\s*(\d+)", ead)
    rec("OK" if m_img and m_img.group(1) == "256" else "FAIL", "H",
        f"image_size={m_img.group(1) if m_img else '?'} (논문 표1: 256)")
    tw = R / "external/efficient_ad_official/models/teacher_small.pth"
    if tw.exists():
        rec("OK", "H", f"teacher_small.pth md5={hashlib.md5(tw.read_bytes()).hexdigest()[:12]} "
                       f"({tw.stat().st_size} B)")
    sh = (R / "scripts/phase0/train_ead_seeds.sh").read_text()
    for key, exp in [("train_steps", "70000"), ("model_size", "small")]:
        got = re.search(rf"--{key} (\S+)", sh)
        rec("OK" if got and got.group(1) == exp else "FAIL", "H",
            f"--{key}={got.group(1) if got else '?'} (기대 {exp})")

    # ---------- 종합 ----------
    print("\n" + "=" * 78)
    print(f"종합: OK {len(OK)} / WARN {len(WARN)} / FAIL {len(FAIL)}")
    if FAIL:
        print("\n실패 항목 (실험 무효 가능):")
        for f in FAIL:
            print("  -", f)
    if WARN:
        print("\n경고 항목 (해석 시 주의):")
        for w in WARN:
            print("  -", w)
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
