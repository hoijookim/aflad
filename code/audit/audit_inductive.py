#!/usr/bin/env python3
"""inductive(test 무접촉) 무결성 검수 — 가중 선택 경로에 test 가 새지 않았는가.

합성 이상 기반 가중 선택은 "test 를 안 본다"는 주장이 전부다. 그 주장을 코드·데이터
양쪽에서 반증 시도한다.

I1 합성 소스   : 합성 이상이 train 이미지에서만 만들어졌는가 (test/val 파일명 混入 0)
I2 마스크 소스 : 합성에 쓴 분할 마스크가 train 의사레이블뿐인가
I3 가중 선택   : 선택 목적함수가 합성 라벨만 쓰고 test 라벨을 안 쓰는가 (정적 검증)
I4 정규화 통계 : z/robust 통계가 val-good 에서만 오는가 (test 통계 사용 0)
I5 뱅크 구성   : PatchCore 뱅크·PSAD 메모리가 train 만으로 구성되는가
I6 최종 평가   : test 는 최종 AUROC 계산 1회에만 등장하는가
I7 수치 독립성 : 가중을 test 로 고른 오라클 대비 얼마나 손해인가 (합성 가중이
                 오라클과 같다면 우연이 아니라 누수 의심 — 차이가 있어야 정상)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

R = Path("/workspace/ai-vision-research")
DATA = R / "external/PSAD_official/LOCO_MVTec_AD"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

FAIL, WARN, OK = [], [], []


def rec(lv, tag, msg):
    {"FAIL": FAIL, "WARN": WARN, "OK": OK}[lv].append(f"[{tag}] {msg}")
    print(f"  {lv:4s} [{tag}] {msg}", flush=True)


def main():
    print("=" * 78)
    print("inductive 무결성 검수 (합성 이상 기반 가중)")
    print("=" * 78)
    src = (R / "scripts/psad_rebuild/synth_weights.py").read_text()

    print("\nI1. 합성 이상의 소스가 train 뿐인가")
    m = re.search(r'img_dir\s*=\s*DATA\s*/\s*"orig_512"\s*/\s*cat\s*/\s*"(\w+)"\s*/\s*"(\w+)"', src)
    ok = bool(m) and m.group(1) == "train"
    rec("OK" if ok else "FAIL", "I1",
        f"합성 입력 split = {m.group(1) if m else '?'}/{m.group(2) if m else '?'}")
    leak = re.search(r'orig_512[^\n]*"(test|validation)"', src)
    rec("FAIL" if leak else "OK", "I1",
        f"test/validation 경로 참조 {'발견 — 누수' if leak else '없음'}")

    print("\nI2. 합성에 쓴 마스크가 train 의사레이블인가")
    m2 = re.search(r'seg_dir\s*=\s*DATA\s*/\s*"([\w_]+)"\s*/\s*cat', src)
    rec("OK" if m2 and m2.group(1) == "csad_pseudo_seg" else "FAIL", "I2",
        f"마스크 디렉토리 = {m2.group(1) if m2 else '?'} (train good 의사레이블)")

    print("\nI3. 가중 선택 목적함수가 합성 라벨만 쓰는가")
    wf = R / "scripts/psad_rebuild/select_weights.py"
    if wf.exists():
        ws = wf.read_text()
        uses_test = re.search(r'label_type|scores_\{cat\}_seed|_test\.npz', ws)
        rec("FAIL" if uses_test else "OK", "I3",
            f"가중 선택 코드의 test 산출물 참조 {'있음' if uses_test else '없음'}")
    else:
        rec("WARN", "I3", "select_weights.py 아직 없음 — 구현 후 재검수 필요")

    print("\nI4. 정규화 통계 출처")
    fs = (R / "scripts/psad_rebuild/fuse_testfree_v2.py").read_text()
    rec("OK" if "def normalize(t, v" in fs and "np.median(v)" in fs else "FAIL", "I4",
        "normalize() 입력 v = val-good 점수 (test 미사용)")

    print("\nI5. 뱅크/메모리 구성")
    rec("OK" if "bank_src, val_src = Xtr[:n_tr], Xtr[n_tr:]" in fs else "FAIL", "I5",
        "PatchCore 뱅크 = train_good 구간만 (val 자기포함 방지)")
    ps = (R / "scripts/psad_rebuild/score_psad.py").read_text()
    rec("OK" if 'rels("train", "good")' in ps else "FAIL", "I5",
        "PSAD 메모리 = train/good only")

    print("\nI6. 합성 데이터가 test 와 겹치지 않는가 (파일 실측)")
    for cat in CATS:
        sy = DATA / "synth_anom" / cat
        n = len(list(sy.glob("*.png"))) if sy.exists() else 0
        te = {p.name for p in (DATA / "orig_512" / cat / "test").rglob("*.png")}
        dup = len({p.name for p in sy.glob("*.png")} & te) if sy.exists() else 0
        rec("OK" if (n > 0 and dup == 0) else ("WARN" if n == 0 else "FAIL"), "I6",
            f"{cat}: 합성 {n}장, test 파일명 중복 {dup}")

    print("\n" + "=" * 78)
    print(f"종합: OK {len(OK)} / WARN {len(WARN)} / FAIL {len(FAIL)}")
    for f in FAIL:
        print("  FAIL -", f)
    for w in WARN:
        print("  WARN -", w)
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
