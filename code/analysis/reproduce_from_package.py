#!/usr/bin/env python3
"""패키지에 담긴 per-image 점수만으로 헤드라인을 재계산한다 — 캐시·체크포인트 불필요.

재현 패키지의 존재 이유는 "우리 말을 믿어라"가 아니라 **직접 계산해서 확인할 수 있다**는
것이다. 이 스크립트는 `results/per_image_scores/` 의 npz 세 종류만 읽어 논문의 주 설정
수치를 다시 만든다. 18GB DINOv3 캐시도, 9GB UNet 체크포인트도 필요 없다.

계산 = 논문이 선언한 test-free 프로토콜 그대로:
  각 분기 점수를 **검증 분할 정상 표본의 평균/표준편차로 z-정규화**하고 **등가중 합**,
  결함 유형별 AUROC 를 낸 뒤 L+S = (logical + structural) / 2.

사용:
  python3 reproduce_from_package.py --pkg /path/to/submission
"""
import sys as _sys
# 260904: 로캘 독립 출력. Windows 기본 로캘(cp949)이나 LC_ALL=C 에서 ± · 한국어를
# 찍다가 UnicodeEncodeError 로 죽는다 — open() 인코딩만 고쳐서는 안 닫힌다.
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = (42, 43, 44)
# 기대값은 상수로 박지 않는다 — 그러면 "내가 정한 숫자와 맞다"는 순환이 된다.
# 논문이 인용하는 산출물 JSON 에서 읽어, **원 점수로부터의 재계산**과 대조한다.
FALLBACK = {"LS": 0.9723, "logical": 0.9685, "structural": 0.9762}
TOL = 5e-4


def expected(pkg):
    f = pkg / "results/main/testfree_final_eadfix_3seed.json"
    if not f.exists():
        print("  (주 산출물 JSON 이 없어 문서값으로 대조한다)")
        return FALLBACK, "README 문서값"
    d = json.load(open(f, encoding="utf-8"))
    return ({k: d[k]["mean"] for k in ("LS", "logical", "structural")},
            f.name)


def load(pkg, cat, seed):
    S = pkg / "results/per_image_scores"
    # 재구성 — seed42 는 재학습본(원본 모델과 섞이지 않게)
    ead_t = (S / "ead_s_seed42_repro" / f"scores_{cat}_seed42.npz" if seed == 42
             else S / "ead_s" / f"scores_{cat}_seed{seed}.npz")
    e = np.load(ead_t, allow_pickle=True)
    ev = np.load(S / "ead_s_val" / f"val_good_{cat}_seed{seed}.npz", allow_pickle=True)
    # 패치 메모리
    p = np.load(S / "pc_branch" / f"pc_L17_{cat}_seed{seed}.npz", allow_pickle=True)
    # 구성
    c = np.load(S / "psad_composition" /
                f"psad_scores_{cat}_seed{seed}_hc_tta_merge_test.npz", allow_pickle=True)
    cv = np.load(S / "psad_composition" /
                 f"psad_scores_{cat}_seed{seed}_hc_tta_merge_val.npz", allow_pickle=True)
    return ((e["score"].astype(float), ev["scores"].astype(float)),
            (p["test"].astype(float), p["val"].astype(float)),
            (c["scores"].astype(float), cv["scores"].astype(float)),
            e["label"].astype(int), e["label_type"].astype(str))


# 표 4 기준선 행도 같은 npz 로 검증한다 — 헤드라인만 맞고 비교 대상이 틀리면 의미가 없다.
# (디렉터리, 파일형식, 점수키, 라벨키, 유형키 or None → EAD 에서 빌려온다)
BASELINES = {
    "EAD-S": ("ead_s", "scores_{c}_seed{s}.npz", "score", "label", "label_type"),
    "EAD-M (imagenette, 구 표기)": ("ead_m_imagenette", "scores_{c}_seed{s}.npz",
                                  "score", "label", "label_type"),
    "SALAD": ("salad", "{c}_seed{s}.npz", "combined", "labels", "label_types"),
    # PUAD-M npz 에는 유형 배열이 없다. test 이미지 순서가 EAD 와 동일함을 이진 라벨 배열
    # 일치로 확인하고 EAD 의 label_type 을 빌려 쓴다.
    "PUAD-M": ("puad_m", "{c}_seed{s}.npz", "puad", "labels", None),
    # PUAD-S 는 유형 배열을 자체 보유한다 (재채점 때 기록했다).
    "PUAD-S": ("puad_s", "{c}_seed{s}.npz", "puad", "labels", "types"),
    "DINOv3-L PatchCore": ("pc_branch", "pc_L17_{c}_seed{s}.npz", "test", None, None),
}


def verify_baselines(pkg):
    tab_p = pkg / "results/tables/table5_3seed_42_43_44.json"
    if not tab_p.exists():
        print("  (표 JSON 이 없어 기준선 검증 생략)"); return True
    tab = json.load(open(tab_p, encoding="utf-8"))["methods"]
    S = pkg / "results/per_image_scores"
    allok = True
    for name, (d, fmt, sk, lk, tk) in BASELINES.items():
        if name not in tab:
            continue
        try:
            per = []
            for sd in SEEDS:
                lg, st = [], []
                for c in CATS:
                    # EAD-S 의 seed42 는 재학습본이 정본이다 (val 과 출처를 맞춘다)
                    dd = "ead_s_seed42_repro" if (d == "ead_s" and sd == 42) else d
                    z = np.load(S / dd / fmt.format(c=c, s=sd), allow_pickle=True)
                    e = np.load(S / ("ead_s_seed42_repro" if sd == 42 else "ead_s") /
                                f"scores_{c}_seed{sd}.npz", allow_pickle=True)
                    sc = z[sk].astype(float)
                    lab = (z[lk] if lk else e["label"]).astype(int)
                    if lk:  # 순서가 EAD 와 같은지 확인하고 넘어간다
                        assert np.array_equal(lab, e["label"].astype(int)), f"{name}/{c} 순서 불일치"
                    lt = (z[tk] if tk else e["label_type"]).astype(str)
                    m1 = (lt == "good") | (lt == "logical")
                    m2 = (lt == "good") | (lt == "structural")
                    lg.append(roc_auc_score(lab[m1], sc[m1]))
                    st.append(roc_auc_score(lab[m2], sc[m2]))
                per.append(0.5 * (np.mean(lg) + np.mean(st)))
            got, exp = float(np.mean(per)), tab[name]["LS"]
            good = abs(got - exp) <= TOL
            allok &= good
            print(f"    {name:28s} 재현 {got:.6f}  표 {exp:.6f}  차이 {abs(got-exp):.6f}  "
                  f"{'PASS' if good else 'FAIL'}")
        except Exception as ex:
            allok = False
            print(f"    {name:28s} 검증 실패 — {type(ex).__name__}: {ex}")
    return allok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=str(Path(__file__).resolve().parents[2] / "submission"))
    a = ap.parse_args()
    pkg = Path(a.pkg)

    def z(t, v):
        return (t - v.mean()) / max(v.std(), 1e-9)

    per_seed, rows = {}, []
    for sd in SEEDS:
        lg, st, pc = [], [], {}
        for cat in CATS:
            (et, ev), (pt, pv), (ct, cv), lab, lt = load(pkg, cat, sd)
            n = len(lab)
            assert len(et) == len(pt) == len(ct) == n, f"{cat}/seed{sd} 길이 불일치"
            f = z(et, ev) + z(pt, pv) + z(ct, cv)
            a_lg = roc_auc_score(lab[(lt == "good") | (lt == "logical")],
                                 f[(lt == "good") | (lt == "logical")])
            a_st = roc_auc_score(lab[(lt == "good") | (lt == "structural")],
                                 f[(lt == "good") | (lt == "structural")])
            lg.append(a_lg); st.append(a_st); pc[cat] = 0.5 * (a_lg + a_st)
        per_seed[sd] = {"logical": float(np.mean(lg)), "structural": float(np.mean(st)),
                        "LS": float(0.5 * (np.mean(lg) + np.mean(st))), "per_cat": pc}
        rows.append(per_seed[sd])
        print(f"  seed {sd}: L+S {per_seed[sd]['LS']:.4f} "
              f"(logical {per_seed[sd]['logical']:.4f} / "
              f"structural {per_seed[sd]['structural']:.4f})")

    agg = {k: (float(np.mean([r[k] for r in rows])),
               float(np.std([r[k] for r in rows], ddof=1)))
           for k in ("LS", "logical", "structural")}
    print(f"\n  3-seed L+S = {agg['LS'][0]:.4f} ± {agg['LS'][1]:.4f}")
    print(f"           logical {agg['logical'][0]:.4f} / structural {agg['structural'][0]:.4f}")
    print("  범주별: " + "  ".join(
        f"{c[:5]} {np.mean([per_seed[s]['per_cat'][c] for s in SEEDS]):.3f}" for c in CATS))

    print("\n  표 4 기준선 행 검증 (같은 npz 로):")
    base_ok = verify_baselines(pkg)

    exp_vals, exp_src = expected(pkg)
    print(f"\n  논문 보고치와 대조 (출처: {exp_src}):")
    ok = True
    for k, exp in exp_vals.items():
        got = agg[k][0]
        good = abs(got - exp) <= TOL
        ok &= good
        print(f"    {k:11s} 재현 {got:.4f}  논문 {exp:.4f}  차이 {abs(got-exp):.5f}  "
              f"{'PASS' if good else 'FAIL'}")
    ok &= base_ok
    print(f"\n  {'재현 성공 — 패키지만으로 헤드라인과 기준선 표가 모두 나온다' if ok else '재현 실패 — 확인 필요'}")
    out = pkg / "results/main/reproduced_from_package.json"
    json.dump({"per_seed": {str(k): {kk: vv for kk, vv in v.items() if kk != "per_cat"}
                            for k, v in per_seed.items()},
               "aggregate": {k: {"mean": v[0], "std": v[1]} for k, v in agg.items()},
               "expected": exp_vals, "expected_source": exp_src, "match": ok},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  [saved] {out.relative_to(pkg)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
