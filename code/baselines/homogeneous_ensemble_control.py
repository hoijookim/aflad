#!/usr/bin/env python3
"""요청 ② — 동종 앙상블 대조군. 이득이 "세 신호" 인가 "그냥 셋" 인가.

표 2 에는 융합 방법이 셋(주 설정 / best-single / raw-mean)뿐이고 **동종 대조군이 없다.**
"EAD 를 시드 셋으로 앙상블해도 비슷하지 않나" 에 답이 없고, 그 질문은 기여 1 을 직접 겨눈다.

규칙은 주 설정과 동일하다: 각 멤버를 **공식 검증셋 정상 표본**의 평균·표준편차로 z-정규화한 뒤
등가중 합. 로그 스케일·로버스트 통계·클리핑 없음. 지표는 L+S.

표준편차 문제 (요청서가 미리 지적한 것)
  3-멤버 동종은 시드 셋을 다 쓰므로 시드 std 를 낼 멤버가 남지 않는다. 두 가지로 답한다.
  (a) 3-멤버: 동종은 단일 값 + **test 이미지 부트스트랩 95% CI**. CI 는 이종 행에도 같이
      계산하므로 두 행이 같은 기준으로 비교된다.
  (b) **2-멤버 급을 추가한다** — 여기서는 양쪽 모두 조합이 3개씩 나와 **std 가 정직하게 잡힌다**
      (이종: EAD+PC, EAD+Comp, PC+Comp / 동종: 시드쌍 3개). 멤버 수를 맞춘 상태에서
      이종성만 바꾼 비교라 (a) 보다 통제가 깨끗하다.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
NBOOT = 2000
OUT = R / "reports/countgd/homogeneous_ensemble_control.json"


def load(cat, seed):
    """(분기 -> (test, val)) 를 돌려준다. val 은 z-정규화의 기준값이다."""
    ead_dir = (R / "reports/phase0/ead_repro_npz" if seed == 42
               else R / "reports/phase0/efficient_ad_official_small/npz")
    ze = np.load(ead_dir / f"scores_{cat}_seed{seed}.npz", allow_pickle=True)
    zev = np.load(R / "reports/phase0/efficient_ad_official_small_seeds_std_val"
                  / f"val_good_{cat}_seed{seed}.npz", allow_pickle=True)
    zp = np.load(R / f"reports/countgd/pc_branch_scores/pc_L17_{cat}_seed{seed}.npz", allow_pickle=True)
    base = (R / "external/PSAD_official/LOCO_MVTec_AD/rebuild_scores"
            / f"psad_scores_{cat}_seed{seed}_hc_tta_merge")
    zs = np.load(f"{base}_test.npz", allow_pickle=True)
    zsv = np.load(f"{base}_val.npz", allow_pickle=True)
    return ({"EAD": (np.asarray(ze["score"], float), np.asarray(zev[zev.files[0]], float)),
             "PC": (np.asarray(zp["test"], float), np.asarray(zp["val"], float)),
             "Comp": (np.asarray(zs["scores"], float), np.asarray(zsv["scores"], float))},
            np.array([str(x) for x in ze["label_type"]]))


def z(t, v):
    return (t - v.mean()) / (v.std() + 1e-12)


def ls(score, lt):
    g = lt == "good"
    au = lambda m: float(roc_auc_score(np.r_[np.zeros(g.sum()), np.ones(m.sum())],
                                       np.r_[score[g], score[m]]))
    return 0.5 * (au(lt == "logical") + au(lt == "structural"))


def ls_boot(score, lt, rng):
    """test 이미지 부트스트랩. 정상/논리/구조를 각각 층화 재표본한다."""
    gi, li, si = (np.where(lt == t)[0] for t in ("good", "logical", "structural"))
    out = []
    for _ in range(NBOOT):
        g = rng.choice(gi, len(gi), True); l = rng.choice(li, len(li), True); s = rng.choice(si, len(si), True)
        try:
            a = roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(l))], np.r_[score[g], score[l]])
            b = roc_auc_score(np.r_[np.zeros(len(g)), np.ones(len(s))], np.r_[score[g], score[s]])
            out.append(0.5 * (a + b))
        except ValueError:
            pass
    return out


def main():
    D = {c: {s: load(c, s) for s in SEEDS} for c in CATS}
    LT = {c: D[c][SEEDS[0]][1] for c in CATS}
    for c in CATS:                       # 라벨 순서가 시드 간 동일한지
        for s in SEEDS:
            assert list(D[c][s][1]) == list(LT[c]), f"{c}/seed{s} 라벨 순서 불일치"

    def fused(cat, members):
        """members: [(branch, seed), ...] → z 합 점수"""
        out = None
        for b, s in members:
            t, v = D[cat][s][0][b]
            out = z(t, v) if out is None else out + z(t, v)
        return out

    def mean_ls(members_fn):
        """범주 평균 L+S. members_fn(cat) -> 멤버 목록"""
        return float(np.mean([ls(fused(c, members_fn(c)), LT[c]) for c in CATS]))

    rows = {}

    # ---------- 3-멤버 ----------
    het3 = [mean_ls(lambda c, s=s: [("EAD", s), ("PC", s), ("Comp", s)]) for s in SEEDS]
    rows["heterogeneous_3branch (주 설정)"] = {
        "LS_mean": float(np.mean(het3)), "LS_std": float(np.std(het3, ddof=1)),
        "per_seed": dict(zip(map(str, SEEDS), het3)), "n_members": 3,
        "variation_unit": "시드 (분기 3종 고정)"}
    for b, label in (("EAD", "EAD-S 시드 3개"), ("PC", "PC 뱅크 추첨 3회"), ("Comp", "구성 분기 시드 3개")):
        v = mean_ls(lambda c, b=b: [(b, s) for s in SEEDS])
        rows[f"homogeneous_3x_{b} ({label})"] = {
            "LS_mean": v, "LS_std": None, "n_members": 3,
            "variation_unit": "없음 — 시드 셋을 멤버로 소진했다",
            "delta_vs_heterogeneous": v - float(np.mean(het3))}

    # ---------- 2-멤버 (양쪽 다 std 가 난다) ----------
    het2 = {}
    for pair in itertools.combinations(("EAD", "PC", "Comp"), 2):
        vals = [mean_ls(lambda c, p=pair, s=s: [(p[0], s), (p[1], s)]) for s in SEEDS]
        het2["+".join(pair)] = vals
    hv = [v for vals in het2.values() for v in vals]
    rows["heterogeneous_2branch (3조합 × 3시드)"] = {
        "LS_mean": float(np.mean(hv)), "LS_std": float(np.std(hv, ddof=1)), "n_members": 2,
        "per_pair": {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                     for k, v in het2.items()},
        "variation_unit": "조합 × 시드"}
    for b in ("EAD", "PC", "Comp"):
        vals = [mean_ls(lambda c, b=b, sp=sp: [(b, sp[0]), (b, sp[1])])
                for sp in itertools.combinations(SEEDS, 2)]
        rows[f"homogeneous_2x_{b} (시드쌍 3개)"] = {
            "LS_mean": float(np.mean(vals)), "LS_std": float(np.std(vals, ddof=1)), "n_members": 2,
            "per_pair": {f"{a}+{b2}": v for (a, b2), v in
                         zip(itertools.combinations(SEEDS, 2), vals)},
            "variation_unit": "시드쌍",
            "delta_vs_heterogeneous_2": float(np.mean(vals)) - float(np.mean(hv))}

    # ---------- 단일 멤버 (기준선) ----------
    for b in ("EAD", "PC", "Comp"):
        vals = [mean_ls(lambda c, b=b, s=s: [(b, s)]) for s in SEEDS]
        rows[f"single_{b}"] = {"LS_mean": float(np.mean(vals)),
                              "LS_std": float(np.std(vals, ddof=1)), "n_members": 1,
                              "variation_unit": "시드"}

    # ---------- 부트스트랩 CI (3-멤버 행들을 같은 기준으로) ----------
    rng = np.random.default_rng(0)
    boot = {}
    specs = {"heterogeneous_3branch (주 설정)": lambda c: [("EAD", 42), ("PC", 42), ("Comp", 42)]}
    for b in ("EAD", "PC", "Comp"):
        specs[f"homogeneous_3x_{b}"] = lambda c, b=b: [(b, s) for s in SEEDS]
    for name, fn in specs.items():
        per_cat = [ls_boot(fused(c, fn(c)), LT[c], rng) for c in CATS]
        n = min(len(x) for x in per_cat)
        m = np.mean([x[:n] for x in per_cat], axis=0)
        boot[name] = {"mean": float(m.mean()),
                      "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                      "n_boot": int(n)}
    boot["_note"] = ("test 이미지 층화 부트스트랩. 이종 행은 시드 42 기준(시드 std 는 위 표에 있다). "
                     "동종과 이종을 같은 불확실성 기준으로 견주기 위한 보조 수치다.")

    res = {"rows": rows, "bootstrap_ci": boot,
           "rule": "각 멤버를 공식 검증셋 정상 표본 통계로 z-정규화한 뒤 등가중 합. 지표 L+S.",
           "note": "3-멤버 동종은 시드를 멤버로 소진해 시드 std 가 없다. 2-멤버 급은 이종·동종 "
                   "양쪽에서 조합이 3개씩 나와 std 가 잡히므로, 멤버 수를 맞춘 통제 비교로 그쪽을 "
                   "함께 본다."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)

    print("=== 동종 vs 이종 (L+S, 5범주 평균)")
    print(f"  {'구성':46s}{'멤버':>4s}{'L+S':>9s}{'std':>9s}{'Δ vs 이종':>11s}")
    for k, v in rows.items():
        sd = f"{v['LS_std']:.4f}" if v.get("LS_std") is not None else "   —  "
        d = v.get("delta_vs_heterogeneous", v.get("delta_vs_heterogeneous_2"))
        dd = f"{d:+.4f}" if d is not None else ""
        print(f"  {k:46s}{v['n_members']:>4d}{v['LS_mean']:>9.4f}{sd:>9s}{dd:>11s}")
    print("\n=== 부트스트랩 95% CI (3-멤버, 같은 기준)")
    for k, v in boot.items():
        if k.startswith("_"):
            continue
        print(f"  {k:40s} {v['mean']:.4f}  [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
