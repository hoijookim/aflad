#!/usr/bin/env python3
"""표 5 베이스라인을 시드 집합 {42,43,44} 로 재집계한다.

배경: 논문 표 5 는 방법마다 시드 집합이 달랐다 (AFLAD 5-seed {42-46}, SALAD 3,
PUAD-S 1, EAD-M 3, ComAD 3 ...). 검수에서도 "test-free 변형만 구식 3시드"가 지적됐다.
시드 집합을 {42,43,44} 로 통일한다.

여기서는 **시드별 원값이 남아 있어 재실행이 필요 없는** 방법만 처리한다.
원값이 없는 방법(EAD-M seed43/44, SALAD)은 목록만 내고 건드리지 않는다 —
없는 수치를 채워 넣지 않는다.

입력은 전부 이미 계산된 시드별 점수이고, 새로 test 를 조회하지 않는다.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np

R = Path("/workspace/ai-vision-research")
TARGET = (42, 43, 44)
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]


def agg(vals):
    v = np.asarray(vals, float)
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else None, "n": len(v)}


def from_per_seed_per_cat(d, key):
    """{'seed42': {cat: v, ...}, ...} 형태에서 3시드 L+S 를 뽑는다."""
    src = d.get(key, {})
    per_seed = {}
    for s in TARGET:
        row = src.get(f"seed{s}") or src.get(str(s))
        if not row:
            return None
        per_seed[s] = float(np.mean([row[c] for c in CATS if c in row]))
    return per_seed


def main():
    out = {"target_seeds": list(TARGET), "methods": {}, "unavailable": {}}

    # ---- ComAD (이미 42/43/44) ----
    f = R / "reports/path_y/comad/comad_3seed.json"
    if f.exists():
        d = json.load(open(f))
        ps = from_per_seed_per_cat(d, "per_seed_per_cat")
        if ps:
            out["methods"]["ComAD"] = {"per_seed": ps, **agg(list(ps.values())),
                                       "source": str(f.relative_to(R))}

    # ---- PUAD (0,42,43,44,1234 보유) ----
    f = R / "reports/path_y/puad_5seed_allcats/puad_5seed_all_cats.json"
    if f.exists():
        d = json.load(open(f))
        pc = d.get("per_cat_per_seed", {})
        # 구조: per_cat_per_seed[cat][seedX] = {"ead_auroc":.., "puad_auroc":..}
        # 주의 — 여기 있는 것은 **pooled AUROC** 이고 표 5 가 쓰는 L+S(논리·구조 분리)가
        # 아니다. L+S 시드별 원값이 없으므로 pooled 로만 재집계하고 그 사실을 명시한다.
        ps = {}
        for sd in TARGET:
            vals = [pc[c][f"seed{sd}"]["puad_auroc"] for c in CATS
                    if c in pc and f"seed{sd}" in pc[c] and "puad_auroc" in pc[c][f"seed{sd}"]]
            if len(vals) == len(CATS):
                ps[sd] = float(np.mean(vals))
        if len(ps) == len(TARGET):
            out["methods"]["PUAD (pooled, L+S 아님)"] = {
                "per_seed": ps, **agg(list(ps.values())),
                "metric": "pooled AUROC", "source": str(f.relative_to(R))}
            out["unavailable"]["PUAD (L+S)"] = (
                "시드별 pooled 만 있고 논리/구조 분리값이 없다 — 표 5 의 L+S 를 "
                "42/43/44 로 재집계하려면 시드별 재채점이 필요하다.")
        else:
            out["unavailable"]["PUAD"] = f"시드별 5범주 완비 실패 (확보 {sorted(ps)})"

    # ---- AFLAD 주 설정 (42-46 -> 42/43/44) ----
    f = R / "reports/countgd/fusion_hc_percat_perseed.json"
    if f.exists():
        d = json.load(open(f))
        s = json.dumps(d)
        ps = {}
        for sd in TARGET:
            m = re.search(rf'"(?:seed)?{sd}"\s*:\s*\{{([^}}]*)\}}', s)
            if not m:
                continue
            vals = [float(x) for x in re.findall(r':\s*([0-9.]+)', m.group(1))]
            if vals:
                ps[sd] = float(np.mean(vals))
        if len(ps) == len(TARGET):
            out["methods"]["AFLAD_hc"] = {"per_seed": ps, **agg(list(ps.values())),
                                          "source": str(f.relative_to(R))}
        else:
            out["unavailable"]["AFLAD_hc"] = "per-seed 구조 파싱 실패 — 수동 확인 필요"

    # ---- hc/hcp 절제 (42-46 보유) ----
    f = R / "reports/countgd/ablation_psad_hc_vs_hcp.json"
    if f.exists():
        d = json.load(open(f))
        for k in ("hc", "hcp"):
            fp = d.get(k, {}).get("fusion_per_seed", {})
            ps = {s: float(fp[str(s)]) for s in TARGET if str(s) in fp}
            if len(ps) == len(TARGET):
                out["methods"][f"AFLAD_{k}_CV"] = {"per_seed": ps, **agg(list(ps.values())),
                                                   "source": str(f.relative_to(R))}

    # ---- 재실행이 필요한 것 ----
    ead_m = sorted({int(x) for x in re.findall(
        r'seed(\d+)', " ".join(glob.glob(str(R / "reports/phase0/efficient_ad_official_medium/**/*.npz"),
                                         recursive=True)))})
    out["unavailable"]["EAD-M"] = f"보유 시드 {ead_m} — 43/44 학습 필요 (약 6시간)"
    out["unavailable"]["SALAD"] = ("저자 사전학습 가중치 추론이라 시드 변동이 없다. "
                                   "3시드화하려면 SALAD 자체 학습이 필요하다(수일). "
                                   "표기 'n_seed 3' 자체를 재검토해야 한다.")

    print(f"{'방법':16s} {'42/43/44 L+S':>16s}  시드별")
    for k, v in out["methods"].items():
        sd = f" ± {v['std']:.4f}" if v["std"] is not None else ""
        print(f"{k:16s} {v['mean']:10.4f}{sd:6s}  " +
              " ".join(f"s{s} {x:.4f}" for s, x in v["per_seed"].items()))
    print("\n재집계 불가:")
    for k, v in out["unavailable"].items():
        print(f"  {k:10s} {v}")
    p = R / "reports/countgd/realign_3seed_42_43_44.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
