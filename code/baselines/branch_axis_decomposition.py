#!/usr/bin/env python3
"""요청 ① — 세 분기 각각의 논리/구조/개수축 AUROC (시드 42/43/44).

왜 필요한가
  논문 사슬이 "§4.2 개수축은 readout 에 묶인다 → §3.1.3 그래서 구성 분기를 둔다 → §4.3 0.9723"
  인데 가운데가 비어 있다. **우리 분기별 축 분해가 논문 어디에도 없다.** 표 4 는 분기별 L+S 만,
  표 3 은 방법별 축 분해만 담는다. §4.2(P79)가 스스로 "표 C2 의 대리 readout 은 구성 분기와
  구별된다" 고 적고 있어 연결이 끊겼음을 논문이 명시하는 셈이다.

무엇을 하나
  새 학습도 새 추론도 없다. 저장된 per-image 점수를 축으로 다시 가른다.
  축 라벨은 부록 C 와 **같은 경로**로 만든다 — LOCO defects_config.json 의 pixel_value→defect_name
  과 이미지별 GT 마스크에서 하위유형을 읽고, 사전등록된 STRAND(결과 보기 전 LOCK)로 축을 준다.
  strand 정의는 scripts/countgd/footprint_resolved_allcat.py 에서 그대로 import 한다 — 다시
  타이핑하면 사전등록이 깨진다.

AUROC 는 순위 통계라 z-정규화가 결과를 바꾸지 않는다. 분기 점수를 그대로 쓴다.
융합(AFLAD) 행은 논문 규칙대로 z(val 정상 통계) 합으로 만든다.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
LOCO = R / "datasets/MVTecLOCO"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = [42, 43, 44]
OUT = R / "reports/countgd/branch_axis_decomposition.json"
MIN_N = 4          # 부록 C 와 동일 (n<4 하위유형 제외)

# 사전등록된 subtype→strand 를 원 스크립트에서 그대로 가져온다.
# exec 하지 않고 AST 로 해당 대입문만 읽는다 — 원 스크립트는 import 시 실험을 다시 돌려
# 산출물을 덮어쓴다(실제로 한 번 덮었고 되돌렸다). 두 dict 는 순수 리터럴이라 안전하다.
def _locked_dicts(src: Path, names):
    import ast
    tree = ast.parse(src.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in names:
                out[t.id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"사전등록 dict 를 찾지 못했다: {missing}")
    return out

_locked = _locked_dicts(R / "scripts/countgd/footprint_resolved_allcat.py", ("STRAND", "EXPECT"))
STRAND, EXPECT = _locked["STRAND"], _locked["EXPECT"]


def branch_paths(cat, seed):
    ead = (R / "reports/phase0/ead_repro_npz" if seed == 42
           else R / "reports/phase0/efficient_ad_official_small/npz") / f"scores_{cat}_seed{seed}.npz"
    pc = R / f"reports/countgd/pc_branch_scores/pc_L17_{cat}_seed{seed}.npz"
    ps = (R / "external/PSAD_official/LOCO_MVTec_AD/rebuild_scores"
          / f"psad_scores_{cat}_seed{seed}_hc_tta_merge_test.npz")
    psv = str(ps).replace("_test.npz", "_val.npz")
    return ead, pc, ps, Path(psv)


def subtypes_for(cat):
    """논리 test 이미지 각각의 하위유형. 부록 C 와 같은 유도 경로."""
    cfg = {d["pixel_value"]: d["defect_name"] for d in json.load(open(LOCO / cat / "defects_config.json"))}
    gt = LOCO / cat / "ground_truth/logical_anomalies"
    out = []
    for p in sorted((LOCO / cat / "test/logical_anomalies").glob("*.png")):
        d = gt / p.stem
        pv = set()
        if d.is_dir():
            for m in d.glob("*.png"):
                pv |= set(np.unique(np.array(Image.open(m))).tolist())
        names = sorted(cfg[v] for v in pv if v in cfg)
        out.append(names[0] if len(names) == 1 else "MIXED")
    return out


def auc(score, good_idx, pos_idx):
    pos = np.asarray(pos_idx)
    return float(roc_auc_score(np.r_[np.zeros(len(good_idx)), np.ones(len(pos))],
                               np.r_[score[good_idx], score[pos]]))


def main():
    per = {c: {} for c in CATS}
    provenance = {}
    for cat in CATS:
        subs = subtypes_for(cat)
        for seed in SEEDS:
            ead_p, pc_p, ps_p, psv_p = branch_paths(cat, seed)
            ze = np.load(ead_p, allow_pickle=True)
            zp = np.load(pc_p, allow_pickle=True)
            zs = np.load(ps_p, allow_pickle=True)
            zsv = np.load(psv_p, allow_pickle=True)

            # --- 정렬 검증. PSAD 의 paths 가 유일한 절대 근거다.
            paths = [str(x) for x in zs["paths"]]
            want = ["good" if "/good/" in q else "logical" if "logical" in q else "structural"
                    for q in paths]
            lt = np.array([str(x) for x in ze["label_type"]])
            assert list(lt) == want, f"{cat}/seed{seed}: EAD 와 PSAD 의 이미지 순서가 다르다"
            assert len(zp["test"]) == len(lt), f"{cat}/seed{seed}: PC 길이 불일치"
            log_names = [Path(q).name for q in paths if "logical" in q]
            assert log_names == [p.name for p in sorted((LOCO / cat / "test/logical_anomalies").glob("*.png"))], \
                f"{cat}: 논리 이미지 순서가 하위유형 라벨 순서와 다르다"

            g = np.where(lt == "good")[0]
            lp = np.where(lt == "logical")[0]
            sp = np.where(lt == "structural")[0]

            # --- 축별 인덱스 (strand 로 묶는다. 부록 C 는 하위유형 단위였다)
            by_strand = {}
            for i, nm in enumerate(subs):
                if nm == "MIXED":
                    continue
                st = STRAND[cat].get(nm)
                if st:
                    by_strand.setdefault(st, []).append(lp[i])

            ev, pv = np.asarray(zp["val"], float), None  # (이름 혼동 방지용 자리)
            e_t = np.asarray(ze["score"], float)
            p_t = np.asarray(zp["test"], float)
            c_t = np.asarray(zs["scores"], float)
            # 융합은 논문 규칙 그대로: 각 분기를 val 정상 통계로 z-정규화한 뒤 등가중 합
            ead_val = np.load(R / "reports/phase0/efficient_ad_official_small_seeds_std_val"
                              / f"val_good_{cat}_seed{seed}.npz", allow_pickle=True)
            e_v = np.asarray(ead_val[ead_val.files[0]], float)
            z = lambda t, v: (t - v.mean()) / (v.std() + 1e-12)
            f_t = z(e_t, e_v) + z(p_t, np.asarray(zp["val"], float)) + z(c_t, np.asarray(zsv["scores"], float))

            row = {}
            for bname, s in (("EAD", e_t), ("PC", p_t), ("Comp", c_t), ("AFLAD", f_t)):
                r = {"logical": auc(s, g, lp), "structural": auc(s, g, sp)}
                for st, idx in by_strand.items():
                    if len(idx) >= MIN_N:
                        r[f"strand_{st}"] = auc(s, g, idx)
                row[bname] = r
            per[cat][seed] = row
            if seed == SEEDS[0]:
                cnt = Counter(STRAND[cat].get(n, "?") for n in subs if n != "MIXED")
                provenance[cat] = {"n_good": len(g), "n_logical": len(lp), "n_structural": len(sp),
                                   "n_mixed": sum(1 for n in subs if n == "MIXED"),
                                   "strand_image_counts": dict(cnt)}

    # --- 시드 집계
    def agg(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}

    branches = ["EAD", "PC", "Comp", "AFLAD"]
    keys = sorted({k for c in CATS for s in SEEDS for b in branches for k in per[c][s][b]})
    percat = {c: {b: {k: agg([per[c][s][b][k] for s in SEEDS])
                      for k in keys if all(k in per[c][s][b] for s in SEEDS)}
                  for b in branches} for c in CATS}
    overall = {b: {k: agg([float(np.mean([per[c][s][b][k] for c in CATS if k in per[c][s][b]]))
                           for s in SEEDS])
                   for k in keys} for b in branches}

    res = {"per_category": percat, "overall_mean_over_categories": overall,
           "per_seed_raw": {c: {str(s): per[c][s] for s in SEEDS} for c in CATS},
           "provenance": provenance,
           "strand_expect": EXPECT,
           "note": "축 라벨은 LOCO defects_config.json + 이미지별 GT 마스크에서 유도했고 "
                   "subtype→strand 는 scripts/countgd/footprint_resolved_allcat.py 의 "
                   "사전등록 STRAND 를 import 해 그대로 썼다. MIXED 와 strand 이미지 수 <4 는 제외. "
                   "AUROC 는 순위 통계라 z-정규화 여부가 결과를 바꾸지 않는다.",
           }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=2)

    # --- 표
    def cell(d, k):
        return f"{d[k]['mean']:.4f}" if k in d else "  —   "
    print("=== 분기 × 축 (5범주 평균, 시드 42/43/44)")
    hdr = ["logical", "structural", "strand_cardinality", "strand_presence",
           "strand_quantity", "strand_damage"]
    print(f"  {'branch':7s}" + "".join(f"{h.replace('strand_',''):>13s}" for h in hdr))
    for b in branches:
        print(f"  {b:7s}" + "".join(f"{overall[b][h]['mean']:>13.4f}" if h in overall[b]
                                    else f"{'—':>13s}" for h in hdr))
    print("\n=== 범주별 (개수축 = cardinality strand)")
    print(f"  {'category':22s}{'EAD log':>9s}{'PC log':>9s}{'Comp log':>9s} | "
          f"{'EAD card':>9s}{'PC card':>9s}{'Comp card':>10s}{'AFLAD card':>11s}")
    for c in CATS:
        d = percat[c]
        print(f"  {c:22s}" + "".join(f"{cell(d[b],'logical'):>9s}" for b in ("EAD", "PC", "Comp"))
              + " | " + "".join(f"{cell(d[b],'strand_cardinality'):>9s}" for b in ("EAD", "PC", "Comp"))
              + f"{cell(d['AFLAD'],'strand_cardinality'):>11s}")
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
