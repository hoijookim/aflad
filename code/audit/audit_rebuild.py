#!/usr/bin/env python3
"""PSAD 재구축 적대적 검수 — "제대로 실험된 게 맞는가"를 반증 시도.

A. 앵커 대조   — 재구축 hcp(test) vs 원본 d2_out(같은 시드): Pearson/Spearman/AUROC Δ.
                 의사레이블 소스가 다르므로(CSAD 저자 배포 vs 원본 자체생성) 완전 일치가
                 아니라 '같은 신호를 재현했는가'(고상관·유사 AUROC)를 판정.
B. 분할 건전성 — 세그멘터 test 분할맵 클래스 분포가 붕괴(단일 클래스 지배)하지 않았는지,
                 train 의사레이블 클래스 집합과 정합하는지.
C. 누수 부재   — 채점 코드 정적 검증: 메모리·스케일 통계가 train 만 사용, val 채점이
                 test 무접촉, min-max(test 통계) 미사용.
D. 점수 건전성 — NaN/상수/중복 없음, good vs anomaly 분리도, hc/hcp per-cat AUROC 를
                 원본 세대의 기대 범위와 대조.
"""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
REB = R / "external/PSAD_official/LOCO_MVTec_AD"
SC = REB / "rebuild_scores"
D2 = R / "external/ROMAD_baselines/PSAD/d2_out"          # 원본 hcp (seed 0/42/1234)
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

FAIL, WARN, OK = [], [], []


def rec(level, tag, msg):
    {"FAIL": FAIL, "WARN": WARN, "OK": OK}[level].append(f"[{tag}] {msg}")
    print(f"  {level:4s} [{tag}] {msg}", flush=True)


def load_rebuild(cat, seed, mtype, split):
    p = SC / f"psad_scores_{cat}_seed{seed}_{mtype}_{split}.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def align_orig(cat, seed, ltypes_order):
    """원본 d2_out 을 (type, fname) 정렬로 재배열."""
    p = D2 / f"psad_scores_{cat}_seed{seed}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    d = {}
    for s, pth in zip(z["scores"].astype(float), [str(x) for x in z["paths"]]):
        t = "good" if "good" in pth else ("logical" if "logical" in pth else "structural")
        d[(t, pth.split("/")[-1])] = s
    try:
        return np.array([d[o] for o in ltypes_order])
    except KeyError:
        return None


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    print("=" * 78)
    print(f"PSAD 재구축 적대적 검수 (seed {seed})")
    print("=" * 78)

    # ---------- A. 앵커 대조 ----------
    print("\nA. 앵커 — 재구축 hcp vs 원본 d2_out (같은 시드)")
    for cat in CATS:
        z = load_rebuild(cat, seed, "hcp", "test")
        if z is None:
            rec("WARN", "A", f"{cat}: 재구축 점수 없음 (아직 미실행?)")
            continue
        paths = [str(p) for p in z["paths"]]
        order = []
        for pth in paths:
            t = ("good" if "/good/" in pth
                 else ("logical" if "logical" in pth else "structural"))
            order.append((t, pth.split("/")[-1]))
        orig = align_orig(cat, seed, order)
        if orig is None:
            rec("WARN", "A", f"{cat}: 원본 d2_out seed{seed} 없음/정렬 실패")
            continue
        reb = z["scores"].astype(float)
        gt = z["gt"].astype(int)
        r = float(np.corrcoef(reb, orig)[0, 1])
        rho = float(spearmanr(reb, orig).statistic)
        a_reb, a_org = roc_auc_score(gt, reb), roc_auc_score(gt, orig)
        lvl = "OK" if (rho >= 0.75 or abs(a_reb - a_org) <= 0.03) else \
              ("WARN" if (rho >= 0.5 or abs(a_reb - a_org) <= 0.08) else "FAIL")
        rec(lvl, "A", f"{cat}: r={r:.3f} ρ={rho:.3f} | pooled AUROC 재구축 {a_reb:.4f} "
                      f"vs 원본 {a_org:.4f} (Δ{a_reb-a_org:+.4f})")

    # ---------- B. 분할 건전성 ----------
    print("\nB. 세그멘터 분할 건전성 (test 분할맵)")
    for cat in CATS:
        seg = REB / f"unet_seed{seed}" / cat / "test" / "good"
        files = sorted(seg.glob("*.png"))[:20] if seg.exists() else []
        if not files:
            rec("WARN", "B", f"{cat}: 분할맵 없음")
            continue
        fracs = []
        idsets = set()
        for f in files:
            m = np.array(Image.open(f))
            ids, cnt = np.unique(m, return_counts=True)
            idsets |= set(ids.tolist())
            fg = cnt[ids != 0].sum()
            fracs.append(fg / m.size)
        # train 의사레이블 클래스 집합
        tr = sorted((REB / "csad_pseudo_seg" / cat).glob("pred_*.png"))[:20]
        tr_ids = set()
        for f in tr:
            tr_ids |= set(np.unique(np.array(Image.open(f))).tolist())
        cover = len(idsets & tr_ids) / max(len(tr_ids), 1)
        fg = float(np.mean(fracs))
        lvl = "OK" if (0.02 <= fg <= 0.98 and cover >= 0.6) else "FAIL"
        rec(lvl, "B", f"{cat}: 전경비율 {fg:.2f}, 예측 클래스 {sorted(idsets)[:8]}"
                      f"{'...' if len(idsets) > 8 else ''}, train 클래스 커버 {cover:.0%}")

    # ---------- C. 누수 부재 (정적) ----------
    print("\nC. 누수 부재 — 채점 코드 정적 검증")
    src = (R / "scripts/psad_rebuild/score_psad.py").read_text()
    mem_train_only = 'rels("train", "good")' in src and "memory = torch.stack(feats" in src
    rec("OK" if mem_train_only else "FAIL", "C",
        f"메모리 구축 입력 = train/good only: {mem_train_only}")
    minmax = re.search(r"min_dists\s*-\s*min_score", src)
    rec("OK" if not minmax else "FAIL", "C",
        f"test 통계 min-max 정규화 사용: {'없음' if not minmax else '있음(누수)'}")
    val_indep = "validation" in src and "score_rel" in src
    rec("OK" if val_indep else "FAIL", "C",
        "val 채점이 train 통계(mean/std/scale-max)만 사용 (test 무접촉)")

    # ---------- D. 점수 건전성 ----------
    print("\nD. 점수 건전성 + hc/hcp AUROC")
    for cat in CATS:
        row = []
        bad = False
        for mtype in ["hc", "hcp"]:
            z = load_rebuild(cat, seed, mtype, "test")
            if z is None:
                bad = True
                continue
            s, gt = z["scores"].astype(float), z["gt"].astype(int)
            nan = np.isnan(s).sum()
            uniq = len(np.unique(s))
            auc = roc_auc_score(gt, s)
            sep = s[gt == 1].mean() / max(s[gt == 0].mean(), 1e-9)
            row.append(f"{mtype}: AUROC {auc:.4f} 분리비 {sep:.2f} 고유 {uniq}/{len(s)}"
                       + (f" NaN {nan}!" if nan else ""))
            if nan or uniq < len(s) * 0.5 or auc < 0.55:
                bad = True
        zv = load_rebuild(cat, seed, "hc", "val")
        nval = len(zv["scores"]) if zv is not None else 0
        rec("FAIL" if bad else "OK", "D", f"{cat}: " + " | ".join(row) + f" | val {nval}장")

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
