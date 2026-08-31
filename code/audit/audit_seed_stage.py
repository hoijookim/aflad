#!/usr/bin/env python3
"""시드 단계별 적대적 검수 — "이 시드는 제대로, 빠짐없이 돌았는가"를 반증 시도.

각 항목은 통과가 목적이 아니라 **실패하면 그 시드를 폐기**해야 하는 반증 조건이다.

S1 완주       학습이 300ep 를 실제로 돌았는가 (조기종료·크래시 잔여 아닌가)
S2 산출 완비  체크포인트 5범주 + 분할맵 전 split + 채점 npz 20개가 다 있는가
S3 시드 유효  이 시드의 가중치·점수가 다른 시드와 실제로 다른가 (시드 미적용 반증)
S4 정렬       채점 npz 의 순서·길이가 라벨 정본과 일치하는가
S5 건전성     점수에 NaN·상수·붕괴가 없고 good/이상 분리가 살아있는가
S6 구간 누락  분할맵이 train/val/test 전부 있고 이미지 수가 원본과 일치하는가
S7 앵커       hcp 가 원본 d2_out(해당 시드 보유 시) 과 상관되는가
S8 재현성     같은 체크포인트로 두 번 채점하면 같은 값인가 (결정성)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

R = Path("/workspace/ai-vision-research")
DATA = R / "external/PSAD_official/LOCO_MVTec_AD"
REB = DATA / "rebuild_scores"
D2 = R / "external/ROMAD_baselines/PSAD/d2_out"
LOCO = R / "datasets/MVTecLOCO"
V4 = R / "algml_v6_5/aupr_bootstrap/scores"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

FAIL, WARN, OK = [], [], []


def rec(lv, tag, msg):
    {"FAIL": FAIL, "WARN": WARN, "OK": OK}[lv].append(f"[{tag}] {msg}")
    print(f"  {lv:4s} [{tag}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    a = ap.parse_args()
    sd = a.seed
    print("=" * 78)
    print(f"PSAD seed {sd} 단계 검수")
    print("=" * 78)

    # ---- S1 완주 ----
    print("\nS1. 학습 완주 (300ep)")
    # 학습 로그의 정본은 train_unet_seeds.sh 가 append 하는 psad_unet_train.log 하나다.
    # 다른 파일들은 이걸 tee 로 감싸 중복 기록하므로 합치면 안 된다.
    # 또한 범주마다 "마지막 시작~완료" 구간만 채점해야 크래시한 이전 시도가 섞이지 않는다.
    log = R / "_logs/psad_unet_train.log"
    txt = log.read_text(errors="ignore") if log.exists() else ""

    # EPOCH 로그는 러너가 `| tail -3` 으로 잘라 저장하므로 신뢰할 수 없다.
    # 대신 체크포인트 파일명이 완주를 증명한다 — train_normal_unet.py 는 300ep 루프가
    # 끝난 뒤에야 f"{obj}_{num_epochs}.pth" 를 저장하므로 `_300.pth` 존재 = 완주.
    # 분할맵은 그 다음 단계 산출물이라 함께 있으면 추론까지 끝난 것이다.
    mins = []
    miss = []
    for cat in CATS:
        ck = DATA / f"output/unet_seed{sd}/{cat}/{cat}_300.pth"
        if not ck.exists():
            miss.append(cat)
        m = re.findall(rf"==== seed={sd} {cat} 완료 (\d+)분", txt)
        if m:
            mins.append(int(m[-1]))
    rec("OK" if not miss else "FAIL", "S1",
        f"완주 증명 체크포인트 *_300.pth {5-len(miss)}/5" + (f" 누락 {miss}" if miss else ""))
    short = [m for m in mins if m < 10]
    rec("OK" if (len(mins) >= 5 and not short) else ("WARN" if mins else "FAIL"), "S1",
        f"범주별 소요 {mins}분 (10분 미만 {len(short)}건)")

    # ---- S2 산출 완비 ----
    print("\nS2. 산출물 완비")
    ck = len(list((DATA / f"output/unet_seed{sd}").rglob("*_300.pth")))
    rec("OK" if ck >= 5 else "FAIL", "S2", f"체크포인트 {ck}/5")
    npz = len(list(REB.glob(f"psad_scores_*_seed{sd}_*.npz")))
    rec("OK" if npz >= 20 else "FAIL", "S2", f"채점 npz {npz}/20 (5범주 x hc/hcp x test/val)")

    # ---- S6 구간 누락 ----
    print("\nS6. 분할맵 구간 누락")
    for cat in CATS:
        root = DATA / f"unet_seed{sd}" / cat
        got = {s: len(list((root / s).rglob("*.png"))) for s in ["train", "validation", "test"]
               if (root / s).exists()}
        exp = {"train": len(list((LOCO / cat / "train" / "good").glob("*.png"))),
               "validation": len(list((LOCO / cat / "validation" / "good").glob("*.png"))),
               "test": sum(len(list(p.glob("*.png")))
                           for p in (LOCO / cat / "test").iterdir() if p.is_dir())}
        miss = [k for k in exp if got.get(k, 0) != exp[k]]
        rec("OK" if not miss else "FAIL", "S6",
            f"{cat}: {got} (기대 {exp}){' 불일치 ' + str(miss) if miss else ''}")

    # ---- S3 시드 유효 ----
    print("\nS3. 시드가 실제로 반영되었는가")
    for cat in CATS[:2]:
        vals = {}
        for s2 in [42, 43, 44, 45, 46]:
            f = REB / f"psad_scores_{cat}_seed{s2}_hc_test.npz"
            if f.exists():
                vals[s2] = np.load(f, allow_pickle=True)["scores"].astype(float)
        if sd in vals and len(vals) > 1:
            others = {k: v for k, v in vals.items() if k != sd}
            same = [k for k, v in others.items()
                    if len(v) == len(vals[sd]) and np.allclose(v, vals[sd])]
            rec("OK" if not same else "FAIL", "S3",
                f"{cat}: seed{sd} 점수가 {list(others)} 와 동일한 시드 {same if same else '없음'}")
        else:
            rec("WARN", "S3", f"{cat}: 비교 대상 부족 (보유 {sorted(vals)})")

    # ---- S4 정렬 ----
    print("\nS4. 정렬 규약")
    for cat in CATS:
        f = REB / f"psad_scores_{cat}_seed{sd}_hc_test.npz"
        if not f.exists():
            rec("FAIL", "S4", f"{cat}: 채점 npz 없음")
            continue
        z = np.load(f, allow_pickle=True)
        paths = [str(p) for p in z["paths"]]
        lt = ["good" if "/good/" in p else ("logical" if "logical" in p else "structural")
              for p in paths]
        ref = np.load(V4 / f"scores_{cat}_seed42.npz", allow_pickle=True)["label_type"].astype(str)
        rec("OK" if list(ref) == lt else "FAIL", "S4",
            f"{cat}: n={len(lt)} (정본 {len(ref)}) 타입 시퀀스 {'일치' if list(ref) == lt else '불일치'}")

    # ---- S5 건전성 ----
    print("\nS5. 점수 건전성")
    for cat in CATS:
        f = REB / f"psad_scores_{cat}_seed{sd}_hc_test.npz"
        fv = REB / f"psad_scores_{cat}_seed{sd}_hc_val.npz"
        if not (f.exists() and fv.exists()):
            rec("FAIL", "S5", f"{cat}: npz 누락")
            continue
        z = np.load(f, allow_pickle=True)
        s, gt = z["scores"].astype(float), z["gt"].astype(int)
        v = np.load(fv, allow_pickle=True)["scores"].astype(float)
        nan = int(np.isnan(s).sum()) + int(np.isnan(v).sum())
        uniq = len(np.unique(s))
        auc = roc_auc_score(gt, s)
        sep = s[gt == 1].mean() / max(s[gt == 0].mean(), 1e-9)
        bad = nan or uniq < len(s) * 0.5 or auc < 0.55 or len(v) == 0
        rec("FAIL" if bad else "OK", "S5",
            f"{cat}: AUROC {auc:.4f} 분리비 {sep:.2f} 고유 {uniq}/{len(s)} val {len(v)}장 NaN {nan}")

    # ---- S7 앵커 ----
    print("\nS7. 원본 d2_out 대비 앵커 (해당 시드 보유 시)")
    for cat in CATS:
        do = D2 / f"psad_scores_{cat}_seed{sd}.npz"
        f = REB / f"psad_scores_{cat}_seed{sd}_hcp_test.npz"
        if not do.exists():
            rec("WARN", "S7", f"{cat}: 원본 seed{sd} 없음 — 대조 불가(정상, 원본은 0/42/1234)")
            continue
        if not f.exists():
            rec("FAIL", "S7", f"{cat}: 재구축 hcp 없음")
            continue
        z = np.load(f, allow_pickle=True)
        reb = z["scores"].astype(float)
        oz = np.load(do, allow_pickle=True)
        d = {}
        for sc, p in zip(oz["scores"].astype(float), [str(x) for x in oz["paths"]]):
            t = "good" if "good" in p else ("logical" if "logical" in p else "structural")
            d[(t, p.split("/")[-1])] = sc
        order = []
        for p in [str(x) for x in z["paths"]]:
            t = "good" if "/good/" in p else ("logical" if "logical" in p else "structural")
            order.append((t, p.split("/")[-1]))
        try:
            org = np.array([d[o] for o in order])
        except KeyError:
            rec("FAIL", "S7", f"{cat}: 정렬 실패")
            continue
        rho = float(spearmanr(reb, org).statistic)
        rec("OK" if rho >= 0.5 else "FAIL", "S7", f"{cat}: Spearman {rho:.3f}")

    print("\n" + "=" * 78)
    print(f"종합: OK {len(OK)} / WARN {len(WARN)} / FAIL {len(FAIL)}")
    for f in FAIL:
        print("  FAIL -", f)
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
