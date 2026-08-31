#!/usr/bin/env python3
"""요청 ⑩ — **추론 경로 기준** 파라미터 수 실측.

## 260821 실측과 무엇이 다른가

260821(`measure_param_counts.py`)은 **backbone 합**을 냈다. 요청 ⑩은 추론 경로를 묻는다.
구성 분기에서 어긋나는데, **요청서가 예상한 두 건 외에 한 건이 더 있다.**

| # | 항목 | 요청서 예상 | 실제 |
|---|---|---|---|
| 가 | 분할기 디코더·좌표 채널 누락 | 과소 | **맞다** (아래 §2) |
| 나 | 임베딩 ResNet-101 전체 계수 | 과대 | **부분적으로만 맞다** (아래 §3) |
| **다** | **분할기 인코더도 전체 계수** | (언급 없음) | **과대** — WRN-101-2 의 layer4·fc 는 **호출조차 안 된다** |

## 세 가지 정의를 분리해야 한다

"추론 파라미터" 가 한 가지가 아니다. 두 하위 모듈의 **성격이 다르기 때문**이다.

`CNNSegmenter.forward`(train_normal_unet.py:112)는 `enc.conv1/bn1/relu/maxpool/layer1/
layer2/layer3` 를 **직접 호출**한다. `enc.layer4` 와 `enc.fc` 는 생성만 되고 **실행되지
않는다.** (어제 3-f 검수에서 두 실행의 `enc.layer4.*` 가 비트 단위로 같고 BN 이 초기값
그대로인 것을 확인했다 — 순전파를 한 번도 안 탔다는 직접 증거다.)

`psad.Encoder.extract_ft`(psad.py:58)는 `_ = self.model(x_t)` 로 **resnet101 전체 forward
를 돌린다.** layer4 는 훅으로 잡혀 보간까지 되지만 `ft = cat([f0,f1,f2])` 에서 빠져
**버려진다.** fc 도 실행된다.

그래서 세 가지가 다른 값이 된다:

```
loaded    체크포인트/가중치에서 메모리에 올라가는 것
executed  forward 에서 실제로 연산되는 것
effective 최종 점수에 기여하는 것
```

- 분할기 인코더: layer4·fc 가 loaded 이지만 executed 아님
- 임베딩: layer4·fc 가 executed 이지만 effective 아님 (계산하고 버린다)

요청서 §2.2 는 "실제 forward 를 타는 부분만" 이라 했는데, 임베딩은 **전체가 forward 를
탄다.** 그러므로 그 지시대로 세면 `executed` 가 아니라 `effective` 를 세는 것이 된다.
어느 것을 원고에 쓸지는 논문 세션 판단이므로 **셋 다 낸다.**

## 동결/학습

`requires_grad` 가 아니라 **본 파이프라인에서 실제로 갱신되는지**로 나눈다.
- 분할기는 `pretrained=False`(scratch)로 300 epoch 학습 → 인코더 포함 전체가 학습
- 임베딩 `Encoder` 는 `requires_grad=False` 이고 갱신 없음 → 동결
- EAD teacher 는 공개 가중치 그대로 → 동결. student·AE 는 70k steps 학습
- DINOv3-L 은 미세조정 없음 → 동결
"""
import json
from pathlib import Path

import torch
import torchvision

REPO = Path("/workspace/ai-vision-research")
OUT = REPO / "reports/countgd/param_counts_inference.json"
SEG_CKPT = (REPO / "external/PSAD_official/LOCO_MVTec_AD/output/unet_seed42"
            / "breakfast_box/breakfast_box_300.pth")
EAD_CKPT = (REPO / "reports/phase0/efficient_ad_official_small_seeds_std"
            / "breakfast_box_seed44/trainings/mvtec_loco/breakfast_box")
NUM_CLS = {"breakfast_box": 7, "juice_bottle": 9, "pushpins": 26,
           "screw_bag": 7, "splicing_connectors": 10}


def n(m):
    return int(sum(p.numel() for p in m.parameters()))


def by_prefix(model, prefixes):
    """접두사로 파라미터를 고른다 — 부분 모듈을 세는 가장 확실한 방법."""
    return int(sum(p.numel() for k, p in model.named_parameters()
                   if any(k.startswith(x) for x in prefixes)))


def main():
    rep = {}

    # ── 구성 분기: 분할기 (CNNSegmenter) ────────────────────────────────
    wrn = torchvision.models.wide_resnet101_2()
    used = ["conv1.", "bn1.", "layer1.", "layer2.", "layer3."]     # forward 가 직접 호출
    dead = ["layer4.", "fc."]                                      # 생성만 되고 미호출
    seg_enc_all, seg_enc_used = n(wrn), by_prefix(wrn, used)
    seg_enc_dead = by_prefix(wrn, dead)
    assert seg_enc_used + seg_enc_dead == seg_enc_all, "인코더 분해가 전체와 안 맞는다"

    # 디코더 — train_normal_unet.py:99-101. 좌표는 conv1 의 입력 2채널로만 들어간다.
    dec = {}
    for cat, k in NUM_CLS.items():
        c1 = torch.nn.Conv2d(1024 + 2, 512, 3, 1, 1)
        c2 = torch.nn.Conv2d(512, 256, 3, 1, 1)
        c3 = torch.nn.Conv2d(256, k, 1)
        dec[cat] = {"conv1": n(c1), "conv2": n(c2), "conv3": n(c3),
                    "sum": n(c1) + n(c2) + n(c3)}
    # 좌표 채널의 기여 = conv1 입력 2채널분. 좌표 자체는 파라미터가 없다(concat).
    coord_only = 2 * 512 * 3 * 3

    # 체크포인트로 교차 검증 — 실제 저장된 것과 맞는가
    ck = torch.load(SEG_CKPT, map_location="cpu", weights_only=False)
    sd = ck.state_dict() if hasattr(ck, "state_dict") else ck
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    # state_dict 는 BN 버퍼(running_mean/var/num_batches_tracked)를 포함하고
    # parameters() 는 포함하지 않는다. 파라미터끼리 비교해야 한다.
    BUF = ("running_mean", "running_var", "num_batches_tracked")
    par = {k: v for k, v in sd.items()
           if torch.is_tensor(v) and not k.endswith(BUF)}
    ck_total = int(sum(v.numel() for v in par.values()))
    ck_buffers = int(sum(v.numel() for k, v in sd.items()
                         if torch.is_tensor(v) and k.endswith(BUF)))
    ck_enc_dead = int(sum(v.numel() for k, v in par.items()
                          if k.startswith("enc.layer4") or k.startswith("enc.fc")))

    # ── 구성 분기: 임베딩 (psad.Encoder) ────────────────────────────────
    r101 = torchvision.models.resnet101()
    emb_all = n(r101)
    emb_eff = by_prefix(r101, used)          # 출력에 기여 = stem + layer1-3
    emb_dead = by_prefix(r101, dead)         # 실행되지만 버려짐
    assert emb_eff + emb_dead == emb_all

    # ── 재구성 분기 (EfficientAD-S) ────────────────────────────────────
    rec = {}
    for f, tag in (("teacher_final.pth", "teacher_PDN"),
                   ("student_final.pth", "student_PDN"),
                   ("autoencoder_final.pth", "autoencoder")):
        rec[tag] = n(torch.load(EAD_CKPT / f, map_location="cpu", weights_only=False))

    # ── 패치 메모리 분기 (DINOv3-L/16) ─────────────────────────────────
    from transformers import AutoModel
    pm = n(AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m"))

    # ── 집계 ───────────────────────────────────────────────────────────
    dmean = sum(v["sum"] for v in dec.values()) / len(dec)
    dmin, dmax = min(v["sum"] for v in dec.values()), max(v["sum"] for v in dec.values())

    def totals(seg_enc, emb):
        """학습/동결/사장 3분할.

        **260830 정정**: 이전 판은 `loaded` 에서 분할기 인코더 전체(126.89M)를 학습으로
        셌다. 틀렸다 — `enc.layer4·fc`(44.02M)는 forward 를 타지 않으므로 gradient 가
        없고 **한 번도 갱신되지 않는다.** 3-f 복제 실행으로 직접 확인했다: 두 독립 학습
        사이에서 layer4+fc 32/32 텐서가 비트 동일이고 나머지 288개는 0/288 동일이다
        (5범주 전부). 학습된 것과 아닌 것이 완전히 갈린다.

        따라서 **학습 94.14M 은 세 정의 모두에서 같다.** 정의에 따라 달라지는 것은
        총합과 '사장' 항목의 유무뿐이다.
        """
        dead = seg_enc_all - seg_enc_used if seg_enc == seg_enc_all else 0
        trained = rec["student_PDN"] + rec["autoencoder"] + (seg_enc - dead) + dmean
        frozen = rec["teacher_PDN"] + pm + emb
        return {"trained": trained, "frozen": frozen, "dead_never_trained": dead,
                "not_trained": frozen + dead, "sum": trained + frozen + dead}

    rep = {
        "note": ("요청 ⑩. '추론 파라미터' 가 한 가지가 아니라 셋이다 — loaded(메모리) / "
                 "executed(연산) / effective(출력 기여). 분할기 인코더는 layer4·fc 가 "
                 "loaded 이나 executed 가 아니고(forward 가 직접 호출하지 않는다), "
                 "임베딩은 layer4·fc 가 executed 이나 effective 가 아니다(계산하고 버린다). "
                 "요청서 §2.2 의 '실제 forward 를 타는 부분만' 은 임베딩에서는 effective 를 "
                 "가리킨다 — 임베딩은 전체가 forward 를 탄다."),
        "composition": {
            "segmenter_encoder_WRN101_2": {
                "loaded": seg_enc_all, "executed": seg_enc_used, "effective": seg_enc_used,
                "dead_layer4_fc": seg_enc_dead,
                "evidence": "train_normal_unet.py:113-119 가 layer1~3 만 직접 호출"},
            "segmenter_decoder": {
                "per_category": dec, "mean": dmean, "min": dmin, "max": dmax,
                "coord_channel_contribution": coord_only,
                "note": "좌표는 파라미터가 없다(2채널 concat). 기여는 conv1 입력 2채널분뿐."},
            "segmenter_checkpoint_crosscheck": {
                "checkpoint_total": ck_total, "checkpoint_layer4_fc": ck_enc_dead,
                "checkpoint_buffers_excluded": ck_buffers,
                "expected_total": seg_enc_all + dec["breakfast_box"]["sum"],
                "match": ck_total == seg_enc_all + dec["breakfast_box"]["sum"],
                "note": "state_dict 는 BN 버퍼를 포함하므로 파라미터만 비교한다"},
            "embedding_ResNet101": {
                "loaded": emb_all, "executed": emb_all, "effective": emb_eff,
                "computed_but_discarded_layer4_fc": emb_dead,
                "evidence": "psad.py:58 `_ = self.model(x_t)` 전체 forward · "
                            "psad.py 의 ft = cat([f0,f1,f2]) 에서 layer4 출력 제외"},
        },
        "reconstruction_EAD_S": {**rec, "sum": sum(rec.values())},
        "patch_memory_DINOv3_L16": pm,
        "per_branch": {
            "reconstruction": {"trained": rec["student_PDN"] + rec["autoencoder"],
                               "frozen": rec["teacher_PDN"],
                               "sum": sum(rec.values())},
            "patch_memory": {"trained": 0, "frozen": pm, "sum": pm},
            "composition": {
                "loaded": {"trained": seg_enc_used + dmean, "frozen": emb_all,
                           "dead_never_trained": seg_enc_all - seg_enc_used,
                           "sum": seg_enc_all + dmean + emb_all},
                "executed": {"trained": seg_enc_used + dmean, "frozen": emb_all,
                             "sum": seg_enc_used + dmean + emb_all},
                "effective": {"trained": seg_enc_used + dmean, "frozen": emb_eff,
                              "sum": seg_enc_used + dmean + emb_eff}},
        },
        "totals": {
            "loaded": totals(seg_enc_all, emb_all),
            "executed": totals(seg_enc_used, emb_all),
            "effective": totals(seg_enc_used, emb_eff),
        },
        "frozen_trained_basis": {
            "frozen": ["DINOv3-L/16 (미세조정 없음)", "EAD teacher PDN (공개 가중치)",
                       "ResNet-101 임베딩 (requires_grad=False, 갱신 없음)"],
            "trained": ["EAD student PDN (70k steps)", "EAD autoencoder (70k steps)",
                        "분할기 전체 (pretrained=False, 300 epochs)"],
        },
        "prior_260821_backbone_sum": {"composition": 171.44e6, "note": "정의가 다르다"},
    }

    M = 1e6
    print("\n=== 요청 ⑩ 추론 파라미터 실측 ===\n")
    print(f"{'항목':<34}{'loaded':>12}{'executed':>12}{'effective':>12}")
    print(f"{'분할기 인코더 WRN-101-2':<30}{seg_enc_all/M:>12.2f}"
          f"{seg_enc_used/M:>12.2f}{seg_enc_used/M:>12.2f}   (layer4+fc {seg_enc_dead/M:.2f}M 미호출)")
    print(f"{'분할기 디코더 (범주 평균)':<30}{dmean/M:>12.2f}{dmean/M:>12.2f}{dmean/M:>12.2f}"
          f"   ({dmin/M:.2f}~{dmax/M:.2f}M)")
    print(f"{'임베딩 ResNet-101':<32}{emb_all/M:>12.2f}{emb_all/M:>12.2f}{emb_eff/M:>12.2f}"
          f"   (layer4+fc {emb_dead/M:.2f}M 실행 후 폐기)")
    print(f"{'재구성 EAD-S':<35}{sum(rec.values())/M:>12.2f}"
          f"{sum(rec.values())/M:>12.2f}{sum(rec.values())/M:>12.2f}")
    print(f"{'패치 메모리 DINOv3-L/16':<31}{pm/M:>12.2f}{pm/M:>12.2f}{pm/M:>12.2f}")
    print()
    for k, v in rep["totals"].items():
        d = v["dead_never_trained"]
        print(f"  전체 [{k:<9}]  학습 {v['trained']/M:>6.2f}M · 동결 {v['frozen']/M:>7.2f}M"
              + (f" · 사장 {d/M:>5.2f}M" if d else " " * 14)
              + f" · 합 {v['sum']/M:>7.2f}M   학습 안 되는 것 "
              f"{v['not_trained']/M:>7.2f}M ({v['not_trained']/v['sum']:.0%})")
    cc = rep["composition"]["segmenter_checkpoint_crosscheck"]
    print(f"\n  체크포인트 교차검증(파라미터만): 저장 {ck_total/M:.2f}M vs "
          f"모델 정의 {cc['expected_total']/M:.2f}M -> {'일치' if cc['match'] else '불일치'}"
          f"  [BN 버퍼 {ck_buffers/M:.2f}M 제외]")
    print(f"  (그중 layer4+fc {ck_enc_dead/M:.2f}M 는 저장되지만 실행되지 않는다)")

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n  [saved] {OUT}")


if __name__ == "__main__":
    main()
