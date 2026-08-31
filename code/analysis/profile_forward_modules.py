#!/usr/bin/env python3
"""요청 ⑩ 보강 — `executed` / `effective` 를 **실측**으로 확정한다.

## 왜

회신서가 두 주장을 **코드 판독**으로만 세웠다:

  (가) 분할기 인코더의 `enc.layer4`·`enc.fc` 는 **실행되지 않는다**
       (`CNNSegmenter.forward` 가 layer1~3 만 직접 호출)
  (나) 임베딩 `resnet101` 은 **전체가 실행된다**
       (`Encoder.extract_ft` 가 `_ = self.model(x_t)`), 다만 layer4 출력은 버려진다

(가)는 3-f 검수의 비트 단위 증거가 뒷받침한다(두 독립 학습 실행의 `enc.layer4.*` 가
동일하고 BN 이 초기값 그대로 = 순전파 미실행). (나)는 세 줄 판독이 전부다.

**forward hook 은 모듈의 forward 가 실제로 돌 때만 발화한다.** 그러므로 훅을 걸고
진짜 입력을 한 번 통과시키면 실행 여부가 직접 관측된다. 판독이 아니라 측정이다.

## 무엇을 보나

각 모듈에 훅을 걸고 forward 1회 실행 → 발화 여부와 출력 shape 을 기록한다.
그리고 발화한/안 한 모듈의 파라미터 수를 세어 회신서 수치와 대조한다.

기대(회신서 주장이 맞다면):

```
분할기   enc.layer1~3  발화       enc.layer4·fc  미발화
임베딩   layer1~4      발화       fc             발화
```

(나)의 임베딩에서 layer4 가 발화하되 그 출력이 최종 `ft` 에 안 들어가는 것은
`Encoder.forward` 가 `ft = cat([f0,f1,f2])` 직후 `return ft` 하는 것으로 이미 확정돼
있다(psad.py:80-82). 여기서는 **실행되는가** 만 잰다.
"""
import json
import sys
from pathlib import Path

import torch

R = Path("/workspace/ai-vision-research")
SHIMS = R / "scripts/psad_rebuild/_shims"
PSAD = R / "external/PSAD_official"
OUT = R / "reports/countgd/forward_profile.json"


def hook_fired(model, names, run):
    """모듈별 발화 여부를 관측한다. 훅은 forward 가 실제로 돌 때만 불린다."""
    fired, shapes, handles = {n: False for n in names}, {}, []

    def mk(n):
        def f(m, i, o):
            fired[n] = True
            shapes[n] = tuple(o.shape) if torch.is_tensor(o) else type(o).__name__
        return f

    mods = dict(model.named_modules())
    for n in names:
        if n not in mods:
            fired[n] = None            # 모듈 자체가 없음 — 미발화와 구분한다
            continue
        handles.append(mods[n].register_forward_hook(mk(n)))
    try:
        run()
    finally:
        for h in handles:
            h.remove()
    return fired, shapes


def nparam(model, prefixes):
    return int(sum(p.numel() for k, p in model.named_parameters()
                   if any(k.startswith(x) for x in prefixes)))


def main():
    sys.path.insert(0, str(PSAD))
    sys.path.append(str(SHIMS))          # apex 스텁
    rep = {}

    # ── (가) 분할기 CNNSegmenter ──────────────────────────────────────
    import train_normal_unet as T
    seg = T.CNNSegmenter(num_cls=7, use_coord=True, pretrained=False).eval()
    x = torch.randn(1, 3, 256, 256)
    coord = torch.randn(1, 2, 256, 256)
    names = [f"enc.layer{i}" for i in (1, 2, 3, 4)] + ["enc.fc", "conv1", "conv2", "conv3"]
    with torch.no_grad():
        fired, shapes = hook_fired(seg, names, lambda: seg(x, coord))
    rep["segmenter"] = {
        "fired": fired, "out_shapes": {k: str(v) for k, v in shapes.items()},
        "params_executed": nparam(seg, ["enc.conv1.", "enc.bn1.", "enc.layer1.",
                                        "enc.layer2.", "enc.layer3.",
                                        "conv1.", "conv2.", "conv3."]),
        "params_not_executed": nparam(seg, ["enc.layer4.", "enc.fc."])}

    print("\n=== (가) 분할기 CNNSegmenter — forward 1회 ===")
    for n in names:
        f = fired[n]
        tag = "발화" if f else ("모듈 없음" if f is None else "**미발화**")
        print(f"  {n:<14}{tag:<12}{shapes.get(n, '')}")
    print(f"  실행 파라미터 {rep['segmenter']['params_executed']/1e6:.2f}M · "
          f"미실행 {rep['segmenter']['params_not_executed']/1e6:.2f}M")

    # ── (나) 임베딩 psad.Encoder ──────────────────────────────────────
    import os
    os.chdir(PSAD)
    import psad as P
    enc = P.Encoder().eval()
    names2 = [f"model.layer{i}" for i in (1, 2, 3, 4)] + ["model.fc", "model.avgpool"]
    with torch.no_grad():
        fired2, shapes2 = hook_fired(enc, names2, lambda: enc(torch.randn(1, 3, 256, 256)))
    rep["embedding"] = {
        "fired": fired2, "out_shapes": {k: str(v) for k, v in shapes2.items()},
        "params_executed": nparam(enc, ["model."]),
        "params_effective": nparam(enc, ["model.conv1.", "model.bn1.", "model.layer1.",
                                         "model.layer2.", "model.layer3."]),
        "params_executed_but_discarded": nparam(enc, ["model.layer4.", "model.fc."])}

    print("\n=== (나) 임베딩 psad.Encoder — forward 1회 ===")
    for n in names2:
        f = fired2[n]
        tag = "발화" if f else ("모듈 없음" if f is None else "**미발화**")
        print(f"  {n:<16}{tag:<12}{shapes2.get(n, '')}")
    e = rep["embedding"]
    print(f"  실행 {e['params_executed']/1e6:.2f}M · 그중 출력 기여 "
          f"{e['params_effective']/1e6:.2f}M · 실행 후 폐기 "
          f"{e['params_executed_but_discarded']/1e6:.2f}M")

    # ── 판정 ──────────────────────────────────────────────────────────
    s, m = rep["segmenter"]["fired"], rep["embedding"]["fired"]
    c1 = all(s[f"enc.layer{i}"] for i in (1, 2, 3)) and not s["enc.layer4"] and not s["enc.fc"]
    c2 = all(m[f"model.layer{i}"] for i in (1, 2, 3, 4)) and m["model.fc"]
    rep["verdict"] = {
        "segmenter_layer4_fc_not_executed": bool(c1),
        "embedding_full_forward_executed": bool(c2),
        "both": bool(c1 and c2)}
    print(f"\n=== 판정 ===")
    print(f"  (가) 분할기 layer4·fc 미실행           {'PASS' if c1 else 'FAIL'}")
    print(f"  (나) 임베딩 전체 forward 실행          {'PASS' if c2 else 'FAIL'}")
    print(f"  ==> 회신서의 코드 판독이 실측으로 {'확정됨' if c1 and c2 else '반박됨'}")

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"  [saved] {OUT}")
    sys.exit(0 if (c1 and c2) else 1)


if __name__ == "__main__":
    main()
