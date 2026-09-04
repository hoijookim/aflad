#!/usr/bin/env python3
"""패키지 커버리지 **2축** — 수치가 아니라 **근거 파일**이 들어 있는가.

## 왜 필요한가 — 1축이 못 보는 것

`audit_package_coverage.py` 는 원고의 수치를 패키지 JSON 에서 찾는다. 그런데 논문
세션이 요청 ⑯ 에서 정확히 짚었다:

> "원고가 근거로 삼는 파일이 패키지에 있는지는 수치 감사가 못 본다
>  (24 라는 숫자는 어디에나 있다)."

§3.1.2 의 **"학습·검증 데이터만 쓰는 기준을 미리 정해 두고 24개 계층 전부를 비교해
정했다"** 가 그 예다. 이 문장의 근거는 사전등록 문서와 전수 스윕 결과인데, `24` 라는
숫자는 아무 데서나 매칭되므로 1축은 **커버된 것으로 오판한다.** 실제로 두 파일 다
패키지에 없었다.

**주장의 종류가 다르면 감사도 달라야 한다** — 수치 주장은 값으로, 절차 주장은
**그 절차를 기록한 파일**로 검증된다.

## 무엇을 보나

1. **사전등록 문서** — 저장소의 `PREREG_*` 가 패키지에 있는가.
   설계 선택을 "미리 정한 기준으로 했다" 고 쓰는 이상, 그 기준서가 없으면 확인 불가다.
2. **전이 폐포** — 패키지 안의 문서가 **참조하는 파일**이 또 패키지에 있는가.
   사전등록서가 결과 JSON 을 가리키는데 그 JSON 이 없으면 반쪽이다.
3. **원고가 이름으로 부르는 산출물** — docx 에 `*.json` / `*.py` 가 적혀 있으면 그 파일.

없는 것을 나열만 한다. **무엇을 담을지는 사람이 정한다** — 기각된 탐색까지 전부 넣으면
패키지가 저장소가 되어 버린다.
"""
import json
import re
import sys
from pathlib import Path

R = Path("/workspace/ai-vision-research")
PKG = R / "submission"
OUT = R / "reports/countgd/evidence_file_audit.json"


def pkg_basenames():
    return {p.name for p in PKG.rglob("*") if p.is_file()}


def main():
    have = pkg_basenames()
    rep, gaps = {}, []

    # ── 1. 사전등록 문서 ──────────────────────────────────────────────
    pregs = sorted({p for d in ("docs", "reports/countgd")
                    for p in (R / d).glob("PREREG_*.md")}, key=lambda p: p.name)
    miss1 = [p for p in pregs if p.name not in have]
    rep["prereg"] = {"total": len(pregs), "missing": [p.name for p in miss1]}
    print(f"\n=== 1. 사전등록 문서 {len(pregs)}개 중 패키지에 없는 것 {len(miss1)}개 ===")
    for p in miss1:
        # 첫 제목 줄을 함께 보여 준다 — 담을지 말지 판단 재료
        head = next((l.lstrip("# ").strip() for l in p.read_text(encoding="utf-8").splitlines()
                     if l.startswith("#")), "")
        print(f"  {p.name:<44} {head[:52]}")
        gaps.append(p.name)

    # ── 2. 패키지 문서가 참조하는 파일 (전이 폐포) ────────────────────
    refs = {}
    for d in PKG.rglob("*.md"):
        t = d.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"[\w/\.-]+\.(?:json|py|sh|npz|md)", t):
            n = Path(m.group()).name
            if n not in have:
                refs.setdefault(n, set()).add(d.relative_to(PKG).as_posix())
    # 저장소에 실재하는 것만 — 없는 이름은 오탈자·예시라 의미 없다.
    # 이름마다 rglob 하면(특히 R.rglob 은 15GB 전체를 훑는다) 끝나지 않는다.
    # **인덱스를 한 번만** 만들어 집합 조회로 바꾼다.
    idx = set()
    for d in ("reports", "scripts", "docs", "results"):
        dd = R / d
        if dd.is_dir():
            idx |= {f.name for f in dd.rglob("*") if f.is_file()}
    real = {n: sorted(v) for n, v in refs.items() if n in idx}
    rep["referenced_but_absent"] = real
    print(f"\n=== 2. 패키지 문서가 참조하는데 패키지에 없는 파일 {len(real)}개 ===")
    for n, where in sorted(real.items())[:20]:
        print(f"  {n:<44} <- {where[0]}")
        gaps.append(n)
    if len(real) > 20:
        print(f"  … 외 {len(real)-20}개")

    # ── 3. 원고가 이름으로 부르는 산출물 ──────────────────────────────
    try:
        import docx
        dx = sorted(R.glob("docs/AFLAD_MDPI_*full.docx"),
                    key=lambda p: p.stat().st_mtime)[-1]
        txt = "\n".join(p.text for p in docx.Document(dx).paragraphs)
        named = {Path(m.group()).name for m in
                 re.finditer(r"[\w/\.-]+\.(?:json|py|sh)", txt)}
        miss3 = sorted(n for n in named if n not in have)
        rep["named_in_manuscript"] = {"docx": dx.name, "missing": miss3}
        print(f"\n=== 3. 원고가 이름으로 부르는데 패키지에 없는 것 {len(miss3)}개 ===")
        for n in miss3:
            print(f"  {n}")
            gaps.append(n)
    except Exception as e:
        rep["named_in_manuscript"] = {"error": str(e)}
        print(f"\n=== 3. 건너뜀: {e}")

    uniq = sorted(set(gaps))
    rep["gap_count"] = len(uniq)
    rep["gaps"] = uniq
    print(f"\n  ==> 근거 파일 공백 {len(uniq)}개 (중복 제거)")
    print(f"      담을지 말지는 사람이 정한다 — 기각된 탐색까지 전부 넣으면")
    print(f"      패키지가 저장소가 된다.")
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [saved] {OUT}")


if __name__ == "__main__":
    main()
