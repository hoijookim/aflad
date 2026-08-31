#!/usr/bin/env python3
"""제출 패키지 **커버리지 전수조사** — 원고의 수치 중 패키지가 뒷받침하지 못하는 것.

## 무엇을 하나

원고(docx)의 본문·표·캡션에서 **소수 수치를 전부 뽑아**, 패키지의 JSON 산출물에서
그 값을 찾을 수 있는지 본다. 못 찾으면 "패키지만으로는 검증 불가" 후보다.

## 왜 이 방식인가

"코드가 있나" 를 세는 것은 약하다 — 코드가 있어도 그 코드가 만든 **값**이 없으면
독자가 대조할 수 없다. 반대로 값이 있으면 최소한 대조는 된다.
그래서 **값 기준**으로 센다.

## 판정 규칙

- 0.xxx / 0.xxxx 형태만 본다(AUROC·AP·F1·std). 연도·절 번호·인용은 걸러진다.
- 패키지 JSON 을 문자열로 펼쳐 **반올림 자릿수를 낮춰 가며** 찾는다
  (원고 0.9723 ↔ JSON 0.9723054922575867).
- 못 찾은 값은 **원고 문맥과 함께** 출력한다 — 사람이 판단할 몫이다.
  자동 판정으로 끝내면 오탐을 걸러낼 수 없다.
"""
import json
import re
import sys
from pathlib import Path

R = Path("/workspace/ai-vision-research")
PKG = R / "submission"
OUT = R / "reports/countgd/package_coverage_audit.json"


def pkg_values():
    """패키지 JSON 의 모든 수치를 문자열 집합으로. 반올림 대조를 위해 여러 자릿수로."""
    vals = set()
    files = 0
    for f in PKG.rglob("*.json"):
        try:
            txt = f.read_text()
        except Exception:
            continue
        files += 1
        for m in re.finditer(r"-?\d+\.\d+", txt):
            try:
                v = float(m.group())
            except ValueError:
                continue
            for nd in (2, 3, 4):
                vals.add(f"{round(v, nd):.{nd}f}")
                if 0 < v < 1:                      # 0.9723 ↔ 97.23 표기 대응
                    vals.add(f"{round(v * 100, nd - 2):.{max(nd-2,0)}f}")
    return vals, files


def paper_numbers(docx_path):
    """본문 문단 + 표 셀에서 0.xx~ 형태 수치를 문맥과 함께."""
    import docx
    d = docx.Document(docx_path)
    items = []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        for m in re.finditer(r"(?<![\d.])0\.\d{2,4}(?![\d])", t):
            items.append({"value": m.group(), "where": "본문",
                          "ctx": t[max(0, m.start() - 60):m.end() + 40]})
    for ti, tb in enumerate(d.tables):
        for r_ in tb.rows:
            cells = [c.text.strip() for c in r_.cells]
            for c in cells:
                for m in re.finditer(r"(?<![\d.])0\.\d{2,4}(?![\d])", c):
                    items.append({"value": m.group(), "where": f"표{ti}",
                                  "ctx": " | ".join(x[:18] for x in cells[:5])})
    return items


def main():
    docx_path = sorted(R.glob("docs/AFLAD_MDPI_*full.docx"),
                       key=lambda p: p.stat().st_mtime)[-1]
    print(f"원고: {docx_path.name}")
    vals, nf = pkg_values()
    print(f"패키지 JSON {nf}개에서 수치 {len(vals):,}개 수집\n")

    items = paper_numbers(docx_path)
    uniq = {}
    for it in items:
        uniq.setdefault(it["value"], it)
    print(f"원고 수치 {len(items)}개 (중복 제거 {len(uniq)}개)")

    miss = {v: it for v, it in uniq.items() if v not in vals}
    print(f"패키지에서 못 찾은 값 **{len(miss)}개** "
          f"(커버리지 {1 - len(miss)/len(uniq):.1%})\n")

    by_where = {}
    for v, it in miss.items():
        by_where.setdefault(it["where"], []).append((v, it["ctx"]))
    for w in sorted(by_where, key=lambda x: -len(by_where[x])):
        print(f"── {w} ({len(by_where[w])}개) ──")
        for v, ctx in sorted(by_where[w])[:14]:
            print(f"  {v}   …{ctx[:88]}")
        if len(by_where[w]) > 14:
            print(f"  … 외 {len(by_where[w])-14}개")
        print()

    OUT.write_text(json.dumps({
        "docx": docx_path.name, "pkg_json_files": nf,
        "paper_unique_values": len(uniq), "missing": len(miss),
        "coverage": 1 - len(miss) / len(uniq),
        "missing_detail": {v: it for v, it in miss.items()}}, ensure_ascii=False, indent=2))
    print(f"  [saved] {OUT}")


if __name__ == "__main__":
    main()
