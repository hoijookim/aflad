#!/usr/bin/env python3
"""제출 패키지 적대적 검수 — 조립이 끝났다고 끝난 게 아니다.

빌드 스크립트는 "복사했다"까지만 보장한다. 이 스크립트는 그다음을 묻는다:
**패키지를 받은 사람이 실제로 검증할 수 있는가?**

검수 항목
  1. 융합 입력 완비    — 세 분기의 test·val 점수가 모두 있는가 (하나만 빠져도 재현 불가)
  2. MANIFEST 무결성   — 기재된 SHA-256 이 실제 파일과 맞는가
  3. README 경로       — 문서가 가리키는 파일이 실재하는가
  4. 코드 실행 가능성  — 파이썬 구문 오류, 셸 구문 오류
  5. 비밀정보         — 공개 레포에 올라가면 안 되는 것
  6. git 추적 가능성   — .gitignore 가 점수 파일을 지우지 않는가 (레포로 올리면 사라진다)
  7. 상태 표시        — 철회된 수치가 현행과 구분되는가
  8. 헤드라인 재현     — reproduce_from_package.py 가 통과하는가

사용: python3 scripts/audit_submission_package.py [--pkg submission]
"""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

R = Path("/workspace/ai-vision-research")
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
SEEDS = (42, 43, 44)


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()[:16]


class Audit:
    def __init__(self):
        self.fail = []

    def check(self, name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            self.fail.append(name)
        return ok


def c1_fusion_inputs(P, A):
    """세 분기의 test·val 이 5범주 × 3시드 전부 있어야 한다."""
    S = P / "results/per_image_scores"
    need = []
    for c in CATS:
        for s in SEEDS:
            need += [
                S / ("ead_s_seed42_repro" if s == 42 else "ead_s") / f"scores_{c}_seed{s}.npz",
                S / "ead_s_val" / f"val_good_{c}_seed{s}.npz",
                S / "pc_branch" / f"pc_L17_{c}_seed{s}.npz",
                S / "psad_composition" / f"psad_scores_{c}_seed{s}_hc_tta_merge_test.npz",
                S / "psad_composition" / f"psad_scores_{c}_seed{s}_hc_tta_merge_val.npz",
            ]
    miss = [p for p in need if not p.exists()]
    A.check("융합 입력 완비", not miss,
            f"{len(need)-len(miss)}/{len(need)}" + (f" · 결측 예 {miss[0].name}" if miss else ""))


def c2_manifest(P, A):
    m = P / "MANIFEST.md"
    if not A.check("MANIFEST 존재", m.exists()):
        return
    rows = re.findall(r"\| `([^`]+)` \| `[^`]+` \| [\d,]+ \| `([0-9a-f]+)` \|", m.read_text())
    bad = [d for d, h in rows if not (P / d).exists() or sha(P / d) != h]
    A.check("MANIFEST 해시 무결성", not bad, f"{len(rows)-len(bad)}/{len(rows)} 일치")


def c3_readme_paths(P, A):
    rp = P / "README.md"
    if not A.check("README 존재", rp.exists()):
        return
    refs = {r for r in re.findall(r"`((?:code|results|docs)/[^`]+)`", rp.read_text())}
    miss = []
    for r in refs:
        r = r.split()[0]                      # 명령줄 인자 제거
        if "*" in r:                          # 글롭은 하나라도 맞으면 통과
            base = Path(r)
            if not list((P / base.parent).glob(base.name)):
                miss.append(r)
        elif not (P / r).exists():
            miss.append(r)
    A.check("README 가 가리키는 경로 실재", not miss,
            f"{len(refs)-len(miss)}/{len(refs)}" + (f" · 결측 {miss[:3]}" if miss else ""))


def c4_code_runs(P, A):
    py = list((P / "code").rglob("*.py"))
    bad = []
    for p in py:
        try:
            ast.parse(p.read_text())
        except SyntaxError as e:
            bad.append(f"{p.name}:{e.lineno}")
    A.check("파이썬 구문", not bad, f"{len(py)-len(bad)}/{len(py)}" + (f" · {bad}" if bad else ""))
    sh = list((P / "code").rglob("*.sh"))
    bad = [p.name for p in sh if subprocess.run(["bash", "-n", str(p)],
                                                capture_output=True).returncode]
    A.check("셸 구문", not bad, f"{len(sh)-len(bad)}/{len(sh)}" + (f" · {bad}" if bad else ""))


def c5_secrets(P, A):
    pat = re.compile(r"webhook|discord\.com/api|api[_-]?key|secret|password|AKIA[0-9A-Z]{16}", re.I)
    hits = []
    for p in P.rglob("*"):
        # 검수 스크립트 자신은 제외한다 — 탐지 패턴 문자열이 그대로 걸린다.
        if (not p.is_file() or p.suffix in (".npz", ".pt")
                or p.name in ("MANIFEST.md", Path(__file__).name)):
            continue
        for i, ln in enumerate(p.read_text(errors="replace").split("\n"), 1):
            if pat.search(ln):
                hits.append(f"{p.relative_to(P)}:{i}")
    A.check("비밀정보 없음", not hits, f"{hits[:3]}" if hits else "")


def c6_git_tracked(P, A):
    """*.npz 를 무시하는 저장소가 많다. 그러면 레포로 올라간 패키지는 재현이 안 된다."""
    npz = list((P / "results/per_image_scores").rglob("*.npz"))
    if not A.check("점수 파일 존재", bool(npz)):
        return
    out = subprocess.run(["git", "-C", str(R), "check-ignore"] + [str(p) for p in npz[:50]],
                         capture_output=True, text=True)
    ignored = [l for l in out.stdout.split("\n") if l.strip()]
    A.check("git 이 점수 파일을 추적", not ignored,
            f"{len(ignored)}개가 .gitignore 에 걸림" if ignored else f"표본 {min(50,len(npz))}개 통과")


def c7_status_flags(P, A):
    f = P / "results/tables/table5_3seed_42_43_44.json"
    if not A.check("기준선 표 존재", f.exists()):
        return
    m = json.load(open(f))["methods"]
    # 철회된 수치(0.98xx 대)가 상태 표시 없이 있으면 현행과 구분되지 않는다
    unflagged = [k for k, v in m.items() if v["LS"] > 0.975 and "status" not in v
                 and "주 설정" not in k]
    A.check("철회 수치에 상태 표시", not unflagged, f"미표시 {unflagged}" if unflagged else "")


def c8_reproduce(P, A):
    sc = P / "code/analysis/reproduce_from_package.py"
    if not A.check("재현 스크립트 존재", sc.exists()):
        return
    r = subprocess.run([sys.executable, str(sc), "--pkg", str(P)], capture_output=True, text=True)
    tail = [l for l in r.stdout.split("\n") if "PASS" in l or "FAIL" in l or "재현" in l]
    A.check("헤드라인·기준선 재현", r.returncode == 0, f"{len(tail)}행 검증")
    for l in tail:
        print(f"        {l.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=str(R / "submission"))
    P = Path(ap.parse_args().pkg)
    A = Audit()
    print(f"제출 패키지 검수 — {P}\n")
    for fn in (c1_fusion_inputs, c2_manifest, c3_readme_paths, c4_code_runs,
               c5_secrets, c6_git_tracked, c7_status_flags, c8_reproduce):
        fn(P, A)
    print(f"\n{'전 항목 통과' if not A.fail else f'{len(A.fail)}건 실패: ' + ', '.join(A.fail)}")
    return 0 if not A.fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
