# 저장소 경로 재지정

여기 담긴 스크립트는 **실제로 실행된 그대로**다. 따라서 저장소 루트가
`/workspace/ai-vision-research` 로 하드코딩돼 있다(75개 중 70개). 실행 기록을
사후 편집하지 않으려고 그대로 뒀다.

다른 위치에서 돌리려면 한 줄로 바꾼다:

```bash
grep -rl /workspace/ai-vision-research . | xargs sed -i "s|/workspace/ai-vision-research|$PWD|g"
```

예외: `analysis/reproduce_from_package.py` 는 절대경로가 없다. 패키지만으로 헤드라인을
재계산하는 이 스크립트는 어디서든 그대로 돈다.

```bash
python3 code/analysis/reproduce_from_package.py --pkg .
```
