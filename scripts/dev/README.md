# scripts/dev — 스케줄 스크리닝 도구 (채점과 무관, RULES §5 참조)

| 파일 | 용도 |
|---|---|
| `dump_schedules.sh <TAG>` | 세 커널을 `--exact --dump-schedule`로 컴파일해 `target/schedules/<TAG>_<kernel>.json`에 저장하고 makespan 요약 출력 |
| `makespan.py <TAG> [kernel...]` | makespan, context별 busy 합, 최장 노드 8개 |
| `bysrc.py <TAG>` | 소스 라인별 cycle 집계 (병목 위치 찾기) |
| `dmacmp.py <kernel> <TAG>...` | 태그 간 DMA 노드 duration/util 비교 |
| `gaps.py <TAG> [kernel...]` | DMA 엔진 유휴 구간 상위 12개와 그 구멍을 끝내는 명령의 소스 줄 (V253·V260의 구멍을 찾은 도구) |
| `census.py <TAG> [kernel...]` | book DMA 규칙 census: DMA 컨텍스트를 가진 tensor-unit pass, 명령이 여럿인 DMA 소스 줄, switch·최장·꼬리 노드 |
| `build.sh <branch> <tag> [repo]` | **pod:** `origin/<branch>`를 checkout하고 **release** 테스트 바이너리를 빌드. `arena.sh`가 올리는 것은 release다 — `--no-run`만 쓰면 옛 바이너리가 측정된다 |
| `pairjobs.sh <repo> <tag> <kernel> <base> <test> <N>` | **pod:** 짝비교 Arena 잡 N개 — 잡마다 두 변형의 median·차이·PASS/FAIL 수, 끝에 부호검정 p. **FAIL이 있는 잡의 시간은 믿지 않는다** |
| `multijobs.sh <repo> <tag> <kernel> <base> <variant...>` | **pod:** pairjobs.sh의 다변형판 — 한 잡 안에서 여러 변형을 base와 짝비교 (N은 환경변수, 기본 16). 잡마다 변형별 차이, 끝에 변형별 부호검정 p |
| `draw.sh <branch> <N> [srcdir]` | **pod:** 제출 전용 clone(기본 `/root/lab3`)에서 직렬 draw N회. 도는 동안 그 clone에서 checkout 금지 |

원격 pod에서는 `/root/env.sh`를 source한 뒤 저장소 루트에서 실행한다:

```sh
scripts/dev/dump_schedules.sh V12
python3 scripts/dev/bysrc.py V12
```

pod clone 역할(2026-09-11): `/root/lab`·`/root/lab2` = 짝비교 실험(배치 중에는 그 clone에서 빌드 금지 — 잡이 바뀐
바이너리를 올린다), `/root/lab3` = 제출 검증·draw, `/root/furiosa-opt-gemma4-12B` = overnight driver. pod에는 GitHub
push 자격이 없으니 로컬에서 `git fetch ssh://root@<pod>:<port>/root/lab <branch>:<branch>` 후 push한다.
