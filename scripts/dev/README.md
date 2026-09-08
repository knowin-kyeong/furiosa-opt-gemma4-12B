# scripts/dev — 스케줄 스크리닝 도구 (채점과 무관, RULES §5 참조)

| 파일 | 용도 |
|---|---|
| `dump_schedules.sh <TAG>` | 세 커널을 `--exact --dump-schedule`로 컴파일해 `target/schedules/<TAG>_<kernel>.json`에 저장하고 makespan 요약 출력 |
| `makespan.py <TAG> [kernel...]` | makespan, context별 busy 합, 최장 노드 8개 |
| `bysrc.py <TAG>` | 소스 라인별 cycle 집계 (병목 위치 찾기) |
| `dmacmp.py <kernel> <TAG>...` | 태그 간 DMA 노드 duration/util 비교 |

원격 pod에서는 `/root/env.sh`를 source한 뒤 저장소 루트에서 실행한다:

```sh
scripts/dev/dump_schedules.sh V12
python3 scripts/dev/bysrc.py V12
```
