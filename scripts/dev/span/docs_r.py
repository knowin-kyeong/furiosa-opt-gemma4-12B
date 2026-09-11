"""docs_r.py -- RULES 10.0r + SOTA afternoon note/draws for the 2026-09-11 afternoon round (run in the repo root)."""
import io


def rd(p):
    return io.open(p, encoding='utf-8').read()


def wr(p, s):
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)


RULES_NOTE = """### 10.0r 2026-09-11 오후 — 동기화의 정체, 순서를 강제하는 도구, 그리고 네 번의 기각 (가장 최신, 여기서 시작할 것)

코드 SOTA는 그대로 **`V273_submit` `331104b`**, 공식 최고 **7.0314**. V273 draw는 오후에 6회 더 뽑았다(6.4623 / 6.6097 / 6.6501 / 6.7763 / 6.5899 / …) — 기록 경신 없음.

**실측으로 확정한 사실 (RESULTS `V284`~`V288`, 메모리 `hardware-span-profiler`)**

1. **0.6.0은 모든 `DmaStore`(scatter 포함) 뒤에 `ExplicitSync`를 넣는다.** PE C 프로그램(`DUMP_PE_PROGRAM=<dir>` → `code_0.c`)에서 이것은
   두 클러스터 master PE 간 IPC다: `sync_intra_chip_cluster(n)`이 상대 클러스터 pe4에 코어 인터럽트로 타임스탬프 n을 push하고,
   `wait_sync_intra_chip_cluster(n)`이 상대의 push를 기다린다. 두 호출은 서로 다른 queue block에 흩어져 있고 양 클러스터 프로그램이 대칭이다.
2. **DMA 명령은 전역 FIFO로 완료된다** — span 로그 8개, 같은 launch 안 DMA 쌍 18,928개 중 나중에 발행된 명령이 먼저 끝난 경우 0건.
   그리고 한 클러스터의 TUC 큐 안에서 **TU pass 뒤에 놓인 DMA 발행은 그 pass가 끝날 때까지 늦어진다**(qkv V1 로드는 pass B 종료 순간 6,375에 발행).
3. **임계 경로** (`scratchpad/crit.py`, V273 warm launch): qkv 103k = **DMA 82k** + 동기화 12k + TU 5k · attn 47k = DMA 30k + **동기화 12k** + TU 4.5k ·
   ffn 289k = DMA 217k + **TU 85k** + 동기화 17k. qkv는 사실상 DMA FIFO 전체가 임계 경로다.
4. **두 클러스터에 쓰는 로드는 한 클러스터 로드보다 1.84× 빠르다** (V284: attn을 한 클러스터로 옮기자 동기화는 사라졌지만 tile0 18.5k → 34.1k, 순 +17.5%).
5. **qkv weight 로드 속도는 슬라이스당 런에 달렸다**: 8행(프로덕션) 505~520 · 16행 612 · 32행 630 B/cycle (V286, span 직접 측정).
   그러나 레이아웃을 바꾸면 스케줄러의 명령 순서가 바뀌어 긴 동기화(16.5~17k)가 V2 로드 앞에 섰다 — V287 +2.9% (12/12).
6. **동기화 길이는 push와 wait 사이의 명령 구성에 달려 있지 않다** (V285: 사이를 비워도 10.4~15.5k, +3.9%).
   **store를 합치면 합친 동기화가 FIFO의 큰 로드 완료에 묶일 수 있다** (V288: rope store 둘 → 하나, 동기화가 31.7k/32.9k이고 두 launch 모두 Q 로드가 끝나는 cycle ±10에 풀림, +8%).

**새 도구 (pod `/root/tk/`, scratchpad)**

- `BEAM_SEARCH_TRACE_DUMP_PATH=<file>` — 스케줄러 노드 이름(`T29 := Dma.StoD(...)`, `T30 := ClusterSync(T29)`); 최종 경로 추출 코드는 scratchpad `ffn_path.txt` 생성부.
- `SCHEDULER_MANUAL_ORDERING_PATH=<file>` — 한 줄에 `T<a> -> T<b>`. **빌드의 모든 커널에 적용되고 캐시로 격리되지 않는다**(없는 노드면 "no producer found").
  ⇒ `one.sh <kernel> <order_file> <tag> <pairs>`: 테스트가 한 커널만 참조하게 해 그 커널만 컴파일하고(fixture 가드 `assert_every_expectation_is_tested` 무력화),
  순서 없는/있는 두 바이너리를 잡마다 번갈아 돌린다. `blocks.py <code_0.c>` — PE 프로그램 queue block 요약.
- span 분석: `map.py`(span↔정적 명령 1:1), `crit.py`(임계 경로와 제약), `rates.py`(명령별 실물 B/cycle). 추가 로드 탐침은 V208 관용구(한 행 TRF staging으로 살림).

**다음 세션에게**

- 가장 큰 미지수이자 비용은 **동기화가 무엇에 묶이는가**다(커널마다 12~40k). V288은 "먼저 발행된 큰 로드의 완료"를 가리키지만 attn 중간 동기화와 h2의 rope 동기화는
  DMA가 놀 때 끝났다. `rs`/base PE 프로그램 블록(`blocks.py`)과 span의 동기화 종료 시점을 대조해 규칙을 세우고, 규칙이 서면 store를 큰 로드 **앞**으로 옮기는 소스 구조를 설계할 것.
- 레이아웃·store 개수를 바꾸는 후보는 **정적 스케줄의 DMA·동기화 순서부터** 비교할 것 — 네 번의 기각(V284·V285·V287·V288)이 전부 "부분 최적화는 맞았는데 순서가 바뀌었다"였다.
- 리더보드(2026-09-11 11시): 1위 #663 7.2836 (91,777 / 39,296 / 269,387), 2위 우리 7.0314, 3위 monkey 7.0170 (ffn 263,908).

"""

p = 'RULES.md'
s = rd(p)
anchor = '### 10.0q 2026-09-11'
assert s.count(anchor) == 1
s = s.replace(anchor, RULES_NOTE + anchor, 1)
s = s.replace('### 10.0q 2026-09-11 — 실물 타임라인을 얻었다: 동기화 대기와 명령 순서 (가장 최신, 여기서 시작할 것)',
              '### 10.0q 2026-09-11 — 실물 타임라인을 얻었다: 동기화 대기와 명령 순서', 1)
wr(p, s)

SOTA_NOTE = """>
> **같은 날 오후 — 동기화의 정체와 순서 강제 도구, 채택 없음.** PE C 프로그램에서 ExplicitSync가 두 클러스터 master PE 간 IPC임을 확인했고,
> DMA가 전역 FIFO임을 span 18,928쌍으로 확정했으며, 스케줄러 순서를 강제하는 환경변수(`SCHEDULER_MANUAL_ORDERING_PATH`)와 한 커널만 도는 A/B 하네스를 만들었다.
> 네 구조 변경은 모두 기각: attn 한 클러스터(V284 +17.5%), 순서 강제(V285 +3.9%), qkv 절반 live 슬라이스(V287 +2.9% — 로드는 20% 빨라졌다),
> rope store 병합(V288 +8% — 합친 동기화가 Q 로드 완료에 묶였다). 자세한 것은 RULES §10.0r.
"""
p = 'SOTA.md'
s = rd(p)
anchor = '> attn launch 분산과 draw 복권의 주성분이다(RULES §10.0q). 이 라운드의 두 구조 변경(V281, V282)은 기각, 채택 없음.\n'
assert s.count(anchor) == 1, 'sota anchor'
s = s.replace(anchor, anchor + SOTA_NOTE, 1)
row_anchor = '| 2026-09-11 01:36~01:42 UTC | (3회) | **`V273_submit` `331104b`** |'
i = s.index(row_anchor)
j = s.index('\n', i) + 1
row = ('| 2026-09-11 04:04~04:14 · 10:25~10:46 UTC | (10회) | `V273_submit` `331104b` | — | — | — | '
       '6.4183 / 6.6584 / 6.2922 / 6.7103 · 6.4623 / 6.6097 / 6.6501 / 6.7763 / 6.5899 / … | 같은 코드 추가 draw — 기록 7.0314를 넘지 못함 |\n')
s = s[:j] + row + s[j:]
wr(p, s)
print('RULES 10.0r + SOTA updated')
