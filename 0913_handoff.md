# 세션 이관 문서 — 2026-09-13 (KST 03:15 · UTC 09-12 18:15 기준)

> **추가 (2026-09-13 UTC 00:20):** 이 문서의 §1~§2 상태는 지났다. **RULES §10.0x부터 읽는다.**
> - draw 대상: `V378_submit` = V368(V313 대칭 attn) + V369(qkv `÷rms` 제거) + V371(qkv x 한 조각, Stage 1 전용).
> - 판정 기준: attn · qkv에 draw 꼬리 지표 추가(`scripts/dev/paired_tails.py`).
> - 새로 닫힌 것: V373(ffn down 정렬 청크, 컴파일러 ICE), V372(attn 로컬 RMS, 꼬리 악화).

새 세션은 **이 문서 → RULES.md §10.0w → RESULTS.md(V361~V367 행) → SOTA.md 첫 항목** 순서로 읽는다.
문서 최신본은 브랜치 `V236_sweep_attnout_tiles`에 있다(origin push 완료). 이 문서가 담는 것:

1. 현재 상태
2. 코드 SOTA 사슬
3. 이번 세션 실험
4. 새로 확인한 사실
5. 커널별 임계 경로
6. 닫힌 길과 남은 후보
7. 인프라 · 절차
8. 재개 체크리스트

---

## 1. 현재 상태

| 항목 | 값 |
|---|---|
| 공식 순위 | **2위 / 21팀** — Goat Chovy #1557 **7.4197** (`a9c7c0d6`, V313_submit draw: qkv 90,189 · attn 38,347 · ffn 265,737) |
| 1위 | vinxst **7.4769** (`97bbe09c`: 89,174 · **33,437** · 301,213) — 우리보다 +0.77%. attn이 구조적으로 앞선다(RULES §10.0u) |
| 3위 | Participant #663 **7.3402** (`0acab4e3`: 92,560 · 36,768 · 278,919). 4위 monkey ≈7.02 |
| 코드 SOTA | **`V367_submit`** (d94d994) — Arena 25/25 ×2 (job 23353 + rerun) |
| draw | **V367_submit 배치 1 진행 중**(UTC 17:53 시작, `drawchain_follow.sh V367_submit 1 8` = 12회 × 8배치): 지금까지 7.1182 · 6.8743 |
| 직전 draw | V360_submit 12회(UTC 16:04~17:53) 평균 6.894 · 최고 **7.1385**(`e3540caa`: 87,932 · 42,746 · 274,556) · V350_submit 12회 최고 7.1187 |
| pod에서 도는 것 | **draw 체인뿐**(follow PID 616619 → drawloop2 PID 617982). 실험 체인 · Arena 잡 없음 |

점수 = 세 커널 speedup의 기하평균. baseline cycle은 qkv 250,514 · attn 404,633 · ffn 3,703,473.
공식 점수는 같은 코드로도 draw마다 σ≈3% 흔들리고, 리더보드에는 최고값만 남는다.

---

## 2. 코드 SOTA 사슬

| 제출 브랜치 | 추가된 것 | 짝비교 |
|---|---|---|
| `V313_submit` | 공식 7.4197을 낸 코드 | — |
| `V340_submit` | attn 비균등 타일(tile1은 클러스터 0 전용) | 24/32 −1.2% |
| `V348_submit` | ffn down x 조각을 Lane에 두고 pass A 안에서 합침 | 16/16 −1.27% |
| `V350_submit` | + geglu hi/lo store 하나 | 16/16, 합산 −2.28% |
| `V360_submit` | qkv head norm 셋 · RoPE 결과 pass 둘을 `commit_cast` | 25/32 −1.6% |
| **`V367_submit`** | **ffn 꼬리 곱 `down_global × out_scale`을 post-FF norm에 접음(V366 T1)** | **12/16 −0.46%** |

- **제출 브랜치 규칙:** `src/device/`와 `src/ops.rs`의 함수 본문만 바꾼다. 새 함수를 추가하고 본문의 호출만 교체한다.
- **채택 절차:**
  1. 실험 브랜치에서 짝비교 16잡(한 잡 안에서 arm 교대, 분 단위 회전, 부호검정)
  2. 제출 브랜치 작성
  3. `subverify.sh`(Arena 잡 + `rngd rerun` 모두 25/25)
  4. draw 전환(§7)

---

## 3. 이번 세션 실험 (09-12 밤 ~ 09-13 새벽)

| 실험 | 커널 | 내용 | 정적 makespan | 실물 짝비교 | 판정 |
|---|---|---|---|---|---|
| V352y | attn | 최종 pass `commit_cast`, 세 번째 배치 | — | 6/16 +0.17% → 합산 24/48 | 기각(중립) |
| V361 | qkv | RoPE를 Vector Engine pair mode로(출력 반쪽마다 pass 하나) | +821 | 0/16 +5.52% | 기각 |
| V362 | qkv | RoPE 행을 on-chip 인덱스로 두 클러스터에 `dma_gather_unscaled` | −204 | 두 배치 합산 11/32 ≈ +0.5% | 기각(중립) |
| V363 | qkv | HBM 인덱스(head마다 한 칸)를 store 하나로 + scaled gather 둘 | +856 | 6/16 +0.89% | 기각 |
| V364 | qkv | x 경로 첫 pass를 Q weight에 의존시켜(`FmaF(gate, 0, x)`) Q 로드를 맨 앞으로 | +1,057 | 5/16 +0.63% | 기각 |
| V365 | ffn | 순서 강제로 down_scale · down1 로드를 geglu store 앞에(두 바이너리 인접 잡 교대) | d1 +883 | 0/8 +3.61% | 기각 |
| **V366** | ffn | 꼬리 곱을 post-FF norm으로: g를 앞에서 한 번 만들고, mean-square는 `x·g`, rms는 g로 나눔 | **−401** | **12/16 −1,234 (−0.46%), 중앙값 −1,645** | **채택** |
| V367_submit | 제출 | V360_submit + V366 | — | Arena 25/25 ×2 | 코드 SOTA |

- 모든 실험의 정확도는 PASS다. 변형이 첫 launch인 잡으로 따로 확인했다.
- 정적 makespan의 부호는 V361 · V363 · V364 · V365 · V366에서 실물과 같았다. V362는 정적 −204, 실물은 중립이었다.
- ⇒ 구조 · 순서 후보는 컴파일 덤프(`cargo furiosa-opt compile ops::<fn> --exact --dump-schedule`)로 먼저 거른다.
- 행별 상세와 로그 위치는 RESULTS.md, 서사는 RULES §10.0w.

---

## 4. 새로 확인한 컴파일러 · 하드웨어 사실

**Vector Engine pair mode (V361):**
- 한 노드에 group VRF는 하나만 쓸 수 있다(`mir: this node needs more slots than a VE pass has`). 둘이면 ALU를 나눈 두 노드로 쪼갠다.
- resize된 축(`Ds = 128 # 256`)은 fetch에서 `/` `%`로 쪼갤 수 없다(`visa: Fetch: a live input axis is left unread`).
  - 크기 1 타일(`tile::<m![Ds / 128], 1, …>`)로 live 축을 `Ds % 128`로 만든다.
- 런타임 DM 입력 둘을 `begin_interleaved`로 섞을 수 없다(`lir: … only one main fetch base can be overridden`).
  - `unsafe { view().reshape() }`로 `[half, …]`를 만들고 반쪽 축을 Time 최내측에 둔다.

**Commit · gather:**
- Commit Adapter의 valid 바이트는 8/16/24/32 B만 된다(i32 한 개 = 4 B는 `m![1 # 2]`로).
- `dma_gather_unscaled`(DmTensor 인덱스)는 lowering에서 **호출마다 인덱스 store + ExplicitSync + scaled gather**가 된다.
- 인덱스에 head 축이 있는 scaled gather는 head 슬라이스마다 정확히 쓴다. V50의 막힌 경우는 HBM 값 하나짜리 인덱스였다.

**API 형태(std 0.6.0):**
- ternary 피연산자는 `(f32, f32)` · `(&Vrf, f32)` · `(Stash, f32)`뿐이다. VRF 둘을 받는 `FmaF`는 없다.
- `Branched::rf(TagGuard, …)` · `TagGuard::matches([BitReq; 4])`가 있다(커널에서 미사용).
- FetchCast는 f8e4m3→f32 · f8e5m2→f32 · bf16→f32 · f32→bf16이다. `fetch_table_lookup` 뒤에 `fetch_cast`를 둘 수 있다.
- `DmTensorView::reshape`(unsafe, 재묶음)는 있고, `dma_scatter`는 DmTensor 전용이다(뷰 불가).

**스케줄 · 하드웨어:**
- 순서 강제(`SCHEDULER_MANUAL_ORDERING_PATH`, `T<a> -> T<b>`)의 이름은 **빔 추적의 노드 이름**이고, 스케줄 JSON의 tensor id와 다르다.
  - 이름 얻기: `BEAM_SEARCH_TRACE_DUMP_PATH` + `/root/tk/beampath.py`.
  - 순서를 반영하려면 커널 캐시(`target/furiosa-opt/.../*ops::<kernel>.*`)를 지워야 한다. `.bin` md5로 반영 여부를 확인한다.
- ExplicitSync 중에도 이미 발행된 DMA는 돈다. 그러나 store 앞에 로드를 몰아 동기화 창을 채우면 store만 늦어진다(V365 +3.6%).
- x 경로 TU pass를 로드 뒤로 보내도 TUC 대기는 다음 로드들 앞으로 옮겨 갈 뿐이고, 겹친 로드는 HBM 대역폭을 나눈다(V364).
- fixture의 f8 weight는 exponent band로 만든 코드라 NaN 코드가 없다(`scripts/fixture_prng.py`).

메모리 파일로도 남겼다: `vector-pair-mode-compile-limits` · `tuc-queue-blocks-dma` · `third-chain-and-leaderboard-facts`.

---

## 5. 커널별 실물 임계 경로 (warm launch, 클러스터 0 span)

도구는 `scripts/dev/span/crit.py`다. `ssh pod 'python3 - <log> "<label>" 1 <schedule.json>' < scripts/dev/span/crit.py` 형태로 쓴다.

### qkv — 88.6k (V355b_r3 cq#1 = V360/V367 qkv)

- **DMA FIFO 72k**
  - weight: Q 27.5k · K 14.0k · V 13.8k
  - 작은 명령 ~20k: x · rms 로드, norm scale/weight 로드, rope gather · store, cs 로드, k scatter
- rope sin store의 ExplicitSync 3.4k, rope TU 꼬리 ~6.6k.
- Q 로드 발행이 9.5k로 늦다. 머리의 x 경로 pass들이 PE 순서에서 작은 로드들 사이에 끼어 FIFO 유휴가 ~3.8k다.
- V 경로는 81.2k에 끝나고, rope/k 경로가 끝을 정한다.

### attn — 45.2k (V352y c#1 ≈ V360/V367 attn)

- tile0 로드 19.2k → contraction0 3.4k → store 1.0k
- → tile1 로드 10.3k(클러스터 0 단독, 305 B/cycle) → contraction1 4.6k
- → 꼬리 행 DM→DM 1.9k → norm 로드 1.1k → 꼬리 pass 1.9k → store 0.9k

### ffn — 266.4k (V366_r3 t1#1 = V367 ffn)

- geglu store 둘 → **ExplicitSync 6.5k**(클러스터 1 지연)
- → inv_s reload 1.0k
- → LUT 테이블 1.9k → **down1 19.4k** → 테이블 2.0k → **down2 18.6k** → 테이블 2.8k → **down3 13.9k**
- → pass A 5.2k → pass B 2.3k
- → down store 2.1k → sync 1.0k → reload 1.1k
- → norm 꼬리 2.9k → store 0.9k
- LUT 테이블 로드는 LUT pass마다 컴파일러가 낸다.

---

## 6. 닫힌 길과 남은 후보

**닫힌 길(다시 하지 말 것):**

- **qkv RoPE staging 대안:**
  - store 하나(V288) · 순서(V291b)
  - pair mode(V361) · unscaled gather(V362) · 공유 HBM 인덱스(V363)
  - 생산(gather 둘 + store 둘 + 로드 하나)이 국소 최적이다.
- **qkv 머리:**
  - 소스 순서(V319)
  - 의존성 게이트(V364)
  - 융합 norm(V353 fn +1.9%). fc −0.9%는 cq와 비가산.
- **ffn:**
  - 동기화 창 채우기(V365)
  - down 한 명령 로드(V305 +5%)
  - up/gate 한 버퍼(V359 +5.8%)
  - 20행 이상 down 타일: pass B scale VRF가 20행이면 9.6 KB > 8 KB
  - 5타일 · 3타일(V314)
- **attn:**
  - 클러스터별 norm · 스칼라 병합(V255 +4.8%, V281)
  - 한 클러스터(V284 +17.5%)
  - 꼬리 행 HBM store(V351 +7%)
  - sub vector chain → VRF(V352 b 값 틀림)
  - 최종 pass `commit_cast`(V352 c 중립)

**남은 후보(모두 작다):**

- **ffn O1b:** geglu 출력을 bf16 store 하나로 두고 down 레이아웃에서 스케일 · hi/lo를 한다.
  - inv_s store · reload · ring-256 switch가 사라진다. T1 뒤 사슬 위 몫 ≈ −1k.
  - 다만 hi/lo pass가 512 슬라이스 × 1,920원소로 커진다. 상세는 `0912_ffn_gap_analysis.md`.
- **attn D:** 256 B 정렬 reducing 레이아웃. −0.2~−0.5k, 패딩 쓰레기 NaN(V346) · 패딩 슬라이스 ICE(V296) 위험. 상세는 `0912_attn_gap_analysis.md`.
- **qkv:** 머리 TUC 대기는 x 경로 pass 수를 줄여야만 준다. V353 fc를 V367 위에서 다시 잴 수는 있다(기대 낮음).

**전망:**
- LB 8.0에는 7.4197 대비 +7.8%가 필요하다.
- 남은 코드 후보는 커널 하나에 0.5% 안팎이다.
- 단기에는 V367 draw가 기록 갱신 가능성이 가장 크다. V360 draw 최고 7.1385, 공식 최고는 V313 draw 7.4197.

---

## 7. 인프라와 절차

**pod:**
- 접속: `ssh -p 41008 -i ~/.ssh/id_ed25519 root@213.192.2.99`
- 환경: `. /root/env.sh`, cargo-furiosa-opt 0.6.0 고정.

**lab:**

| lab | 브랜치 · 용도 |
|---|---|
| lab | V360_submit(정적 스크린) |
| lab2 | V366 |
| lab4 | V364 |
| lab5 | V365_C |
| lab6 | V365_A |
| lab7 | V367_submit(제출 검증) |
| lab8 | pe313 |
| draw_src | V367_submit(draw 전용, 배치 중 건드리지 말 것) |
| **lab3** | 옛 draw 소스 — **절대 건드리지 말 것** |

- 실험에 쓸 수 있는 lab: lab · lab2 · lab4 · lab5 · lab6.
- **lab4의 git origin은 `/root/lab3`**이라 origin에서 fetch하면 안 된다.

**브랜치를 lab에 넣는 법:**
1. 로컬에서 GitHub origin(`knowin-kyeong/furiosa-opt-gemma4-12B`)에 `<branch>`와 `<branch>:refs/heads/tmp_<TAG>`를 push한다.
2. lab에서 `git fetch -q https://github.com/knowin-kyeong/furiosa-opt-gemma4-12B.git +tmp_<TAG>:tmp_<TAG>`.

**fixture:**
- `ref/fixtures.safetensors`는 git 밖 파일이다. lab5 · lab6은 lab3 파일로의 symlink, lab7은 사본이다.
- 새 lab에 없으면 `arena.sh: missing submission artifact`.

**Arena:**
- 계정당 동시 2잡까지. `arena_retry.sh`가 quota 초과 시 60 s마다 재시도한다.
- 같은 바이너리 반복은 `rngd rerun <job>`으로 한다. `rngd remove`는 사용자에게 먼저 묻는다.

**스크립트** (pod `/root/tk`, 사본은 저장소 `scripts/dev/tk/`):

| 스크립트 | 쓰임 |
|---|---|
| `vchain.sh LAB BRANCH TAG KERNEL ARM...` | 체크아웃 → 정적 덤프(`SV<TAG>_<arm>.json`) → 빌드 → 첫 잡 정확도 게이트 → rerun 15 |
| `rerun_n.sh JOB TAG N KREGEX` | rerun N회, `<TAG>_rerun_summary.txt` |
| `subverify.sh LAB BRANCH TAG` | 제출 브랜치 검증(`tmp_<TAG>` 필요): 빌드 → 잡 → rerun, 25/25 ×2 |
| `pairab.sh LA LB TAG N` · `pairab_ja.sh JOBA LB TAG N` | 두 바이너리를 동시 rerun 쌍으로 비교(순서 강제 탐침용) |
| `build_ordered.sh LAB BRANCH REF ORDER REFLAB` | 순서 강제 빌드 + `.bin` md5 비교 |
| `screen_ffn_down.sh` · `beam_ffn360.sh` | 순서 정적 스크린 · 빔 추적 예시 |
| `v360scores.sh` · `v367scores.sh` | draw 점수 수집 → `/root/tk/v36x_scores.txt` |
| `/root/drawchain_follow.sh BR B0 N` · `/root/drawloop2.sh BR N GAP` | draw 체인 |
| `/root/drawswitch_v367.sh` | draw 전환 스크립트의 틀 |

**하네스 도구:**
- `set_arms.py <tests/test_kernels.rs> <shim fn> <SWEEP_CONST> tag:ops_fn...`: arm 교체, 4행 회전 스윕.
- `mk_qkv_harness.py`: qkv 전용 span 하네스.
- 저장소에 있는 하네스 원본: qkv는 `V361_qkv_rope_pair_mode:tests/test_kernels.rs`, ffn은 `V365_ffn_down_tiles_before_geglu_sync:tests/test_kernels.rs`(생산 arm만).

**분석:**
- 짝비교: `ssh pod 'python3 - <TAG> <kernel> prod <arm>...' < scripts/dev/paired_medians.py`
- span: `scripts/dev/span/{crit,spanlist,map,schedorder}.py`

**draw 전환 절차** (새 제출 브랜치 NEW):
1. `git -C /root/draw_src fetch -q <github url> +NEW:NEW` — ref만 받고 체크아웃하지 않는다.
2. 돌던 follow 루프를 **PID로** 끊는다. `pkill -f`는 ssh 원격 셸도 죽인다.
3. `drawswitch_v367.sh`를 복사해 NEW로 바꾸고 `setsid nohup bash /root/drawswitch_<new>.sh > log 2>&1 < /dev/null &`로 띄운다(돌던 배치가 끝나면 넘어간다).
4. `sed 's/V367_submit/NEW/g; s/v367/vNEW/g' /root/tk/v367scores.sh > /root/tk/vNEWscores.sh`.

**리더보드:** pod에서 `curl -s https://micro2026-api.duckdns.org:7777/api/leaderboard`. 행마다 teamName · qkvCycles · attentionCycles · ffnCycles가 있다.

---

## 8. 재개 체크리스트

1. 이 문서 → RULES §10.0w → RESULTS.md(V361~V367) → SOTA.md 첫 항목을 읽는다.
2. draw 체인이 살아 있는지 본다: `ssh pod 'ps -eo pid,etime,args | grep -E "[d]rawloop|[d]rawchain_follow"'`. V367_submit 8배치가 끝나면 follow가 멈춘다.
3. 점수를 모은다: `ssh pod 'bash /root/tk/v367scores.sh; cat /root/tk/v367_scores.txt'`. 리더보드도 확인한다.
4. 새 실험을 할 때:
   - 이름은 `V368_<설명>`, 가설 하나.
   - **RESULTS.md에 빈 행을 먼저 등록하고 commit · push한 뒤** 코딩한다. main에는 커밋하지 않는다.
   - 정적 덤프로 먼저 거르고, `vchain.sh`로 16잡 짝비교, 변형이 첫 launch인 잡의 정확도를 확인한다.
5. 로컬 scratchpad의 worktree(wt351~wt367)는 세션 임시 폴더라 사라진다. 브랜치는 모두 origin에 있으니 `git worktree prune`으로 정리한다.
