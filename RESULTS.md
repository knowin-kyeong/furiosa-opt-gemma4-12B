# RESULTS.md — 실험 기록 (전체)

> 규칙은 [RULES.md](RULES.md). 성공·실패 **모두** 기록한다. 중복 실험 방지와
> A/B 비교가 이 문서의 존재 이유다. 실측(RNGD cycles)이 없는 결과로는 "채택" 판정을
> 내리지 않는다 (RULES §5). **새 가설은 코드보다 먼저 여기에 슬롯으로 등록한다** (RULES §2.1).

## 요약 보드

기준: `V0_baseline`. 점수 = 3커널 speedup의 기하평균. makespan은 정적 스케줄(스크리닝용),
RNGD cycles만 점수다.

| 브랜치 | 분기점 | 가설 한 줄 | qkv makespan | attn_out makespan | ffn makespan | 기하평균 (makespan 기준) | RNGD 실측 | 정확도 | 상태 |
|---|---|---|---:|---:|---:|---:|:---:|:---:|:---:|
| `V0_baseline` | `main` | 원본 skeleton (기준) | 116,583 | 194,020 | 1,693,200 | 1.000 | — | — | **기준** |
| `V1_ffn_down_chunked_dequant` | `V0_baseline` | down proj: 슬라이스별 L/8 청크만 dequant, 재배치 DMA 제거 | 116,583 | 194,020 | **609,223** | **1.406** | — | — | makespan 측정 |
| `V2_attnout_rows_over_256_slices` | `V1` | O proj: 32→256 슬라이스 (H/60 × Qs/1024), 4-way inter-slice reduce | 116,583 | **106,461** | 609,223 | **1.723** | — | — | makespan 측정 |
| `V7_qkv_x_replicate_via_hbm` | `V2` | x 복제를 switch(62k)/DM→DM DMA 대신 HBM 경유 로드로 (qkv: `HbmTensor::new()` 스크래치, attn_out: 입력 HBM에서 청크 직접 로드) | **95,433** | **58,015** | 609,223 | **2.250** | — | — | makespan 측정 |
| `V8_weight_rows_interleaved_dma` | `V7` | weight 행을 `rows % 256`으로 슬라이스에 교차 배치해 HBM→DM DMA를 DMN/슬라이스 인터리빙 (580 → ~2,000 B/cycle 목표) | — | — | — | — | — | — | 설계됨 |
| `V9_ffn_x_via_hbm` | `V7` | ffn의 x→Replicated DM→DM DMA(54k)를 V7 기법(HBM 스크래치 경유, 18.4k)으로 | 95,433 | 58,015 | **574,845** | **2.295** | — | — | makespan 측정 |
| `V3_qkv_hsplit_no_broadcast` | `V0_baseline` | QKV: H를 8슬라이스로 분할해 x 전체 브로드캐스트(62k) 제거, inter-slice reduce | — | — | — | — | — | — | 보류 (V7 우선; 아래 참조) |
| `V4_attnout_qsplit_no_broadcast` | `V2` | O proj: Qs를 8슬라이스로 분할해 x 브로드캐스트(66k) 제거 | — | — | — | — | — | — | 설계됨 |
| `V5_lut_in_contract_chain` | SOTA | f8→bf16 table lookup을 별도 pass 없이 contraction 체인 안에서 수행 | — | — | — | — | — | — | 설계됨 |
| `V6_ffn_upgate_overlap` | `V9` | FFN weight를 4행 타일로 스트리밍(LUT+cast를 fetch 단계에 융합), up/gate 인터리브, down 더블버퍼, geglu 직접 relayout, scale 타일화 | 95,433 | 58,015 | **412,304** | **2.559** | — | — | makespan 측정 |

> 단위: cycle. `—` 미측정. `FAIL(accuracy)` tolerance 위반. 상태 전이는 RULES §2.1.

## 현재 SOTA

실측(RNGD) 기준: `V0_baseline` (아직 실측 없음). **makespan 기준 잠정 선두: `V6_ffn_upgate_overlap`**
(V1+V2+V7+V9+V6 누적, 기하평균 2.559×). 자세한 서사는 [SOTA.md](SOTA.md).

## 죽은 길 (다시 시도하지 말 것)

- **switch 기반 브로드캐스트 변형으로 x 복제 비용 줄이기** — book(Switch Engine)이
  명시: 모든 SwitchConfig의 비용 = `ring_size × Time × flits_per_packet`. 1개 슬라이스의
  7.5KB(bf16 [H])를 256 슬라이스로 뿌리면 어떤 변형이든 256 × 240 = 61,440 cycle.
  V0의 62,215/66,311이 정확히 이 값이다. ring 크기 조작, Broadcast01, CustomBroadcast
  모두 동일 비용. **바이트 수 자체를 줄이는 것(축 분할)만이 답.**
- **DMA로 Replicated 매핑에 복제** — FFN이 이미 이렇게 하고 있고(ops.rs:227) 54,432 cycle.
  512 슬라이스 × ~106 cycle의 디스크립터 비용에 묶인다. switch 대비 8k 이득뿐.
- **contract_lane으로 reduction 축 일부를 미축약 상태로 남기기** (per-block scale을
  contraction 뒤에 적용하려는 시도) — book(Lane Folder): "reduction axes cannot be
  partially preserved". FFN f4의 16-블록 scale은 contraction 전에 곱해야 한다.

- **슬라이스 축에 패딩(예: `m![H / 16 # 256]` = 240 실제 + 16 패딩)** — DMA `to_dm`가
  `internal compiler error: split (inner_size: 64) is not valid on shape([H_16=240])`로 죽는다.
  DMA는 슬라이스 축을 64 단위로 쪼개므로 **실제 슬라이스 수가 64의 배수(사실상 256)** 여야 한다.
  `1 # 8`, `1 # 32` 같은 패딩은 곱해서 256이 되니 괜찮다. 행 수를 슬라이스에 나눌 때는
  H=3840 → 15행 × 256, Qs=4096 → 16행 × 256, L=15360 → 60행 × 256 처럼 정확히 나눠야 한다.
  (V2 1차 시도에서 확인, 2026-09-09)

- **"rmsnorm 출력(`Slice = m![1 # 256]`)이 이미 256 슬라이스에 복제되어 있으니
  `broadcast_hidden`을 `unsafe reshape`로 대체"** (0909_baseline.md 축 A 관찰 1) — 근거가
  반대다. FFN은 같은 `Slice → Replicated` 변환을 `x.to_dm(tdma)`로 하는데 512 슬라이스 전부에
  쓰느라 54,432 cycle을 쓴다(ops.rs:227). 패딩 슬라이스에 유효 데이터가 있었다면 컴파일러가
  복사를 생략했을 것이다. rmsnorm의 마지막 `Broadcast1{slice1: 8}`는 ring 8 안에서만 모은다
  (attn_out에서 744 cycle). 즉 x는 8개 슬라이스에만 있다. Arena 없이 정확도 검증이 불가능한
  변경이므로 실측 전에는 시도하지 않는다.

## 컴파일러가 강제하는 매핑 제약 (V2 구현 중 확인, 2026-09-09)

한 번씩 다 부딪힌 것들이다. 새 레이아웃을 설계할 때 먼저 대조할 것.

| 제약 | 에러 메시지 | 의미 |
|---|---|---|
| 슬라이스 축 live 개수는 **2의 거듭제곱** | `split (inner_size: 64) is not valid on shape([H_16=240])`; 192는 `cannot find the across_index` (`[H_1280=3, H_20=64]`) | `m![H / 16 # 256]`(240), `m![H / 20 # 256]`·`m![H / 60, H / 20 % 3 # 4]`(192) 모두 불가. 패딩을 안쪽 인자로 써도 정규화되면 같다. 베이스라인 live 수는 전부 256/32/8 |
| DmTensor 슬라이스당 element 크기는 8 B 배수 | `in-slice element extent is 30 B, not a multiple of the SRAM access width 8` | 15 × bf16 = 30 B 불가 |
| VRF는 슬라이스당 8 KB | `VRF data (15360 bytes) exceeds register file capacity (8192 bytes per slice)` | f32 [H] 벡터는 통째로 못 올림 → 1920 단위 타일 |
| transpose 패킷의 실제 행은 최대 4 | `output packet beyond max_in_rows (4) must be 1 # n pure padding` | 행 수는 4의 배수로 시간축 분해 (`H / 4 % k`, `H % 4 # 16`) |
| transpose 입력 시간축은 패딩 없는 축이어야 일관 | `cannot place in_rows in Time (H % 15) consistently with OutTime (H % 15 # 16 / 4)` | 15처럼 4로 안 나뉘는 행 수는 transpose 불가 |
| fetch는 패딩된 행 축을 stride 못 함 | `lower_fetch_unit: failed to stride_exact` | 2-D 타일의 행 축을 `# 16`으로 패딩해 fetch 불가 |
| DMA relayout은 패딩된 패킷을 stride 못 함 | `Condition failed: points ... b % a == 0` | `m![H / 3 % 5, H % 3 # 4]` → Slice `m![H]` DMA 불가 |
| collect 출력 패킷은 정확히 32 B | `Collect output packet must be exactly 32 bytes (one flit)` | bf16 16개 / f32 8개 / f8 32개 |
| collect 시간축은 switch가 만든 축까지 합친 형태 | `Collect time mismatch. Expected: H / 15, got: H / 3` | `fetch::<m![1], big>` → switch → `collect::<합쳐진 축, 32 B>` (rmsnorm gather 패턴) |
| commit_trim 패킷은 8/16/24/32 B | `commit_trim output packet must be one of [8, 16, 24, 32] bytes, got 6` | bf16 4/8/12/16개 |
| `m![H % 15 # 16 / 4]` 같은 패딩 축 분해는 **파싱은 됨** | (위 transpose 에러가 그 표현을 정상 출력) | 엔진별 지원 여부는 따로 확인 |

결론: 행 수를 슬라이스에 나눌 때 **행/슬라이스가 4의 배수**이고 **live 슬라이스 수가 64의 배수**인
조합을 고른다. H=3840이면 20행 × 192, 60행 × 64, 120행 × 32 (60·120은 8의 배수라 vector pass도 됨).
Qs=4096이면 16 × 256. L=15360이면 60 × 256.

## V0에서 측정된 하드웨어 상수 (슬라이스당, 스케줄 기준)

| 항목 | 값 | 근거 |
|---|---|---|
| `fetch_table_lookup` pass | ≈ 6.2 elem/cycle | qkv Q: 16×3840 elem / 9,863 cyc |
| vector scale pass (f32) | ≈ 3.9 elem/cycle (4 lane) | ffn down: 2×15360 / 7,961 cyc |
| contraction | ≈ 15 MAC/cycle | qkv Q: 16×3840 / 4,105 cyc |
| HBM→DM DmaLoad 실효 | ≈ 580 B/cycle (피크 2,048의 28%) | Q weight 15.7MB / 26,475 cyc; util 평균 0.06–0.11 |
| DM→DM 복제 디스크립터 | ≈ 106 cycle/슬라이스 | 54,432 / 512 |
| switch 브로드캐스트 | ring × flits (bf16 [H]: 61,440) | book 공식 = 실측 |

---

# 실험 상세

각 실험은 아래 템플릿으로 추가한다. **최신 실험을 위에 쌓는다.**

<!-- ===================== 템플릿 (복사해서 쓸 것) =====================
## V{n}_{description}

- **상태:** 설계됨 / 구현됨 / makespan 측정 / 채택 / 기각 / 보류
- **분기점:** `V{k}_{...}` (커밋 `abc1234`)
- **가설:** (무엇이 병목이라고 보았고, 무엇을 바꾸면 왜 줄어든다고 예상했는가)
- **변경 파일:** `src/device/...` (RULES §4 허용 범위 확인 완료)
- **공유 코드 영향:** 없음 / `shared/mlp.rs` 수정 → vision/audio 경로 동시 영향

### 측정

| 커널 | makespan (before → after) | RNGD cycles (before → after) | speedup |
|---|---|---|---:|
| `sliding_project_qkv` | | | |
| `sliding_attention_output` | | | |
| `decoder_feedforward` | | | |
| **기하평균** | | | |

- **정확도:** PASS / FAIL(어느 커널, 어느 tolerance를 얼마나 넘겼는지)
- **측정 방식:** RNGD 실측 / makespan only
- **지배 context (변경 후):** MainContext xx% / DmaEngine xx% / SubContext xx%

### 판정

- **이유:**
- **배운 것:**
- **다음 후보:**
======================================================================= -->

## V6_ffn_upgate_overlap

- **상태:** makespan 측정 (2026-09-09). 브랜치 안에서 단계별로 누적.
- **분기점:** `V9_ffn_x_via_hbm` (`5258498`)
- **가설(원문):** up/gate 직렬화 해소 + ROWS_PER_PASS 튜닝. 실제로는 "FFN의 DMA와 Main을 겹치게
  만드는 모든 것"으로 확장됐다.
- **변경 파일:** `src/device/shared/mlp.rs` (`project_up_and_gate`, `project_down`, `feedforward`)
- **공유 코드 영향:** vision/audio MLP 경로 동시 영향 (Stage 1 세 커널 중 ffn만 변함, qkv/attn_out 불변 확인)

### 단계별 측정 (ffn makespan)

| 단계 | 변경 | ffn | Δ |
|---|---|---:|---:|
| V9 | (출발점) | 574,845 | |
| V6-b | 행렬 전체 f8 사본 제거 + `fetch_table_lookup → fetch_cast` 융합 체인, 두 행렬 동시 상주, 인터리브 | 578,083 | +3k (DMA 138k가 Main 앞에 통째로) |
| V6-(0,1,4) | weight를 pass별 4행 타일로 로드 + up/gate 인터리브 | 522,379 | −56k |
| V6-c | down: 반복당 타일 2개(더블버퍼) + geglu 출력 → ByColumns 직접 relayout + down scale 선발행 | 465,394 | −57k |
| V6-d | scale도 pass별 타일, 첫 반복 peel | 462,004 | −3k |
| V6-e | dequant/contract 분리, x_trf를 첫 dequant 뒤에 생성 (head 38k→8k) | 462,004 | 0 (임계 경로가 버퍼 사이클) |
| V6-f | 반복당 4타일 선로드 | 412,304 | |

### 컴파일러/스케줄러 제약 (추가 발견)

| 시도 | 결과 |
|---|---|
| 캐리 텐서 프리페치 (`next = load(i+1)` 루프 밖으로 전달, 마지막에 `DmTensor::new()` 더미) | `V1-Modulo failed for all operator schedule heuristics` |
| `Vec<DmTensor>` | `unsupported type in device function: *const u8` |
| `std::array::from_fn(\|i\| ...)` | `internal compiler error: not yet implemented: closure` |
| `for i in 1..PASSES` | `A loop was not rewritten to for` → `0..CONST`만 허용 (`const REST = PASSES-1; for k in 0..REST`) |
| `ROWS_PER_PASS = 12` | scale VRF 11,520 B > 8 KB. 60의 약수·4의 배수·VRF 한도 → 4만 가능 |
| `fetch → fetch_table_lookup::<f8> → fetch_cast::<f32> → vector` (f4 weight) | **컴파일됨.** f8 사본 pass(18k×2 + 26k) 제거 |
| DMA 발행 순서 | 스케줄러가 소비 시점 기준으로 재배치 — 소스에서 앞당겨 발행해도 무의미. 소비자를 앞당겨야 함 |
| 같은 DM 버퍼 재사용 | WAR로 다음 로드가 이전 contract 뒤로 밀림 → 반복 안에 타일 여러 개를 동시에 살려 다른 주소를 받게 함 |

### 판정: makespan 측정 (실측 대기)

- **정확도:** dequant 수식 동일(LUT→f32→×scale→bf16), 누산 순서 동일. geglu 직접 relayout은 데이터 이동만.
- **지배 context (V6-e):** DMA 74% / Main 56% / Vector 56%. DMA busy 342k 중 pass당 `DmaLoad ?`(설명 없음, 4 KB, 838 cycle)가
  90개 ≈ 75k — LUT/엔진 설정 테이블로 추정, pass 수에 비례.
- **배운 것:** FFN은 이제 DMA-bound. 남은 레버는 (1) 버퍼 깊이(4타일), (2) pass 수 자체(ROWS_PER_PASS 제약에 막힘),
  (3) Sub 컨텍스트로 scale pass 절반 이전(LUT는 Main 전용이라 LUT를 별도 pass로 되돌려야 함 — 순이득 불확실).
- **다음 후보:** V5(attn_out/qkv LUT 융합, tail 단축) · attn_out/qkv weight 타일화(tail 단축) · Sub 오프로드.

## V5_lut_in_contract_chain

- **상태:** 설계됨 (2026-09-09)
- **분기점:** 당시 SOTA
- **가설:** V0는 f8 weight를 `fetch → fetch_table_lookup::<bf16> → collect → commit`으로 DM에
  bf16 사본을 만든 뒤 다시 `fetch → collect → contract`한다. book(Fetch Adapter):
  "all stages preserve mapping and can route directly to Switch/Collect/Contraction
  without intermediate commits". 따라서 `fetch → fetch_table_lookup → collect → contract_outer`
  한 체인으로 LUT pass(qkv 20k, attn_out 78k→V2 후 ~10k, ffn up/gate 36k)와 bf16 DM 사본을
  제거할 수 있다. contraction 스트리밍 피연산자는 bf16(LUT 출력)이므로 TRF의 bf16 x와 같은 family.
- **변경 파일:** `src/device/sliding/projection.rs`, `src/device/shared/mlp.rs`
- **공유 코드 영향:** mlp.rs 부분은 vision/audio 영향.
- **리스크:** 빌더 타입 상태가 LUT 출력 → contract를 실제로 허용하는지는 컴파일해봐야 안다.
  f4→f8 LUT(ffn)는 그 뒤에 block scale 곱이 필요하므로 이 실험 범위 밖(f8 weight만).
- **예상:** qkv −15k, attn_out −8k(V2 후), ffn −30k.

## V4_attnout_qsplit_no_broadcast

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V2` (V2가 채택되면 그 위에)
- **가설:** V3와 같은 원리를 O projection에 적용. Qs=4096을 8슬라이스로 분할(512/슬라이스),
  행은 32그룹 × 120행. x는 512 bf16 = 32 flit만 각 슬라이스에 필요 → 복제 비용
  ring-8 스테이지 + 256×32 ≈ 10k (V0 66,311). 8-way inter-slice reduce로 partial 합산.
- **변경 파일:** `src/device/sliding/projection.rs` (`project_output`), `src/ops.rs` 본문
- **공유 코드 영향:** 없음 (`layout::broadcast_sliding_heads`는 사용 중단, 삭제는 하지 않음)
- **예상:** attn_out −55k.

## V3_qkv_hsplit_no_broadcast

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V0_baseline` (독립 A/B)
- **가설:** qkv makespan의 53%(62,215)가 `layout::broadcast_hidden` 한 노드다. 이 값은 switch
  브로드캐스트의 이론 하한(256 ring × 240 flit)이므로 switch 변형으로는 못 줄인다("죽은 길").
  대신 **reduction 축 H를 8슬라이스로 분할**한다: Q weight를 32행그룹(128행) × 8 H-청크(480열)로
  슬라이스에 배치하면 각 슬라이스는 x의 480개(960B = 30 flit)만 필요. x 복제 = 1→8 슬라이스
  청크 분배(ring 8 × 240 ≈ 2k) + 청크를 32그룹에 브로드캐스트(256 × 30 ≈ 7.7k) ≈ 10k.
  partial [128행]을 `vector_inter_slice_reduce`로 8-way 합산(project_down이 이미 쓰는 패턴).
  K/V(2048행)는 256행그룹... 8행×H → 32그룹(64행) × 8청크로 동일 적용.
- **변경 파일:** `src/device/sliding/projection.rs`, `src/ops.rs` 본문 (broadcast_hidden 호출 제거)
- **공유 코드 영향:** 없음 (`layout.rs`의 broadcast 함수는 full/vision 경로가 계속 사용 — 건드리지 않음)
- **리스크:** H-청크 480 f8 = 480B는 256B 정렬이 안 됨 → DmaLoad 2× 페널티 가능(현재 DMA는
  Main 뒤에 숨어 있고 util 0.06이라 여유 있음). weight 접근이 strided가 되어 HBM row
  switch 비용 증가 가능. 스케줄의 DMA util로 확인.
- **예상:** qkv 116k → ~60k.

## V8_weight_rows_interleaved_dma

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V7`
- **가설:** V7 이후 qkv(DMA 88%)와 attn_out(DMA 57%)은 weight DmaLoad가 임계 경로다. 실효
  580 B/cycle은 피크 2,048의 28%이고 util 0.06–0.11. book(Memory Performance): "DMN interleaving:
  alternate across 2 DMNs per cluster, else 50% loss", "Slice interleaving: spread across 32 slices
  per DMN". 현재 레이아웃은 슬라이스당 61 KB(qkv Q) / 60 KB(attn_out) **연속 블록**이라 한
  슬라이스·한 DMN에 순차 기록된다. 행을 4행 블록 단위로 슬라이스에 교차 배치
  (`m![H / 4 % 64, Qs / 1024]`, element `m![H / 256, H % 4, ...]`)하면 HBM 순차 읽기가 연속
  슬라이스로 번갈아 들어간다. 4행 블록인 이유: transpose 패킷(4행)과 `to_dm` permutation(8 B 단위)
  제약. 먼저 attn_out에서 프로브, 효과 있으면 qkv(Q/K/V)와 ffn(up/gate/down)에 확장.
- **변경 파일:** `src/device/sliding/projection.rs` (매핑만), 이후 `mlp.rs`
- **공유 코드 영향:** 1차는 없음
- **리스크:** DMA 엔진이 디스크립터를 어떻게 병렬화하는지 모른다 — 효과가 0일 수 있다.
  출력 벡터가 블록 permutation된 채 나오므로 `to_dm` relayout이 8 B 조각 960개를 옮겨야 한다.
- **예상:** attn_out weight DMA 26.7k → 10k 이하면 성공.

## V7_qkv_x_replicate_via_hbm

- **상태:** makespan 측정 (2026-09-09)
- **분기점:** `V2_attnout_rows_over_256_slices` (`6403c3a`)
- **가설:** x 복제 비용 — switch 61k(하한), DM→DM DMA ~70 B/cycle — 대신 HBM→DM 로드.
  프로브 결과 커널 안에서 `HbmTensor::new()`가 **컴파일된다**. attn_out은 입력 `x`가 이미
  HBM `[Ns, Gs, Ds]`이므로 `unsafe { x.view().reshape() }`로 `HbmTensorView<m![Qs]>`를 만들어
  `project_output`이 슬라이스별 1024-청크를 직접 로드한다.
- **변경 파일:** `src/ops.rs` 본문 2곳, `src/device/sliding/projection.rs` (`project_output` 시그니처)
- **공유 코드 영향:** 없음 (`layout::broadcast_*`는 미사용으로 남음 — dead_code 경고만)

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup (makespan) |
|---|---|---|---:|
| `sliding_project_qkv` | 116,583 → **95,433** | — | 1.222 |
| `sliding_attention_output` | 106,461 → **58,015** | — | 1.835 |
| `decoder_feedforward` | 609,223 → 609,223 | — | 1.000 |
| **기하평균 (V0 대비 누적)** | | | **2.250** |

- **정확도:** — (Arena 대기). 수치 변화 없음(복제 경로만 변경). qkv의 스크래치는 별도
  `HbmTensor::new()`라 `q_out`과 무관.
- **측정 방식:** makespan only
- **지배 context (변경 후):** qkv DMA 88.4% (Q 26,475 / x 18,400 / K,V 13,512×2 직렬) · Main 51.9%;
  attn_out DMA 57.1% (weight 26,734) · Main 35.2%.
- **HBM→DM Replicated 로드 실효:** 3.9 MB / 18,400 = 213 B/cycle (DM→DM 72 B/cycle의 3배,
  HBM 순차 로드 580 B/cycle의 1/3 — 같은 7.5 KB를 512번 읽는 패턴 한계로 추정).

### 판정: makespan 측정 (실측 대기)

- **배운 것:** (1) 커널 내 HBM 스크래치 할당 가능 → 레이아웃 변환의 자유도가 크게 늘었다.
  (2) 두 attention 커널 모두 이제 **weight DMA-bound** — 다음은 DMA 접근 패턴(V8).
  (3) FFN의 x→Replicated DmaStos 54k도 같은 기법으로 ~18k로 줄일 수 있다 (V9로 등록).
- **다음 후보:** V8(DMA 인터리빙) → V9(ffn x via HBM) → V6(up/gate 오버랩) → V5(LUT 융합).

## V2_attnout_rows_over_256_slices

- **상태:** makespan 측정 (2026-09-09)
- **분기점:** `V1_ffn_down_chunked_dequant` (`f588d2a`) — attn_out은 mlp.rs를 쓰지 않으므로 V0 분기와 동일
- **가설:** 원문은 위 요약 보드. 최종 구현은 15행×256이 아니라 **H/60 × Qs/1024 = 64 × 4 슬라이스**
  (슬라이스당 60행 × 1024열, 4-way `vector_inter_slice_reduce`). 8번의 컴파일 실패를 거쳐 도달했고
  그 제약들은 "컴파일러가 강제하는 매핑 제약" 표에 정리했다.
- **변경 파일:** `src/device/sliding/projection.rs` (`project_output` 재작성, `output_partial`·
  `add_partials` 삭제, `apply_output_channel_scale`은 Slice 레이아웃 1920-타일 2회로)
- **공유 코드 영향:** 없음

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup (makespan) |
|---|---|---|---:|
| `sliding_project_qkv` | 116,583 → 116,583 | — | 1.000 |
| `sliding_attention_output` | 194,020 → **106,461** | — | **1.822** |
| `decoder_feedforward` | 609,223 → 609,223 | — | 1.000 |
| **기하평균 (V0 대비 누적)** | | | **1.723** |

- **정확도:** — (Arena 대기). 수치 변화: Qs 합산이 bf16 partial 3회 add(f32 왕복)에서 f32
  inter-slice reduce 1회로 → 오히려 정밀해짐.
- **측정 방식:** makespan only
- **지배 context (변경 후):** Main 81.4% / DMA 45.5% / Vector 9.9%. 잔여 상위:
  broadcast_sliding_heads 66,311 / weight DmaLoad 26,734 (브로드캐스트와 겹침) /
  x.to_dm 16,128 / LUT 9,863 / contract 4,105.

### 판정: makespan 측정 (실측 대기)

- **배운 것:** (1) live 슬라이스 수는 2의 거듭제곱, 행/슬라이스는 4의 배수 — H=3840에서는
  60·120만 가능. (2) 두 축을 함께 슬라이스에 나누고 inter-slice reduce로 합치는 것이
  "행 수가 안 나눠지는" 문제의 정답. (3) DM→DM 분배 DMA는 ~65-70 B/cycle — x 분배 16k는
  V7에서 HBM 경유로 줄일 것. (4) attn_out은 이제 x 분배(82k)가 makespan의 77%.
- **다음 후보:** V7 → V5(LUT 체인 융합; LUT 9.9k) → V6.

## V1_ffn_down_chunked_dequant

- **상태:** makespan 측정 (2026-09-09) — ffn 1,693,200 → **609,223** (2.779×). 첫 컴파일 통과.
  잔여 상위: x→Replicated DmaStos 54,432 / up·gate weight DmaLoad 49,157×2 / down scale DmaLoad 18,998 /
  down weight DmaLoad 30×2,473=74k / down scale pass 30×2,201=66k / up·gate scale pass 61,815×2.
  이제 ffn은 Main 52.9% / DMA 51.0% — DMA와 Main이 대등. 정확도: 누산 순서 불변(같은 8-way reduce).
- **분기점:** `V0_baseline`
- **가설:** ffn makespan 1.69M 중 **down projection이 ~1.2M**이다:
  - mlp.rs:393 scale pass 60 × 7,961 = 477,660 (28%) — 슬라이스당 2행 × L=15360 전체를 dequant.
  - mlp.rs:409/411 DMA 재배치 60 × 4,275 × 2 = 513,000 (30%) — `DownRows`(행 전체) →
    `DownRowsByColumns`(L을 8슬라이스로 분할)로 옮기는 DM→DM 복사.
  - mlp.rs:372 LUT pass 159,780. 정작 contraction(:419)은 22,560뿐.
  `DownRows = m![H/120, 1#8]`이라 8슬라이스 그룹이 같은 행 전체를 받아 dequant한 뒤 각자
  1/8만 쓴다 → **vector 작업이 필요량의 8배**. 처음부터 각 슬라이스가 자기 (4행 × L/1920)
  청크만 HBM에서 받아(`DownRowsByColumns` 레이아웃으로 직접 `to_dm`) 그 청크만 LUT·scale하면
  scale pass ≈ 60k, LUT ≈ 20k, 재배치 DMA 0.
- **변경 파일:** `src/device/shared/mlp.rs` (`project_down` 루프; `HALVES_PER_PASS` 제거)
- **공유 코드 영향:** `shared/mlp.rs` → vision/audio MLP 경로 동시 영향. 세 커널 makespan 기록.
- **리스크:** 행당 L/8 청크 = 960B(f4)는 256B 정렬이 안 됨 → HBM read 2× 페널티 가능. 현재
  down weight DmaLoad는 81k(4.8%)라 2배가 되어도 재배치 513k 제거가 압도. block scale
  `[H, L/16]`도 같은 방식으로 청크 로드(120 scale/행/슬라이스).
- **예상:** ffn 1.69M → 0.6~0.7M (**단독으로 기하평균 ~1.35×**).

## V0_baseline

- **상태:** makespan 측정 (2026-09-09). RNGD 실측은 Arena 승인 후.
- **분기점:** `main` (`4bf1bac`)
- **가설:** 없음 — 기준선. 커널 코드 원본 그대로.
- **변경 파일:** 없음 (문서·환경 파일만)
- **공유 코드 영향:** 없음

### 측정

| 커널 | makespan | instructions | RNGD cycles |
|---|---:|---:|---|
| `sliding_project_qkv` | 116,583 | 131 | — |
| `sliding_attention_output` | 194,020 | 163 | — |
| `decoder_feedforward` | 1,693,200 | 1,802 | — |

- **정확도:** — (Arena 승인 후 `./scripts/rngd_test.sh`)
- **측정 방식:** makespan only (`cargo furiosa-opt compile --exact --dump-schedule`, cargo-furiosa-opt 0.6.0)
- **지배 context:**
  - qkv: Main 95.9% / DMA 56.9% / Vector 17.4% / Sub 9.3%
  - attn_out: Main 93.8% / Vector 19.5% / DMA 18.5% / Sub 6.3%
  - ffn: Main 51.3% / DMA 49.8% / Vector 39.7% / Sub 7.7%

### 소스 라인별 cycle 분해 (상위)

**qkv (116,583)**

| cycle | n | %  | 무엇 |
|---:|---:|---:|---|
| 62,215 | 1 | 53.4 | `layout.rs:16` broadcast_hidden (switch ring 256) |
| 27,024 | 2 | 23.2 | `projection.rs:93` K/V weight DmaLoad |
| 26,475 | 1 | 22.7 | `projection.rs:24` Q weight DmaLoad |
| 10,126 | 2 | 8.7 | `projection.rs:94` K/V LUT pass |
| 9,863 | 1 | 8.5 | `projection.rs:25` Q LUT pass |
| 4,370 | 2 | 3.7 | `projection.rs:103` K/V contract |
| 4,105 | 1 | 3.5 | `projection.rs:34` Q contract |

**attn_out (194,020)**

| cycle | n | % | 무엇 |
|---:|---:|---:|---|
| 77,852 | 4 | 40.1 | `projection.rs:216` O weight LUT pass ×4 청크 |
| 66,311 | 1 | 34.2 | `layout.rs:32` broadcast_sliding_heads |
| 31,780 | 4 | 16.4 | `projection.rs:224` contract ×4 |
| 27,864 | 4 | 14.4 | `projection.rs:212` weight DmaLoad ×4 |

**ffn (1,693,200)**

| cycle | n | % | 무엇 |
|---:|---:|---:|---|
| 477,660 | 60 | 28.2 | `mlp.rs:393` down weight scale pass |
| 256,500 | 60 | 15.1 | `mlp.rs:409` down weight 재배치 DmaStos |
| 256,500 | 60 | 15.1 | `mlp.rs:411` down weight `to_dm_view` DmaStos |
| 159,780 | 60 | 9.4 | `mlp.rs:372` down LUT pass |
| 81,000 | 60 | 4.8 | `mlp.rs:368` down weight DmaLoad |
| 61,815 ×2 | 15 ×2 | 7.3 | `mlp.rs:57/:120` up/gate scale pass |
| 54,432 | 1 | 3.2 | `ops.rs:227` x → Replicated DmaStos |
| 49,157 ×2 | 1 ×2 | 5.8 | `mlp.rs:31/:94` up/gate weight DmaLoad |
| 22,560 | 30 | 1.3 | `mlp.rs:419` down contract |
| 18,263 ×2 | 1 ×2 | 2.2 | `mlp.rs:32/:95` up/gate LUT pass |

### 판정: **기준**

- **배운 것:**
  1. 세 커널 모두 **MainContext-bound**이며, 그 Main 시간의 대부분은 contraction이 아니라
     **dequant pass와 브로드캐스트**다. contraction 자체는 qkv 8.5k, attn_out 32k, ffn 60k 정도.
  2. HBM 실효 대역폭이 피크의 28%(util 0.06~0.11) — DMA는 Main 뒤에 숨어 있어 지금은
     병목이 아니지만 Main을 줄이면 곧 드러난다. 그때는 접근 패턴(정렬·strided)이 다음 과제.
  3. ffn의 down projection 구조(행 전체 복제 후 8-way 분할)가 전체 점수의 최대 레버.
- **다음 후보:** V1 → V2 → V5 → V3/V4 → V6 순.
