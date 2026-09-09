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
| `V8_weight_rows_interleaved_dma` | `V7` | weight 행을 4행 블록으로 슬라이스에 교차 배치해 HBM→DM DMA 인터리빙 | 95,433 | 59,225 | 609,223 | 2.235 | — | — | **기각** (makespan; DMA 노드 불변) |
| `V10_attn_weight_tiles_fused_lut` | `V6` | attn_out weight 5×12행 타일 선로드 + f8→bf16 LUT를 contraction 체인에 융합(V5 흡수); qkv는 융합만(타일화는 역효과) | **93,127** | **53,848** | 412,304 | **2.646** | — | — | makespan 측정 |
| `V13_ffn_dma_trims` | `V12` | (a) geglu 출력을 HBM 경유로 ByColumns 로드 **채택**; (b) scale 행렬당 1회 로드는 head 증가로 **기각**(354,617) | 93,127 | 50,110 | **348,874** | **2.866** | — | — | makespan 측정 |
| `V28_ffn_16row_tiles` | `V27` | FFN 타일을 12행×5 → 16/16/16/12행×4로 (scale VRF 7.7 KB < 8 KB): 행렬당 DmaLoad 고정비·`?` 838·Sub 프리로드 1개씩 제거 | — | — | — | — | — | — | 설계됨 |
| `V27_attnout_scale_in_rmsnorm` | `V26` | attn_out 채널 scale(64 디스크립터 로드 1,318이 DMA 큐 선두에서 첫 타일을 막음)을 epilogue 대신 post-attn rmsnorm 두 pass에 접어 넣어 로드를 tail로 | — | — | — | — | — | — | 설계됨 |
| `V26_qkv_hsplit_x_halved` | `V25` | qkv 투영을 H/1920 열 반으로 나눠(Q 16행×1920, K/V 8행×1920, 2-way inter-slice reduce) 복제 x 바이트를 절반으로 (x 로드 5.4k, x_trf 1.2k) | — | — | — | — | — | — | 설계됨 |
| `V25_ffn_geglu_two_clusters` | `V24` | up/gate의 HBM hop(store 4,535×2 + load 1,328×2 + geglu 출력 store 2,219 = 13.9k)을 없앤다: global scale을 contraction epilogue에 접고, geglu를 두 클러스터 reduce 출력 레이아웃(슬라이스당 60행, V18식 패딩 패킷)에서 직접 수행 | — | — | — | — | — | — | 설계됨 |
| `V24_gather_before_hbm_store` | `V23` | (a) 투영 출력을 HBM에 쓰기 전 클러스터 안 switch gather → **기각**(live 슬라이스가 8칸 간격이라 ring 256, 2,055 = store 절감분); (b) post-norm 피연산자(x·residual)를 HBM에서 ReducingSlices로 직접 로드, ffn global scale을 8슬라이스 레이아웃에서 → **채택** | **50,981** | **30,037** | **179,922** | **5.180** | — | — | makespan 측정 |
| `V23_ffn_whole_scale_loads` | `V22` | FFN block scale을 pass 타일(15개, 36k) 대신 행렬당 1회 로드(슬라이스당 7.2 KB)로; V13b 재시도 | 51,631 | 30,487 | **181,625** | **5.131** | — | — | makespan 측정 |
| `V22_attnout_weight_tiles` | `V21` | attn_out weight를 20행 타일 3개로 선로드해 융합 LUT+contract(5k)를 DMA(13.5k)와 겹침 | 51,631 | **30,487** | 186,976 | **5.082** | — | — | makespan 측정 |
| `V21_qkv_rope_tables_early` | `V20` | cos/sin gather·HBM hop·VRF 스테이징을 커널 맨 앞에서 수행해 q/k rope가 V 로드와 겹치게 | 51,631 | — | — | — | — | — | **기각** (불변; 스케줄러가 gather 하나만 앞당김) |
| `V20_qkv_tail_per_cluster` | `V19` | q/k/v의 HBM hop 제거: 클러스터별 ring-64 switch gather로 헤드/슬라이스 레이아웃을 만들고 후처리·저장·scatter를 두 클러스터에서 병렬 수행 | **51,631** | 31,113 | 186,976 | **5.048** | — | — | makespan 측정 |
| `V19_qkv_tail_heads_layout` | `V18` | qkv 후처리(q/k/v rmsnorm·rope·저장)를 헤드별 슬라이스 분산 레이아웃에서 수행 — HBM hop에서 직접 그 레이아웃으로 로드, rope의 InterTranspose·Broadcast1 제거 | **55,811** | 31,113 | 186,976 | **4.919** | — | — | makespan 측정 |
| `V18_attnout_scale_in_epilogue` | `V17` | attn_out 채널 scale을 contraction 체인 epilogue(inter-slice reduce 뒤 vector 곱)로 접어 넣어 tail의 scale pass 2개 제거 | 59,216 | **31,113** | 186,976 | **4.822** | — | — | makespan 측정 |
| `V17_qkv_hoist_weight_loads` | `V16` | qkv: Q weight의 LUT pass를 rmsnorm 앞에 발행해 Q 로드가 DMA 큐 선두로 (K/V까지 같은 방식은 무효) | **59,216** | 34,776 | 186,976 | **4.649** | — | — | makespan 측정 |
| `V16_rmsnorm_fused_residual` | `V15` | post-attn/post-FF rmsnorm의 마지막 vector pass에 residual add(+layer gate)를 접어 넣어 tail pass 제거 (새 함수 `normalize_add[_gate]`) | 60,412 | **34,776** | **186,976** | **4.618** | — | — | makespan 측정 |
| `V15_x_replicate_hbm_copies` | `V14` | qkv의 x 복제 로드가 같은 7.5 KB HBM 구간을 512번 읽는 패턴(420 B/cycle) → x를 HBM에 8부 쓰고 슬라이스별로 다른 사본 읽기 | **60,412** | 38,240 | 191,122 | **4.434** | — | — | makespan 측정 |
| `V14_two_clusters` | `V13` | 모든 DM 텐서가 `Cluster = m![1 # 2]`(클러스터 1개만 live)였다. 세 커널의 투영을 두 클러스터에 실제로 나눠 512 슬라이스 사용; DMA 처리량·연산 2배 | **73,445** | **38,240** | **191,122** | **4.147** | — | — | makespan 측정 |
| `V12_ffn_rows_per_pass_12` | `V11` | FFN `ROWS_PER_PASS` 4→12: up/gate는 1920열 절반씩 dequant(scale VRF 5.8 KB), down은 그대로; pass 수 90→30 | 93,127 | 50,110 | **352,164** | **2.857** | — | — | makespan 측정 |
| `V11_residual_1920_tiles` | `V10` | `residual::add`를 480×8 타일에서 1920×2 타일로 (공유 코드) | 93,127 | **50,110** | **408,566** | **2.716** | — | — | makespan 측정 |
| `V9_ffn_x_via_hbm` | `V7` | ffn의 x→Replicated DM→DM DMA(54k)를 V7 기법(HBM 스크래치 경유, 18.4k)으로 | 95,433 | 58,015 | **574,845** | **2.295** | — | — | makespan 측정 |
| `V3_qkv_hsplit_no_broadcast` | `V0_baseline` | QKV: H를 8슬라이스로 분할해 x 전체 브로드캐스트(62k) 제거, inter-slice reduce | — | — | — | — | — | — | 보류 (V7 우선; 아래 참조) |
| `V4_attnout_qsplit_no_broadcast` | `V2` | O proj: Qs를 8슬라이스로 분할해 x 브로드캐스트(66k) 제거 | — | — | — | — | — | — | 설계됨 |
| `V5_lut_in_contract_chain` | SOTA | f8→bf16 table lookup을 별도 pass 없이 contraction 체인 안에서 수행 | — | — | — | — | — | — | V10에 흡수 |
| `V6_ffn_upgate_overlap` | `V9` | FFN weight를 4행 타일로 스트리밍(LUT+cast를 fetch 단계에 융합), up/gate 인터리브, down 더블버퍼, geglu 직접 relayout, scale 타일화 | 95,433 | 58,015 | **412,304** | **2.559** | — | — | makespan 측정 |

> 단위: cycle. `—` 미측정. `FAIL(accuracy)` tolerance 위반. 상태 전이는 RULES §2.1.

## 현재 SOTA

실측(RNGD) 기준: `V0_baseline` (아직 실측 없음). **makespan 기준 잠정 선두: `V24_gather_before_hbm_store`**
(…+V24 누적, 기하평균 5.180×). 자세한 서사는 [SOTA.md](SOTA.md).

## 죽은 길 (다시 시도하지 말 것)

- **switch 기반 브로드캐스트 변형으로 x 복제 비용 줄이기** — book(Switch Engine)이
  명시: 모든 SwitchConfig의 비용 = `ring_size × Time × flits_per_packet`. 1개 슬라이스의
  7.5KB(bf16 [H])를 256 슬라이스로 뿌리면 어떤 변형이든 256 × 240 = 61,440 cycle.
  V0의 62,215/66,311이 정확히 이 값이다. ring 크기 조작, Broadcast01, CustomBroadcast
  모두 동일 비용. **바이트 수 자체를 줄이는 것(축 분할)만이 답.**
- **DMA로 Replicated 매핑에 복제** — FFN이 이미 이렇게 하고 있고(ops.rs:227) 54,432 cycle.
  512 슬라이스 × ~106 cycle의 디스크립터 비용에 묶인다. switch 대비 8k 이득뿐.
- **inter-slice reduce 출력(`m![X, 1 # k]`)을 switch로 클러스터 안에서 모아 HBM store 디스크립터를 줄이기** (V24a/b) —
  VRU reduce 축은 innermost 강제(`VRU reduce axes must be innermost in Partitioning`)라 live 슬라이스가 k칸 간격으로
  남고, `Broadcast1 { slice1: 32, slice0: 8 }`은 ring 256으로 취급되어 비용 = 256 × Time × flit(+~800 고정):
  4원소 패킷(Time 15) 4,615, 12원소 패킷(Time 5) 2,055. store 절감(2,510 → 458)과 정확히 상쇄. Main이 노는
  구간(ffn up/gate)에서만 쓸모가 있으나 V25가 hop 자체를 없애므로 불필요.
- **contract_lane으로 reduction 축 일부를 미축약 상태로 남기기** (per-block scale을
  contraction 뒤에 적용하려는 시도) — book(Lane Folder): "reduction axes cannot be
  partially preserved". FFN f4의 16-블록 scale은 contraction 전에 곱해야 한다.

- **슬라이스 축에 패딩(예: `m![H / 16 # 256]` = 240 실제 + 16 패딩)** — DMA `to_dm`가
  `internal compiler error: split (inner_size: 64) is not valid on shape([H_16=240])`로 죽는다.
  이후 192(=3×64)도 실패해 결론은 **live 슬라이스 수가 2의 거듭제곱**(아래 제약 표 참조).
  `1 # 8`, `1 # 32` 같은 패딩은 곱해서 256이 되니 괜찮다. (V2 시도에서 확인, 2026-09-09)

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

결론: 행 수를 슬라이스에 나눌 때 **행/슬라이스가 4의 배수**이고 **live 슬라이스 수가 2의 거듭제곱**인
조합을 고른다. H=3840이면 60행 × 64, 120행 × 32 (둘 다 8의 배수라 vector pass도 됨) — 256 슬라이스를
다 쓰려면 다른 축(Qs, L)과 함께 나눠 inter-slice reduce로 합친다(V2가 그 예). Qs=4096이면 16 × 256.
L=15360이면 60 × 256.

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

## V24_gather_before_hbm_store

- **상태:** makespan 측정 (2026-09-09)
- **분기점:** `V23_ffn_whole_scale_loads` (`fd20ecd`)
- **가설(원문):** V23 스케줄에서 HBM store가 디스크립터 비용에 묶여 있다(util 0.2~1%): ffn up/gate 결과 store
  4,535×2(슬라이스당 120 B × 256 디스크립터), ffn down·attn_out 출력 store 2,510(64 디스크립터). 단일 디스크립터
  store는 337~434. (a) 클러스터 안에서 switch ring gather로 슬라이스 1개에 모은 뒤 클러스터당 1 디스크립터로
  쓴다. (b) 이어지는 post-norm은 `Slice`로 로드(547) 후 `ReducingSlices`로 재배치(449)하던 것을 HBM에서
  ReducingSlices로 직접 로드한다.
- **변경 파일:** `src/device/shared/rmsnorm.rs`(`ReducingSlices` 공개, `load_reducing`, `normalize_reduced` /
  `normalize_add_reduced` / `normalize_add_gate_reduced` 추가; 기존 함수는 래퍼로), `src/device/shared/mlp.rs`
  (`project_down`·`feedforward`가 HBM/ReducingSlices 반환, global scale pass를 8슬라이스에서), `src/device/sliding/projection.rs`
  (`project_output`이 HBM 반환), `src/ops.rs`
- **공유 코드 영향:** rmsnorm 기존 함수 의미 불변(리팩터). mlp.rs 호출처는 `decoder_feedforward`뿐.

### 측정

| 단계 | qkv | attn_out | ffn | 비고 |
|---|---:|---:|---:|---|
| V23 | 51,631 | 30,487 | 181,625 | |
| (a) attn_out: `Broadcast1 { slice1: 32, slice0: 8 }` gather, 4원소 패킷 | | 33,050 | | switch 4,615 (ring 256 × Time 15), store 2,510 → 458 |
| (a) 12원소 패킷(Time 5, commit_trim 24 B) | | 30,483 | | switch 2,055 — store 절감분과 상쇄. **기각** |
| (b) 직접 로드 (세 커널) | **50,981** | **30,037** | **179,922** | qkv −650 (head의 relayout 449 + 로드 차), attn_out −450, ffn −1,703 (relayout 449×2 + global scale pass 1,241 → ~200 + 잔여) |
| **기하평균 (V0 대비 누적)** | | | | **5.180** |

- **정확도:** 수치 동일(로드 경로·pass 레이아웃만 변경; global scale 곱은 같은 f32 연산).
- **측정 방식:** makespan only.
- **컴파일 교훈:** `vector_inter_slice_reduce`는 innermost 슬라이스 축만 줄인다 — 바깥 축을 reduce하려는 레이아웃
  (`m![1 # 8, H / 60 % 32]`)은 visa 단계에서 `VRU reduce axes must be innermost in Partitioning`. `SwitchConfig`의
  `CustomBroadcast { ring_size }`는 패딩 축·순열을 지원하지만(std 테스트 참조) stride가 있는 live 집합은 ring 전체
  비용을 낸다.

### 판정: makespan 측정 (실측 대기)

- **배운 것:** HBM store 비용은 디스크립터 수와 레이아웃 슬롯 수에 묶이고, 그것을 줄이는 switch는 ring 크기에
  묶인다(둘 다 ~2k). hop을 줄이는 정답은 "모으기"가 아니라 **hop 자체를 없애거나(V25) 소비자 레이아웃으로 직접
  로드하기(V24b)**.

## V25_ffn_geglu_two_clusters

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V24_gather_before_hbm_store`
- **가설:** ffn DMA 165k 중 up/gate → geglu → down 사이의 hop이 13.9k: up/gate 결과 store 4,535×2(`m![L / 60 % 128, 1 # 2]`,
  256 디스크립터), geglu 입력 로드 1,328×2, geglu 출력 store 2,219. up과 gate는 같은 슬라이스에 같은 60행이 있으므로
  geglu를 그 레이아웃에서 직접 계산하면 앞의 네 DMA가 사라진다. 60 f32 = 7.5 패킷이라 8-wide 패킷에 안 맞는 문제는
  V18의 패딩 패킷 기법(`fetch::<m![L / 4 % 15], m![L % 4 # 8]>` → `narrow_split::<m![L / 4 % 15, 1 # 2], m![L % 4]>`)으로
  푼다. up/gate global scale은 contraction epilogue(inter-slice reduce 뒤 MulF scalar VRF)에 접어 넣어 geglu의
  scale pass 2개도 없앤다. geglu 출력 store는 256 디스크립터(4,535)로 남지만, 순이득 ≈ −9k. 2단계로 출력 앞에
  ring-4 pair gather(`Broadcast1 { slice1: 2, slice0: 2 }`, 비용 ~60+고정)를 붙여 128 디스크립터(2,219)로 줄이는 것을
  시도한다.
- **변경 파일:** `src/device/shared/mlp.rs`(`contract_up_gate_rows` epilogue, `geglu` 대체, `feedforward`)
- **공유 코드 영향:** mlp.rs 호출처는 `decoder_feedforward`뿐.
- **정확도:** 곱셈 순서 동일(sum × global_scale → bf16 → × gelu); 중간 bf16 반올림이 한 번 줄어 오히려 정밀.
- **예상:** ffn −9k ~ −11k.

## V26_qkv_hsplit_x_halved

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V25_ffn_geglu_two_clusters`
- **가설:** qkv DMA 큐 47k 중 복제 x 로드 5.4k(3.9 MB)와 x_trf 스테이징 1.2k(임계 경로)는 슬라이스당 x 전체
  [H]를 필요로 해서 생긴다. 투영을 H/1920 열 반으로 나누면(Q: `m![Qs / 16 % 128, H / 1920]` 16행×1920열,
  K/V: `m![Ps / 8 % 128, H / 1920]` 8행×1920열, 2-way `vector_inter_slice_reduce`) 슬라이스당 x가 절반이라
  복제 로드 바이트가 절반(V15의 8부 사본 기법 유지: `m![Dummy256 / 16, Dummy8, H / 1920]`), x_trf도 절반.
  weight DMA는 바이트 동일, 행 세그먼트 2배(모델상 +0.2 cycle/세그먼트 ≈ +1.7k)라 순이득은 x 쪽에서 나온다.
  헤드 gather는 reduce 출력의 `1 # 2` 패딩이 섞인 ring-64(`Broadcast1 { slice1: 64, slice0: 2 }` → ring 128)가 된다.
- **변경 파일:** `src/device/sliding/projection.rs`, `src/ops.rs`
- **예상:** qkv −2k ~ −3k.

## V27_attnout_scale_in_rmsnorm

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V26_qkv_hsplit_x_halved`
- **가설:** attn_out 타임라인 head 3.75k = x 청크 594 + `?` 838 + **채널 scale 로드 1,318**(64 디스크립터 × 120 B)
  뒤에야 첫 weight 타일이 시작된다. scale은 V18에서 contraction epilogue로 옮겨 tile0 pass의 소비자가
  됐기 때문에 DMA 큐 선두에 선다. scale 곱을 post-attn rmsnorm의 두 pass(mean_square 앞의 MulF, 마지막
  정규화 pass의 MulF)에 접어 넣으면 scale 로드는 ReducingSlices 8 디스크립터(~550)로 tail에서 마지막 타일
  pass와 겹치고, epilogue의 narrow_split/MulF/widen이 빠진다. rms(s⊙x)는 scale된 값으로 계산하므로 수치 동일.
- **변경 파일:** `src/device/sliding/projection.rs`, `src/device/shared/rmsnorm.rs`(`normalize_scaled_add_reduced` 추가), `src/ops.rs`
- **예상:** attn_out −1.3k.

## V28_ffn_16row_tiles

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V27_attnout_scale_in_rmsnorm`
- **가설:** FFN DMA 165k 중 타일당 고정비(DmaLoad 디스크립터 고정 ~550 + `?` 838) × 15. V12에서 ROWS_PER_PASS=12가
  scale VRF 한도(8 KB)로 정해졌지만 16행 × 120 f32 = 7.7 KB도 들어간다. 60 = 16+16+16+12로 행렬당 타일 4개
  (합계 12)로 하면 고정비 3개(≈4.2k)와 Sub 프리로드 3개가 빠진다. transpose는 `= 16 / 4`.
- **변경 파일:** `src/device/shared/mlp.rs`
- **공유 코드 영향:** vision/audio MLP 경로
- **예상:** ffn −4k.

## V13_ffn_dma_trims

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V12_ffn_rows_per_pass_12`
- **가설:** V12에서 down 구간은 타일당 DMA 9.5k(weight 6.3k + scale 2.4k + `?` 0.8k) vs Main 7.75k로 DMA-bound.
  DMA 큐에서 줄일 수 있는 것: (a) geglu 출력의 DM→DM relayout 11,687(mlp.rs:364, down 타일 로드 사이에 끼어
  Main을 5k 멈춤) → V7처럼 HBM에 30 KB 쓰고 ByColumns로 로드(예상 ~3k); (b) scale 타일 로드 20×2,716 + 10×2,394
  = 51k — 120~240 B 행이라 300~540 B/cycle. 행렬당 한 번(14 KB/슬라이스)이면 V0 실측 8.3k×2 + V6c 19k = 36k.
  단, 큰 로드가 첫 타일 앞에 오면 head가 늘 수 있다(V6d에서 4행 타일 때는 상쇄됨).
- **변경 파일:** `src/device/shared/mlp.rs`
- **공유 코드 영향:** vision/audio MLP 경로
- **예상:** (a) −7k, (b) −10k~−15k.

### 측정

| 단계 | ffn makespan | 판정 |
|---|---:|---|
| (a) geglu → HBM 스크래치 → ByColumns 로드 | 352,164 → **348,874** | 채택 |
| (b) + scale 행렬당 1회 로드 (`.view().tile()`로 pass별 참조) | 354,617 | **기각** — DMA busy 273k→258k인데 스케줄러가 8.3k scale 로드를 첫 타일 앞에 배치, Main 시작 15k→20.6k |

- **컴파일 교훈:** 이미 행 타일된 뷰에 열 타일을 다시 걸 때는 매핑에 바깥 패딩을 함께 적는다
  (`m![L % 60 = 12 # 60, H / 16 = 120 # 240]`), 아니면 `Output shape mismatch for IndexAccess`.
- **판정:** makespan 측정 (실측 대기), 누적 기하평균 2.866.

## V15_x_replicate_hbm_copies

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V14_two_clusters`
- **가설:** qkv의 `x_hbm.to_dm()`(Replicated, 512 슬라이스)는 7.7 MB를 18,400 cycle에 옮겨 420 B/cycle —
  같은 바이트 수의 down 타일 로드(930 B/cycle)의 절반이다. 디스크립터 수는 같으므로 원인은 **512개
  디스크립터가 전부 같은 7.5 KB HBM 구간을 읽는 것**(book: "channel interleaving: spread across 32 channels";
  7.5 KB는 채널 몇 개에만 걸친다)으로 추정. x를 `HbmTensor<bf16, Chip, m![Dummy8, H]>`에 8부 쓰고
  (8 디스크립터), DM 슬라이스 매핑 `m![Dummy32, Dummy8]`로 슬라이스마다 다른 사본을 읽게 하면 채널이
  분산된다.
- **변경 파일:** `src/ops.rs` (qkv 본문), 필요 시 `layout.rs`
- **리스크:** HBM 쪽 Dummy 축과 DM 쪽 Dummy 축의 대응을 DSL이 "사본 선택"으로 해석하는지 미지수.
- **예상:** qkv −9k (18.4k → ~9k).

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `sliding_project_qkv` | 73,445 → **60,412** | — | 1.216 |
| **기하평균 (V0 대비 누적)** | | | **4.434** |

- x 복제 DmaLoad **18,400 → 5,367** (util 0.157 → 0.539). `HbmTensor<bf16, Chip, m![Dummy8, H]>`에 8부 쓰고
  DM 슬라이스 매핑 `m![Dummy256 / 8, Dummy8]`로 로드(HBM에 있는 Dummy 축은 사본 선택, 없는 축은 복제) 후
  `unsafe reshape`로 `Replicated` 타입으로 되돌림. axes.rs는 채점에서 무시되므로 새 Dummy 축을 추가하지 않고
  `Dummy256 / 8`로 32를 만들었다.
- **정확도:** 수치 동일. **측정 방식:** makespan only.

### 판정: makespan 측정 (실측 대기)

- **배운 것:** 복제 로드의 병목은 디스크립터 수가 아니라 **같은 HBM 구간의 반복 읽기**(채널 집중)였다.
  attn_out(x 2k)과 ffn(x 3.4k)의 청크 로드도 같은 원리가 적용될 수 있으나 절대량이 작다.

## V17_qkv_hoist_weight_loads

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V16_rmsnorm_fused_residual`
- **동기:** 기하평균 점수에서 커널별 배율이 qkv 1.93× / attn_out 5.58× / ffn 9.06×로 qkv가 가장 뒤처진다
  (사용자 지시: 덜 개선된 커널에 집중).
- **가설:** V16 qkv 타임라인(60.4k): 0–6.6k rmsnorm, 6.7–12k x 복제 로드, **12–25.5k Q weight 로드**, 26–33k K,
  37–44k V, 이후 rope/rmsnorm tail 11k. weight 로드는 x와 무관한데 소스에서 `project_query` 안에서 발행되어
  x 뒤에 줄을 선다. 스케줄러는 소스 순서상 앞의 DMA를 뒤로 미루지 않으므로(V13b) 세 weight `to_dm`을
  rmsnorm 앞에서 발행하면 Q 로드 13.5k가 rmsnorm·x 스테이징(6.6k)과 겹치고 K/V도 연달아 흐른다.
- **변경 파일:** `src/device/sliding/projection.rs`(로더 함수 분리), `src/ops.rs`(qkv 본문 순서)
- **예상:** qkv 60.4k → ~50k.

### 측정

| 변형 | qkv | 비고 |
|---|---:|---|
| (a) 세 weight `to_dm`만 rmsnorm 앞으로 이동 | 60,412 | **불변** — 스케줄러는 발행 순서가 아니라 소비자 기준으로 DMA를 배치 |
| (b) Q만 LUT pass 분리(소비자를 rmsnorm 앞에 발행), contraction은 bf16에서 | **59,216** | Q 로드 3.6–17.1k로 선두, LUT 5k가 x 로드와 겹침. 그러나 K 로드는 여전히 32k 시작(7.7k 공백) |
| (c) K/V도 LUT 분리 + x_trf를 그 앞에 스테이징 | 60,375 | K/V 로드·LUT가 Q contract 뒤로 밀림 — Main도 프로그램 순서를 따르지 않음 |

- **채택:** (b). 기하평균 4.649.
- **배운 것:** 스케줄러는 DMA도 Main도 리스트 스케줄링(자체 우선순위)이며 소스 순서로 제어되지 않는다.
  DMA를 앞당기려면 **그 DMA를 소비하는 명령이 의존성 그래프상 일찍 필요해져야** 한다. qkv의 잔여
  59k는 DMA 36k(Q 13.5 + x 5.4 + K 7 + V 7 + 소량) + 마지막 로드 뒤 tail(V pass·normalize·scatter ≈ 8k)
  + 스케줄러 slack(~10k: q/k/v rmsnorm·rope 소형 pass들이 직렬). 하한은 ~50k.

## V19_qkv_tail_heads_layout

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V18_attnout_scale_in_epilogue`
- **동기:** qkv가 기하평균상 가장 뒤처짐(1.97×). V17 타임라인의 마지막 로드 뒤 tail 13k는 q/k/v 후처리 pass
  10여 개(각 0.3~1.3k)가 **슬라이스 1개**(Slice 레이아웃)에서 직렬로 도는 것 + K의 HBM hop이 V 로드 뒤에 막힌 것.
- **가설:** rope는 이미 `KvHeadsAcrossSlices = m![1 # 32, Ns]`(헤드당 슬라이스 1개)로 옮겨서 계산한다.
  투영 결과를 HBM hop에서 곧장 그 레이아웃으로 로드하고, q/k/v rmsnorm을 그 레이아웃용으로 다시 쓰고
  (`normalize_*_heads`; Ds 축약이 슬라이스 안에서 끝남), rope의 앞 InterTranspose(q 2.3k, k 1.3k)와 뒤 Broadcast1을
  없애며, q 저장·k/v scatter를 그 레이아웃에서 직접 하면 pass당 데이터가 1/8이 되고 전환 pass 4개가 사라진다.
- **부수 실험:** V19a — K/V 투영 순서를 바꿔 tail 길이를 조정하려 했으나 스케줄이 완전히 동일(스케줄러는 소스
  순서 무시). 기각.
- **변경 파일:** `src/device/layout.rs`(`HeadSlices`), `src/device/sliding/rmsnorm.rs`(함수 추가),
  `src/device/sliding/rope.rs`(`apply_rope_heads` 추가), `src/device/sliding/projection.rs`(반환 레이아웃), `src/ops.rs`
- **예상:** qkv −6k ~ −8k.

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `sliding_project_qkv` | 59,216 → **55,811** | — | 1.061 |
| **기하평균 (V0 대비 누적)** | | | **4.919** |

- 첫 컴파일 통과. 후처리 pass들이 1,304/1,287 → 345~409 cycle로 줄었고 InterTranspose 2개·Broadcast1 2개가
  사라졌다. 남은 tail(V 로드 종료 45.5k → 55.8k)은 V contract 2.7k + V scale 1k + **K/V의 HBM hop이 V weight
  로드 뒤에 DMA 큐에서 막혀** k 후처리가 48.4k에야 시작하는 것 + scatter 2개.
- **정확도:** 연산 동일(레이아웃만 변경).

### 판정: makespan 측정 (실측 대기)

## V20_qkv_tail_per_cluster

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V19_qkv_tail_heads_layout`
- **가설:** q/k/v 결과를 한 클러스터로 모으는 HBM hop(각 store+load ≈ 1.5k, 그리고 DMA 큐에서 큰 weight 로드
  뒤에 막힘)을 없앤다. 각 클러스터는 이미 4개 헤드(2048/1024 = 4 × Ds)를 갖고 있으므로, 클러스터 안에서
  ring-64 `Broadcast1{slice1: 64}` switch(64 슬라이스 × 8행 → 슬라이스당 1헤드, ~64 cycle)로 헤드/슬라이스
  레이아웃 `(m![Ns / 4], m![Ns % 4, 1 # 64])`을 만들고, `normalize_*_heads`·`apply_rope_heads`를 (C, S) 제네릭으로
  바꿔 두 클러스터에서 4헤드씩 병렬 처리한 뒤, q 저장·k/v scatter를 그 레이아웃에서 직접 한다. cos/sin은
  HBM 스크래치를 거쳐 양 클러스터에 복제.
- **변경 파일:** `src/device/sliding/{projection,rmsnorm,rope}.rs`, `src/ops.rs`
- **리스크:** 두 클러스터 레이아웃에서의 `dma_scatter`/`to_hbm_view`; padded `1 # 64` 슬라이스 축에서의 vector pass.
- **예상:** qkv −5k (~50k).

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `sliding_project_qkv` | 55,811 → **51,631** | — | 1.081 |
| **기하평균 (V0 대비 누적)** | | | **5.048** |

- 첫 컴파일 통과. ring-64 gather는 스케줄에 보이지 않을 만큼 작다. DMA 91% busy(47k) — qkv는 이제 DMA-bound.
  V 로드 종료(43.6k) 뒤 tail 8k: V contract 2.7k + cos/sin gather(43.7k, V 로드 뒤에 큐잉됨)를 기다리는 q/k rope
  + scatter 2개.
- **정확도:** 연산 동일. `dma_scatter`/`to_hbm_view`가 두 클러스터 레이아웃에서 컴파일됨(실측으로 확인 필요).

### 판정: makespan 측정 (실측 대기)

## V22_attnout_weight_tiles

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V21_qkv_rope_tables_early` (= V20 + rope 별칭 수정; V21 코드는 되돌림)
- **가설:** attn_out(31.1k)은 weight 로드 13.5k 뒤에 융합 LUT+contract 5k가 직렬로 붙는다. V10처럼 행을 타일로
  나눠 전부 선로드하면 타일 k의 pass가 타일 k+1의 로드와 겹친다. 두 클러스터 레이아웃(슬라이스당 60행)에서
  20행 × 3타일(4의 배수 ✓, `= 20 / 4` transpose). 타일당 `DmaLoad ?` 838이 2개 늘지만 DMA가 임계가 아니어서 흡수.
- **변경 파일:** `src/device/sliding/projection.rs`
- **예상:** attn_out −3k (~28k).

### 측정: attn_out 31,113 → **30,487** (−0.6k), 기하평균 5.082

- 타일 DMA 4,870×3 = 14.6k(통짜 13.5k보다 1.1k 많음) + 타일 사이 `DmaLoad ?` 838 ×2 → 겹침 이득 3.2k 중 2.6k 상쇄.
  채택하되 기대에 못 미침.

### 판정: makespan 측정 (실측 대기)

## V23_ffn_whole_scale_loads

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V22_attnout_weight_tiles`
- **가설:** V14 이후 FFN DMA 173k 중 scale 타일 15 × 2,394 = 36k(11 MB, 슬라이스당 12행 × 120 B 조각 → 300 B/cycle).
  행렬당 한 번(슬라이스당 60 × 120 B = 7.2 KB, 3.7 MB)이면 ~5k씩 3회 = 15k. V13b는 단일 클러스터·첫 타일 앞 배치로
  head가 8k 늘어 실패했지만, 지금은 첫 타일이 6.3k이고 up/gate 두 행렬의 scale 로드가 각 ~5k라 판단 재시도.
- **변경 파일:** `src/device/shared/mlp.rs`
- **예상:** ffn −10k ~ −20k (또는 head 증가로 무효).

### 측정: ffn 186,976 → **181,625** (−5.4k), 기하평균 5.131

- 행렬당 scale 로드는 9,774(3.7 MB → 380 B/cycle: 120 B 행 조각이라 여전히 느림) ×3 ≈ 29k vs 36k. 이번엔 첫
  타일(6.3k) 뒤에 배치되어 head 증가 없음. DMA busy 173k → 166k.

### 판정: makespan 측정 (실측 대기)



## V21_qkv_rope_tables_early

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V20_qkv_tail_per_cluster`
- **가설:** rope의 cos/sin gather(934×2)와 HBM hop이 V weight 로드 뒤에 DMA 큐잉되어 q/k rope가 46.6k에야
  시작한다. gather→hop→VRF 스테이징을 커널 맨 앞(Q LUT 뒤)에서 발행하면 Sub이 한가해 일찍 잡히고(V17에서
  Main 소비자가 DMA를 앞당긴 것과 같은 원리) q/k rope가 V 로드 중에 끝나 tail이 V 경로만 남는다.
- **변경 파일:** `src/device/sliding/rope.rs`(`stage_rope_tables`/`RopeTables`), `src/ops.rs`
- **예상:** qkv −2k ~ −3k.

### 측정: **51,631 → 51,631 (불변) — 기각**

- cos gather는 18k로 앞당겨졌으나 sin gather는 여전히 43.7k(V 로드 뒤). 순서를 바꿔도(sin 먼저) 첫 번째 것만
  앞당겨진다 — 스케줄러가 weight 로드 앞에 두는 소형 DMA를 하나로 제한하는 듯. 코드는 되돌렸다.
- **부수 발견:** device 함수 안에서 **사용자 struct 생성은 ICE**(`not yet implemented: RopeTables { .. }`) —
  여러 값을 돌려줄 때는 튜플을 쓴다.
- V20 커밋의 rope.rs에 `HeadSlices` import 정리로 생긴 컴파일 오류가 있었다 → 이 브랜치와 V20 브랜치에 수정 커밋.




## V18_attnout_scale_in_epilogue

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V17_qkv_hoist_weight_loads`
- **가설:** attn_out tail(13k/34.8k)에서 채널 scale pass가 1920 타일 2개(Sub 743 + Main 761 ×2 ≈ 3k)를 쓴다.
  contraction 체인은 `contract_lane → vector_init → inter_slice_reduce → vector_final`로 끝나는데, book이
  InterFirst(reducer 뒤 intra chain) 순서를 지원하므로 reduce 뒤에 `narrow_split → MulF(scale_vrf) →
  widen_concat`을 붙이면 slice당 60행의 scale 곱이 epilogue에서 끝나 tail pass가 사라진다. scale VRF는
  행당 1값을 `m![H % 60, 1 # 8]` 패딩 패킷으로 스테이징.
- **변경 파일:** `src/device/sliding/projection.rs`
- **리스크:** contract 출력 패킷(`1 # 8`)의 narrow_split 분해(`m![H % 60, 1 # 2], m![1 # 4]`)와 scale VRF
  매핑 일치가 미검증.
- **예상:** attn_out −3k (8.6%).

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `sliding_attention_output` | 34,776 → **31,113** | — | 1.118 |
| **기하평균 (V0 대비 누적)** | | | **4.822** |

- 첫 시도에 컴파일. scale VRF: DM `m![H % 60]`을 `fetch::<m![H % 60], m![1 # 8]>() → fetch_cast → collect`로
  행당 패딩 패킷 스테이징; epilogue `inter_slice_reduce → intra_slice_tag → narrow_split::<m![H % 60, 1 # 2],
  m![1 # 4]> → MulF → widen_concat::<m![H % 60], m![1 # 8]> → vector_final`. 같은 기법을 qkv의 Q/K/V scale
  pass(각 ~1k+)에도 적용 가능 → V19 후보.
- **정확도:** scale을 f32 단계에서 곱하므로 이전(bf16 재로드 후 곱)보다 정밀.

### 판정: makespan 측정 (실측 대기)



## V16_rmsnorm_fused_residual

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V15_x_replicate_hbm_copies`
- **가설:** attn_out tail 16k와 ffn tail에는 rmsnorm(ReducingSlices 레이아웃, Main pass 4개) 뒤에 residual add
  (1920 타일 2개: Sub 743 + Main 521 ×2 ≈ 2.5k)와 ffn의 layer gate pass(~1.2k)가 별도로 붙는다. rmsnorm의
  마지막 정규화 pass는 이미 vector 체인(DivF rms, MulF weight)이므로 residual VRF(480 f32 = 1.9 KB)를
  `vector_clip(Add)`로, gate를 `MulF(Mul1)`로 접어 넣으면 tail pass들이 사라진다. `normalize`는 그대로 두고
  `normalize_add`, `normalize_add_gate`를 추가(full/vision/audio 경로 무영향).
- **변경 파일:** `src/device/shared/rmsnorm.rs`(함수 추가), `src/ops.rs`(attn_out·ffn 본문)
- **예상:** attn_out −2.5k, ffn −3.5k.

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `sliding_attention_output` | 38,240 → **34,776** | — | 1.100 |
| `decoder_feedforward` | 191,122 → **186,976** | — | 1.022 |
| **기하평균 (V0 대비 누적)** | | | **4.618** |

- **컴파일 교훈:** `vector_clip`은 `vector_narrow_split` 이전(8-wide) 단계 전용 — 정규화 곱셈 뒤에는
  `vector_fp_binary(FpBinaryOp::AddF, &vrf)`를 쓴다(`no method named vector_clip ... Way4`).
- **정확도:** 연산 순서 동일(정규화 → ×w → +residual → ×gate), 중간 bf16 반올림이 한 번 줄어 오히려 정밀.
- **측정 방식:** makespan only

### 판정: makespan 측정 (실측 대기)



## V14_two_clusters

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V13_ffn_dma_trims`
- **가설:** `src/device/layout.rs`의 `Cluster = m![1 # 2]`는 클러스터 축 "1 live + 1 패딩"이다. 즉 세 커널
  모두 칩의 두 클러스터 중 하나(256 슬라이스, DM 128 MB, DMN 8개 = 1 KB/cycle)만 쓴다. 관측 DMA 최고치
  ~590 B/cycle(256 슬라이스 기준)이 1 KB/cycle 한도 아래에 있는 것과 부합. 행(또는 열)을 클러스터 축
  `m![H / 1920]` 같은 실제 분할로 바꾸면 슬라이스당 작업이 절반, DMA 대역폭 2배가 될 수 있다.
  프로브: attn_out을 Cluster `m![H / 1920]` × Slice `m![H / 60 % 32, Qs / 512]`(60행 × 512열, 8-way reduce)로.
- **변경 파일:** `src/device/sliding/projection.rs` (프로브), 성공 시 `layout.rs`·`mlp.rs`·rmsnorm 등 전반
- **리스크:** 클러스터 축의 실제 분할을 DSL/DMA가 지원하는지, 클러스터 간 gather(`to_dm` → `Cluster 1#2`)가
  되는지 미지수. `#[device(chip = 1)]`과 HBM 텐서의 `Chip` 매핑은 그대로.
- **예상:** 성공 시 attn_out weight DMA 29k → ~15k, Main 절반. 세 커널 전부에 적용 가능.

### 측정 (단계별, 브랜치 안에서 누적)

| 단계 | 변경 | qkv | attn_out | ffn |
|---|---|---:|---:|---:|
| V13 | (출발점) | 93,127 | 50,110 | 348,874 |
| attn_out | Cluster `H/1920` × Slice `H/60 % 32, Qs/512`, 60×512/슬라이스, 8-way reduce, HBM gather | | **38,240** | |
| ffn down | Cluster `H/1920` × Slice `H/60 % 32, L/1920`, 12행 타일 5개 선로드 | | | 291,679 |
| ffn up/gate | Cluster `L/7680` × Slice `L/60 % 128, H/1920`, 60×1920/슬라이스(dequant 1회), 2-way reduce, HBM으로 geglu 레이아웃 복귀; x는 슬라이스당 절반만 | | | **191,122** |
| qkv v1 | Q 8행·K/V 4행 두 클러스터, 결과를 512 슬라이스에서 직접 HBM 저장 | 176,882 (악화) | | |
| qkv v2 | x를 `BothClusters=m![Dummy2]`로 1회 로드해 reshape 공유; 클러스터 안 switch gather 후 HBM 2 디스크립터 | **73,445** | | |
| **기하평균 (V0 대비 누적)** | | | | **4.147** |

- **결정적 관측:** 같은 12행 down 타일 로드가 256 슬라이스 때 6,326 cycle, 512 슬라이스(2배 바이트) 때도
  6,326 cycle → **DMA 처리량이 정확히 2배**. attn_out weight 15.7 MB 전체가 13,512 cycle(util 0.86).
- **제약:** 클러스터 간 DM→DM DMA는 `synchronization_checker`가 거부(`T13 is used by DmaCommand#O8, while
  not synchronized from other clusters`); `DmTensor::to_dm`은 Cluster 타입을 못 바꾸고(`to_dm_view`만 가능).
  **HBM 경유 gather는 통과**한다. 작은 조각(슬라이스당 8~16 B)을 512 슬라이스에서 HBM으로 직접 쓰면
  디스크립터 비용으로 37k — 먼저 클러스터 안에서 switch로 모아 클러스터당 1개 조각으로 쓴다.
- **정확도:** 수치 동일(행 분할·partial 합산 f32). 전체 테스트 바이너리 빌드 통과.
- **측정 방식:** makespan only
- **지배 context:** qkv DMA 84% (x 복제 18.4k, Q 13.5k, K/V 7k×2) / attn_out DMA 56% + tail 16k /
  ffn DMA 90% (weight 타일 6.3k×15, scale 2.4k×15, HBM hop 4.5k×2+2.2k).

### 판정: makespan 측정 (실측 대기)

- **배운 것:** 하루 종일 싸운 "DMA-bound"의 절반은 칩의 절반만 쓰고 있었기 때문이었다. 베이스라인의
  `Cluster = m![1 # 2]`를 의심하지 않은 것이 가장 비싼 가정이었다.
- **다음 후보:** qkv x 복제(18.4k; Q를 H-split으로 바꾸면 절반), attn_out tail 16k(scale·rmsnorm·residual),
  ffn scale 타일 36k·`DmaLoad ?`, geglu(단일 클러스터 vector 작업)도 두 클러스터로.


## V12_ffn_rows_per_pass_12

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V11_residual_1920_tiles`
- **가설:** V6 이후 FFN은 DMA 83% busy이고, dequant pass마다 설명 없는 4 KB `DmaLoad ?`(838 cycle)가 하나씩
  붙어 90 pass × 838 ≈ 75k가 DMA 큐를 차지한다. 명령 수(스케줄러 오버헤드, Sub 프리로드 ×90)도 pass에 비례.
  `ROWS_PER_PASS`를 12로 올리면 pass가 30개로 줄어 이 고정비가 1/3이 된다. 12행이 막혔던 이유는 scale VRF
  (12 × 240 × 4 B = 11.5 KB > 8 KB)였는데, up/gate의 dequant를 1920열 절반씩 두 번 하면 scale이 5.8 KB로
  들어간다(LUT pass 수는 60→... 절반 분할 때문에 up/gate는 10×2=20, down 10 → 총 30). down은 L 청크가
  1920이라 12행 scale이 그대로 5.8 KB.
- **변경 파일:** `src/device/shared/mlp.rs`
- **공유 코드 영향:** vision/audio MLP 경로 동시 영향
- **리스크:** 12행 transpose `m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]` 문법 미검증 → 실패 시 contraction만
  4행 서브체인 3개로. DM: 타일 4개 동시 상주 시 f4 92 KB + bf16 92 KB (up/gate) — 여유.
- **예상:** ffn −40k ~ −70k.

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `decoder_feedforward` | 408,566 → **352,164** | — | 1.160 |
| **기하평균 (V0 대비 누적)** | | | **2.857** |

- **정확도:** 수치 동일(행 묶음만 변경). 전체 테스트 바이너리 빌드 통과.
- **측정 방식:** makespan only. 명령 수 1,112 → 632.
- **지배 context:** DMA 79.6% / Main 67.9% / Vector 67.8%. 소스별: up/gate 타일 DmaLoad 10×10,271 / down 타일
  10×6,326 / 절반 dequant 20×6,040 / down dequant 10×6,040 / up·gate contract 10×3,145 / scale 타일 DMA 51k /
  `DmaLoad ?` 30×838 = 25k (V6의 75k에서) / x 복제 18.4k / geglu relayout 11.7k.
- **컴파일 교훈:** 열 타일 뷰(`H = 1920 # 3840`)는 fetch에서 `/ 32`로 분해 불가("live input axis is left unread")
  → 패킷 전체 `m![H = 1920]`로 fetch하고 collect에서 flit로 쪼갠다. 12행 transpose
  `m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]`는 된다.

### 판정: makespan 측정 (실측 대기)

- **배운 것:** pass당 고정비가 실제로 컸다(−56k). dequant는 여전히 vector 엔진(≈3.8 elem/cycle)에 묶여
  Main 180k. 다음은 Sub 컨텍스트에 vector 작업을 나누는 것(LUT는 Main 전용이므로 f8 pass 별도, 12.6 elem/cycle).

## V11_residual_1920_tiles

- **상태:** makespan 측정 (2026-09-09)
- **분기점:** `V10_attn_weight_tiles_fused_lut`
- **가설:** attn_out과 ffn 모두 `shared::residual::add`로 끝나는데, 480 원소 × 8 타일마다 Sub 프리로드(383) +
  Main add(341)가 붙어 ~5.8k. 1920 × 2 타일(f32 7.5 KB < VRF 8 KB)이면 ~1.5k.
- **변경 파일:** `src/device/shared/residual.rs` (`add`만; `add_vision`은 그대로)
- **공유 코드 영향:** full attention 경로(`full_attention_output`)도 `residual::add`를 쓴다 — Stage 2에서 같이 이득.

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `sliding_project_qkv` | 93,127 → 93,127 | — | 1.000 |
| `sliding_attention_output` | 53,848 → **50,110** | — | 1.075 |
| `decoder_feedforward` | 412,304 → **408,566** | — | 1.009 |
| **기하평균 (V0 대비 누적)** | | | **2.716** |

- **정확도:** 수치 동일(타일 폭만 변경).
- **측정 방식:** makespan only

### 판정: makespan 측정 (실측 대기)

## V10_attn_weight_tiles_fused_lut

- **상태:** makespan 측정 (2026-09-09)
- **분기점:** `V6_ffn_upgate_overlap` (`14b428d`)
- **가설:** 위 요약 보드 참조. V5(LUT 융합)를 흡수.
- **변경 파일:** `src/device/sliding/projection.rs` (`project_output`, `project_query`, `project_one_kv_matrix`)
- **공유 코드 영향:** 없음

### 측정

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `sliding_project_qkv` | 95,433 → **93,127** | — | 1.025 |
| `sliding_attention_output` | 58,015 → **53,848** | — | 1.077 |
| `decoder_feedforward` | 412,304 → 412,304 | — | 1.000 |
| **기하평균 (V0 대비 누적)** | | | **2.646** |

- **변형 비교 (attn_out):** 12행×5타일을 루프 안에서 로드 61,787(버퍼 재사용으로 DMA 직렬화) → 5타일
  선로드 + LUT 융합 **53,848** → 5타일 선로드 + LUT 분리 54,969. 융합 pass는 타일당 2,183 (분리 2,800).
- **변형 비교 (qkv):** Q 4×4행 + K/V 2×4행 타일 + 융합 = **97,866 (악화)** — DMA 93% busy라 타일당
  `DmaLoad ?` 838이 큐만 늘림. 융합만 적용 = 93,127.
- **정확도:** 수치 동일(LUT 결과를 DM에 쓰지 않고 바로 contraction).
- **측정 방식:** makespan only
- **지배 context:** attn_out DMA 71.8% (x 2.1k + 타일 5×5,785 + `?` 5×838) / Main 32%; qkv DMA 90.6%.

### 판정: makespan 측정 (실측 대기)

- **배운 것:** (1) `fetch → fetch_table_lookup::<bf16> → collect → contract_outer` 한 체인은 **컴파일된다**
  (V5 확인). (2) 타일화는 Main이 임계일 때만 이득; DMA-bound 커널에서는 타일당 고정 DMA 비용(838)이 손해.
  (3) attn_out tail 15k(relayout, scale, rmsnorm, residual)는 별도 표적 → V11.
- **다음 후보:** FFN ROWS_PER_PASS=12(scale VRF를 반으로 나눠 로드) · FFN scale pass Sub 오프로드 ·
  rmsnorm 경량화(공유) · qkv x 복제 18.4k.

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

### 측정 (attn_out만 프로브)

| 커널 | makespan (before → after) | RNGD cycles | speedup |
|---|---|---|---:|
| `sliding_attention_output` | 58,015 → 59,225 | — | 0.980 |

- weight DmaLoad: **26,734 → 26,734, util 0.433 → 0.433 (완전히 동일)**. 출력 permutation 때문에
  `contraction.to_dm` relayout만 952 → 2,162로 늘었다.
- **측정 방식:** makespan only

### 판정: **기각** (스케줄러의 DMA 비용 모델은 슬라이스 배치 순서를 보지 않는다)

- **이유:** 정적 스케줄에서 DmaLoad 비용은 바이트 수와 디스크립터 형태로만 정해진다. 실물에서는
  DMN 인터리빙이 영향을 줄 수 있으나 검증 수단이 없다. Arena 실측이 되면 V7 vs V8 브랜치를 그대로
  A/B 제출할 가치는 있다(둘 다 push됨). 코드는 `V8_weight_rows_interleaved_dma` 브랜치에만 있다.
- **배운 것:** makespan 관점에서 DMA는 **바이트 수**와 **겹침**으로만 줄어든다.

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
