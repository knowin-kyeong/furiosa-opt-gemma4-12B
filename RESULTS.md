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
| `V1_ffn_down_chunked_dequant` | `V0_baseline` | down proj: 슬라이스별 L/8 청크만 dequant, 재배치 DMA 제거 | — | — | — | — | — | — | 설계됨 |
| `V2_attnout_rows_over_256_slices` | `V0_baseline` | O proj: 32→256 슬라이스 (15행/슬라이스), Qs 청킹·partial add 제거 | — | — | — | — | — | — | 설계됨 |
| `V3_qkv_hsplit_no_broadcast` | `V0_baseline` | QKV: H를 8슬라이스로 분할해 x 전체 브로드캐스트(62k) 제거, inter-slice reduce | — | — | — | — | — | — | 설계됨 |
| `V4_attnout_qsplit_no_broadcast` | `V2` | O proj: Qs를 8슬라이스로 분할해 x 브로드캐스트(66k) 제거 | — | — | — | — | — | — | 설계됨 |
| `V5_lut_in_contract_chain` | SOTA | f8→bf16 table lookup을 별도 pass 없이 contraction 체인 안에서 수행 | — | — | — | — | — | — | 설계됨 |
| `V6_ffn_upgate_overlap` | SOTA | up/gate 루프 인터리브 + ROWS_PER_PASS 튜닝으로 DMA/Main 오버랩 | — | — | — | — | — | — | 설계됨 |

> 단위: cycle. `—` 미측정. `FAIL(accuracy)` tolerance 위반. 상태 전이는 RULES §2.1.

## 현재 SOTA

`V0_baseline` — 자세한 서사는 [SOTA.md](SOTA.md).

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

- **상태:** 설계됨 (2026-09-09)
- **분기점:** 당시 SOTA
- **가설:** V0에서 up과 gate는 완전히 직렬이다 — gate weight DMA(49k)는 up 경로가 끝나야
  시작하고, 각각 LUT 18k + scale pass 15×4.1k = 62k가 뒤따른다. 두 루프를 pass 단위로
  인터리브하면 gate DMA가 up의 Main 작업과 겹친다. 또 `ROWS_PER_PASS=4`(15 pass)는 슬라이스당
  bf16 타일 30KB만 쓰므로 8~12로 올려 instruction 수를 줄일 여지가 있다.
- **변경 파일:** `src/device/shared/mlp.rs` (`project_up_and_gate`)
- **공유 코드 영향:** `shared/mlp.rs` → vision/audio MLP 경로 동시 영향. 세 커널 makespan 모두 기록.
- **예상:** ffn −50k ~ −100k.

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

## V2_attnout_rows_over_256_slices

- **상태:** 설계됨 (2026-09-09)
- **분기점:** `V0_baseline`
- **가설:** `project_output`의 `HiddenRows = m![H/120, 1#8]`는 32개 행그룹 × 8 패딩 슬라이스다.
  스케줄 JSON에서 weight 타일이 512 슬라이스 전체에 할당되지만 슬라이스당 120행 × 1024열을
  처리하고 있다(LUT pass 19,463 = qkv Q pass(16행×3840)의 정확히 2배 → 슬라이스당 122,880 elem).
  즉 **행이 32개 슬라이스에만 실질 분배**되어 Qs를 4×1024로 청킹해야 했고(120×4096 bf16 =
  983KB > DM 512KB), 4번의 LUT(78k) + 4번의 contract(32k) + 3번의 partial add가 생겼다.
  `m![H/15]`(256 슬라이스 × 15행)로 바꾸면 슬라이스당 15×4096 f8 = 61KB, bf16 123KB로
  청킹 불필요 → LUT 1회(~10k) + contract 1회(~4k), add_partials 삭제.
- **변경 파일:** `src/device/sliding/projection.rs` (`project_output`, `output_partial`,
  `apply_output_channel_scale`; `HiddenRows` 타입)
- **공유 코드 영향:** 없음 (`sliding/`만; `full/projection.rs`는 별도 구현)
- **예상:** attn_out 194k → ~98k. 브로드캐스트 66k는 V4에서.

## V1_ffn_down_chunked_dequant

- **상태:** 설계됨 (2026-09-09)
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
