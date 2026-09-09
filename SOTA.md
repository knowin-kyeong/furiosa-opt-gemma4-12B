# SOTA.md — Road to SOTA

> **신기록이 나올 때만 갱신하는 문서.** 실패한 실험과 중간 시도는
> [RESULTS.md](RESULTS.md)에 있다. 여기는 "어떻게 여기까지 왔는가"의 서사만 남긴다.
> 추후 report의 원본이 된다.

## 현재 SOTA

| 항목 | 값 |
|---|---|
| 브랜치 | `V0_baseline` |
| 커밋 | (아래 이력 참조) |
| `sliding_project_qkv` | makespan 116,583 (RNGD 실측 대기) |
| `sliding_attention_output` | makespan 194,020 (RNGD 실측 대기) |
| `decoder_feedforward` | makespan 1,693,200 (RNGD 실측 대기) |
| **기하평균 speedup** | 1.000 (기준) |
| 측정일 | 2026-09-09 (makespan) |

## 잠정 (makespan 기준, 실측 전)

| 브랜치 | qkv | attn_out | ffn | 기하평균 |
|---|---:|---:|---:|---:|
| `V7_qkv_x_replicate_via_hbm` (V1+V2+V7 누적) | 95,433 | 58,015 | 609,223 | 2.250× |
| `V6_ffn_upgate_overlap` (V1+V2+V7+V9+V6 누적) | 95,433 | 58,015 | 412,304 | 2.559× |
| `V11_residual_1920_tiles` (V1+V2+V7+V9+V6+V10+V11 누적) | 93,127 | 50,110 | 408,566 | 2.716× |
| `V12_ffn_rows_per_pass_12` (…+V12 누적) | 93,127 | 50,110 | 352,164 | 2.857× |
| `V13_ffn_dma_trims` (…+V13 누적) | 93,127 | 50,110 | 348,874 | 2.866× |
| `V14_two_clusters` (…+V14 누적) | 73,445 | 38,240 | 191,122 | 4.147× |
| `V15_x_replicate_hbm_copies` (…+V15 누적) | 60,412 | 38,240 | 191,122 | 4.434× |
| `V16_rmsnorm_fused_residual` (…+V16 누적) | 60,412 | 34,776 | 186,976 | 4.618× |
| `V17_qkv_hoist_weight_loads` (…+V17 누적) | 59,216 | 34,776 | 186,976 | 4.649× |
| `V18_attnout_scale_in_epilogue` (…+V18 누적) | 59,216 | 31,113 | 186,976 | 4.822× |
| `V19_qkv_tail_heads_layout` (…+V19 누적) | 55,811 | 31,113 | 186,976 | 4.919× |
| `V20_qkv_tail_per_cluster` (…+V20 누적) | 51,631 | 31,113 | 186,976 | 5.048× |
| `V22_attnout_weight_tiles` (…+V22 누적) | 51,631 | 30,487 | 186,976 | 5.082× |
| `V23_ffn_whole_scale_loads` (…+V23 누적) | 51,631 | 30,487 | 181,625 | 5.131× |
| `V24_gather_before_hbm_store` (…+V24 누적) | 50,981 | 30,037 | 179,922 | 5.180× |
| `V25_ffn_geglu_two_clusters` (…+V25 누적) | 50,981 | 30,037 | 170,158 | 5.277× |
| `V28_ffn_tile_shapes` (…+V28 누적; V26·V27 미구현) | 50,981 | 30,037 | 168,757 | 5.292× |
| `V29_ffn_block_scale_after_contract` (…+V29 누적) | 50,981 | 30,037 | 165,733 | 5.324× |
| `V30_qkv_rope_tables_direct_gather` (…+V30 누적) | 48,638 | 30,037 | 165,733 | 5.408× |
| `V31_qkv_f8_contraction_no_lut` (…+V31 누적) | 46,541 | 30,037 | 165,733 | 5.488× |
| `V32_attnout_f8_contraction_no_lut` (…+V32 누적) | 46,541 | 29,318 | 165,733 | 5.533× |
| `V33_attnout_immediate_scale` (…+V33 누적) | 46,541 | 28,002 | 165,733 | 5.618× |
| `V35_attnout_trunc_split` (…+V35 누적; V34 기각·V35에 흡수) | 46,541 | 27,954 | 165,733 | 5.621× |
| `V36_qkv_scales_in_head_norms` (…+V36 누적) | 45,744 | 27,954 | 165,733 | 5.654× |
| `V37_ffn_upgate_pass_a_big_tiles` (…+V37 누적) | 45,744 | 27,954 | 158,712 | **5.736×** |

실측이 나오기 전까지는 SOTA가 아니다. Arena 승인 후 `./scripts/rngd_test.sh`로 확정한다.

## 갱신 이력

각 갱신은 아래 형식으로 **아래에 추가**한다 (시간순, 오래된 것이 위).

<!-- 템플릿
### V{n}_{description} — 기하평균 {x.xxx}× (이전 {y.yyy}× 대비 +{z}%)

- **날짜:**
- **한 줄:** (무엇을 바꿔서 이겼는가)
- **병목이었던 것:** (schedule 근거)
- **왜 통했는가:** (기전. "빨라졌다"가 아니라 "왜 빨라졌는가")
- **커널별 기여:** qkv {a}× / attn_out {b}× / ffn {c}×
- **포기한 것:** (정확도 여유, 코드 복잡도, Stage 2 경로 영향 등 trade-off)
-->

### V0_baseline — 기준선

- **날짜:** 2026-09-09
- **한 줄:** 원본 skeleton에 실험 인프라(RULES/RESULTS/SOTA/requirements)를 붙인 출발점.
  커널 코드 변경 없음.
- **상태:** 실측 대기. 최초 RNGD 측정값이 나오면 위 표와 RESULTS.md를 함께 채운다.

## 서사 메모 (report용)

최종 리포트에서 쓸 관점을 미리 모아둔다.

- Stage 1 점수가 **기하평균**이라는 점이 전략을 규정한다 — 한 커널의 극단적 최적화보다
  세 커널의 균형 잡힌 개선이 유리하다.
- 세 커널이 공유하는 코드(`device/shared/rmsnorm.rs`, `device/shared/mlp.rs`)가 있어,
  공유 코드 개선은 레버리지가 크지만 A/B 해석과 Stage 2 회귀 위험이 함께 커진다.
- `decoder_feedforward`의 tolerance(`0.01`)가 가장 빡빡해, 정밀도 trade-off의 여지가
  커널마다 다르다.
