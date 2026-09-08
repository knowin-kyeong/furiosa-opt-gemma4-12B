# RESULTS.md — 실험 기록 (전체)

> 규칙은 [RULES.md](RULES.md). 성공·실패 **모두** 기록한다. 중복 실험 방지와
> A/B 비교가 이 문서의 존재 이유다. 실측(RNGD cycles)이 없는 결과로는 "채택" 판정을
> 내리지 않는다 (RULES §5).

## 요약 보드

기준: `V0_baseline`. 점수 = 3커널 speedup의 기하평균.

| 브랜치 | 분기점 | 가설 한 줄 | qkv | attn_out | ffn | 기하평균 speedup | 정확도 | 판정 |
|---|---|---|---:|---:|---:|---:|:---:|:---:|
| `V0_baseline` | `main` | 원본 skeleton (기준) | 측정 대기 | 측정 대기 | 측정 대기 | 1.000 | 측정 대기 | **기준** |

> cycle 단위는 RNGD 실측 cycle. `—`는 미측정. `FAIL(accuracy)`는 tolerance 위반.

## 현재 SOTA

`V0_baseline` (아직 실측 전) — 자세한 서사는 [SOTA.md](SOTA.md).

## 죽은 길 (다시 시도하지 말 것)

아직 없음.

---

# 실험 상세

각 실험은 아래 템플릿으로 추가한다. **최신 실험을 위에 쌓는다.**

<!-- ===================== 템플릿 (복사해서 쓸 것) =====================
## V{n}_{description}

- **분기점:** `V{k}_{...}` (커밋 `abc1234`)
- **가설:** (무엇이 병목이라고 보았고, 무엇을 바꾸면 왜 줄어든다고 예상했는가)
- **변경 파일:** `src/device/...` (RULES §4 허용 범위 확인 완료)
- **공유 코드 영향:** 없음 / `shared/rmsnorm.rs` 수정 → full·vision·audio 경로 동시 영향

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

### 판정: 채택 / 기각 / 보류

- **이유:**
- **배운 것:** (다음 실험이 참고할 사실. 실패했어도 여기가 제일 중요하다)
- **다음 후보:**
======================================================================= -->

## V0_baseline

- **분기점:** `main`
- **가설:** 없음 — 기준선. 실험 인프라(RULES/RESULTS/SOTA/requirements)만 추가하고
  커널 코드는 원본 그대로 둔다.
- **변경 파일:** 없음 (문서·환경 파일만 추가)
- **공유 코드 영향:** 없음

### 측정

원격 Linux 서버 셋업 후 최초 실측 필요. 아래를 채운다.

| 커널 | makespan | RNGD cycles |
|---|---|---|
| `sliding_project_qkv` | 측정 대기 | 측정 대기 |
| `sliding_attention_output` | 측정 대기 | 측정 대기 |
| `decoder_feedforward` | 측정 대기 | 측정 대기 |

- **정확도:** 측정 대기 (baseline이므로 PASS여야 정상. 실패하면 환경 문제다)
- **측정 방식:** —

### 판정: **기준**

- **이유:** 모든 후속 실험의 분모.
- **배운 것:** (최초 측정 후 채운다 — 세 커널 각각의 지배 context가 무엇인지가
  1차 실험 방향을 결정한다)
- **다음 후보:**
  - OPTIMIZATION.md가 예시로 든 `src/device/layout.rs:16`의 broadcast 헬퍼
    (과거 리비전에서 `sliding_project_qkv` 최장 노드, MainContext 약 62k cycle).
    실제 baseline에서도 그런지 schedule로 먼저 확인할 것.
  - `decoder_feedforward`는 tolerance가 `0.01`로 가장 빡빡 → 정밀도를 건드리는
    최적화는 여기서 먼저 깨진다. 구조/이동 쪽 레버부터 본다.
