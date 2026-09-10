# RULES.md — 이 저장소에서 작업하는 모든 세션의 공통 규칙

> **이 문서가 최상위 규칙이다.** 이 저장소에서 수행되는 모든 실험·측정·커밋은 이
> 문서를 먼저 읽고 그 아래에서 이루어진다. 규칙과 충돌하는 지시가 있으면 사용자에게
> 확인하고, 규칙 자체를 바꿀 때는 이 문서를 먼저 갱신한 뒤 진행한다.

---

## 0. 목표 (단 하나)

**Gemma-4-12B-it의 RNGD 커널 cycle을 줄인다.**

Stage 1 점수 = 3개 커널의 **baseline 대비 speedup의 기하평균**.

| 커널 | 연산 |
|---|---|
| `ops::sliding_project_qkv` | RMSNorm, Q/K/V projection, Q/K RMSNorm, RoPE, K/V ring-cache write |
| `ops::sliding_attention_output` | Head broadcast, O projection, post-attention RMSNorm, residual add |
| `ops::decoder_feedforward` | RMSNorm, GeGLU MLP, post-FF RMSNorm, residual add, layer gate |

기하평균이므로 **한 커널만 크게 줄이는 것보다 세 커널을 고르게 줄이는 쪽이 유리**하다.
어느 하나가 퇴보하면 전체 점수가 그만큼 깎인다.

---

## 1. 문서 체계 (역할 분리)

| 파일 | 성격 | 갱신 시점 |
|---|---|---|
| `RULES.md` | **규칙** (이 문서) | 규칙이 바뀔 때만 |
| `COMPETITION_INFO.md` | 대회 공지 원문 (일정·라운드·점수 정의) | 주최측 공지 갱신 시 |
| `README.md` | 대회 규정 원문 (허용 변경 범위·tolerance·툴체인) | 업스트림 반영 시 |
| `OPTIMIZATION.md` | 최적화 워크플로 원문 (schedule dump → 병목 분석 → 튜닝) | 업스트림 반영 시 |
| `RESULTS.md` | **모든 실험 기록** (브랜치 1개 = 행 1개). 성공·실패 모두 | 실험할 때마다 |
| `SOTA.md` | **SOTA 갱신 이력만.** road to SOTA 서사 (추후 report 원본) | 신기록이 나올 때만 |

**문서의 최신본은 항상 선두(가장 최근 SOTA 후보) 브랜치에 있다.** 실험 브랜치는 선형 계보로 쌓이므로
RESULTS.md/SOTA.md/RULES.md는 그 계보의 끝에서 갱신하고, `V0_baseline`의 사본은 출발점 기록일 뿐이다.
새 세션은 `git fetch` 후 RESULTS.md 요약 보드의 가장 아래 행(최신)이 있는 브랜치를 체크아웃해서 시작한다.

**RESULTS.md는 실패도 반드시 기록한다.** 이 문서의 존재 이유가 중복 실험 방지이므로,
"해봤는데 안 됐다"가 "안 해봤다"보다 훨씬 가치 있다.

---

## 2. 매 실험 시작 전 필수 절차

성능 개선 요청을 받으면 **코드를 건드리기 전에** 다음을 순서대로 한다.

1. `RULES.md` (이 문서) 확인 — 특히 §4 금지사항.
2. `COMPETITION_INFO.md` 확인 — 일정·점수 정의 변경 여부.
3. `README.md`의 *Stage 1 rules: skeleton contract* 확인 — 허용 변경 범위(§4).
4. **`RESULTS.md` 전체를 읽는다** — 같은 아이디어를 이미 시도했는지, 어떤 병목이
   남아 있는지, 어떤 방향이 죽은 길로 판명됐는지.
5. `SOTA.md` 확인 — 현재 기준선이 되는 브랜치/커밋이 무엇인지.
6. 그 다음에 가설을 세우고 브랜치를 판다.

**한 브랜치 = 한 가설.** A/B 비교가 성립하지 않는 변경 묶음(무관한 최적화 2개를 한
브랜치에 섞기)은 금지한다. 어느 쪽이 효과였는지 알 수 없게 되면 그 실험은 버린 것과 같다.

### 2.1 중복 설계 방지: 설계 즉시 RESULTS.md에 빈 슬롯을 만든다

실험은 **코드를 쓰기 전에** 등록된다. 여러 세션이 동시에 돌 수 있으므로, "측정이 끝나면
기록한다"는 방식으로는 같은 가설을 두 세션이 동시에 파는 사고를 막지 못한다.

1. 가설이 정해지면 **먼저** `RESULTS.md` 요약 보드에 행을 추가하고 상세 섹션을 템플릿으로
   채운다. cycle 칸은 `—`(미측정), 판정은 `설계됨`으로 둔다.
2. 그 상태로 **즉시 commit + push** 한다 (브랜치 생성 전이라도 `V0_baseline` 또는 현재
   SOTA 브랜치에 문서 커밋으로 올린다). push된 순간부터 그 가설은 "점유됨"이다.
3. 다른 세션은 새 가설을 세우기 전에 `git fetch` 후 RESULTS.md의 `설계됨` 행을 확인하고,
   이미 점유된 가설은 건드리지 않는다. 같은 가설을 다른 각도로 파고 싶으면 **새 V번호**를
   받고 분기점·차이를 명시한다.
4. `V{n}` 번호는 RESULTS.md 요약 보드에서 **가장 큰 번호 + 1**로 받는다. 번호 충돌이
   나면(동시 push) 나중에 push한 쪽이 번호를 바꾼다.
5. 판정 상태 전이: `설계됨` → `구현됨`(코드 push, 미측정) → `makespan 측정` →
   `채택 / 기각 / 보류`(RNGD 실측 후). 각 전이마다 RESULTS.md를 갱신하고 push한다.

빈 슬롯은 부채가 아니라 **예약**이다. 설계만 해두고 구현하지 못한 슬롯도 지우지 않는다 —
"왜 안 했는지"를 한 줄 남기면 다음 세션의 판단 재료가 된다.

---

## 3. 브랜치 규칙

### 3.1 명명

```
V{version_num}_{description}
```

- `version_num`: 0부터 단조 증가하는 정수. 재사용 금지.
- `description`: 소문자 snake_case, 무엇을 바꿨는지 한눈에 (예: `V3_qkv_fused_rmsnorm`).
- 예: `V0_baseline`, `V1_qkv_broadcast_removal`, `V2_ffn_tile_reshape`

### 3.2 분기 기준

- **`main`은 절대 건드리지 않는다.** upstream 동기화 전용이다. `main`에 커밋·rebase·
  force-push 금지.
- `V0_baseline`이 모든 실험의 뿌리다. 문서 인프라(RULES/RESULTS/SOTA/requirements)는
  여기에만 있다.
- 신규 실험은 **현재 SOTA 브랜치에서 분기**하는 것을 원칙으로 한다 (누적 개선).
  독립적인 A/B 비교가 목적이면 `V0_baseline`에서 분기하고, RESULTS.md에 분기점을 명시한다.
- **어느 브랜치에서 분기했는지 RESULTS.md에 반드시 적는다.** 이게 없으면 cycle 숫자를
  비교할 수 없다.

### 3.3 커밋

- 실험 브랜치의 커밋 메시지 첫 줄: `V{n}: {한 줄 요약}`
- 측정 결과가 나오면 RESULTS.md 갱신을 **별도 커밋**으로 남긴다 (코드 diff를 깨끗하게 유지).
- 브랜치는 지우지 않는다. 실패한 실험도 근거로 남는다.
- **브랜치를 만들면 곧바로 `origin`에 push한다.** 작업 서버가 휘발성 컨테이너
  디스크이므로 push되지 않은 것은 언제든 사라질 수 있다 (§6.2.1).

---

## 4. 금지사항 / 허용 범위 (RESTRICTION)

> 출처: `README.md` — *Stage 1 rules: skeleton contract*. **위반 시 채점 자체가 안 된다.**

### 4.1 채점에 반영되는 = 수정해도 되는 곳

- `src/device/` **전체**
- `src/ops.rs`, `src/ops_vision.rs`, `src/ops_audio.rs`의 **함수 본문(body)만**

### 4.2 수정해도 무시되는 곳 (= 여기서 최적화해봐야 점수에 반영 안 됨)

- `src/axes.rs`, `src/host/`, `src/api/`, `src/bin/`, `src/lib.rs`, `tests/`

> **주의:** 무시된다는 것은 "채점 서버가 baseline 버전으로 되돌린다"는 뜻이다.
> 로컬에서 `src/host/`를 고쳐 빨라졌다면 그건 **가짜 개선**이다. 이 파일들에 의존하는
> 최적화는 설계 단계에서 배제한다.

### 4.3 절대 규칙

1. **`#[device]` 함수의 이름·파라미터·타입·반환형을 바꾸지 않는다.** 시그니처가
   평가자 계약(evaluator contract)이다.
2. **`src/ops.rs`, `src/ops_vision.rs`, `src/ops_audio.rs`는 크레이트 루트에 그대로 둔다.**
   컴파일된 커널 이름에 `module_path!()`가 들어가므로 경로가 바뀌면 채점이 실패한다.
3. **`tests/test_kernels.rs`를 고쳐서 통과시키지 않는다.** 채점 서버는 자기 버전을 쓴다.
4. **정확도는 hard gate다.** 틀리면 아무리 빨라도 0점.

### 4.4 정확도 tolerance

| 커널 | Absolute | Relative |
|---|---:|---:|
| `sliding_project_qkv` | `0.04` | `1e-2` |
| `sliding_attention_output` | `0.05` | `1e-2` |
| `decoder_feedforward` | `0.01` | `1e-2` |

`decoder_feedforward`가 가장 빡빡하다(`0.01`). FFN에서 정밀도를 낮추는 방향의
최적화는 여기서 먼저 깨진다.

### 4.5 공유 코드 주의

`src/device/shared/rmsnorm.rs`와 `src/device/shared/mlp.rs`는 sliding 경로뿐 아니라
**full attention, vision, audio 경로에서도 쓰인다.** 여기를 고치면:

- Stage 1 세 커널 중 둘 이상이 동시에 움직인다 → A/B 해석이 어려워진다.
- Stage 2(E2E)에서 다른 경로를 망가뜨릴 수 있다.

공유 코드를 고칠 때는 RESULTS.md에 **영향받는 모든 커널의 cycle을 함께 기록**한다.

---

## 5. 측정 규칙 (무엇을 숫자로 믿을 것인가)

### 5.1 두 종류의 숫자를 구분한다

| 지표 | 얻는 법 | 성격 |
|---|---|---|
| **Makespan** (schedule 정적 span) | `cargo furiosa-opt compile ... --dump-schedule` | 개발용 빠른 신호. **점수 아님** |
| **RNGD cycles** (실측) | `./scripts/rngd_test.sh` (Arena 제출) | **이것만 점수다** |

- makespan은 반복이 싸므로 탐색·스크리닝에 쓴다.
- **RESULTS.md의 "채택" 판정은 반드시 RNGD 실측 cycle로만 내린다.**
- makespan만 있는 결과는 RESULTS.md에 `측정: makespan only`로 명시하고, 채택 여부는
  `보류`로 둔다.

### 5.2 절대 하면 안 되는 것

- 재현 가능한 schedule 비교나 RNGD 근거 없이 "개선됐다"고 보고하지 않는다
  (OPTIMIZATION.md §5 명시).
- 정확도 검증(테스트 통과)을 건너뛴 cycle 수치는 무효다. 정확도가 hard gate이므로
  **틀린 커널의 cycle은 기록조차 하지 않는다** (`FAIL(accuracy)`로만 남긴다).

### 5.3 측정 커맨드

```sh
# 0) 최초 1회: 레퍼런스 픽스처 생성 (ref/fixtures.safetensors 없을 때)
python3 scripts/generate_references.py

# 1) schedule dump (커널 하나씩. --exact 필수 — 이름이 다른 커널의 prefix일 수 있음)
mkdir -p target/schedules
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
    --dump-schedule target/schedules/V{n}_sliding_project_qkv.json
cargo furiosa-opt compile ops::sliding_attention_output --exact \
    --dump-schedule target/schedules/V{n}_sliding_attention_output.json
cargo furiosa-opt compile ops::decoder_feedforward --exact \
    --dump-schedule target/schedules/V{n}_decoder_feedforward.json

# 2) 실측 (Arena 제출 → 정확도 + 실제 cycle)
./scripts/rngd_test.sh
```

- schedule JSON은 `target/schedules/V{n}_{kernel}.json`으로 **버전 접두사를 붙여 보관**한다
  (`target/`은 gitignore 대상이므로, 비교가 끝나면 수치를 RESULTS.md로 옮긴다).
- makespan = `max(instruction.lifetime.end)` (schedule JSON에서 직접 계산 가능).

---

### 5.4 제출 예산 (2026-09-09 밤, 사용자 지시)

- **공식 리더보드 제출(`moa-submitter submit`)은 하루 5회로 간주한다.** 확실한 개선(실측으로 확인된 것)만 올린다.
  리더보드는 팀별 최고 유효 결과를 보이므로, 같은 코드를 다시 올리는 것은 노이즈를 수확할 뿐이다 — 그것도 예산 안에서만.
- Arena 잡도 남발하지 않는다. **측정 바이너리는 한 잡에 여러 변형을 warm 반복으로 담는다**(tests/ 변경은 채점에 무시된다):
  같은 프로세스에서 두 번째 실행부터는 qkv도 ±0.7%라 1% 차이가 한 잡에서 판정된다. cold(첫 실행) 수치는 채점 환경의 것이고
  ±8% 흔들리므로 A/B에 쓰지 않는다.

## 6. 실행 환경

### 6.1 결론: **GPU는 필요 없다**

이 대회의 연산 타깃은 NVIDIA GPU가 아니라 **FuriosaAI RNGD NPU**이며, 실측은
`rngd` CLI로 **Arena 원격 서버에 제출**해서 이루어진다. 로컬/원격 개발 머신이 하는 일은
**Rust 크로스 컴파일 + 스케줄 분석 + 잡 제출**뿐이다. 3090이든 H100이든 점수에 아무
영향이 없다.

`python3 scripts/generate_references.py`가 torch를 쓰지만, 48개 레이어를 돌리지 않고
작은 텐서만 다루므로 **CPU torch로 충분**하다 (픽스처는 약 120KB).

### 6.2 RunPod 사양 권장

| 항목 | 권장 | 이유 |
|---|---|---|
| 인스턴스 종류 | **CPU pod** | GPU 불필요 |
| OS | **x86_64 Ubuntu 22.04+ / GLIBC 2.34+** | README 툴체인 요구사항 (필수) |
| vCPU | 8~16 | Rust 릴리즈 빌드 시간이 전부 |
| RAM | **32GB 이상** | rustc + furiosa-opt 컴파일 피크 |
| Disk | **80~100GB** | `target/` 디렉터리와 툴체인 |
| GPU | 불필요 | — |

> RunPod에서 CPU pod를 못 쓰는 상황이면, **가장 싼 GPU 등급(RTX A4000 등)** 을 고른다.
> 3090(24GB)도 물론 동작하지만 **VRAM은 이 작업과 무관하므로 돈 낭비**다.
> GPU를 고를 때 봐야 할 것은 VRAM이 아니라 **함께 딸려오는 RAM/디스크/vCPU**다.
> 반드시 x86_64 Ubuntu 22.04+ 이미지를 고를 것 (ARM 인스턴스는 툴체인 미지원).

**확정 인스턴스 (2026-09-09):** RunPod CPU pod / Ubuntu 22.04 / 8 vCPU / 32GB RAM / 80GB disk.

RunPod에서 반드시 확인할 것:

1. **아키텍처가 x86_64인지.** RunPod에는 ARM(Ampere/Graviton 계열) CPU pod도 있다.
   ARM에 걸리면 `cargo-furiosa-opt`가 아예 설치되지 않는다. 접속 직후 `uname -m`이
   `x86_64`인지 먼저 확인하고, 아니면 인스턴스를 갈아탄다.
2. **디스크 배분.** RunPod은 디스크를 *container disk*(휘발성)와 *volume disk*
   (`/workspace`, 영속)로 나눈다. **이 프로젝트는 `/workspace`를 쓰지 않고 컨테이너
   디스크에서 작업하기로 했다** (2026-09-09 결정). 따라서 80GB 중 대부분을
   **container disk에 배정**해야 한다 — volume에 몰아주면 정작 빌드할 공간이 없다.

   대략적인 용량: rustup 툴체인 ~2GB, cargo registry ~3GB, `target/release` 5~15GB,
   CPU torch ~1GB. 부족해지면 `cargo clean`(전부 날림) 대신 `rm -rf target/debug`부터 한다.

### 6.2.1 컨테이너 디스크는 휘발성이다 — git이 유일한 안전장치

컨테이너 디스크의 내용은 pod을 terminate하면 **전부 사라진다** (stop만 해도 보존이
보장되지 않는다). 재빌드는 시간만 들면 되지만, **실험 코드와 측정 결과는 복구 불가능하다.**

그래서 다음이 규칙이다:

1. **실험 브랜치는 만들자마자 push한다.** 측정이 끝날 때까지 기다리지 않는다.

   ```sh
   git checkout -b V{n}_{description}
   git push -u origin V{n}_{description}     # 코드를 쓰기 전에 먼저
   ```

2. **RESULTS.md를 갱신하면 즉시 commit + push한다.** 실측 cycle 숫자는 Arena 잡을
   다시 돌려야만 얻을 수 있다. 로컬에만 있는 측정 결과 = 아직 없는 결과로 취급한다.

3. **pod을 내리기 전에 `git status`가 깨끗한지, `git log origin/{branch}..HEAD`가
   비어 있는지 확인한다.** 하나라도 남아 있으면 그 작업은 날아간다.

4. `target/schedules/*.json`은 gitignore된 `target/` 아래에 있어 push되지 않는다.
   보존이 필요한 makespan 수치는 **JSON이 아니라 RESULTS.md에 숫자로 옮겨 적는다.**

원격: `origin` = `knowin-kyeong/furiosa-opt-gemma4-12B` (우리 fork, push 대상),
`upstream` = `HoseongLee/furiosa-opt-gemma4-12B` (주최측 원본, **절대 push 금지**).

### 6.3 원격 서버 초기 셋업

```sh
# 시스템
sudo apt update
sudo apt install -y build-essential libclang-dev gcc-aarch64-linux-gnu \
                    python3 python3-pip git curl

# Rust 툴체인 (rust-toolchain.toml이 nightly-2026-05-01로 고정)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup toolchain install nightly-2026-05-01
cargo +nightly-2026-05-01 install cargo-binstall
cargo +nightly-2026-05-01 binstall -y cargo-furiosa-opt
cargo install furiosa-schedule-viewer

# Arena CLI
cargo binstall -y furiosa-arena-cli

# Python (픽스처 생성용, CPU only)
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt`는 저장소 루트에 있다. 새 원격 서버를 띄울 때마다 이 절차를 따른다.
현재 pod에는 위 절차를 자동화한 `/root/setup_pod.sh`가 있고, 로그는 `/root/setup.log`다.

#### 6.3.1 실제로 걸린 함정 (2026-09-09, RunPod Ubuntu 22.04 이미지에서 확인)

**① `rngd`라는 바이너리는 없다.** `furiosa-arena-cli` 0.8.0은 `furiosa-arena`로 설치된다.
README와 `scripts/rngd_test.sh`는 `rngd`를 부르므로 심링크가 필요하다. 서브커맨드
(`login/submit/status/logs/list/cancel`)와 `submit --name/--entrypoint/--timeout` 플래그는
스크립트와 일치하므로 심링크만으로 동작한다.

```sh
ln -sf ~/.cargo/bin/furiosa-arena ~/.cargo/bin/rngd
```

**② URL 환경변수가 두 개다.** `rngd_test.sh`는 `RNGD_URL`이 *설정되어 있는지만* 검사하고,
실제 바이너리는 `FURIOSA_ARENA_URL`(또는 `--url`)을 읽는다. 둘 다 같은 값으로 둔다.
URL은 이전 README 버전에 있던 `https://arena.furiosa.ai`.

**③ `python3`과 `pip3`가 다른 파이썬이다.** RunPod 이미지에서 `/usr/bin/python3`은 3.10,
`/usr/local/bin/pip3`는 3.12의 pip다. `pip3 install`한 torch는 3.12에만 들어가므로
`python3 scripts/generate_references.py`가 `ModuleNotFoundError: torch`로 죽는다.
시스템 `python3`을 바꾸지 말고(apt가 의존) PATH 앞에 3.12 심링크를 둔다.

**④ 로그인은 사람이 해야 한다.** `furiosa-arena login`은 GitHub device flow(브라우저에서
코드 입력)라 자동화가 안 된다. 로그인 전에는 `health`조차 실패한다.

**⑤ 실행시간 상한은 70초다.** `rngd submit --timeout`이 70을 넘으면 컨트롤러가
`Error: timeout_sec 3000 exceeds server maximum 70`으로 **거부**한다. `scripts/rngd_test.sh`의
기본값은 1800이라 그대로 쓰면 아무것도 제출되지 않는다. 세 커널 테스트는 70초 안에 끝난다.

**⑥ `rngd_test.sh`는 제출 에러를 삼킨다.** `submit_output=$(rngd submit …)`가 `set -e` 아래
있어서, 거부되면 `echo "$submit_output"` 전에 스크립트가 죽는다. 증상은 "빈 로그"다. 무인 체인은
이 때문에 드라이버 전용 `arena.sh`(§11.2)로 제출한다.

**⑦ 로그인 계정은 `knowin-kyeong`이다** (2026-09-09 완료, 토큰은 pod의
`/root/.config/furiosa-arena/token`). pod가 사라지면 device flow를 다시 해야 한다.

위 ①~③은 `/root/env.sh`에 모아두었고 `~/.bashrc`가 이를 source한다.
**비대화형 ssh 명령은 `.bashrc`를 읽지 않으므로** 원격 명령 앞에 항상 `. /root/env.sh`를 붙인다.

```sh
# /root/env.sh
. "$HOME/.cargo/env"
export PATH="$HOME/.local/bin:$PATH"     # ~/.local/bin/python3 -> /usr/bin/python3.12
export RNGD_URL=https://arena.furiosa.ai
export FURIOSA_ARENA_URL=https://arena.furiosa.ai
```

#### 6.3.2 설치 확인된 버전

| 도구 | 버전 |
|---|---|
| rustc | 1.97.0-nightly (2026-04-30) = `nightly-2026-05-01` |
| cargo-furiosa-opt | 0.7.0 (crate 의존성은 `furiosa-opt-std 0.6.0`) |
| furiosa-arena-cli | 0.8.0 |
| python3 (env.sh 적용 후) | 3.12.13 |
| torch | 2.14.0+cpu |

원격 작업 디렉터리: `/root/furiosa-opt-gemma4-12B` (origin clone, `V0_baseline` 체크아웃).

### 6.4 로컬(Windows)에서 하는 일 / 안 하는 일

- **한다:** 코드 편집, 문서 작성, git 브랜치 관리, schedule JSON 분석.
- **안 한다:** 빌드·테스트·측정. 툴체인이 Linux x86_64 전용이므로 Windows에서
  `cargo furiosa-opt`를 돌리려 하지 않는다. 모든 측정은 원격 Linux 서버에서 한다.

---

## 7. 실험 사이클 (표준 절차)

```
1. RESULTS.md 읽기 → 중복 아닌지, 어떤 병목이 남았는지 확인
2. 가설 세우기 (예: "sliding_project_qkv의 broadcast_hidden이 MainContext를 96% 점유 →
                    브로드캐스트를 제거하면 X% 줄어든다")
3. git checkout {기준 브랜치} && git checkout -b V{n}_{description}
4. 코드 수정 (§4 허용 범위 안에서만)
5. 원격 서버에서 schedule dump → makespan 확인 (빠른 스크리닝)
6. 유망하면 ./scripts/rngd_test.sh → 정확도 + 실측 cycle
7. RESULTS.md에 결과 기록 (실패도 기록) + 채택/기각/보류 판정
8. SOTA 갱신이면 SOTA.md에 서사 추가
9. 커밋
```

### 7.1 채택 기준

| 판정 | 조건 |
|---|---|
| **채택** | 정확도 통과 + 3커널 기하평균 speedup이 기준 브랜치 대비 개선. 다음 실험의 기준 브랜치가 된다 |
| **기각** | 정확도 실패, 또는 기하평균이 나빠짐 |
| **보류** | makespan만 측정, 또는 실측 노이즈 범위 내(±1% 미만)의 변화 |

---

## 8. 병목 분석 참고 (OPTIMIZATION.md 요약)

주요 context와 그것이 병목일 때의 의미:

- **`DmaEngine`** — 가중치/활성값 이동이 한계. DMA 노드는 duration뿐 아니라 `util`을 본다.
  낮은 util은 대역폭 한계가 아니라 **접근 패턴 문제**(partition 교차, strided,
  non-contiguous, 정렬/HBM bank conflict)일 가능성이 높다.
- **`SubContext`** — 레지스터 파일 스테이징/프리로드가 한계.
- **`MainContext` / `VectorEngine`** — softmax, RMSNorm, cast 같은 벡터 연산이 한계.

`MainContext`와 `SubContext`는 Tensor Unit 파이프라인을 두고 경합한다. 따라서 오버랩에는
상한이 있다: 최악은 둘의 합, 이상적이면 큰 쪽에 수렴.

레버 선택:
1. 현재 자원이 병목 → **실행 엔진 경로** 변경
2. 불필요한 직렬 작업 / 안 맞는 reduction → **타일·split shape** 변경
3. 이동 또는 주소 의존성이 한계 → **매핑·패딩·전송 경계** 변경

---

## 9. 일정 (COMPETITION_INFO.md 기준)

| 항목 | 날짜 |
|---|---|
| 등록 마감 | 2026-09-15 |
| **Kernel Optimization Round (Stage 1)** | 2026-09-01 ~ **2026-09-25** |
| 결선 진출자 발표 | 2026-09-30 |
| Model Optimization Round (Stage 2) | 2026-10-01 ~ 2026-10-25 |
| 시상 (MICRO 2026, Athens) | 2026-11-01 |

**현재 우선순위는 Stage 1(커널)이다.** Stage 2는 진출 후에 착수한다.

### 9.1 아직 TBD인 것 (확정되면 이 문서를 갱신할 것)

- 채점 서버 URL·인증 방식
- 제출 아카이브 형식·제출 커맨드·제출 마감·팀당 최대 제출 횟수
- 세 커널 cycle을 합산하는 정확한 공식 (COMPETITION_INFO는 "기하평균"이라 명시)

---

## 10. 세션 간 인계

새 세션이 이 저장소에서 작업을 시작하면:

1. `RULES.md`(이 문서) → `RESULTS.md` → `SOTA.md` 순서로 읽는다.
2. `git branch -a`로 현재까지의 실험 목록을 본다.
3. 현재 SOTA 브랜치를 기준으로 다음 가설을 세운다.
4. 절대 `main`에 커밋하지 않는다.

### 10.0g 2026-09-10 밤 — 리더보드가 목표를 확정했다 (가장 최신, 여기서 시작할 것)

**공개 리더보드(`curl -k https://micro2026-api.duckdns.org:7777/api/leaderboard`)에 팀이 늘었다.**

| 팀 | qkv | attn_out | ffn | 점수 |
|---|---:|---:|---:|---:|
| **우리(V209, Arena)** | **102,800** | 52,670 | 294,574 | ≈6.30 |
| Participant #905 | 119,177 | **46,318** | 314,820 | 6.0 |
| Pulbitmaru | 131,489 | 50,617 | **289,085** | 5.8 |
| 세 팀 최고치 조합 | 102,800 | 46,318 | 289,085 | **6.487** |

**우리 qkv는 압도적 1위**(2위보다 14% 빠르다)이고, 지는 곳은 attn_out(**−6,352 필요, −12%**)과
ffn(**−5,489 필요, −1.9%**)이다. **둘 다 남이 이미 도달한 값이므로 열려 있는 것이 확실하다.**
6.5는 이 둘을 따라잡으면 나온다 — 새 축이 필요하다는 이전 판단은 리더보드를 보기 전 이야기였다.

**"연산 유닛을 다르게 점유해 병렬화" 축은 이 DSL에서 닫혀 있다(V223·V224).**
`ctx.sub`는 contraction도 vector pass도 못 돌린다 — `to_trf`·`to_vrf` 스테이징 전용이다.
ffn pass A는 `fetch_table_lookup`이 Main 전용이라(V223), attn_out의 타일 contraction은
Multi 형상 fetch sequencer가 `CommandTuExec::verified failed`로 거부돼서(V224) 각각 막힌다.
V40c(ffn pass B on sub)까지 합치면 셋 다 같은 결론이다. **실재하는 병렬성은 DMA ∥ Main 하나이고,
그것을 쓴 것이 V206(ffn ring 4, −16k)이다.** 같은 손잡이를 attn_out·qkv에서 다시 돌려볼 가치가 있다.

**V219(pass 비용 교정)는 설계 결함으로 판정 불가다.** n0가 매 반복의 첫 변형이라 cold 전이를 뒤집어썼다.
순서 편향이 없는 n16 − n8만 보면 pass당 ≈554지만 n8 − n4는 부호가 반대다. **재설계 시 변형 순서를
균형화하고 노이즈가 1/6인 attn_out에서 잴 것.** 그때까지 V220·V221(pass 융합)의 기대값은 미확정이다.

### 10.0f 2026-09-10 밤 — qkv 스트림은 465 B/cycle에 고정이다 (가장 최신, 여기서 시작할 것)

**V217(채널 분산 head 레이아웃)은 기각됐고, 그 이유가 qkv 전체를 설명한다.**
투영은 설계대로 동작한다(weight 로드 13,383으로 프로덕션 13,512보다 싸다). 그런데 **store가 556,076**이다:
`q_out`이 `m![Ns, Gs, Ds]`라 채널 분산에서는 슬라이스마다 2 B 쓰기가 8군데로 흩어지고 HBM 비정렬 쓰기는 RMW ~50×다.

**일반 법칙(이제 확정):** store가 싸려면 슬라이스가 출력의 연속 구간을 가져야 하고, 그러면 슬라이스는
한 head의 연속 채널 C개를 갖는다. `256슬라이스 ÷ (클러스터당 head쌍 8) = head당 32슬라이스 → C = 8`
→ **weight 런 = 8 × 3,840 = 30,720 B = 프로덕션 레이아웃 그 자체**. 3,840 B 런에는 C=1, 즉 4,096슬라이스가
필요한데 칩에는 512개뿐이다. **런 길이와 head gather의 ring 크기는 같은 손잡이이고, 짧은 런은 언제나
더 비싼 ring으로 되갚는다.** V215가 잰 −6.5k는 실재하지만 **가져올 수 없다.**

**따라서 qkv = 스트림 67.7k(31.46 MB ÷ 465, 고정) + 나머지 ≈33k.** 남은 레버는 나머지뿐이다:
미측정 슬롯 **V178**(head-norm 9 → 5 pass, ≈−2k)과 **V179**(RoPE ternary, ≈−1k). 둘 다 tail에 있고,
tail은 weight 스트림이 끝난 뒤라 노출돼 있다(V189/V192가 같은 자리에서 −3,883을 실측했다).

**6.5까지의 산수 (갱신).** Arena 기준 현재 **6.30**(V209). V178+V179가 qkv를 98k로 만들면 **≈6.36**.
ffn·attn_out에는 측정된 미수확 항목이 없다(V213 정렬은 store 파편화로, V216 런 길이는 부호가 반대라 닫혔다).
**즉 현재 알려진 축만으로는 6.35~6.4가 상한이고, 6.5에는 아직 발견되지 않은 축이 필요하다.**

**V218이 남은 레버의 정체를 확정했다: pass의 크기가 아니라 개수다.** qkv tail(head RMSNorm ×3 + RoPE)은
실물 **16,012**(정적 7,435의 2.15배)인데, **head RMSNorm만 떼면 실물 −6,833에 정적은 −1,263 = 5.4배**다.
클럭비 2.0을 크게 넘으므로 초과분은 일이 아니라 **PE core의 in-order pass 발행 비용**이고, pass당 실물
≈600 cycle이다. 따라서 **V178**(q/k/v head-norm의 mean-square·sqrt를 합친 `[4, Ds]` 버퍼에서 한 pass씩,
9 → 5 pass)과 **V179**(RoPE ternary, 3 → 2 pass)는 이제 추정이 아니라 **측정된 근거**를 갖는다: 6 pass ≈ **−3.6k**.
그러면 qkv 101k → 97.4k, Arena 기하평균 **≈6.375**.

### 10.0e 2026-09-10 밤 — 런 길이는 커널마다 부호가 다르다 (가장 최신, 여기서 시작할 것)

**점수 기준을 섞지 말 것.** V200의 6.229와 V209의 "6.05"는 다른 기준이다. Arena 기준(세 커널 모두 Arena 실측)으로
V200 = 6.229, **V209 = 6.30**. 공식 draw 기준(V204가 받은 나쁜 attn_out draw 56,845가 반복된다고 가정)으로 V209 ≈ 6.05.
**attn_out draw 하나(49.9k vs 56.8k)가 기하평균을 4% 흔든다.**

**런 길이 실험 두 개, 결과가 정반대다(둘 다 같은 잡 안 짝비교, 정적 makespan은 둘을 구분 못 함).**
- **qkv(V215): 짧은 게 이긴다.** 순환 행 매핑으로 30,720 B 런 1개 → **3,840 B 런 8개**(명령 수는 그대로 1개,
  H=3840=15×256이라 행 경계는 항상 정렬): 6쌍 중 5승, **평균 −3,240 / 15.73 MB** = 465 → **514 B/cycle**.
- **ffn up/gate(V216): 긴 게 이긴다.** 같은 변환(57,600 B 런 1개 → 3,840 B 런 15개)에서 6쌍 중 5패,
  **평균 +5,867**. **V181의 통짜 런이 옳았다.**
→ **런 길이 곡선은 커널마다 다르다. 한 커널의 결과를 다른 커널로 이식하지 말 것.**

**그리고 두 이득 모두 구조가 막고 있다.**
- qkv의 −6.5k는 순환 매핑에서 head h의 256행이 256슬라이스로 흩어져 ring-64 gather가 ring-256(61k)이 되기 때문에
  그대로는 못 쓴다. 4행 연속을 지키는 절충(15,360 B 런)은 ring-64와 호환되지만 이득이 ≈1k뿐이다.
  **전체를 가지려면 head 레이아웃까지 바꿔야 한다 → `V217` (설계 등록됨, 남은 최대 판돈).**
- ffn down의 정렬 이득(V211 −16k)은 2청크 = 슬라이스당 15행을 강요하고, 15행 출력(30 B)은 `SRAM access width 8`에
  걸리며 패딩하면 **SIGABRT**다. 타일별 4행 버퍼 4개로 우회하면 컴파일은 되지만 **store 4개가 총 +40k**다 —
  8청크는 출력이 live 슬라이스 32개 × 120 B인데 2청크는 128개 × 30 B라 store가 4배 파편화된다. **닫힌다.**

**6.5까지의 산수.** Arena 기준 현재 6.30. 6.5 = +3.2%이므로 한 커널 −9.5% 또는 세 커널 −3.2%가 필요하다.
남은 측정된 미수확 항목은 **qkv 런 길이 −6.5%(V217이 열어야 함)** 하나뿐이고, 그 외 V178(head-norm 9→5 pass,
≈−2k)·V179(RoPE ternary, ≈−1k)가 미측정으로 남아 있다. **V217 없이는 6.4 근처가 상한이다.**

**V217의 게이팅 질문은 이미 통과했다(`V217_probe_reduce_256`):** `vector_inter_slice_reduce::<m![1 # 256], …>`가
lowering을 통과하고 정적 비용도 무시할 수준이다(저장소 최대는 지금까지 32슬라이스였다). 즉 채널 분산 head
레이아웃에서 head RMSNorm을 switch 없이 VRU로 할 수 있다. 제약은 Way8 → Way4 `vector_narrow_split`을
reduce 앞에 걸어야 한다는 것뿐이다. **다음 세션은 이 브랜치에서 시작해 `project_query`부터 바꾼다.**

### 10.0d 2026-09-10 저녁 — V209 채택, 그리고 정렬 벌점의 발견 (가장 최신, 여기서 시작할 것)

**Arena 후보: `V209_ffn_ring4_production`** (V204_submit + ffn x broadcast ring 4). 잡 5개, 15/15 PASS,
**ffn 317,445 → 294,574**, qkv·attn_out 불변 → 공식 draw 가정 기하평균 **≈ 6.05**(현 공식 5.9032, +2.5%).
`src/device/`만 바뀌므로 그대로 제출 가능. **제출은 사용자 지시 대기.**

**리더보드(공개 API `curl -k https://micro2026-api.duckdns.org:7777/api/leaderboard`): 등록 팀은 둘뿐이다.**
우리(Goat Chovy #1557)와 Participant #905(123,470 / 53,028 / 315,050 = 5.6669). 상대 ffn도 같은 벽 앞이다.

**새 실측 사실 두 개.**
1. **256 B 정렬 벌점은 실재하고 크다(V211, 3/3 잡).** 같은 29.5 MB를 down weight 레이아웃으로 추가 로드했을 때
   현행 8청크(960 B 런, 6개가 미정렬) **399 B/cycle** vs 2청크(3,840 B 런, 전부 정렬) **509 B/cycle** —
   **15,951 cycle 차이**. 비율 1.28은 "런이 granule 4개 대신 5개를 건드린다"의 5/4와 정확히 맞는다.
   **ffn은 지금 이 벌점을 물고 있다.**
2. **겹치는 타일은 합법이고 정확하다(V214).** 마지막 down 타일을 16@48+4@56 대신 **16@44**로 두면 행 44~47이
   두 번 계산·기록되는데 결과가 비트 단위로 같다(PASS, max\|Δ\| 동일). 대가는 겹친 행 수만큼의 계산(+7.6k).
   → **4의 배수가 아닌 행 수를 4행 transpose 규칙 아래서 덮는 법**이 생겼다.

**그런데 (1)의 적용(V213)은 컴파일러가 막는다.** 2청크 레이아웃은 슬라이스당 15행이라 출력 버퍼가 30 B가 되고
(`not a multiple of the SRAM access width 8`), 32 B나 40 B로 패딩하면 **컴파일러가 SIGABRT로 죽는다** —
V202b와 같은 크래시다. 구현은 `V213_ffn_down_two_chunks`에 전부 있다. 다음 세션은 (a) `DownRows2`의
live 슬라이스 2개짜리 inter-slice reduce를 의심해 stride를 바꿔보거나, (b) cargo-furiosa-opt 0.7.0으로
재시도하거나, (c) 출력 패킹을 transpose 없이 재구성한다. **이것이 남은 최대 판돈(−16k 이상)이다.**

**이번에 구조적으로 닫힌 것.**
- **x의 hi/lo 분해는 하드웨어 요구사항이다.** Lane으로 옮기는 길(V210)은 Lane fold 규칙(Interleaved는 OutPacket이
  정확히 `Lane # 8`, Sequential은 Lane이 OutTime 맨 끝)과 `StoVrf` 거부로 막히고, bf16 고정 피연산자(V212)는
  `bf16: ContractionWeight<f8e4m3>` · `f8e4m3: FetchCast<bf16>` 부재로 막힌다.
- V는 정말로 RMSNorm된다(`gemma4.py`의 `v_norm = RMSNorm(head_dim, eps, with_scale=False)`) — 지울 수 없다.

### 10.0c 2026-09-10 오후 — 실물 비용 모델을 바꾼 측정 (가장 최신, 여기서 시작할 것)

**한 줄:** 벽은 대역폭이 아니었다. **커널 시간의 절반은 바이트와 무관한 고정비이고, 그 정체는 커널마다 다르다.**

**① attn_out의 스트림은 이미 최고 속도다 (V205, 잡 4개).** O-weight 타일의 슬라이스당 행 수만
120/88/60/32/16으로 바꿔 `cycles = a + bytes/R`를 맞췄다 → **R = 645 B/cycle**(잡별 581/656/683/686),
절편 **27,000**. 즉 attn_out 50k 중 **27k(54%)가 바이트와 무관**하다. 같은 R을 적용하면
qkv 52k / ffn 155k가 고정비다. **접근 패턴(DMN·정렬·인터리브) 축은 여기서 닫힌다 — 최대 6%짜리다.**

**② 런 길이 곡선은 2,048 B 너머에서도 계속 떨어진다 (V208, 잡 3개).** q_weight를 한 번 더 로드한
증분: qkv 자신의 **30,720 B 런 = 465 B/cycle**, 같은 바이트의 **1,920 B 런 = 385 B/cycle**.
곡선 전체: **256 B 618 · 512 B 595 · 1024 B 557 · 2048 B 547 · 30,720 B 465**.
qkv의 31.46 MB는 465 B/cy면 67.7k = 실측 101k의 3분의 2. **그런데 H를 쪼개 런을 줄이는 길은 닫혀 있다**
— H=3840=15×256이라 2의 거듭제곱 분할(1920/960/480)이 전부 256 B 정렬을 깬다(V208 p1이 실측으로 확인).

**③ 더하기와 빼기가 비대칭이다 (V207 vs V208).** qkv에서 K+V 15.73 MB를 **빼면** 6.5~12.2k만 줄고,
같은 바이트를 **더하면** 33.9k가 는다. 빼면 숨어 있던 계산이 드러난다는 뜻이다 →
**qkv는 DMA와 계산이 상당히 겹쳐 있고 구속 자원은 계산 쪽이다.**

**④ ffn은 MainContext가 구속 자원이고, 스케줄은 그것을 모른다 (V206, ffn 3/3 잡).**
x ring broadcast를 ring 32 → **ring 4**(클러스터당 사본 8 → 64)로 바꾸면 ffn **−16,351 (−5.3%)**.
**정적 makespan은 +144로 정반대를 예측한다.** ffn Main 83.5k 정적 vs DMA 128k인데, 스케줄이
"Main은 스트림 뒤에 숨는다"고 보는 것이 실물에서는 성립하지 않는다.
qkv에 같은 변경은 무효~약간 손해다(Main이 스트림에 비해 작고, 사본마다 x2 영역을 다시 읽는
V158 병리가 이득을 먹는다). **프로덕션: `V209_ffn_ring4_production`** (src/device만, 검증 중).

**그래서 다음 세션이 팔 곳 (기대값 순).**
1. **ffn MainContext 83.5k.** 최대 항목은 `contract_up_gate_full` **36,526 정적**(up/gate pass A,
   LUT f4→f8 + contraction), 다음이 down pass A 19,315, geglu 5,848. `contract_lane`이
   `m![H / 16 % 4 # 8]`로 **8차선 중 4개만 쓴다** — `Dummy2`(x hi/lo 재생)를 시간축에서 차선축으로
   옮기면 가중치 fetch가 절반이 될 수 있다. **미검증, 가장 큰 판돈.**
2. **`begin_interleaved`** — 저장소가 한 번도 쓴 적 없는 API. "identical mapping인 두 텐서를 한
   sequencer 연산으로 합친다". ffn up/gate pass A와 qkv K/V가 정확히 그 형태다.
3. **qkv 계산 90k대.** ring 스윕은 실패했으나 Main 자체는 여전히 구속 자원이다. 미측정 슬롯
   V178(head-norm 9→5 pass)·V179(RoPE ternary)가 여기 해당한다.
4. attn_out 고정비 27k — 출력 store(2,040 정적, util 0.003)와 HBM 왕복 tail. V45가 이 슬롯이다.

**측정 방법 메모.** 바이트-스케일링 사다리(구조 고정, 바이트만 변경)와 탐침(같은 로드를 한 번 더)은
서로 다른 것을 잰다. 사다리는 **구속 자원**을, 탐침은 **한계 전송률**을 준다. 둘이 어긋나면 그
자체가 오버랩의 증거다. 정적 makespan은 **커널 안에서만** 유효하고, ffn Main처럼 스케줄이 숨는다고
보는 자원에는 **틀린다**.

### 10.0b 2026-09-10 메모리 대역폭 캠페인이 남긴 상태 (최신)

**공식 SOTA: `V204_submit` = 5.9032** (moa-submitter `46bc9eed`, 2026-09-10 05:12 UTC:
qkv 101,132 / attn_out 56,845 / ffn 317,445). 커밋 `3c381d2`. 새 실험은 여기서 분기한다.
측정본은 `V200_production`(같은 채점 코드 + 한 잡 안 A/B 대조군), Arena 14잡 기준 6.229.

**제출본은 `src/device/`만 고친다.** RULES §4.1은 `src/ops.rs`에서 **함수 본문만** 허용한다.
측정 브랜치는 A/B 대조군 함수를 ops.rs에 넣어도 되지만, 제출 브랜치는 넣지 않는다(V200 → V204가 그 차이).

**이 캠페인이 확정한 것 (RESULTS "메모리 대역폭 캠페인" 절에 표로 있음).**
- **DMA 비용 곡선(V197, 실측):** 128 B 런 319 B/cy · **256 B 618(최적)** · 512 B 595 · 1024 B 557 · 2048 B 547.
  256 B 미만은 정확히 2× 벌점이고, **256 B보다 길어도 완만히 나빠진다**. "길수록 좋다"는 틀렸다.
- **256 B 런은 분할 축이 2의 거듭제곱일 때만 쓸 수 있다**(청크 오프셋 정렬). attn_out(Qs=4096) 가능 → V198 −6.5%.
  qkv·ffn up/gate(H=3840), ffn down(청크폭 960)은 불가.
- **DMA 명령 병합은 커널 선두에서만 값어치가 있다.** 선두의 스테이징 store를 하나 없애면 ffn −11k(6/6 잡),
  같은 병합을 중반 geglu store에 하면 0(V201).
- **커널은 순수 DMA 바운드가 아니다.** 같은 15.73 MB를 추가로 읽으면 26,430 cycle(595 B/cy)이라
  "커널 전체 ÷ 바이트 = 310 B/cy"는 비-DMA 시간이 섞인 과소평가다.

**이번에 닫힌 길 (다시 열지 말 것).**
- DMN 인터리빙(book "alternate across 2 DMNs, else 50% loss") — 슬라이스 축을 뒤집으면 4/4 잡에서 **+6%**(V196). V8 종결.
- ffn down 벌크 로드 1개 — 파이프라이닝 손실로 정적 +4.6k, 2·3분할은 **컴파일러 SIGABRT**(V202).
- attn_out 타일 재분할 — 88/32가 이미 최적, 1타일은 +2.9k(V203).
- geglu store 병합(V201), attn_out store 병합(V199에서 +3.0k).

**구조적으로 닫힌 것 (근거와 함께).**
- **바이트 축소 불가:** 채점 fixture가 가중치를 PRNG로 합성한다(`prng::f4_nibbles`/`f8_banded`/`bf16_uniform`).
  균등난수라 sparsity·pruning·low-rank·codebook·엔트로피 코딩이 전부 무효다.
- **디바이스 팬아웃 불가:** `#[device(chip = 1)]`이 이미 `pe = 8`(pe0-3 + pe4-7) 전부를 잡고,
  `CLUSTER_SIZES=[1,2]`/`SLICE_SIZES=[64,128,256]`이라 2×256이 상한. `total_memory 268435456 = 512 × 524288`이 확인.
- **DMA 병렬화 불가:** 로드 경로 자원은 `DmaEngine` 하나이고 1,480개 명령 중 겹치는 쌍이 0.
  `PcieDmaEngine`은 유휴지만 HBM→DM에 타입으로 막혀 있다.
- 스펙 대비: 최고 실효 618 B/cycle = 1,500 B/cycle의 41%. 남은 격차는 커널 코드에서 닿지 않는다.

**운영.** 드라이버 deadline은 `/root/auto/deadline`(epoch)에 있고 지나면 keeper·driver가 함께 종료한다.
되살리려면 `echo $(( $(date +%s) + N )) > /root/auto/deadline; rm -f /root/auto/STOP` 후
`cd /root/auto && setsid nohup bash keeper.sh > keeper.out 2>&1 < /dev/null &`.
ssh 한 줄로 띄울 때는 stdout을 파일로 돌려야 세션이 안 붙잡힌다.
제출은 pod `/root/lab`에서 브랜치 체크아웃 후 `/root/.cargo/bin/moa-submitter submit --source /root/lab`,
확인은 `moa-submitter status` / `moa-submitter log <id>`.

### 10.0a 2026-09-10 밤 3차 체인(12h+)이 남긴 상태

- **공식 SOTA: `V165_qkv_broadcast_attnout_two_tiles` = 5.7576** (moa-submitter `d0239b5b`, 2026-09-09 17:15 UTC: qkv 105,544 / attn_out 53,374 / ffn 349,160). 새 실험은 여기서 분기한다.
- (제출 전 기록) 실측 SOTA 후보: `V165_qkv_broadcast_attnout_two_tiles` (= V82 + qkv x 복제를 16 디스크립터 + ring-32 `CustomBroadcast`로(V158)
  + attn_out O-weight 타일 48/12). cold 6회 median qkv 109.5k / attn_out 53.9k / ffn 350.7k, 전부 PASS → 공식 baseline 기준 ≈5.66.
  **리더보드 제출은 사용자 지시가 있을 때만**(팀 Goat Chovy #1557, 현재 공식 5.086 = V82; 1위 5.667). 제출: pod `/root/lab`에서
  브랜치 체크아웃 후 `moa-submitter submit --source /root/lab` (로그인 토큰 30일).
- **핵심 발견(RESULTS "실물 비용 모델"):** 커널은 PE ARM 코어 프로그램이고 DMA 디스크립터를 런타임에 직렬 생성 → "같은 작은 영역을
  512 디스크립터가 읽는" 로드가 실물에서 ~45k(정적 18k). qkv x2가 그것이었고 ring broadcast로 −43k. 더미 클러스터 축 위의 switch pass는
  cluster 0만 채운다(실제 축으로 로드·switch 후 reshape).
- **측정 방법:** tests/의 TESTS를 [변형 ×4 ...]로 두는 다중 변형 하네스(cold = 첫 실행, warm median/stdev 요약). Arena는 자유,
  잡당 반복으로 median/stdev를 본다. cold(채점 환경)는 ±5%.
- **닫힌 길(2026-09-09/10):** 스크래치→출력버퍼 hop(V151~153), ffn/attn_out x2 broadcast(V159/161), weight 디스크립터 절반(V160/V160b),
  q scatter(V164, API), rope idx8 gather(V169, 정렬), ffn down 3타일(V166 중립), store 전 switch gather(V167/V168: FAIL·느림).
- 드라이버: `/root/env.sh`에 `AUTO_CPU_TEST=0`, `AUTO_POLL_SECONDS=45`; 큐 문법 `BRANCH[@COMMIT][#TAG]`.

### 10.0 2026-09-09 첫 RNGD 실측 세션이 남긴 상태 (최신)

- **Arena 로그인 완료**(`knowin-kyeong`), 첫 실측 7건 확보. 자세한 표·근본 원인은 RESULTS.md
  §"2026-09-09 첫 RNGD 실측".
- **실측 SOTA는 `V50_drop_false_replication` 4.972×** (149,244 / 56,522 / 354,825, 3/3 PASS, job 15360).
  그 전은 V14의 3.936×. V15~V41은 전부 정확도 FAIL이라 점수가 없다.
- **원인은 하나다: 소스에 없는 축은 DMA write를 복제하지 않는다.** 두 곳이 이 전제에 기대고 있었다 —
  qkv x의 HBM 사본(V15 도입, V31이 16부로, V41이 attn_out·ffn까지 확대)과 V30의 RoPE 테이블 직접 gather.
  store 시간이 안 늘어난 것이 증거이자 함정이었다(스케줄에서 "공짜 복제"로 보였다).
- **V49(사본마다 명시적 store)는 기각.** store 1개당 Core 디스크립터 명령 3개가 in-order로 직렬화되어
  사본 수 어디서도 사본 없는 쪽을 못 이긴다(c=1 58,962 … c=16 144,128).
- **다음 레버리지는 qkv 하나다.** 실측 speedup이 qkv 1.64 vs attn_out 7.17 / ffn 10.45로, 기하평균이므로
  여기가 전부다. 후보: 사본의 실물 A/B(c=2; makespan은 c=1을 고르지만 실측은 사본을 더 좋아한다),
  V30을 복제까지 정직하게 다시 하기, V16~V37 중 미실측 최적화를 V50 위에 다시 쌓기(단 판정은 실측으로만).
- **makespan은 상대 스크리닝으로는 유효하다**(실측/makespan 2.0~2.4로 균일, 순서 보존). 다만 정확도는
  전혀 보지 못한다. **실측 없이 채택 판정을 내리지 않는다** — 이번 사고의 교훈이다.
- 야간 체인은 `/root/auto/STOP`으로 멈춰 있다. 재개하려면 STOP을 지우고 keeper를 다시 띄운다(§11.2).
  큐(`origin/auto_results:auto/queue.txt`)는 V15~V28 계보 순회로 갱신돼 있다(V15가 첫 FAIL일 것이라는
  예측의 확인용이며, 근본 원인은 스케줄 비교로 이미 확정됐다).

### 10.0.1 2026-09-09 밤 세션이 남긴 상태 (그 전 — makespan만 있던 시점)

- **선두 브랜치: `V41_x2_hbm_copies`** (V38 + x2 8부 사본; makespan 45,744 / 27,272 / 157,282, 기하평균 5.808×). 문서
  최신본(RESULTS/SOTA/RULES)도 여기. 계보: … V36 ─ V37 ─ V38 ─ V39(기각) ─ V41. V40(ffn down 타일 수), V42(attn_out 타일 형상),
  V43(qkv V 분할), V26(qkv H-split)은 모두 V39 위의 독립 A/B이며 **기각**(RESULTS 각 섹션에 스케줄러 함정 기록). V44–V48은
  슬롯만.
- **야간 자동 체인이 pod에서 돌고 있다** (§11). 큐(`origin/auto_results:auto/queue.txt`) 순서: V0 → V38 → V41 → 정확도
  이분 탐색 사다리(V14, V7, V29, V31, V32, V35, V23, V22, V13, V2, V1). Arena 로그인 전에는 makespan·전체 빌드만 기록하고
  `arena=pending`으로 남는다.
- **Arena는 로그인되어 있지 않다.** `ssh root@213.192.2.99 -p 41008 '. /root/env.sh; furiosa-arena login'`으로 사람이 GitHub
  device flow를 마치면 체인이 10분 안에 pending 항목을 순서대로 제출한다.
- pod의 결과는 push되지 않는다(자격증명 없음). 다음 세션 첫 일: `scp -r -P 41008 root@213.192.2.99:/root/auto/wt/auto ./auto`
  → 보드 수치를 RESULTS.md로 옮기고 `auto_results`에 커밋. (push를 자동화하려면 사용자가 pod에서 `ssh-keygen`으로 만든
  키를 `gh repo deploy-key add --allow-write`로 등록하고 `git remote set-url --push origin git@github.com:…`.)
- 남은 후보(기대값 순): V45(post-norm을 생산자 레이아웃에서, attn_out·ffn −1k씩), V46(geglu 스칼라 broadcast 병합 −1.2k),
  V44(a)(ffn scale 세그먼트, pair 행 교환), V47(global scale → eps −0.4k), V48(gather/scatter 인덱스 공유). 세 커널 모두
  DMA 큐가 makespan이고 weight 바이트는 HBM 한계라, 이제는 실측(Arena)으로 정적 모델과 실물의 차이를 먼저 봐야 한다.

### 10.1 2026-09-09 세션이 남긴 상태 (다음 세션이 이어받을 것)

**브랜치 계보 (모두 origin에 push됨, 각각 한 가지 변경만 담음):**

```
main ─ V0_baseline ─ V1_ffn_down_chunked_dequant ─ V2_attnout_rows_over_256_slices
                                                     └─ V7_qkv_x_replicate_via_hbm
                                                          ├─ V8_weight_rows_interleaved_dma   (기각, 실물 A/B용으로 보존)
                                                          └─ V9_ffn_x_via_hbm ─ V6_ffn_upgate_overlap ─ V10_attn_weight_tiles_fused_lut ─ V11_residual_1920_tiles
                                                                  ─ V12_ffn_rows_per_pass_12 ─ V13_ffn_dma_trims ─ V14_two_clusters ─ V15_x_replicate_hbm_copies
                                                                  ─ V16_rmsnorm_fused_residual ─ V17_qkv_hoist_weight_loads ─ V18_attnout_scale_in_epilogue
                                                                  ─ V19_qkv_tail_heads_layout ─ V20_qkv_tail_per_cluster ─ V21(기각, 코드 되돌림+rope 별칭 수정)
                                                                  ─ V22_attnout_weight_tiles ─ V23_ffn_whole_scale_loads  ← 선두
                                                                                                                                              (V3, V4, V5는 슬롯만; V5는 V10에 흡수)
```

**makespan (정적, cargo-furiosa-opt 0.6.0):** V0 116,583 / 194,020 / 1,693,200 → V23 51,631 / 30,487 / 181,625
(기하평균 5.131×; 커널별 2.26× / 6.36× / 9.32×). **실측은 하나도 없다.** 정확도도 미검증. V14는 베이스라인이 칩의 두 클러스터 중
하나만 쓰고 있었다는 발견(`Cluster = m![1 # 2]`)을 세 커널에 적용한 것이라 이득이 가장 크고, 실측에서
확인할 가치도 가장 크다.

**Arena 승인이 나면 가장 먼저 할 일 (순서 고정):**

1. pod에서 `. /root/env.sh; cd /root/furiosa-opt-gemma4-12B; git checkout V0_baseline && ./scripts/rngd_test.sh`
   → 세 커널의 **분모**(V0 실측 cycle)와 정확도 PASS 확인. 이게 없으면 아무것도 판정 못 한다.
2. `git checkout V23_ffn_whole_scale_loads && ./scripts/rngd_test.sh` → 정확도와 실측 cycle.
3. V23이 정확도에서 깨지면 계보를 거슬러 이분 탐색: V22 → V20 → V19 → V18 → V16 → V15 → V14 → V13 → … → V1.
   V14 이후 변경은 전부 두 클러스터 레이아웃·HBM 경유 gather에 기대므로, 깨진다면 V14가 첫 용의자다
   (`dma_scatter`/`to_hbm_view`를 두 클러스터 텐서에서 호출하는 V19/V20도 실측 검증 대상). 각 브랜치가
   단일 기전이라 깨진 지점이 곧 원인이다. 수치를 건드린 변경은 없다(V2의 f32 inter-slice reduce는 오히려
   정밀). **유효성 리스크가 가장 큰 것은 V7/V9의 커널 내 `HbmTensor::new()`** — 채점 런타임이 커널 내
   HBM 할당을 거부하면 DM→DM 복제(54k)로 되돌린다.
4. 실측이 나오면 RESULTS.md 요약 보드의 "RNGD 실측"·"정확도" 칸을 채우고 판정을 `채택/기각`으로 바꾼다.
   SOTA.md는 그때 처음으로 갱신한다.

**pod 상태 (`root@213.192.2.99 -p 41008`, 휘발성):** 툴체인·env.sh·헬퍼 스크립트(`/root/*.py`, `/root/*.sh`)는
전부 `scripts/dev/`와 RULES §6.3에 있으므로 pod이 사라져도 §6.3 절차로 30분 안에 복구된다.
`cargo-furiosa-opt`는 **0.6.0을 고정 설치**해야 한다(0.7.0은 crate와 불일치, §6.3.1 참조).

**다음 가설 후보 (RESULTS.md 각 섹션 "다음 후보" 종합, 기대값 순):**

| 후보 | 기대 | 근거 |
|---|---|---|
| geglu를 두 클러스터 레이아웃에서 직접 수행(up/gate HBM hop 12k 제거 + vector 작업 절반) | ffn −15k | 60원소/슬라이스는 8-wide f32 패킷에 안 맞아 pairing 재설계 필요 |
| FFN `DmaLoad ?`(838 × 15 = 12.6k) — pass 수를 더 줄일 수 있는 레이아웃 | ffn −5k | ROWS_PER_PASS>12는 scale VRF(8 KB) 한계 |
| qkv 소형 DMA 14k(cos/sin gather+hop 3.6k, K/V scale 838×2, scatter 929×2) 통합 | qkv −3k | DMA 47k 중 weight/x 33k 외 잔여 |
| attn_out tail(HBM gather 2.5k + rmsnorm 3.5k) | attn_out −2k | |
| V8 실물 A/B (인터리브 레이아웃) | 실측에서만 판단 가능 | 정적 모델은 무반응 |
| `shared::rmsnorm` 경량화 (ReducingSlices 경로 3~4k × 4회) | 세 커널 각 −2k~−3k | 기하평균 레버리지 |
| qkv x 복제 18.4k (H-split, 정렬 리스크) | qkv −10k | §RESULTS V3 보류 사유 참조 |
| V8 실물 A/B (인터리브 레이아웃) | 실측에서만 판단 가능 | 정적 모델은 무반응 |

---

## 11. 무인 실험 체인 (2026-09-09 도입)

랩탑이 꺼져 있어도 pod가 큐의 브랜치를 순서대로 **makespan 덤프 → 전체 크레이트 빌드 → (로그인 시) Arena 제출**하고 기록한다.
코드는 `auto_results` 브랜치의 `scripts/dev/auto/`(driver.sh·keeper.sh·dump.sh·summarize.py·record.py), 설명은 `auto/README.md`.

### 11.1 프로토콜

1. 실험 브랜치를 push한 뒤 `auto_results`의 `auto/queue.txt`에 한 줄(`BRANCH` 또는 `BRANCH@COMMIT`) 추가하고 push한다.
   GitHub 웹 편집기로도 된다. 드라이버는 매 항목 뒤·매 유휴 폴(10분)마다 `origin/auto_results:auto/queue.txt`를 다시 읽는다.
2. 항목마다 pod의 `/root/auto/wt/auto/results/<id>.json`과 `auto/BOARD.md`(makespan 3개, V0 대비 기하평균, 빌드, Arena 판정,
   첫 컴파일 에러)가 갱신된다. 로그 tail은 `auto/logs/<id>.*`.
3. Arena가 로그인돼 있지 않으면 `arena=pending`으로 남고, 로그인이 확인되면 큐 순서대로 재제출된다. 큐에서 지운 항목은 재제출
   되지 않는다(기각된 변형은 큐에서 지워 Arena 시간을 아낀다).
4. 보드의 숫자는 **스크리닝**이다. RESULTS.md로 옮겨 적고, 판정은 §5대로 실측 후에만 한다.

### 11.2 pod 쪽 운영

- 시작: `/root/auto/deadline`(epoch)을 쓰고 `cd /root/auto && setsid nohup bash keeper.sh > keeper.out 2>&1 < /dev/null &`.
  keeper는 driver가 죽으면 60초 뒤 다시 띄운다(driver는 flock으로 1개만).
- 중지: `touch /root/auto/STOP` 후 driver의 자식(`sleep`)부터 죽인다 — 자식이 flock을 물려받는다(`pkill -P <pid>; kill <pid>`).
- 드라이버 갱신: `git show origin/auto_results:scripts/dev/auto/driver.sh > /root/auto/driver.sh` 후 driver만 죽이면 keeper가
  새 스크립트로 재시작한다.
- 결과 worktree(`/root/auto/wt`, 브랜치 `auto_results`)는 **절대 pull/rebase 하지 않는다**(첫 판에 identity 없는 rebase가
  멈춰 체인이 서 있었다). push 자격증명이 없으므로 결과는 `scp`로 가져와 랩탑에서 커밋한다. push까지 자동화하려면 pod에
  deploy key(쓰기)를 등록한다(§10.0).
- 드라이버는 브랜치의 스크립트에 의존하지 않는다(V0_baseline에는 `scripts/dev`가 없다): 덤프는 `/root/auto/dump.sh`,
  Arena는 `/root/auto/arena.sh`(§6.3.1 ⑤⑥ 때문에 상류 `rngd_test.sh`를 쓰지 않는다. 실행시간 상한
  `ARENA_TIMEOUT`(70)과 큐 대기 `ARENA_WAIT`(1800)이 분리돼 있고, 제출 출력과 exit code를 항상 남긴다).
- 스크리닝 클론 `/root/lab`, `/root/lab2`(target 사본)에서 대화형 컴파일을 하면 드라이버와 CPU를 나눠 쓴다(각 ~2분/커널).
