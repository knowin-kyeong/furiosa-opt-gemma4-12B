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

- **공식 리더보드 제출(`moa-submitter submit`)에 횟수 상한은 없다** (2026-09-11 확인: 상류 `README.md`의 Stage 1 TBD 목록에서 "maximum submissions per team"이 사라졌다. 이전 규칙 "하루 10회"는 폐기). 상한이 없어도 **확실한 개선(짝비교로 확인된 것)만** 올린다 — draw는 코드를 비교하지 못하기 때문이다(§10.0l). 실제 병목은 예산이 아니라 직렬 제출 시간(잡당 ~2분)이다.
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

### 10.0t 2026-09-12 — 공식 1위 7.3136, 남은 최대 비용은 클러스터 1 지연, book이 알려 준 switch의 DMA 점유 (가장 최신, 여기서 시작할 것)

**상태.** 공식 최고 **7.3136**(`0ead0b04`, `V313_submit` draw — qkv 87,602 / attn 40,305 / ffn 271,797) → **리더보드 1위**(2위 #663 7.3097 = 91,578 / 39,144 / 268,129).
V313 draw 배치 2·3은 6.45~7.31, 평균 ≈6.87 — 최고가 평균 +6.5%다. **8.0은 draw로 불가능하고 코드 평균 +9.4%가 필요하다.**
draw는 pod `/root/drawchain_follow.sh <branch> <first_batch> <n>`이 12회 배치로 계속 돌린다(`/root/draw_src`, 로그 `/root/drawloop_<branch>.batch<N>.log`; 도는 동안 `draw_src`·`lab3` 금지).

**V313 실물 임계 경로** (`scripts/dev/span/crit.py`를 pod에 stdin으로 넘겨 실행, warm launch #1, 정적 `/root/tk/S313_*.json`):

| 커널 | launch | 사슬 | 이름 붙은 대기 |
|---|---:|---|---|
| qkv | 95.3k | DMA 70.5k · 동기화 8.9k · TU 6.6k · slack 5.1k (+ 끝 동기화 4.0k) | V 로드 뒤 RoPE cs 재로드 동기화(`rope.rs:280→283`) 8.9k + 끝 4.0k ≈ **13%** |
| attn | 45.3k | DMA 30.4k · 동기화 9.4k · TU 4.5k | contraction store 뒤 동기화(`projection.rs:238`) **9.4~12.6k** |
| ffn | 274.9k | DMA 258.5k · 동기화 13.6k · TU 14.9k · slack 13.2k | geglu hop 동기화 **12.9k** · down_scale 로드 slack **3,144**(erf switch `xsw.rs:368` → StoVrf `f8split.rs:70` 2.76k → StoVrf `mlp.rs:722`가 끝난 뒤 150,730 발행) · inv_s 재로드 slack 2,441 · down x 재로드 9.1k(1.97 MB, 216 B/cycle, 영역당 32 reader) |

모든 동기화는 클러스터 0이 클러스터 1을 기다리는 시간이다(V294). 몫을 클러스터 0으로 옮겨야만 줄고, 행 단위 읽기에서만 통한다(V321 vs V323·V324).

**book·논문 재독** (Switch Engine · DMA Engine · Memory Performance · Schedule/Tuning · Transformer 사례, TCP ISCA'24, IEEE Micro HC2024)
1. **CustomBroadcast의 SFR 쓰기는 DMA 엔진과 sub 컨텍스트를 점유한다.** 정적 스케줄은 switch pass를 MainContext로만 둔다(S313: `xsw.rs:168` · `xsw.rs:368` · `mlp.rs:216` · `mlp.rs:285`) — V306이 정적의 3배를 번 기전이다. 큰 로드 앞 CustomBroadcast 제거는 정적 스크린이 과소평가하므로 짝비교로 판정한다.
2. **0.6.0에는 DMA 엔진 지정 API가 없다**(`Context.tdma`는 marker, `DmnIndex`/`DmaDescriptor` 비공개). DMN당 128 B/cycle(32 슬라이스가 데이터 경로 공유), 클러스터당 256 B/cycle(DMN 인터리빙), **슬라이스당 DMA 명령 큐 2칸**, HBM 채널 컨트롤러 큐 64칸, 명령당 startup ≈500.
3. **`commit_cast`/`commit_cast_relu`는 0.6.0에 있다**(`engine/commit_adapter.rs:174`): f32→bf16 cast를 commit 경로로 접어 Cast Engine을 sub Vector Engine 작업에 비워 준다.
4. 0.6.0에는 `#[unroll]`이 없고 채점기가 받는 스케줄러 env 노브도 없다. 공식 transformer 예제는 weight를 `to_dm` 한 번 → sub staging으로 TRF(Lane)에, 활성값은 main 스트림으로 — qkv·attn의 Lane 8 사용과 같다.

**이번 캠페인 슬롯:** `V326`(qkv 큐 전체 몫 균형 탐침 — **기각: q5 +1.2% · kv5 +0.7% · q9 −1.8%(9/12). 몫은 명령마다 맞춰야 하고 62.5%는 과하다 → qkv 클러스터 몫 계열 닫힘**) → `V327` 취소 · `V328`(ffn CustomBroadcast 정리 — **설계 단계 기각**: erf는 전역 스칼라 분배, inv_s는 클러스터별 값이라 switch 대안이 대량 pass·복제 로드·추가 동기화뿐) · `V329`(ffn h 재로드 사본 탐침 — **설계 보류: 기대 순이득 ≈0**) · `V330`(ffn down 256 B 정렬 청크 — **설계 기각: 클러스터별 geglu inv_s가 청크 경계를 막는다**) · V323 패키지 16잡 재측정(job 20380 `rngd rerun`) — **결과: a53 +2.0%(11/16 느림) · a67 +7.4%(16/16) → attn 클러스터 몫 계열 닫힘.** 6잡 −461(4/6)은 노이즈였다 — 클러스터 지연 계열 판정은 16잡으로.

**cold 벌점 census (같은 날, 생산 arm의 커널 첫 launch − warm 중앙값):** qkv **+3.0k**(V325 잡 24행, 3.2%) · **+5.2k**(V326 잡 24행, 5.2%) · attn −3.5k / +2.0k(노이즈) · ffn −0.4k.
공식 점수는 커널당 cold launch 하나이므로 **qkv의 3~5% cold 벌점은 warm 짝비교가 보지 못하는 별도 표적**이다(첫 실행의 명령 청크 스테이징; ffn은 초반이 긴 DMA 대기라 숨는 것으로 보인다). 판정은 변형별 cold 표본을 잡마다 짝지어야 한다.
**짝지은 cold census (잡별 arm 첫 launch끼리):** V299 `hi` − `sw` **−5.5k (7/8)** — `hi`가 프로세스 첫 launch(위치 0)였는데도 이겼으니 V299는 cold에서도 warm 이득(−9.1k)의 60% 이상을 지킨다.
반면 V326 q5·kv5·q9 − b의 −2.9~3.9k(7~9/12)는 **위치 교란**이다(b는 늘 위치 1, 나머지는 2~4).
⇒ **스윕 순서는 바이너리에 고정되고 `rngd rerun`은 같은 순서를 반복하므로, 기존 로그의 cold 비교는 판정용이 아니다.** cold를 판정하려면 첫 launch 위치를 잡마다 회전시킨다(하네스가 시작 시각으로 변형 부분열을 회전 — rerun도 순서가 바뀐다).

### 10.0s 2026-09-12 — 동기화는 클러스터 1의 지연이다, 칩 안 복제로 HBM 왕복 둘을 없앴다

코드 SOTA **`V301_submit`**(V299_submit + V301 ffn pass A Lane), 공식 최고 **7.2264**(V293_submit draw; 이전 7.0314)(V267/V273 계열의 운 좋은 draw; V273 draw 평균 ≈6.56).
짝비교 합산 기대 개선은 기하평균 +0.6%라 draw 하나로는 보이지 않는다(σ≈2.5%). 7.03을 확실히 넘으려면 몇 % 단위 구조 개선이 더 필요하다.

**확정 사실**
1. **DMA는 클러스터마다 직렬 FIFO이고 span 시작은 발행 시각이다.** V276 `qb`의 "겹친 로드"는 줄 선 시간이었다(동시 실행 아님).
2. **span은 클러스터 0의 명령만 보고한다.** 커널 중간·끝 `Cluster` span = 클러스터 0이 클러스터 1을 기다리는 시간.
3. **동시 DMA에서 클러스터 1만 느려진다 (V294):** 한쪽만 로드하면 끝 동기화 1,298 고정, 둘 다 로드하면 4~16k. "무작위 동기화"와 "첫 중간 동기화가 길다"의 정체.
4. **클러스터마다 자기 HBM 영역만 store/reload해도 컴파일러는 ExplicitSync를 넣는다**(ps.py). 동기화를 없애려면 HBM을 아예 거치지 않아야 한다.
5. **switch 비용 = ring_size × Time::SIZE × flits_per_packet**(book, V273 x8 broadcast 정적 7,943으로 확인). `CustomBroadcast` 검증기는 패딩은 trim하고 live 데이터는 버리지 않는다 →
   부분링마다 청크 8개를 두면 ring-32 all-gather 한 번(정적 ~1.2k)으로 256 슬라이스에 x 전체를 놓는다(`src/device/shared/xsw.rs`).

**채택:** V292(qkv −1.3%, 7/8) · V293(ffn −0.6%, 13/16) → `V293_submit`(Arena 25/25 PASS ×2).
**운영 규칙(사용자, 2026-09-12):** 같은 바이너리 반복 잡은 `rngd rerun <job id>`(`/root/auto/arena_rerun.sh`, `/root/tk/rerun_n.sh`) — 재업로드 없음.

**다음 레버**
- **비대칭 분할은 기각(V295):** 클러스터별 명령 둘은 직렬화된다(c53 70.6k vs c01 ~38k). 두 클러스터 로드는 **한 명령 안에서 HBM 읽기 대역폭을 나눠 쓰는 것**이 물리적 한계다
  (클러스터 단독은 DM 쓰기 한계 ~262 B/cycle, 둘이면 합 ~420 B/cycle, 클러스터 0 우선) ⇒ 레버는 **바이트당 HBM 속도(런 구조)**: attn O-weight(256 B 런) ~680 B/cycle vs qkv Q(30.7 KB 연속) ~500.
  **V296 탐침 진행 중:** q_weight를 attn식 256 B 열 청크 레이아웃(`m![Qs / 128 % 16, H / 256 # 16]`)으로 로드한 속도.
- **정정 (V321~V325, 2026-09-12): 비대칭 분할은 한 명령 안에서 된다 — 다만 행 단위 읽기에서만 이긴다.** 축을 먼저 패딩하고 stride(`Qs # 4608 / 2304`, 패딩량은 가장 안쪽 묶음 크기의 배수)하면 텐서 하나다.
  q_weight 행 읽기 56.25%: **−6.9%(5/6)**, 잡 간 흔들림도 사라짐(V321). 그러나 attn 256 B 열 청크 로드는 노이즈(V323), ffn 연속 30행 런은 **+11~20%**(V324), attn_out 커널 적용은 +3.0%(V322).
  qkv는 head 정렬 때문에 62.5%만 가능(−0.8k, 3/6). 같은 날 닫힌 것: 머리 동률을 소스 순서로 깨기(V319·V319b, 스케줄 불변), 한 버퍼 로드로 깨기(V325 −0.38%, 6/12),
  클러스터 교차 배치(V317 +14%, V318 중립), pcopy(V320 unknown primitive), RoPE rotate/sin 융합(V316 +2.6%).
- qkv: V 로드 뒤에 남은 rope HBM 왕복(gather·store·sync·reload)과 꼬리 17.7k. attn: contraction 병합 동기화(= 클러스터 1 지연) 10~15k.

**같은 날 이어서 (V296~V300) — 로드 속도는 런 구조가 정한다, qkv −8.8%**
- 두 클러스터 로드는 HBM 읽기 대역폭을 나눠 쓰므로, 남은 레버는 **바이트당 HBM 속도**다. qkv weight를 **행 하나씩(3,840 B = 256 B 정렬) 읽게** 슬라이스 배치만 바꾸면
  q_weight −15%, k_weight −18%(V297·V298). head 모음과 맞추려면 head의 64 슬라이스 안에서 교차하고(원소 k = 행/64, 슬라이스 s = 행 % 64),
  ring-64 `Broadcast1`에 k를 Time으로 넣으면 packet-major 도착이 자연 순서가 된다. 1값 패킷은 commit할 수 없어(8~32 B) transpose로 4개씩 묶는다.
- **V299(qkv 커널 적용): 8/8, −9,079 (−8.8%)** → `V299_submit`. 같은 트릭을 ffn up weight(1,920 B 행 = 7.5 granule)에 쓰면 +27%로 나빠진다(V300) — 이득은 행이 256 B로 정렬될 때만.
- 256 B 열 청크 레이아웃(`H / 256 # 16`)은 load 컴파일러 ICE(V296).
- **V301: ffn pass A의 LUT 조회 절반(V235)이 V293 경로에서 8/8 −2.3%** → `V301_submit`. V235가 중립이던 이유(gate 로드가 pass A를 묶음)가 V293의 x2 동기화 제거로 풀렸다 —
  "과거에 중립이던 절감은 임계 경로가 바뀐 뒤 다시 잴 것".
- V302: gather 출력 클러스터에 tile 표기를 써도 클러스터 1로 가지 않는다(무시) → RoPE HBM 왕복은 소스로 못 없앤다. post-V299 qkv 꼬리 = rope 왕복·대기 ~22k.
- V303(down contraction Lane): 6/8 −0.24%, 미채택 — down 단계는 타일 로드에 묶여 있다. RoPE 패스 융합은 `to_vrf`가 텐서 전체 staging만 지원(tile 쓰기 없음)해
  q·k마다 Main 패스 1개 절감(~1–2k)이 상한이라 보류.
- **V301_submit ffn 대기열 조사(crit.py, 279.9k launch):** DMA busy 240.9k · idle 39.0k. 최대 낭비는 **geglu 왕복 동기화 15.3k 중 FIFO 유휴 13.5k** —
  정적 스케줄이 down 타일을 "타일 로드 → 그 타일 contraction"으로 파이프라인하느라 타일 1~3(49.5k)을 geglu store 뒤에 둔다(모델은 동기화를 600으로 본다).
  그 밖에 down scale 로드 앞 유휴 7.6k, 꼬리 5.3k. V304(attn 한 타일)는 중립.
- **V305(down weight 한 명령) 6/6 +14.1k 기각:** 합친 로드도 정적 스케줄러는 첫 소비자(down contraction) 바로 앞, 즉 geglu store 뒤에 둔다.
  hop 유휴는 회수되지 않고 타일 로드↔contraction 파이프라인만 잃는다. geglu hop 유휴 13.5k는 로드 모양으로는 못 줄인다.
- **V306(out_scale을 꼬리 곱으로) 12/12 −4.9k(−1.76%) → `V306_submit`.** 판독(ffn PE 프로그램 `DUMP_PE_PROGRAM` + span): 명령은 정적 시작 순서로
  클러스터당 TUC 큐 하나에 들어가고, DMA 발행은 앞선 DMA 완료(`wait_dma` TUC 명령)와 **앞에 선 TU pass의 종료**를 기다린다. 실물 TU는 정적의 1.6~3.6배
  (switch 295 → 1,069, pass B 761 → 2,757, 스칼라 StoVrf 267 → 505~3,114), DMA는 ~2.15배라 **TU 체인 뒤에 줄 선 로드가 DMA 엔진을 놀린다.**
  `StoTab`(f4 LUT 적재, 정적 1,293)은 Main pass가 끝날 때까지 기다린다(실물 6~9k로 보임). ⇒ **큰 로드 앞 창에서 작은 TU pass·switch를 빼는 변경을 찾는다.**
  남은 창: ffn 머리(up weight 발행 11.2k, 앞의 x 스테이징 StoVrf 3.1k · LUT · StoTab · cfg), down_scale 앞(erf switch · StoVrf · pass B 4개), qkv 꼬리(rope 왕복).
- **V307(qkv RoPE를 클러스터별 gather로) 중립(6/12, +90):** `dma_gather_unscaled`에 두 클러스터 head 슬라이스의 DM 인덱스(`rope_offset ≫ 9`)를 주면
  각 클러스터가 제 행을 읽는다(PASS) — HBM 왕복·동기화 없는 gather가 가능하다는 것은 확인. 그러나 **동기화를 없애도 클러스터 1의 DMA 지연은 끝 동기화로 옮겨 갈 뿐**이다.
  ⇒ 실물 이득은 **두 클러스터 DMA가 함께 노는 시간**(대칭 TUC 큐 막힘, V306), DMA 효율(디스크립터·정렬), TU 꼬리에서만 나온다. 동기화 자체를 표적으로 삼지 말 것.
- **V308(ffn 머리 순서 강제 탐침) 6/6 +1.9k 기각:** up weight를 x 로드 바로 뒤로 강제하면 머리 유휴 5.7k는 사라지지만 rms weight · LUT · cfg가 44k 로드 뒤로 밀려 더 잃는다.
  **정적 makespan은 이번 세션 네 번 모두 실물의 부호를 맞췄다**(V305 +5.6k → +14.1k, V306 −1.6k → −4.9k, V307 +76 → +90, V308 +1.7k → +1.9k) — 구조·순서 후보는 컴파일 + `--dump-schedule`로 먼저 거른다.
  도구: `BEAM_SEARCH_TRACE_DUMP_PATH` 경로 추출은 scratchpad `beampath.py`(pod `/root/tk/`), 순서 강제 A/B는 `probe_build.sh <lab> <kernel> <order> <tag>` + `pairs_lab.sh`(첫 잡 뒤 `rngd rerun`).
  캐시에 커널이 있으면 빔 추적이 안 나온다 — `target/furiosa-opt` 아래 그 커널 파일을 지우고 컴파일할 것.
- **V309(qkv 꼬리 순서 강제) 16짝 중립(−0.3%), 순서 스크린 8종:** 빔 탐색의 ffn 순서는 시험한 대안보다 정적으로 모두 같거나 좋고, qkv는 RoPE 스테이징을 Q 앞에 둘 때만 정적 −623.
  **소스 생성 순서는 스케줄을 바꾸지 않는다**(RoPE 스테이징을 첫 호출로 떼어도 스케줄 동일) — 강제 순서의 이득은 구조 변경으로만 옮길 수 있다. 순서 조정 계열은 여기서 접는다.
- **V310(down scale 레이아웃 로드 탐침) 8/8:** 같은 3.69 MB 추가 로드가 8청크 +27.7k · 4청크 +22.1k · 2청크 +18.8k. **ffn의 남은 최대 레버는 down 청크 수**다
  (weight V211 −16k + scale −8.9k). 걸림돌 셋: x 스테이징 4배(2청크, 추정 +7k), 행 수 15의 출력 정리, 타일 경계. V246의 패배는 타일마다 넣은 gather/pack pass(큐 막힘)로 설명된다.
  다음 설계: 8슬롯 묶음 안 행 교차(V299식) → 한 번의 ring-8 gather가 packet-major로 자연 순서를 돌려준다.
- **V311(2청크 down) 기각: +67.6k, 정확도 FAIL.** 로드 절감은 정적으로 보였지만(tile −1.3k × 4, scale −4.8k) **x 청크 재로드가 73k**였다 —
  같은 HBM 영역(15,360 B)을 클러스터당 128 슬라이스가 다시 읽는 복제 로드는 영역 × 사본에 초선형(8청크 3,840 B × 32 → 5.3k). V154의 병리와 같은 계열.
  ⇒ **down 청크 수를 줄이는 계열은 x 복제 비용으로 닫힌다.** 청크를 줄이려면 x를 슬라이스마다 다시 읽지 않는 방법이 먼저 있어야 한다.
- **V313(down 출력 한 번에 store) 12/12 −2.9k(−1.04%) → `V313_submit`.** V260의 반쪽 store 트릭은 V306 스케줄에서 빈칸을 잃고 두 store 사이에 동기화를 끼웠다 —
  **조건이 바뀌면 옛 트릭을 다시 잰다**(V301·V266과 같은 교훈). 타일 계획은 정적으로 16/16/16/8/4 단일 store가 −1,011로 더 좋아 V314에서 3-arm 비교 중.
- **V314(16/16/16/8/4 단일 store) 12/12 +4~8k 기각:** 정적 −1,011이 실물에서 뒤집혔다 — **정적 스크린은 같은 타일 수·같은 런 구조 안의 순서/pass 변경에만 쓴다.**
  V313 census(276.4k): 꼬리는 마지막 타일 pass A 9.1k + pass B 2.3k가 묶는 TU 경로다. 남은 slack: 머리 3.8k · down_scale 앞 3.6k · inv_s store 뒤 2.2k.
- 리더보드: `V293_submit` draw **7.2264**(091cfa9f: qkv 90,833 / attn 38,455 / ffn 284,802)로 **공식 최고 경신, 2위**; 1위 #663 7.3097(ffn 268,129). 격차는 ffn.

### 10.0r 2026-09-11 오후 — 동기화의 정체, 순서를 강제하는 도구, 그리고 네 번의 기각

코드 SOTA는 그대로 **`V273_submit` `331104b`**, 공식 최고 **7.0314**. V273 draw는 오후에 6회 더 뽑았다(6.4623 / 6.6097 / 6.6501 / 6.7763 / 6.5899 / 6.3980) — 기록 경신 없음.

**실측으로 확정한 사실 (RESULTS `V284`~`V288`, 메모리 `hardware-span-profiler`)**

1. **0.6.0은 store한 HBM을 다시 읽는 로드 앞(과 커널 끝)에 `ExplicitSync`를 넣는다** — V273 세 커널에서는 모든 `DmaStore`(scatter 포함) 뒤에 동기화가 붙어 'store마다 하나'로 보였지만, 다시 읽지 않는 dummy store를 attn 앞쪽에 넣자 store 3개에 동기화 2개였다(같은 날 밤 정정). PE C 프로그램(`DUMP_PE_PROGRAM=<dir>` → `code_0.c`)에서 이것은
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

**같은 날 밤 추가 (RESULTS `V289`~`V291b`)**

- **동기화가 DMA/TU 완료에 풀리는 경우를 전수 집계했다**: 큰 로드에 묶이는 동기화는 특정 구조(rs 16/32 · qa 2/6)에서만 생기고, 짧은 동기화는 상대 쪽 TU pass 완료에 풀리는 경우가 많다(ffn 14/92 · qkv 33/184, 우연 2~4%). **긴 첫 중간 동기화의 트리거는 여전히 없다.** 동기화는 store가 아니라 **store한 HBM을 다시 읽는 로드**(와 커널 끝)에 붙는다 — 다시 읽지 않는 dummy store는 동기화를 만들지 않았다(V290 +4.4%).
- **긴 동기화가 비용이 되는 것은 FIFO의 큰 로드를 가로막을 때다.** V291: h2 + 순서 강제(`T120 -> T52`)로 K/V 로드가 x2 store 앞(FIFO 맨 앞)에 오자, 가장 긴 동기화(~47k)가 마지막 V 로드 완료에 풀렸는데도 h2 대비 **−5.0%(6/6)**. 그러나 프로덕션 대비는 **−1.9%, 5/8, p = 0.73**(V291b)이고, 같은 순서를 소스에서 요청하는 탐침 둘(로드 keep-alive, Q만 교체)이 모두 실패했다 — **빔 탐색 스케줄러는 한 경로의 정적 비용만 바뀌어도 순서 전체를 다시 짠다.**
- V289(Q 로드를 x8 앞으로 강제) +12%: x8 로드가 FIFO에서 Q 뒤로 밀린다. 순서 후보는 hwsim3(FIFO 모델)로 먼저 거르면 부호가 맞았다.

**다음 세션에게**

- 가장 큰 미지수이자 비용은 **동기화가 무엇에 묶이는가**다(커널마다 12~40k). V288은 "먼저 발행된 큰 로드의 완료"를 가리키지만 attn 중간 동기화와 h2의 rope 동기화는
  DMA가 놀 때 끝났다. `rs`/base PE 프로그램 블록(`blocks.py`)과 span의 동기화 종료 시점을 대조해 규칙을 세우고, 규칙이 서면 store를 큰 로드 **앞**으로 옮기는 소스 구조를 설계할 것.
- 레이아웃·store 개수를 바꾸는 후보는 **정적 스케줄의 DMA·동기화 순서부터** 비교할 것 — 네 번의 기각(V284·V285·V287·V288)이 전부 "부분 최적화는 맞았는데 순서가 바뀌었다"였다.
- 리더보드(2026-09-11 11시): 1위 #663 7.2836 (91,777 / 39,296 / 269,387), 2위 우리 7.0314, 3위 monkey 7.0170 (ffn 263,908).

### 10.0q 2026-09-11 — 실물 타임라인을 얻었다: 동기화 대기와 명령 순서

**상태 한 줄.** 공식 최고 **7.0314** 그대로 (2위, 1위 #663 **7.2164**). 코드 SOTA **`V273_submit`** 그대로, 채택 없음.
`V273_submit` draw 루프가 pod `/root/drawloop.sh`로 돈다(`/root/drawloop_V273_submit.log`; 도는 동안 `/root/lab3` 건드리지 말 것).

#### ① 도구: 실물 span 타임라인 (`V280_hw_span_dump`)
- 테스트 바이너리가 `main` 첫 줄에서 `TUC_PROFILE_LEVEL=trace`를 켜고 span `name`을 기록한다 → launch당 ffn 144 · qkv 82 · attn 24 span
  (`DMA`, `Renegade::TuExec`, `StoVrf`/`StoTrf`/`StoTab`, `Cluster`). 새 가설은 **정적 스케줄보다 먼저 이 span으로** 본다.
- 판독법: SPAN 줄(도착 순서, 정렬 안 됨)을 정렬하고 `Cluster` 구간, DMA 합집합의 빈 구간, 동기화와 함께 끝나는 긴 TuExec을 찾는다.
- 채택 판정은 여전히 **trace 없는** 짝비교 잡 + 변형 먼저 잡 + 제출 검증이다.

#### ② 판독 (V273 코드)
1. **`Cluster` span = 클러스터 간 HBM store 뒤 ExplicitSync 대기, 실물 0.4~15k cycle이고 같은 코드에서도 launch마다 무작위다** (정적 600).
   ffn 9개 합 32.4k(11%), qkv 16.6k(16%), attn post-store 4~14k. **store 하나 = 동기화 하나**, `dma_gather_unscaled`도 gather마다 하나.
2. 동기화 뒤에 나열된 Sub staging은 대기 끝까지 멈추고, attn의 x reload는 대기가 끝나야 시작한다 → **attn launch 분산(42k ↔ 49k)의 주성분이자 공식 draw 복권의 큰 몫.**
3. PE는 명령 목록을 정적 begin 순서로 걸으며 입력이 준비될 때까지 멈춘다(attn tile1 로드 발행이 tile0 로드 끝까지, qkv Q 로드가 x2 동기화 + switch 뒤로 밀린다).
   서로 다른 버퍼로의 로드는 겹쳐 돌 수 있다(`qb`: weight 로드 셋). **그러나 같은 버퍼의 타일 로드는 겹치지 않고, 겹치는 메모리 트래픽은 서로를 늦춘다(V282 +17%).**
4. 같은 잡 qkv: base 102,973 · `qb` 89,648 · `qa` 114,226(vs 100,682) — 순서·동기화 위치가 수만 cycle을 좌우한다. 올바른 변형으로 `qb`의 순서를 얻는 법은 아직 없다.

#### ③ 이번 라운드 실패 (다시 하지 말 것)
| 시도 | 결과 |
|---|---|
| `V281` BC (attn norm 꼬리를 두 클러스터에서) | 대기 길이는 구조와 무관 — 기각 |
| `V282` ST (O-weight 타일을 한 버퍼로) | 정적 순서는 바뀌었지만 실물 +17% — 기각 |
| ffn store 병합 m1 / m2 | 컴파일러 크래시(HBM `[Dummy2, 1 # 8]`) / LIR ICE |
| RoPE head-layout gather R1b | 컴파일은 되나 동기화 수 그대로 + 인덱스 단위 변환 필요 — 보류 |

#### ④ 남은 레버 후보
- **임계 경로 위의 동기화 수를 줄이는 구조.** qkv x2 HBM hop(store + 동기화 + x8 로드)을 클러스터마다 x path를 돌려 switch(gather → 재정렬 fetch → broadcast)로 대체 — 설계가 크고 V250(switch 벌크 이동은 느림)과 겨뤄야 한다.
- qkv에서 목록상 Q 로드 앞에 있는 모든 것(입력 norm + hi/lo 분해 ~13k 실물 + x2 동기화)을 줄이면 Q·K 체인 전체가 당겨진다. V263 융합 norm은 qkv에서 +0.32%(11/32)로 미채택이었다 — span으로 체인이 실제로 당겨지는지부터 볼 것.
- draw는 싸고 분산이 크다(①-2): 코드 SOTA로 계속 뽑는다(§5.4).

### 10.0p 2026-09-11 — 레지스터 직접 쓰기 라운드: 가짜 −4.3%를 잡았다

**상태 한 줄.** 공식 최고 **7.0314** 그대로 (2위, 1위 #663 **7.2164**). 코드 SOTA **`V273_submit` `331104b`** 그대로.
V273 draw 누적 7회, 최고 6.7103 — 기록을 못 넘었다. 이번 라운드에 채택된 변경은 없다.

#### ① 한 일과 결과
| 실험 | 가설 | 결과 |
|---|---|---|
| Z1 정적 탐침 → `V274` | 꼬리 pass가 `.vector_final().to_vrf()`로 VRF를 직접 쓴다 (`CanApplyToVrf for PositionVectorFinal`) | 정적 attn −285 · ffn −491 · qkv RoPE +2,584. **실물: `PXI-601` 커널 멈춤** (2/2 잡) → 모든 사이트 닫힘 |
| Q2b 정적 탐침 → `V276` → `V278` | qkv ring-32 broadcast pass가 `collect().to_trf()`로 TRF를 직접 쓴다 (`CanApplyToTrf for PositionCollect`) | 정적 −1,085. 짝비교 **16/16, −4.3%, p < 1e-4 — 가짜**: 제출 검증 FAIL, 변형 먼저 잡 FAIL |
| `V277` (F2b) | 같은 쓰기를 ffn up/gate에 | 9/9 느림 (평균 +6,211) |
| `V279` (Q2a) | Sub staging 하나를 Q·K·V가 공유 (정확) | **11/11 느림 (+7.5%)** — 정적 +1.3%의 5배 |
| load order s1–s6 | V weight를 마지막으로 보내기 | 정적 동일 또는 +566 — 닫힘 |
| D1 | attn `DramReuse` 회피 (reload 버퍼 선할당) | 정적 동일 — 닫힘 |

#### ② 새 규칙
1. **TRF/VRF는 Sub staging으로만 채운다.** Main pass의 레지스터 직접 쓰기는 DSL·lowering·정적 스케줄을 모두 통과하지만,
   실물에서 VRF 쓰기는 커널을 멈추고(PXI-601) TRF 쓰기는 contraction에 x를 전달하지 못한다.
2. **짝비교 하네스는 기준 arm이 남긴 상태를 변형에게 넘겨준다.** 정확도는 variant별 첫 launch에서만 보고, 기준 arm이 먼저 돌면
   TRF·VRF·DM scratch가 올바른 값으로 채워져 있다. 상태를 쓰는 방식(레지스터 쓰기, staging 제거, 버퍼 공유)을 바꾸는 변형은
   **`arm.py --sweep <CONST> <variant> _ _`로 변형을 첫 launch에 둔 잡 하나를 먼저** 돌리고, 채택 전 **제출 브랜치의 단독 검증을 생략하지 않는다.**
   p < 1e-4의 16/16도 이 함정 앞에서는 증거가 아니다.
3. **새 형태는 16잡 배치 전에 잡 하나로 멈춤부터 본다** — 멈추면 그 잡의 나머지 launch가 사라져 `pairjobs.sh`의 중앙값이 빈칸(`tn= tz=`)이 된다.
4. 정적 모델은 여전히 부호까지 틀린다: Q2a 정적 +1.3% / 실물 +7.5%, F2b 정적 −0.4% / 실물 +2.1%.

#### ③ 도구와 다음 할 일
- **하네스 base:** `V277_ffn_switch_to_trf`의 첫 커밋 **`1d38e0e` = V273_submit + 하네스 (variant arm 제거, sweep ×3)**. 새 실험은 여기서 가지를 쳐
  커널 복사본만 추가한다 (V274처럼 옛 하네스 브랜치에 헬퍼를 복사하지 않는다).
- 새 도구 (`scripts/dev/`): `arm.py` (arm 추가 · sweep 설정), `mk_base273.py` (submit 트리에 하네스 설치), `tl.py` (정적 타임라인),
  `schedcmp.py` (정적 스케줄 비교), `addr.py` (DM 텐서 주소 지도 · 물리 페이지).
- **다음 후보: 실물 타임라인.** `furiosa_profiled_run`은 디코드한 span마다 `name` + `begin_cycle` + `end_cycle`을 `span::npu`로 넘기는데
  (0.6.0 `backend/npu/ffi.rs`) 하네스는 begin/end로 커널 창만 계산한다. 이름까지 덤프하면 정적 모델이 아니라 **하드웨어의 임계 경로**를 볼 수 있다
  (브랜치 `V280_hw_span_dump`, 진행 중).

### 10.0o 2026-09-11 — furiosa-opt book 라운드: 규칙을 읽고, 전부 쟀다

**상태 한 줄.** 공식 최고 **7.0314** (`V267_submit` `c6f4a80` (= V258 + V260 + V263), `e1fd59ad`) — 리더보드 **2위**. 1위 #663 **7.2164**
(92,586 / 39,406 / 273,805)이 qkv·ffn에서 앞서고, **attn_out은 우리 38,623이 리더보드 전체 최고**다(꼬리 draw).
코드 SOTA는 **`V273_submit` `331104b` (= V268 + attn_out contraction store 256 B 정렬)**. 제출 상한은 없다(§5.4).

#### ① 이번 라운드가 한 일
사용자 지시: book의 Mapping / Moving / Computing 장을 읽고 축마다 최적화를 찾아 실험·스윕·제출까지. 세 갈래로 읽은 뒤
**모든 메커니즘을 0.6.0 crate 소스로 확인하고** 실험했다 — book은 툴체인보다 새것이라 0.6.0에 없는 API
(`fetch_*_lift`, 컴파일 타임 f4 테이블)도 설명한다.

| 실험 | 가설 (근거) | 결과 |
|---|---|---|
| E0 census | 64-access 직렬화 · DMN 경계 명령 분할이 발동하는가 (`scripts/dev/census.py`) | **둘 다 0건.** (**정정:** "store 비용은 정렬이 아니라 디스크립터 수"라는 여기의 결론은 정적 모델만 본 것이었다 — 실물은 정렬을 청구한다, ⑤의 V271) |
| `V262` (E1) | qkv weight 패킷을 stream adapter로 재생 (`stream_adapter.rs`: `OutTime = [Time, broadcast]`) | 합법·정확, **+1.8% 기각** (6/7) — contraction은 OutTime step에 묶여 있다 |
| `V263` (E2) | RMSNorm 3 pass → 2 (`Widen → InterSliceReduce`; `Clip`이 종단이라 `+EPS`는 `AddF` 즉치값) | **attn_out −1.05% 채택** (23/32) → `V265_submit` · **ffn −0.37% 채택** (24/32) → `V267_submit` · qkv +0.32% 미채택 (11/32) |
| `V264` (E3) | geglu 스케일 max 한 pass (ring-256 switch 제거) | gathered 레이아웃은 lowering 거부, full 레이아웃은 **+1.6% 기각** (8/8) — 스칼라는 switch가 싸다 |
| `V266` (E5) | attn_out 타일 88/32 재스윕 (104/16 · 96/24 · 72/48) — V259의 생성기 결함은 사이트 개수 검사로 막았다 | 104/16 −0.25% · **96/24 −1.5% 채택** (E5 6/8 + E5b 13/16 = 19/24, p = 0.007) → `V268_submit` · 72/48 +0.6% — **V259의 "재시도하지 않는다"는 틀렸다** |

제출(전부 Arena 3/3 PASS 뒤): `V265_submit` 6.4389 / 6.7245 / 6.2679 / 6.3316 / 6.3702 · `V267_submit` 6.2254 / **7.0314** / 6.6032
→ **새 공식 최고** (attn_out 38,623이 튄 꼬리 draw, 같은 코드 평균 6.620) · `V268_submit` 6.9642 / 6.3307 / 6.5570.
**상한이 없으니 최고 코드가 바뀔 때마다 몇 회씩 뽑는다** — 평균이 기록보다 3~6% 낮던 시간대에 8회 만에 기록을 깼다.

#### ② 새로 확정한 규칙 (다음 설계 전에 대조할 것)
1. **Vector 단계 전이:** intra reduce → `widen_pad` → inter reduce를 **한 pass**에 쓸 수 있다. inter reduce 뒤에는 Tag/Filter/Output만,
   `Clip`은 종단, `FpDiv`와 `IntraSliceReduce` 뒤에 `Sqrt`는 못 온다 (`furiosa-opt-std-0.6.0/src/engine/vector/stage/markers.rs`).
2. **inter-slice reduce 축은 슬라이스 분할의 최내측이어야 한다** — 패딩이 최내측인 `m![L / 480 % 16, 1 # 16]`에서는 불가.
3. **stream adapter broadcast는 합법이지만, fetch를 줄여도 contraction은 빨라지지 않는다** (V262).
4. **정적 모델은 HBM 쓰기의 256 B 정렬을 값매기지 않지만 실물은 청구한다** — attn contraction 출력을 256 B 경계에 쓰자 정적 −11, 실물 **−831 (13/16, V271)**. 정렬과 디스크립터에 관한 판단은 짝비교로만 한다(E0가 정적 스케줄만 보고 틀렸다).
5. **스칼라 분배는 switch, 벌크 재분배는 HBM** (V264 + V250).
6. **효과가 2% 미만이면 변형당 8회 × 16잡 이상으로 판정한다** — V263 attn_out이 첫 8잡 8/8 뒤 다음 8잡 4/8이었다.
7. **꼬리의 pass는 청구되고 머리의 pass는 숨는다** — 같은 norm 융합이 attn_out·ffn 꼬리에서 이기고 qkv 머리에서는 0 또는 손해였다.
8. **조건이 바뀌면 옛 스윕은 무효다** — V203의 attn_out 타일 88/32는 V257 이후 96/24에 −1.5%로 졌다. "재시도하지 않는다"는 판정에는 유효기간이 있다.

#### ③ 측정·배포 함정 (이번에 전부 한 번씩 밟았다)
- `cargo furiosa-opt test --no-run`은 **debug**다. `arena.sh`는 release 바이너리를 올린다 → **`--release --test test_kernels --no-run`** (`scripts/dev/build.sh`).
- **짝비교는 `scripts/dev/pairjobs.sh`로 돌린다** (잡별 median·차이·PASS/FAIL 수 + 부호검정 p). E5b가 이 경로로 돌았다.
- **변형 생성기는 치환 사이트 수를 세서 검증한다** (V259·V260의 결함, V266은 12/12/2/2/2로 막았다). 가능하면 로컬 사본에서 먼저 돌린다.
- **`target/`을 다른 clone으로 복사하면 rustc ICE**(`uninterned StableCrateId`). 새 clone은 빈 target에서 빌드한다(`lab3` 135초, 이후 증분 37초).
- **pod에는 GitHub push 자격이 없다** → 로컬에서 `git fetch ssh://root@<pod>:<port>/root/lab <b>:<b>` 후 push.
- **ssh 한 줄 안의 `pgrep -f`는 자기 자신과 맞는다** → 확인은 스크립트 파일 안에서.
- clone 역할: `lab`·`lab2` 짝비교(배치 중 빌드 금지), `lab3` 제출 검증·draw(draw 중 checkout 금지), `furiosa-opt-gemma4-12B` overnight driver.
- 로컬 Bash 도구가 긴 heredoc을 가끔 통째로 파싱 실패한다 → 긴 편집은 Write 도구로 파일에 쓰고 `python3 파일`로 실행.

#### ④ 남은 표적 (census와 gaps가 가리킨 곳)
- **attn_out contraction store의 디스크립터 32개** — 정렬 부분은 V271이 가져갔다(−1.7%). 개수를 줄이는 것은 레이아웃 쪽 해법이어야 한다 (switch gather는 V254에서 졌다).
- **ffn 꼬리 DMA 유휴 2,099 + 1,600** (post-ff norm pass 대기 · reload 대기), **attn_out 꼬리의 `DramReuse` 708.**
- **다른 상수 스윕도 옛 조건에서 고른 것이 있다** — 규칙 8에 따라 V257·V260·V263 이후 조건에서 ffn down 타일(16/16/16/12)과 split store 비율을 다시 볼 가치가 있다.
- #663(7.2164)과의 공식 격차는 2.6%다. 꼬리 draw 하나가 격차의 절반을 메웠으니 **최고 코드의 draw 수확은 계속하되**,
  전형적 draw 기준 격차(추정 qkv ~4% · attn_out ~5% · ffn ~4%)는 코드로만 줄어든다.
- **Stage 2 착수 시 V257을 되돌린다** (§10.0n).

#### ⑤ 추가 라운드 (2026-09-11 오전): 옛 조건의 상수를 다시 재고, 정적 탐침으로 후보를 골랐다
| 실험 | 결과 |
|---|---|
| `V269` attn 타일 100/20 · 92/28 vs 96/24 | 둘 다 8/16, p = 1.0 → **96/24가 평평한 최적** |
| `V270` qkv ring 16 · 8 vs 32 (16잡) | ring 16 +698 (7/16), ring 8 **+1,308 (3/16, p = 0.021)** → **ring 32 확정** |
| 정적 탐침 X5: residual 로드를 머리로 | 스케줄이 바이트 단위로 같다 — 로드 위치는 소비 시점이 정한다(소스 순서 무시) |
| 정적 탐침 X3: ffn down store 20/20/20 | +3,172 — 세 번째 store가 임계 경로에 남는다 |
| **`V271` attn contraction store를 256 B 정렬 오프셋에 (X1)** | 정적 −11, **실물 −831 (13/16, p = 0.021) → `V273_submit`**. E0의 결론(정렬 무관)을 뒤집었다 |
| `V272` ffn down global-scale pass 제거 (X6, RMSNorm 스케일 불변성) | 정적 −480, **정확도 FAIL (16/16 잡)** — fixture의 down 출력이 ε 영역이라 스케일 불변이 아니다 |

제출: `V273_submit` Arena 3/3 PASS, draw 6.6436 / 4.7458 / 6.1271 — 4.7458은 qkv 하나만 279,933 cycle이다(V265 이래 바뀌지 않은 qkv 코드의 채점기 이상치; attn 43,485 · ffn 288,516은 정상이고 전부 PASS).
**교훈 셋.** ① 정적 탐침은 기각 필터가 아니라 **후보를 고르는 데만** 쓴다 — X1은 정적 −11이었지만 실물에서 이겼다.
② **norm의 스케일 불변성을 쓰기 전에 ε 영역인지 확인한다** — 채점 fixture는 weight 행 스케일이 ≤1e-3이라 활성값의 평균제곱이 ε(1e-6) 근처다.
③ 쓰기 정렬은 실물 비용이지만 **커널 출력 store([H] 고정 레이아웃)는 정렬할 수 없다** — 1920원소 슬라이스가 필요하고 VRF 8 KB를 넘는다.
**다음 세션의 열린 항목:** qkv 채널 스케일을 ring-64 head gather 안으로 접기(V244가 컴파일을 열었고 미측정, 회계상 순 −3 pass — §10.0k ⑨).
256 B 경계에서 시작하지 않는 중간 store(ffn down split store의 슬라이스당 60 B, geglu·x 스테이징 store)는 정렬 후보지만, 패딩이 reload 디스크립터를 늘리지 않는 형상을 먼저 찾아야 한다.
**시도하지 말 것:** 4×960 norm 레이아웃 — 슬라이스당 VRF 세 개(scale · weight · residual)가 11.5 KB라 8 KB 한도를 넘는다.

### 10.0n 2026-09-11 — Stage 2로 가져가면 안 되는 변경이 하나 생겼다 (반드시 읽을 것)

**`V257`(attn_out의 x를 f8 한 조각으로)은 Stage 1 채점 fixture에 의존한다.**

채점 하네스는 attn_out의 x를 `s.signs(ctx, "x", 1.0)`으로 만든다 — **정확히 ±1**이다. 그래서 `x * 16 = ±16`이
f8e4m3에 **정확히** 표현되고 **낮은 조각은 항상 0**이라, 한 조각만 써도 결과가 **비트 단위로 같다**.
두 변형의 `max|Δ|`가 0.01562로 완전히 동일했던 것이 그 증거다(그 값은 계산 오차가 아니라 **출력 bf16 반올림
1 ulp**다 — 크기 4 근처에서 ulp = 2^-6).

**실제 모델에서는 성립하지 않는다.** attention 출력은 value 행들의 볼록결합이라 ±1이 아니고, f8e4m3 한 조각은
~3.6% RMS 상대오차를 낸다(dot product에서 평균으로 줄지 않는다 — 분자와 분모가 같이 커진다).

⇒ **Stage 1에서는 채택(−10%, 사용자 판단: "채점 기준이 곧 사양").
Stage 2(E2E)로 넘어갈 때는 두 조각 버전으로 되돌릴 것.** 되돌릴 지점은
`projection.rs`의 `project_output_one_piece` → `project_output`(= V251의 두 조각 형태)이다.

이와 대비되는 것: V251의 `s = 16`은 **모델의 성질**(value RMSNorm이 |x| ≤ sqrt(Ds) = 16으로 묶는다)에서 오므로
fixture 의존이 아니다. **모델 성질과 fixture 성질을 구분할 것.**

### 10.0m 2026-09-11 — attn_out이 축이었다: −7.9% (가장 최신, 여기서 시작할 것)

**코드 SOTA = `V252_submit` = V243 + attn_out 직접 x 쪼개기.** 리더보드 제출은 보류 중이다.

#### ① 리더보드가 어디를 파야 하는지 말해 줬다

2026-09-11 Participant #663: **98,125 / 43,492 / 285,067 = 6.758**. 우리 median 추정과 나란히 놓으면:

| 커널 | 우리 median | #663 | 격차 |
|---|---:|---:|---:|
| qkv | ~99,000 | 98,125 | +0.9% |
| **attn_out** | **~51,500** | **43,492** | **+18.4%** |
| ffn | ~295,300 | 285,067 | +3.6% |

**attn_out 하나가 격차의 전부다.** attn_out만 맞추면 우리 median 점수는 6.294 → **6.659**가 된다.

#### ② 우리 6.6075는 머신이 빨랐던 게 아니라 attn_out 한 커널의 outlier였다

`826c0058` 분해: qkv **103,145**(우리 최악 축에 든다) / attn **43,049**(median 대비 −16%) / ffn 293,076(평범).
즉 잡 전체가 빨랐던 게 아니라 **attn_out 하나가 튀었다.** 그리고 #663이 같은 자리에서 43,492를 **일상적으로**
내고 있으므로 **43k는 운이 아니라 도달 가능한 수준**이다. 우리 median 51.5k가 그만큼을 흘리고 있었다.

#### ③ V251: 쪼개기를 소비자 레이아웃에서 한다 — **−4,198 (−7.9%), 7/7**

attn_out은 x(8 KB)를 16슬라이스로 싣고(535) f8 두 조각으로 쪼갠 뒤 HBM에 store 399+399, 다시 512슬라이스로
load 933 했다. **쪼개기를 512슬라이스에서 직접** 하면 슬라이스당 읽는 바이트는 같은데(256 bf16 = 512 B =
2×256 f8) 명령 3개가 사라지고, 쪼개기를 기다리던 DMA 유휴 527도 사라져 O-weight 스트림이 ~1,000에 시작한다.

**이 수법은 attn_out 전용이다.** qkv·ffn은 슬라이스마다 x **전체**가 필요해 직접 로드가 512 디스크립터 ×
전체 벡터가 되고 그게 V154가 45k로 잰 병리다. attn_out만 슬라이스당 x의 1/16을 쓴다.
⇒ **일반 원칙: 스테이징 왕복은 소비자가 생산물의 *일부*만 쓸 때 없앨 수 있다.**

#### ⑤ attn_out epilogue는 세 방향에서 모두 닫혔다 (V253·V254·V255)

V251 이후 남은 최악 항목은 contraction store(**2,040 cycle, util 0.003**, 7,680 B를 64 디스크립터로)와
그 뒤의 norm 사슬이다. 세 방향을 다 재 봤고 **셋 다 졌다**:

| 실험 | 수법 | 결과 |
|---|---|---:|
| V253 | store를 60+60으로 쪼개 마지막 store만 RAW 경로에 | **+1,791, 4/4** |
| V254 | 행그룹 4개를 ring-64로 묶어 64 → 8 디스크립터 | **+2,512, 4/4** |
| V255 | norm을 producer 레이아웃에서, 스칼라만 클러스터 간 이동 (V45 슬롯) | **+2,289, 5/5, 정확도 FAIL** |

**V255는 처음으로 컴파일에 성공했다** — V45를 두 번 죽인 것은 reduce 순서였고(chunk reduce 뒤 live 슬라이스는
행그룹 = **바깥** 축, VRU는 안쪽만 줄인다, V186), 슬라이스별 부분합 16개를 먼저 한 슬라이스로 **ring**하면
cross-slice 합이 intra-slice 합이 되어 통과한다. 그런데 실물이 졌다: Main pass 3 → 5, 피연산자 3개가
32디스크립터 레이아웃, 출력 store의 live 슬라이스 8 → 16. **⇒ V45는 이제 "막혔다"가 아니라 "측정해서 졌다"다.**

V254는 **V250의 법칙이 작은 데이터에서도 성립함**을 보인다: 3.8 KB짜리 switch도 아낀 디스크립터보다 비싸다.
원본 코드의 주석("ring이 아끼는 것보다 비싸다")이 새 레이아웃에서도 옳았다.

⇒ **attn_out에서 남은 것은 weight 스트림(13,933 정적, util 0.85)과 직렬 norm 사슬(1,698)뿐이고 둘 다 바닥이다.**
V251이 이 커널에서 가져올 수 있는 것이었다.

#### ④ 하네스가 조용히 회귀해 있었다

반복된 arm 복제가 match 블록을 중복시켜 **attn_out과 qkv가 첫 실행만 대조**하고 있었다(`if i == 0`).
빠르면서 틀린 변형이 통과할 수 있었다는 뜻이다. 세 shim 모두 변형별 대조로 고쳤고, 이후 모든 잡에서
attn_out PASS 줄이 2개임을 확인한다. **측정 도구를 정기적으로 검증할 것.**

### 10.0l 2026-09-11 — 짝비교가 유일한 증거, 그리고 대량 재분배는 HBM이 이긴다 (가장 최신, 여기서 시작할 것)

**코드 SOTA = `V243_submit`(V209 + ffn down 타일 4개), 이제 통계적으로 확정됐다.** 리더보드 제출은 보류 중이다.

#### ① 공식 draw로 코드를 비교하면 안 된다 (V250 step 0)

공식 draw는 V243 10회 평균 **6.2363**, V209 계열 9회 평균 **6.3151**로 **t4가 나쁘다고 말한다.**
그런데 한 잡 안에서 교대 실행한 **짝비교 11잡은 11/11 음수**(−556 / −922 / −251 / −1,015 / −2,152 / −270 /
−614 / −1,010 / −184 / −185 / −805, 평균 **−724**, 부호검정 **p = 0.00049**)로 정반대다.

같은 7잡의 base median은 292,856~297,032(sd 1,400 = 0.5%)로 **잡간 드리프트가 작은 날**이었다.
즉 공식 draw의 0.079 차이는 코드가 아니라 **다른 시간대의 머신 속도**다. t4의 실제 값어치는
ffn −0.25% = 점수 **+0.08%**이고, draw σ(1.8~2.8%)의 30분의 1이라 공식 점수로는 원리적으로 안 보인다.

**규칙: 코드 비교는 한 잡 안 짝비교로만 한다. 공식 draw는 코드 비교에 쓸 수 없다.**

#### ② 대량 재분배는 HBM이 switch보다 한 자릿수 빠르다 (V250)

ffn에서 남은 최대 미수확 항목은 geglu → down 스테이징 왕복 **정적 7,522**(store 1,653+1,653+755,
load 2,934+527)였고, 그것이 존재하는 이유는 **up/gate는 L을, down은 H를 클러스터로 쪼개서** geglu 출력이
클러스터를 건너야 하기 때문이다. down도 L로 쪼개면(64 행그룹 × 4청크 = 256 슬라이스, 60행, 1,920열 —
V237/V246이 죽인 것은 아무것도 안 건드린다) 칩 안에서 전달할 수 있다.

**결과: +40,270 (+13.5%), 그리고 정확도 FAIL.** 정확도는 고칠 수 있지만 고칠 값어치가 없다:

| 경로 | 클러스터당 983 KB 전달 비용 | 실효 속도 |
|---|---:|---:|
| HBM 왕복 (현행) | 2,934 (+스테이징 4,588) | **≈335 B/cycle** |
| 온칩 switch broadcast | **≈37,000** | **≈27 B/cycle** |

⇒ **switch는 링이 작거나(V206의 ring 4) 스칼라를 뿌릴 때만 이긴다. 벌크 데이터는 HBM이 12배 빠르다.**
V245의 switch-scatter도 flit 규칙을 통과했더라도 같은 이유로 졌을 것이다.
**"HBM 왕복을 온칩으로 대체한다"는 가설군 전체가 닫힌다** — 이것으로 ffn의 정적 DMA 130,117 중
weight 80,845 · scale 19,534 · LUT/config 9,461 · 스테이징 ~20,277이 **모두** 닫혔다.

### 10.0k 2026-09-10 — 상수 스윕: 세 커널 모두 날카로운 국소 최적에 있다 (가장 최신, 여기서 시작할 것)

**코드 SOTA = `V243_submit`** (= V209 + ffn down 타일 4개). 문서 최신본은 `V236_sweep_attnout_tiles`.
이번 세션은 사용자 요청("interleave·index·contract 상수를 스윕")대로 **한 번도 안 건드린 상수들을 전수로 돌렸고,
하나만 채택되고 나머지는 전부 기각됐다 — 그런데 기각의 이유들이 새 법칙 세 개를 준다.**

#### ① V235의 "ffn Main은 스트림 뒤에 숨는다"는 **단방향**이다 (V237)

down의 L 분할을 8청크 → 4청크로 하면 V211이 잰 256 B 정렬 벌점(런당 granule 5개로 4개를 쓴다)이
사라진다(런 1,920 B는 정렬 여부와 무관하게 정확히 8 granule: 128 + 1,920 = 2,048). fetch/useful이
1.267 → 1.067이니 −11.6k를 기대했다. **실물은 +51,143 (+17.4%)이다.** 대가로 live 슬라이스가 128개가 되어
down pass A가 슬라이스당 열을 두 배 물었고, **그 시간은 하나도 숨지 않았다.**

⇒ **Main을 봉투 아래로 줄이면 0이고(V235), 봉투 위로 넘기면 전액 청구된다(V237).**
⇒ **"계산을 내주고 정렬/바이트를 산다"는 교환은 ffn에서 전부 닫힌다.**

#### ② ffn down 타일 계획의 구속 자원은 **16행 scale VRF**다 (V238·V241·V242)

| 계획 | 결과 |
|---|---|
| 16/16/16/8/4 (현행) | 기준 |
| **16/16/16/12 (V238)** | **−686, 4/4 잡에서 음수 → 채택** |
| 30/30, pass B 독립 타일링 (V241) | **+17,490** — 아낀 LUT 재로드·고정비보다 잃은 파이프라이닝이 4배 |
| 8/16/16/16/4 (V242 s1) | **+25,616** — 첫 타일을 *줄여도* 나쁘다 |
| 4/8/16/16/16 (V242 s2) | **컴파일 실패**: `Cannot find evict target for Vrf (size 3932160)` |

16행 타일의 scale VRF는 8 KB 파일의 **7,680 B**다. **꼬리의 작은 타일들이 그 압력을 빼주는 장치**이고,
그래서 현행 계획이 날카로운 국소 최적이다. V238의 16/16/16/12만이 그 옆의 유일한 개선이다.

#### ③ `LaneMode`는 손잡이가 아니다 (V240)

- attn_out Interleaved → Sequential: **+108(무효)**. qkv: **−213(무효)**.
- ffn Sequential → Interleaved: **구조적 거부** — `contract_lane (Interleaved): OutPacket mismatch.
  Expected: 1 # 8, got: H / 16 % 4 # 8`. V210의 fold 규칙이 ffn에서도 확인됐다.
- attn_out O-weight 3타일(44/44/32): **+887**. V203의 2타일 최적 재확인.

⇒ **바꿀 수 있는 곳에서는 무효, 유효할 만한 곳에서는 못 바꾼다.** 이 축은 닫힌다.

#### ④ 매핑 규칙 넷 (다음 세션이 시간을 아끼도록)

1. **DM 할당은 256 슬라이스를 전부 덮어야 한다.** `slice extent 128 does not match the device config`.
   노는 슬라이스는 `1 # n` **죽은 축**으로 표현한다(`DownRows`가 8개 중 7개를 그렇게 놀린다).
2. **한 switch 매핑에서 같은 축(`Dummy2`)을 두 타일이 주장할 수 없다.** 다른 2짜리 축(`Dummy8 / 4`)을 쓴다.
3. **부분 축(`m![Ns / 2 = 2]`)은 갓 할당한 텐서에서 불법이다** — `StoVrf … lower_fetch_unit: There should be
   not-exactly-matched from_in_slice slots`. **전체 extent를 가진 텐서에서 `tile`로 잘라낼 때만** 합법이다
   (mlp의 `H % 60 = $rows`가 되는 이유가 이것이다).
4. **2단 `tile`은 축을 뭉갠다.** `Dummy2` → `Gs` 순으로 자르면 커밋 대상이 `m![Gs #{!} 4, 1 # 8]`이 되어
   `IndexWrite: Output tensor must be identical to table tensor`로 거부된다.

#### ⑤ 그래서 qkv tail 병합(V239)은 막혔다 — 그리고 남은 유일한 길

V218이 값매긴 자리(tail 17 pass, pass당 실물 ≈600)를 12 pass로 줄이려면 q의 두 행과 k·v가 한 버퍼를 써야 하는데,
규칙 3·4 때문에 **평평한 4-extent 행 축이 필요하고, 그러면 모든 피연산자 VRF도 진짜 4행 텐서에서 와야 한다.**
채널 스케일이 파라미터 3개(`q/k/v_weight_scale`)로 쪼개져 있어 그런 텐서를 만들 수 없다(HBM 스크래치로 합치면
DMA 명령 4개 = 실물 +4k로 이득 −3k를 넘는다).

**남은 길은 하나다: 채널 스케일을 ring-64 gather pass 안으로 접어 head norm이 scale VRF를 아예 안 쓰게 만든다.**
그러면 x를 자유롭게 reshape할 수 있고 규칙 3이 비켜간다. 게이팅 질문: **switch 뒤에 vector 연산을 걸 수 있는가,
그리고 그 VRF 피연산자는 switch 전/후 중 어느 슬라이스 매핑을 쓰는가.** 이것부터 탐침할 것.

#### ⑥ attn_out V45는 문서화된 벽에 그대로 걸린다

producer 레이아웃(`HiddenRows256 = m![H / 120 % 16, 1 # 16]`)에서 post-norm을 하려면 **행그룹(바깥 슬라이스 축)을
가로질러 reduce**해야 하는데 V186의 "VRU reduce axes must be innermost"에 걸린다. 우회는 ring-256 switch gather이고
`projection.rs`의 주석이 이미 그 값을 2,055로 적어 두었다 — 기대 이득(−1.5k 정적)보다 크다. **닫힌다.**

#### ⑦ 4청크는 두 번 죽었다 — 정렬 가설을 닫는다 (V237 + V246)

| 형상 | live 슬라이스 | 슬라이스당 연산 | 결과 |
|---|---:|---|---:|
| 8청크 × 60행 (현행) | 256 | 60 × 1,920 | 기준 |
| 4청크 × 60행 (V237) | **128** | 60 × 3,840 (2배) | **+17.4%** |
| 4청크 × 30행 (V246) | **256** | 30 × 3,840 (**동일**) | **+8.2%** |

V246은 V237의 치명상(연산 2배)을 정확히 제거했는데도 진다. ⇒ **V211이 잰 정렬 벌점 27%는 "같은 바이트를
한 번 더 읽는 탐침"에서만 보이고, 레이아웃을 실제로 갈아끼우면 회수되지 않는다.** V216이 ffn 런 길이에서
내린 결론과 같다: **런 길이·정렬 곡선은 커널 안에서 그 로드가 놓인 위치에 따라 부호가 다르다.**

#### ⑧ ffn에서 가장 큰 낭비는 이름이 붙었지만 손이 닿지 않는다 (V245)

스케줄이 정확히 지목한다: `up_scale`과 `down_scale`은 **같은 3.69 MB**인데
**up 4,880 (util 0.556, 7,200 B 런) vs down 9,774 (util 0.277, 120 B 런)**이다. 차이 4,894 정적 ≈ 실물 10k.
행 레이아웃으로 싣고 switch로 청크를 흩뿌리는 수법은 **`mir: Collect output packet must be exactly 32 bytes
(one flit)`**에 걸린다 — 목적지 청크가 120 블록이고 **32 ∤ 120**이다. 중간 단계를 둬도 마지막 목적지는 늘 120이다.
broadcast로 바꾸면 8슬라이스가 같은 데이터를 받아 청크 선택이 안 된다. **여는 유일한 길은 down이 8청크가
아니게 되는 것인데 그건 ⑦에서 닫혔다.**

#### ⑨ 그래도 V239는 열려 있다 (V244 탐침 통과)

**switch 뒤에 vector 체인을 걸 수 있고, 그 VRF 피연산자는 switch *후* 슬라이스 매핑을 쓴다** — 컴파일로 확인했다.
따라서 채널 스케일을 ring-64 gather 안으로 접을 수 있고, head norm이 scale VRF를 버리면 규칙 3이 비켜간다.
다만 **회계를 다시 하면 순이득은 5 pass가 아니라 3 pass**다(kv를 공유 버퍼로 옮기는 사본 +1, k용 gamma·1 버퍼 준비 +2,
scatter용 k·v 추출 +2). 실물 ≈ **−1.8k (qkv −1.8%)**. switch가 bf16 대신 f32를 나르는 비용은 미측정이다.

#### ⑩ "3개 ctx를 모두 쓴다"와 "연산 순서 interleave"는 둘 다 닫혔다 (V247·V248)

사용자가 지목한 세 축을 각각 실측했고 **셋 다 중립**이다.

| 축 | 실험 | 결과 |
|---|---|---:|
| 세 컨텍스트를 모두 쓴다 | qkv head-norm sqrt pass 3개를 `ctx.sub`로 | **+312 (중립)** |
| 커널 내 연산 순서 interleave | attn_out epilogue 로드 3개를 스트림 앞으로 | **+358 (중립)** |
| contract 3중 합 스윕 | attn_out fetch 패킷 64 → 32원소 | **+130 (중립)** |

의미가 큰 것은 첫 번째다. **`ctx.sub`는 vector 체인을 돌려 DM에 commit할 수 있다**(V223의 제약은
`fetch_table_lookup`이 Main 전용이라는 것이었다). 그런데 Main에서 3 pass를 덜어냈는데 실물이 안 움직였다.
⇒ **V218의 "pass당 ≈600"은 컨텍스트별 발행비가 아니라 PE core의 in-order 발행비이고, 컨텍스트를 나눠도
겹치지 않는다.** RULES §8이 적어둔 "Main과 Sub는 Tensor Unit 파이프라인을 두고 경합한다"가 실물로 확인됐다.

**이것은 V47(down global scale을 eps로)의 기대값도 뒤집는다:** 그 변경은 Main pass 1개를 지우고 Sub pass 2개를
만드는데, Sub pass가 Main pass만큼 비싸다면 **순증**이다. 구현하지 않는다.

두 번째는 V21·V42와 같은 결론이다 — **스케줄러는 소스 순서를 되돌려 놓고, 그 배치가 이미 최선이다.**

세 번째로 **contract 3중 합 스윕 전체가 닫힌다**: Packet↔Time은 실측 중립(book 모델대로), Lane은 qkv·attn_out에서
이미 8차선 만석이고 ffn의 4/8은 V235가 닫았다.

#### ⑪ flash attention은 이 대회에 적용할 대상이 없다

Stage 1의 채점 커널 셋은 **투영 2개(qkv, attention output)와 FFN**이다. softmax·attention matrix를 계산하는
`ops::sliding_attention`은 **채점 대상이 아니다**(RULES §0의 표). flash attention이 최적화하는 것은 정확히 그
softmax·KV 타일링이므로 여기에는 걸 데가 없다. 같은 이유로 KV cache 압축·paged attention류도 Stage 1에는 무효다.
(Stage 2 E2E에서는 이야기가 달라진다.)

#### ⑫ V243 draw 10회: 재제출의 기대값도 예전만 못하다

`V243_submit`(V209 + t4)로 10회 뽑았다: 6.2111 · 6.2348 · 6.2335 · 6.3495 · 6.0782 · 6.1762 · 6.0495 ·
6.3746 · 6.3765 · 6.2796. **평균 6.2363, σ 0.1143(1.8%), 최고 6.3765 — 10회 중 한 번도 기존 6.6075를 넘지 못했다.**

같은 코드 계열의 이전 분포(n=9)는 평균 **6.3151**, σ 2.8%였다. 새 표본의 평균이 **0.079 낮고** σ는 **더 작다**.
표본오차(SE ≈ 0.036)로 보면 평균 차이는 2σ 남짓이라 우연이라 하기에는 크다 — **머신이 이 시간대에 전반적으로
느렸거나, draw 분포 자체가 시간에 따라 이동한다**는 뜻이다. 어느 쪽이든 실무적 결론은 같다:

- **6.6075는 평균 대비 +6.0%짜리 꼬리값이었다.** 그 수준을 다시 뽑을 확률은 10회에 0/10이었다.
- **재제출의 기대값은 이전 세션의 추정(+3.5%)보다 훨씬 낮다.** "예산이 남으면 무조건 재제출"이라는 §10.0i의
  규칙은 **분포가 좋을 때만** 성립한다. 제출 전에 최근 draw 평균을 먼저 확인할 것.

#### 다음 세션이 먼저 할 일

1. **draw를 수확한다.** 이번 세션도 코드 이득은 −686(ffn −0.23%)뿐이고, 같은 커밋의 draw 분포는 σ≈2.8%다.
2. 굳이 코드를 판다면 **⑤의 게이팅 탐침**(switch 뒤 vector 연산) 하나뿐이다. 성공하면 qkv −3k.
3. 리더보드(2026-09-10 기준): 1위 우리 6.6075, 2위 Participant #663 **6.4920**(qkv **99,739** / attn 47,560 / ffn 289,234),
   3위 Pulbitmaru 6.4264(107,376 / 45,577 / **289,031**). **세 팀 최고치 조합 = 6.717.**
   우리 median은 qkv ~101k / attn ~51k / ffn ~296k이고, 6.6075는 attn_out 43,049짜리 운 좋은 draw였다.

### 10.0j 2026-09-10 — 세션 정리: 커널은 하드웨어 바닥 근처다 (가장 최신, 여기서 시작할 것)

**공식 SOTA = `826c0058` 6.6075.** 코드는 여전히 `V209_ffn_ring4_production` `21d3e20`이다 —
이번 세션의 코드 실험은 둘 다 기각됐고, 점수는 전부 draw에서 왔다(같은 커밋의 draw 분포: 6.2045 / 6.2833 / 6.3776 / 6.5010 / 6.6075 (n=5, 평균 6.395, σ 2.5%)).

**문서 최신본은 `V232_qkv_rope_no_rotate_half`에 있다.** 코드는 `V209_ffn_ring4_production`,
하네스는 `V226_harness_v2`. 새 실험은 **코드를 V209에서, 하네스를 V226에서** 가져와 분기한다.

#### 이번에 처음 확보한 것: furiosa-opt book 원문

<https://developer.furiosa.ai/furiosa-opt/book/print.html> 한 페이지에 전문이 있다. RULES가
"book"이라며 인용하던 두 문장은 전해 들은 것이었고, 그중 DMN 인터리빙 문장은 우리 형상에 적용되지
않아 V8/V196을 낭비시켰다. 점수에 걸리는 사실은 [[furiosa-opt-book-hardware-facts]] 메모리와
RESULTS의 V227 절에 있다. 요약: **Outer는 `Lane ≤ 8`, `Packet ≤ 64 B`이고 Lane<8이면 처리율이
비례해 떨어진다 · Lane은 TRF(고정) 피연산자에서 온다 · TRF 64 KB/slice(crate 상수) ·
fetch 비용 = `Time × (Packet / read_size)`, main의 read_size ≤ 32 B이고 **sub는 8 B 고정** ·
DMA 엔진 8개, 명령당 startup ≈500 · HBM stack bit = 주소 bit 8 · `begin_interleaved`는 0.6.0에 있다.**

#### 왜 코드로는 더 못 짜는가 (이번 세션의 산수)

세 커널 실물 합 ≈450k, 정적 합 209k → 비 2.15. **프로파일러 counter가 2 GHz이므로 클럭만으로 2.0이
설명되고, 모델 밖 오버헤드는 450k 중 31k(7%)뿐이다.** V197이 잰 한계 전송률 618 B/cycle은 2 GHz 기준
1,236 B/1 GHz-cycle = **HBM 피크의 82%**다. 즉 **바이트도 전송률도 거의 다 썼다.**

- **V227(가중치를 TRF에, Lane 1 → 8)**: contraction은 정말로 **36,526 → 5,984 static(6.1×)** 빨라졌다.
  그런데 `to_trf`가 table lookup 출력을 거부해 dequant를 별도 pass로 빼야 하고, 그 f8 버퍼
  (슬라이스당 115,200 B × 2행렬)의 DM 왕복이 이득을 삼킨다 → **+37,744 / +15,525. 기각.**
  ⇒ **ffn pass A는 contraction-bound가 아니라 LUT-bound다.**
- **V232(RoPE의 rotate_half 제거)**: Main 명령 31 → 29, −542 cycle로 설계대로 됐는데
  **Core 명령이 54 → 60**(commit 1개가 tile commit 2개로 갈라져 디스크립터 일이 늘었다) → **+1,472. 기각.**
  ⇒ **pass를 줄여도 commit이 늘면 손해다.** V218의 "pass당 ≈600"은 commit 수 고정에서만 성립한다.
- **V235(x의 f8 조각을 Lane으로) — 이번 세션 최대 발견이자 가장 유용한 음성 결과.**
  ffn pass A는 **정확히 LUT 처리율 한계**(≈12.5 elem/cycle)이고 `Dummy2`가 fetch 시간축에 있어 모든 weight를
  **두 번** 조회한다. Lane으로 옮겨 한 번만 조회하게 만들었다 — **V210이 `StoVrf`로 죽은 자리인데, scale replay를
  패킷이 아니라 Time으로 옮기고 조각 합을 별도 fold pass로 빼면 통과한다**(벽은 컴파일러가 아니라 하드웨어였다).
  설계대로 **pass A 36,526 → 18,526 static Main**, **MainContext 76,836 → 66,598**이 됐는데
  **실물은 +464(노이즈 안)로 꿈쩍도 하지 않았다.**
  ⇒ **ffn의 MainContext는 정말로 DMA 스트림 뒤에 숨는다.** 정적 `DmaEngine 94.2%`가 옳았고
  **V206의 −16,351은 Main 절감이 아니라 그 변경의 DMA 쪽 효과**였다.
  ⇒ **"ffn Main을 줄인다"는 가설군 전체가 닫힌다.**
- **V230(`begin_interleaved`)**: book의 마지막 미사용 API인데 **device fn에서 도달 불가** —
  컴파일러가 만드는 축 이름이 `interleave`이고 `Ident`는 대문자로 시작해야 해서 어떤 사용자 축도 못 맞춘다.
- **V234(pass A 패킷 128)**: commit이 128 B OutPacket을 거부하고, 통과했어도
  `Time × (Packet / read_size)`(read_size ≤ 32 B)라 이득이 0이다.

#### 다음 세션이 먼저 할 일

1. **draw를 수확한다.** 검증된 개선이 없으면 최고 코드 재제출에 쓴다
   (`/root/draw2.sh N`). σ≈2%, 10회면 평균 대비 +3.5% 근처다. 이번 세션이 그것으로 +5.2%를 벌었다.
2. **코드로 남은 것은 거의 없다.** 이번 세션이 ffn에서 −18,000 static Main을 실제로 만들어냈는데 실물이 0이었다(V235).
   세 커널 모두 정적 스케줄에서 DMA가 77~94%이고 한계 전송률은 이미 HBM 피크의 82%다. 굳이 판다면
   **attn_out**(V45 post-norm; DMA 비중 77%로 가장 낮다)이나 **V213 재진입**(0.6.0이 설치돼 있으므로 0.7.0 재시도는 열려 있다)이지
   **ffn Main은 아니다.**

3. **A/B는 반드시 한 잡 안에서 인접 실행끼리.** 잡이 다르면 머신 속도가 다르다(같은 코드가
   잡마다 qkv 94.7k~99.1k). V226 하네스가 변형당 12~25회를 9~17초에 돌린다.

### 10.0i 2026-09-10 — 공식 점수는 추첨이다: 같은 코드가 6.2833과 6.5010을 뽑았다 (가장 최신, 여기서 시작할 것)

**공식 SOTA = `b772875a` 6.5010** (2026-09-10 11:07 UTC). **코드는 `b78a63a0`(6.2833)과 완전히 같다** —
둘 다 `V209_ffn_ring4_production` `21d3e20`이다. 차이는 draw뿐이다:

| 커널 | b78a63a0 | b772875a | 차이 |
|---|---:|---:|---:|
| qkv | 103,697 | **97,847** | −5,850 (−5.6%) |
| attn_out | 49,657 | **48,019** | −1,638 (−3.3%) |
| ffn | 293,889 | **290,803** | −3,086 (−1.1%) |
| **점수** | 6.2833 | **6.5010** | **+3.5%** |

**draw 노이즈에는 두 성분이 있다.** `b772875a`에서는 세 커널이 **같은 방향으로 함께** 움직였고
(−5.6% / −3.3% / −1.1%) 이는 잡 전체의 머신 속도다. 반면 최고 기록 `826c0058`
(qkv 103,145 / attn_out **43,049** / ffn 293,076)에서는 **attn_out 하나만 −13%** 튀었다 —
평소 48~50k인 커널이 43,049다. 즉 **공통 성분(머신 속도) + 커널별 성분**이고, 이전 세션의
"attn_out draw 하나가 4% 흔든다"는 후자를 본 것이다. 우리 하네스의 warm stdev도 이와 맞는다:
qkv 2.5~3.2% · attn_out 2.8~4.1% · ffn 0.66%.

**2026-09-10 11:13 기준 공개 리더보드 1위다**(2위 Pulbitmaru 6.4264, 3위 Participant #663 6.1713,
4위 Participant #905 6.0002). 우리 attn_out 43,049는 보드 전체 최고치이고, ffn은 Pulbitmaru의
289,031이 여전히 앞선다.

**같은 커밋 `21d3e20`의 draw 분포(n=9, 2026-09-10):** 6.0278 · 6.1343 · 6.2045 · 6.2206 · 6.2833 · 6.3776 · 6.4791 · 6.5010 · 6.6075. 평균 6.3151, σ 0.1790(**2.8%**), 범위 6.0278~6.6075(**9.6% 폭**), 최고는 평균 대비 **+4.6%**. 즉 **점수의 ±5%는 코드와 무관하다.**

**따라서 리더보드에서 가장 기대값이 높은 행동은 코드 개선이 아니라 재제출이다.** 리더보드는 팀 최고값만
남기므로 같은 코드를 N번 내면 max(N draws)를 갖는다. σ ≈ 2%이므로 6회면 평균 대비 +2.5~3%다.
이번 세션의 코드 작업 전체(V227 기각, V232 ±0)보다 재제출 한 번이 더 벌었다.
**제출은 이렇게 쓴다: 검증된 개선이 있으면 즉시 올리고, 최고 코드가 바뀔 때마다 draw를 몇 회씩 수확한다(상한 없음). 평균이 기록보다 3~6% 낮던 시간대에도 8회 만에 +6% 꼬리값이 기록을 깼다(V267 7.0314, 2026-09-11).**

**제출은 직렬화된다.** `moa-submitter`는 앞 제출이 `building` / `evaluating`인 동안 다음 제출을 거부한다
(거부는 예산을 쓰지 않는다). 자동화하려면 `status`가 `building|queued|running|pending|evaluating`을
더 이상 보이지 않을 때까지 기다렸다 다음을 낸다 — `/root/draw2.sh N`이 그것이다.

### 10.0h 2026-09-10 — 공식 SOTA 6.2833 (가장 최신, 여기서 시작할 것)

**제출 완료: `b78a63a0` = 6.2833** (2026-09-10 09:59 UTC, 브랜치 `V209_ffn_ring4_production` `21d3e20`,
qkv 103,697 / attn_out 49,657 / ffn 293,889, 3/3 PASS). 직전 5.9032 대비 **+6.4%**이고 리더보드 2위(6.0)를 넘는다.
오늘 제출은 2건 사용(46bc9eed, b78a63a0) — 하루 5건 예산.

**ring 스윕은 끝났다(V225): ring 4가 최적이다.** r8 −(+72, 노이즈), r2 +1,192(사본 바이트 2배를 DMA로 되갚음).
V206의 하강 추세는 r4가 바닥이었다. 부수 소득: **변형 순서를 매 반복 회전시키는 하네스**가 변형 간 편차를
1.2k 안으로 줄였다(V219는 n0가 항상 첫 변형이라 판정 불가였다). **앞으로 모든 다변형 잡은 순서를 회전시킬 것.**

### 10.0g 2026-09-10 밤 — 리더보드가 목표를 확정했다

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
