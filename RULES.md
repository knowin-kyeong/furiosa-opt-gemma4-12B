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

### 10.1 2026-09-09 세션이 남긴 상태 (다음 세션이 이어받을 것)

**브랜치 계보 (모두 origin에 push됨, 각각 한 가지 변경만 담음):**

```
main ─ V0_baseline ─ V1_ffn_down_chunked_dequant ─ V2_attnout_rows_over_256_slices
                                                     └─ V7_qkv_x_replicate_via_hbm
                                                          ├─ V8_weight_rows_interleaved_dma   (기각, 실물 A/B용으로 보존)
                                                          └─ V9_ffn_x_via_hbm ─ V6_ffn_upgate_overlap ─ V10_attn_weight_tiles_fused_lut ─ V11_residual_1920_tiles
                                                                  ─ V12_ffn_rows_per_pass_12 ─ V13_ffn_dma_trims ─ V14_two_clusters  ← 선두
                                                                                                                                              (V3, V4, V5는 슬롯만; V5는 V10에 흡수)
```

**makespan (정적, cargo-furiosa-opt 0.6.0):** V0 116,583 / 194,020 / 1,693,200 → V14 73,445 / 38,240 / 191,122
(기하평균 4.147×). **실측은 하나도 없다.** 정확도도 미검증. V14는 베이스라인이 칩의 두 클러스터 중
하나만 쓰고 있었다는 발견(`Cluster = m![1 # 2]`)을 세 커널에 적용한 것이라 이득이 가장 크고, 실측에서
확인할 가치도 가장 크다.

**Arena 승인이 나면 가장 먼저 할 일 (순서 고정):**

1. pod에서 `. /root/env.sh; cd /root/furiosa-opt-gemma4-12B; git checkout V0_baseline && ./scripts/rngd_test.sh`
   → 세 커널의 **분모**(V0 실측 cycle)와 정확도 PASS 확인. 이게 없으면 아무것도 판정 못 한다.
2. `git checkout V14_two_clusters && ./scripts/rngd_test.sh` → 정확도와 실측 cycle.
3. V14가 정확도에서 깨지면 계보를 거슬러 이분 탐색: V13 → V12 → V11 → V10 → V6 → V9 → V7 → V2 → V1. 각 브랜치가
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
| qkv x 복제 18.4k: Q를 H-split(클러스터당 H 절반, partial을 HBM에서 합산)으로 | qkv −9k | x가 슬라이스당 절반이면 바이트 절반 |
| geglu·rmsnorm·residual 등 단일 클러스터 vector 단계를 두 클러스터로 | attn_out tail −5k, ffn −5k | V14 원리 |
| FFN scale 타일 36k(120~240 B 행, 낮은 DMA 효율) | ffn −10k | 24행 단위 로드 등 |
| `shared::rmsnorm` 경량화 (ReducingSlices 경로 3~4k × 4회) | 세 커널 각 −2k~−3k | 기하평균 레버리지 |
| qkv x 복제 18.4k (H-split, 정렬 리스크) | qkv −10k | §RESULTS V3 보류 사유 참조 |
| V8 실물 A/B (인터리브 레이아웃) | 실측에서만 판단 가능 | 정적 모델은 무반응 |
