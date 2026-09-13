"""MICRO 2026 MOA — Kernel Optimization Round 공식 리더보드 실시간 감시기.

https://micro2026-moa.github.io/leaderboard.html 이 부르는 API 를 주기적으로 폴링해
  - 팀별 3커널 cycle(nnn.nk)과 그 옆 괄호에 speedup(baseline cycle / 팀 cycle, (n.nnnx)),
    Total(기하평균)은 n.nnnx 로
  - F1 타이밍 화면 색(섹터 = 커널 칸에만): 전체 최고 보라, 이전 제출보다 좋아진 칸 초록, 나빠진 칸 노랑,
    이전 제출이 없는 신규 팀은 세 칸 모두 초록. Total 은 칠하지 않는다. 우리 팀 줄은 연파랑 글자
  - 표 위에 커널별 이론 하한(0912_v360.md §2.6), 표 바로 위에 Theoretical best (커널별 최고 speedup 만 모은 기하평균)
를 셸에 그리고, 변동(신규 팀 / 새 제출 / 순위 이동)이 생기면 로그를 남긴다.

  python leaderboard/leaderboard_watch.py              # 기본 60초 간격
  python leaderboard/leaderboard_watch.py -i 30 --table
  python leaderboard/leaderboard_watch.py --once       # 표 한 번만

API 는 팀마다 '최고 유효 결과' 한 줄만 주고 제출 이력은 주지 않는다. 그래서 '이전 제출'은
이 감시기가 직전에 본 그 팀의 항목이다: submissionId 가 바뀌면 옛 항목을 previous 로 밀어
leaderboard_state.json 에 남긴다(재시작해도 비교가 이어진다). 인증은 필요 없다.
"""

import argparse
import json
import math
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://micro2026-api.duckdns.org:7777/api/leaderboard"
OUR_TEAM = "Goat Chovy #1557"

# 참가팀이 아닌 행. 주최 측 기준선(speedup 1.000x)은 순위·최고값 계산에서 뺀다.
EXCLUDE = {"Baseline"}

KST_OFFSET = timedelta(hours=9)     # submissionId 의 시각은 UTC 다 (V204 46bc9eed = 05:12 UTC).

QKV = "ops::sliding_project_qkv"
ATTN = "ops::sliding_attention_output"
FFN = "ops::decoder_feedforward"

# (cycles 키, 옛 평면 필드명, 표 머리) — 채점 순서 그대로.
KERNELS = [
    (QKV, "qkvCycles", "QKV"),
    (ATTN, "attentionCycles", "AttnOut"),
    (FFN, "ffnCycles", "FFN"),
]

# 응답에 baselineCycles 가 빠졌을 때만 쓰는 공식 baseline cycle (2026-09-09 확인).
BASELINE = {QKV: 250_514, ATTN: 404_633, FFN: 3_703_473}

# 커널별 이론 하한 cycle (0912_v360.md §2.6 표): 가중치를 1.5 TB/s = 1,500 B/cycle 로 다 읽는 시간.
# 계산 시간(125~1,400 cycle)은 이보다 훨씬 작아 뺐다. §2.7 대로 실물 비용은 바이트에 비례하지 않으므로
# 도달 목표가 아니라 기준선이다.
FLOOR = {QKV: 21_000, ATTN: 10_500, FFN: 66_000}

# 감시기 이전에 본 항목. 처음 보는 팀의 previous 를 한 번만 채운다. cycles 가 없으면 Total 만 비교된다.
#   · Goat Chovy: V258_submit 첫 draw 6.8026 (RESULTS.md) — c5dc8d05 6.8279 직전의 리더보드 항목.
#   · Pulbitmaru / #905: 2026-09-10 ~10:00 UTC 리더보드 스냅샷 (메모).
SEED_PREVIOUS = {
    "Goat Chovy #1557": {"cycles": None, "score": 6.8026},
    "Pulbitmaru": {"cycles": {QKV: 131_489, ATTN: 50_617, FFN: 289_085}},
    "Participant #905": {"cycles": {QKV: 119_177, ATTN: 46_318, FFN: 314_820}},
}

# Kernel Optimization Round (COMPETITION_INFO.md 'Sep 1 – Sep 25, 2026'). KST 기준.
# 마감 시각·시간대는 공지에 없어 23:59 로 잡았다. 다르면 --deadline 으로 조정.
START = datetime(2026, 9, 1, 0, 0)
DEADLINE = datetime(2026, 9, 25, 23, 59)

SHOW_N = 20          # 표에 그릴 상위 팀 수. 우리 팀이 밖이면 따로 덧붙인다.
TOP_N = 3            # 상위 3팀은 MOA Workshop 발표 기회 (COMPETITION_INFO.md). 이 등수 아래에 컷 선.
STATE_PATH = Path(__file__).with_name("leaderboard_state.json")

# 열 폭(표시 칸 수). 모든 행/구분선이 이 값들로부터 위치를 계산한다.
RANK_W, NAME_W, VAL_W, TAIL_GAP = 4, 18, 9, 3
KVAL_W = 2 + len("288.2k (13.526x)")                # 커널 칸: cycle + 괄호 speedup, 앞 2칸은 칸 사이
TAIL_W = len("#c0ffdd40  09-11 06:05")
LINE_W = 2 + RANK_W + 1 + NAME_W + len(KERNELS) * KVAL_W + VAL_W + TAIL_GAP + TAIL_W
RULE = "-" * (LINE_W - 2)

ANSI = re.compile(r"\x1b\[[0-9;]*m")

C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "blue": "\033[94m", "cyan": "\033[96m",
    # F1 섹터 색. 칸의 색이 곧 의미라 줄 전체를 칠하지 않는다.
    "purple": "\033[38;5;135m", "green": "\033[92m", "yellow": "\033[93m",
    "onus": "\033[38;5;117m\033[1m",             # 우리 팀 줄 — 연파랑 글자. 섹터 칸 색이 이 색을 덮어쓴다.
}


def paint(s, *styles):
    styles = [k for k in styles if k]
    if not styles:
        return s
    return "".join(C[k] for k in styles) + s + C["reset"]


def paint_line(s, *styles):
    """줄 전체에 스타일 적용. 안쪽에 이미 색이 든 조각(예: 보라 칸)이 있으면
    그 reset 이 바깥 스타일까지 꺼버리므로, reset 마다 바깥 스타일을 다시 건다."""
    base = "".join(C[k] for k in styles if k)
    if not base:
        return s
    return base + s.replace(C["reset"], C["reset"] + base) + C["reset"]


def now():
    return datetime.now().strftime("%H:%M:%S")


def now_kst():
    return datetime.now(timezone.utc).replace(tzinfo=None) + KST_OFFSET


def log(msg):
    print(f"{paint('[' + now() + ']', 'dim')} {msg}", flush=True)


def width(s):
    """한글 등 wide 문자를 2칸으로 세는 표시 폭."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def vwidth(s):
    """색 코드를 뺀 실제 표시 폭."""
    return width(ANSI.sub("", s))


def fit(s, n):
    """표시 폭 n 칸에 맞춘다. 길면 …로 자르고, 짧으면 공백으로 채운다."""
    if width(s) <= n:
        return s + " " * (n - width(s))
    out, w = "", 0
    for ch in s:
        cw = width(ch)
        if w + cw > n - 1:      # 마지막 1칸은 … 자리
            break
        out += ch
        w += cw
    return out + "…" + " " * (n - w - 1)


def lpad(s, n):
    """표시 폭 n 칸에 우측 정렬. 색 코드가 섞여 있어도 된다."""
    return " " * max(0, n - vwidth(s)) + s


def xs(v):
    return f"{v:.3f}x"


def ks(c):
    return f"{c / 1000:.1f}k"


def speedup(base, cycles, k):
    return base[k] / cycles[k]


def geomean(base, cycles):
    return math.exp(sum(math.log(speedup(base, cycles, k)) for k, _, _ in KERNELS) / len(KERNELS))


def to_kst(submission_id):
    """'2026-09-10-21-05-54-460-c0ffdd40'(UTC) → '09-11 06:05'(KST). 형식이 다르면 '—'."""
    try:
        utc = datetime.strptime(submission_id[:19], "%Y-%m-%d-%H-%M-%S")
    except (TypeError, ValueError):
        return "—"
    return (utc + KST_OFFSET).strftime("%m-%d %H:%M")


def fetch():
    """(baseline cycles, {팀명: row}). 순위는 API 순위에서 EXCLUDE 행을 빼고 다시 매긴다."""
    req = urllib.request.Request(API, headers={"Cache-Control": "no-store",
                                               "User-Agent": "leaderboard_watch"})
    with urllib.request.urlopen(req, timeout=15) as res:
        payload = json.loads(res.read().decode("utf-8"))
    if not payload.get("success", True):
        raise ValueError(f"API success=false: {str(payload)[:200]}")

    base = {**BASELINE, **(payload.get("baselineCycles") or {})}
    rows, rank = {}, 0
    for d in sorted(payload.get("data") or [], key=lambda d: d.get("rank", 10**9)):
        if d.get("teamName") in EXCLUDE:
            continue
        cyc = d.get("cycles") or {}
        cycles = {k: int(cyc.get(k, d.get(flat))) for k, flat, _ in KERNELS}
        rank += 1
        sid = d.get("submissionId") or ""
        rows[d["teamName"]] = {
            "team": d["teamName"],
            "rank": rank,
            "submission_id": sid,
            "code": d.get("submissionCode") or sid[-8:] or "—",
            "cycles": cycles,
            "score": float(d["score"]) if d.get("score") is not None else geomean(base, cycles),
            "submitted": to_kst(sid),
        }
    return base, rows


# ── 이전 제출 추적 ─────────────────────────────────────────────────────────────

def load_state(path):
    try:
        state = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        state = {}
    state.setdefault("teams", {})
    return state


def save_state(path, state):
    try:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        log(paint(f"상태 저장 실패: {e}", "red"))


def entry_of(r):
    return {k: r[k] for k in ("submission_id", "code", "cycles", "score", "submitted")}


def seed_entry(seed, base):
    cycles = seed.get("cycles")
    score = seed.get("score")
    if score is None and cycles:
        score = geomean(base, cycles)
    return {"submission_id": None, "code": None, "cycles": cycles, "score": score,
            "submitted": "seed"}


def track(rows, state, base):
    """팀별 current/previous 를 갱신하고 각 row 에 prev(직전 제출 항목 또는 None)를 붙인다."""
    teams = state["teams"]
    for team, r in rows.items():
        st = teams.get(team)
        if st is None:
            seed = SEED_PREVIOUS.get(team)
            st = teams[team] = {"current": entry_of(r),
                                "previous": seed_entry(seed, base) if seed else None}
        elif st["current"]["submission_id"] != r["submission_id"]:
            st["previous"], st["current"] = st["current"], entry_of(r)
        r["prev"] = st["previous"]


# ── 색 판정 ────────────────────────────────────────────────────────────────────

def bests(base, rows):
    """커널별 최소 cycle, 최고 Total, Theoretical best(커널별 최고만 모은 기하평균)."""
    kbest = {k: min(r["cycles"][k] for r in rows) for k, _, _ in KERNELS}
    return kbest, max(r["score"] for r in rows), geomean(base, kbest)


def kernel_style(r, k, kbest):
    c = r["cycles"][k]
    if c == kbest[k]:
        return ("purple", "bold")
    prev = r.get("prev")
    if prev is None:        # 신규 팀: 비교할 이전 제출이 없는 첫 기록 → 세 칸 모두 초록 (보라가 우선)
        return ("green",)
    p = prev.get("cycles")
    if not p:               # seed 처럼 Total 만 알려진 이전 항목 — 칸별로는 비교할 수 없다
        return ()
    return ("green",) if c < p[k] else ("yellow",) if c > p[k] else ()


def delta(new, old, unit="x"):
    """speedup 변화 표기. 좋아지면 초록 ▲, 나빠지면 노랑 ▼."""
    d = new - old
    if abs(d) < 5e-4:
        return paint(f"±0.000{unit}", "dim")
    return paint(f"▲{d:+.3f}{unit}", "green") if d > 0 else paint(f"▼{d:+.3f}{unit}", "yellow")


# ── 출력 ───────────────────────────────────────────────────────────────────────

def progress_lines():
    """라운드 진행 막대 (막대줄, 기간줄). 남은 시간이 짧아질수록 색이 붉어진다."""
    t = now_kst()
    total = (DEADLINE - START).total_seconds()
    frac = min(1.0, max(0.0, (t - START).total_seconds() / total))

    left = DEADLINE - t
    days = max(0, left.days)
    hours = max(0, int(left.total_seconds() // 3600) % 24)

    if left.total_seconds() <= 0:
        color, tail = "dim", "라운드 종료"
    else:
        color = "green" if days >= 10 else "yellow" if days >= 3 else "red"
        tail = f"D-{days}"

    head, pct = "  Kernel Round [", f"] {frac * 100:.1f}%  {tail}"
    bar_w = LINE_W - width(head) - width(pct)
    fill = round(bar_w * frac)
    bar = "█" * fill + "░" * (bar_w - fill)

    span = (f"  {START:%Y-%m-%d %H:%M} 시작 · {DEADLINE:%Y-%m-%d %H:%M} 마감 (KST) · "
            + ("종료됨" if left.total_seconds() <= 0 else f"{days}일 {hours}시간 남음"))
    return (f"{head}{paint(bar, color)}] {frac * 100:.1f}%  {paint(tail, color, 'bold')}",
            paint(span, "dim"))


def kcell(base, cycles, k, style):
    """커널 칸 'nnn.nk (n.nnnx)'. 둘 다 섹터 색을 쓴다. cycle·speedup 을 각각 고정 폭으로 맞춰
    k 와 소수점이 세로로 맞는다. 공백은 칠하지 않는다."""
    cyc = paint(f"{ks(cycles[k]):>6}", *style)
    sp = lpad(paint(f"({xs(speedup(base, cycles, k))})", *style), 9)
    return lpad(f"{cyc} {sp}", KVAL_W)


def rank_cut(rows, n):
    """N위 팀의 row (팀 수가 부족하면 None)."""
    return next((r for r in rows.values() if r["rank"] == n), None)


def cut_rule(score):
    """TOP_N 경계선. 마지막 통과 등수 바로 아래에 깔아 통과선과 컷 점수를 보인다."""
    head = f"  └─ TOP{TOP_N} 컷 {xs(score)} "
    return paint(head + "─" * max(0, LINE_W - width(head)), "cyan")


def render_table(base, rows):
    ordered = sorted(rows.values(), key=lambda r: r["rank"])
    print()
    for ln in progress_lines():
        print(ln)
    if not ordered:
        print(paint("\n  아직 유효 제출이 없다.\n", "dim"), flush=True)
        return

    kbest, _, theo = bests(base, ordered)
    us = rows.get(OUR_TEAM)
    shown = ordered[:SHOW_N]
    hidden = ordered[SHOW_N:]
    if hidden:
        # 생략 구간은 팀 수를 적어 표시(int 마커). 우리 팀이 그 안에 있으면 따로 끌어올린다.
        shown = shown + ([len(hidden) - 1, us] if us and us["rank"] > SHOW_N else [len(hidden)])

    legend = "  ".join([paint("■ 전체 최고", "purple", "bold"),
                        paint("■ 이전 제출보다 개선 · 신규 팀", "green", "bold"),
                        paint("■ 이전 제출보다 악화", "yellow", "bold"),
                        paint("■ 우리 팀", "onus")])
    print(f"\n  {legend}\n")

    # 이론 하한: 참고 기준선이라 섹터 색을 쓰지 않는다. Total 칸은 세 하한의 speedup 기하평균.
    floor_cells = "".join(kcell(base, FLOOR, k, ()) for k, _, _ in KERNELS)
    floor_total = lpad(xs(geomean(base, FLOOR)), VAL_W)
    print(f"  {' ' * RANK_W} {paint(fit('이론 하한', NAME_W), 'bold')}{floor_cells}{floor_total}")

    # Theoretical best: 커널마다 가장 빠른 팀의 speedup 을 모았을 때의 Total (F1 의 이론 최고 랩).
    theo_cells = "".join(kcell(base, kbest, k, ("purple", "bold")) for k, _, _ in KERNELS)
    theo_total = lpad(paint(xs(theo), "bold"), VAL_W)     # Total 은 칠하지 않는다
    print(f"  {' ' * RANK_W} {paint(fit('Theoretical best', NAME_W), 'bold')}{theo_cells}{theo_total}")

    head_vals = "".join(lpad(label, KVAL_W) for _, _, label in KERNELS) + lpad("Total", VAL_W)
    print(paint(f"  {fit('순위', RANK_W)} {fit('팀', NAME_W)}{head_vals}{' ' * TAIL_GAP}제출 (KST)",
                "bold"))
    print(paint("  " + RULE, "dim"))

    for r in shown:
        if isinstance(r, int):
            note = f"이하 {r}팀 생략" if r else "…"
            print(paint(f"  {fit('⋮', RANK_W)} {note}", "dim"))
            continue
        vals = "".join(kcell(base, r["cycles"], k, kernel_style(r, k, kbest)) for k, _, _ in KERNELS)
        total = lpad(xs(r["score"]), VAL_W)          # Total 은 칠하지 않는다 — 색은 섹터 칸에만
        line = (f"  {str(r['rank']):<{RANK_W}} {fit(r['team'], NAME_W)}{vals}{total}"
                f"{' ' * TAIL_GAP}{paint('#' + r['code'], 'dim')}  {r['submitted']}")
        if r["team"] == OUR_TEAM:
            # 줄 전체를 연파랑 글자로 강조한다. 섹터 칸은 자기 색 코드가 줄 색 뒤에 오므로 보라/초록/노랑이
            # 연파랑을 덮어쓰고, 칠하지 않은 칸(순위·팀명·Total·제출)만 연파랑으로 남는다.
            line = "▶" + line[1:]
            print(paint_line(line, "onus"))
        else:
            print(line)

        if r["rank"] == TOP_N:                  # TOP_N 마지막 등수 → 컷 선
            print(cut_rule(r["score"]))

    print(paint("  " + RULE, "dim"))

    if us:
        print(f"  {paint(OUR_TEAM, 'onus')} : {paint(str(us['rank']) + '위', 'bold')}"
              f" / {len(rows)}팀")
    print(flush=True)


def kernel_changes(base, old, new):
    """커널별 'QKV 2.560x → 2.605x ▲+0.045x' 목록."""
    return " · ".join(
        f"{label} {xs(speedup(base, old['cycles'], k))} → {xs(speedup(base, new['cycles'], k))} "
        f"{delta(speedup(base, new['cycles'], k), speedup(base, old['cycles'], k))}"
        for k, _, label in KERNELS)


def diff(base, prev, curr):
    """이전/현재 스냅샷을 비교해 변경 로그 라인 리스트를 만든다."""
    lines = []

    for team, r in curr.items():
        old = prev.get(team)
        if old is None:
            kern = " · ".join(f"{label} {xs(speedup(base, r['cycles'], k))}" for k, _, label in KERNELS)
            lines.append(f"{paint('＋ 신규', 'cyan', 'bold')} {paint(team, 'bold')} {r['rank']}위 진입 · "
                         f"Total {xs(r['score'])} ({kern})")
            continue
        if old["submission_id"] != r["submission_id"]:
            lines.append(f"{paint('제출', 'cyan')} {paint(team, 'bold')} #{r['code']} "
                         f"{paint('(' + r['submitted'] + ')', 'dim')}  Total {xs(old['score'])} → "
                         f"{xs(r['score'])} {delta(r['score'], old['score'])}")
            lines.append(f"      {kernel_changes(base, old, r)}")
        if old["rank"] != r["rank"]:
            up = r["rank"] < old["rank"]
            arrow = paint(f"↑{old['rank']}→{r['rank']}위", "green") if up \
                else paint(f"↓{old['rank']}→{r['rank']}위", "yellow")
            tag = paint("★ 우리팀", "blue", "bold") + " " if team == OUR_TEAM else ""
            lines.append(f"{paint('순위', 'blue')} {tag}{paint(team, 'bold')} {arrow}")

    for team in prev.keys() - curr.keys():
        lines.append(f"{paint('－ 이탈', 'dim')} {team}")

    old_cut, new_cut = rank_cut(prev, TOP_N), rank_cut(curr, TOP_N)
    if old_cut and new_cut and abs(old_cut["score"] - new_cut["score"]) >= 5e-4:
        lines.append(f"{paint(f'TOP{TOP_N} 컷', 'cyan', 'bold')} {xs(old_cut['score'])} → "
                     f"{xs(new_cut['score'])} {delta(new_cut['score'], old_cut['score'])}"
                     f"  (현재 {TOP_N}위: {new_cut['team']})")

    if prev and curr:
        _, _, old_theo = bests(base, list(prev.values()))
        _, _, new_theo = bests(base, list(curr.values()))
        if abs(old_theo - new_theo) >= 5e-4:
            lines.append(f"{paint('Theoretical best', 'purple', 'bold')} "
                         f"{xs(old_theo)} → {xs(new_theo)} {delta(new_theo, old_theo)}")

    return lines


def main():
    global OUR_TEAM, SHOW_N, DEADLINE

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", type=float, default=60, help="폴링 간격(초), 기본 60")
    ap.add_argument("--table", action="store_true", help="변동이 없어도 매번 전체 표 출력")
    ap.add_argument("--once", action="store_true", help="한 번만 표를 찍고 종료")
    ap.add_argument("--team", default=OUR_TEAM, help=f"우리 팀명, 기본 {OUR_TEAM!r}")
    ap.add_argument("-n", "--show", type=int, default=SHOW_N,
                    help=f"표에 그릴 상위 팀 수, 기본 {SHOW_N} (0이면 전체)")
    ap.add_argument("--deadline", default=None, metavar="'Y-M-D H:M'",
                    help=f"라운드 마감(KST), 기본 {DEADLINE:%Y-%m-%d %H:%M}")
    ap.add_argument("--state", type=Path, default=STATE_PATH,
                    help=f"이전 제출 기록 파일, 기본 {STATE_PATH.name}")
    args = ap.parse_args()
    OUR_TEAM = args.team
    SHOW_N = args.show or 10**9
    if args.deadline:
        DEADLINE = datetime.strptime(args.deadline, "%Y-%m-%d %H:%M")

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    state = load_state(args.state)

    def poll():
        """리더보드 취득 + 이전 제출 추적 + 상태 저장. (base, rows) 반환."""
        base, rows = fetch()
        track(rows, state, base)
        save_state(args.state, state)
        return base, rows

    if args.once:
        render_table(*poll())
        return

    print(paint(f"\n  MICRO 2026 MOA Leaderboard · 감시 시작 · {args.interval:g}초 간격 · "
                f"우리 팀 = {OUR_TEAM}  (Ctrl+C 종료)", "bold", "cyan"))

    prev = None
    fails = 0
    live = False        # 하트비트가 줄 끝에 커서를 두고 있는 상태

    def endline():
        """하트비트 줄 위에 다른 출력이 겹쳐 찍히지 않게 줄을 닫는다."""
        nonlocal live
        if live:
            print()
            live = False

    while True:
        try:
            base, curr = poll()
            fails = 0
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError,
                ValueError, KeyError, TypeError) as e:
            fails += 1
            endline()
            log(paint(f"조회 실패 ({fails}회 연속): {e}", "red"))
            time.sleep(min(args.interval, 15))
            continue

        if prev is None:
            log("초기 스냅샷 취득")
            render_table(base, curr)
        else:
            changes = diff(base, prev, curr)
            if changes:
                endline()
                print()
                log(paint(f"◆ 갱신 감지 — {len(changes)}건", "bold", "cyan"))
                for ln in changes:
                    print(f"    {ln}", flush=True)
                render_table(base, curr)
            elif args.table:
                endline()
                render_table(base, curr)
            else:
                # 변동이 없으면 진행 막대를 제자리 갱신한다. 스크롤을 늘리지 않으면서
                # 남은 시간이 계속 살아 있고, 폴링이 도는 중이라는 표시도 겸한다.
                bar, _ = progress_lines()
                print(f"\r{bar}  {paint('[' + now() + ']', 'dim')}", end="", flush=True)
                live = True

        prev = curr
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(paint("\n  감시 종료.\n", "dim"))
