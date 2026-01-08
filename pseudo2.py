#!/usr/bin/env python3
"""
Impressive-looking terminal output generator (Python version).

- Simulates development activity streams (analysis, metrics, monitoring, etc.)
- Output is intentionally "fake complicated code", status bars, and error scenes.
- Runs until:
    a) prints a literal line "x" (stop token), OR
    b) duration ends (--duration), OR
    c) max lines reached (--max-lines) -> will print "x" then exit
- Ctrl-C stops cleanly.

Examples:
  python stakewash.py
  python stakewash.py --dev-type backend --jargon high --complexity extreme --alerts --team
  python stakewash.py --duration 10 --max-lines 400
  python stakewash.py --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


# ---------------------------
# Enums / Types
# ---------------------------

class DevelopmentType(str, Enum):
    backend = "backend"
    frontend = "frontend"
    devops = "devops"
    ml = "ml"
    security = "security"


class JargonLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    extreme = "extreme"


class Complexity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    extreme = "extreme"


@dataclass(frozen=True)
class SessionConfig:
    dev_type: DevelopmentType
    jargon_level: JargonLevel
    complexity: Complexity
    duration_s: int
    alerts_enabled: bool
    project_name: str
    minimal_output: bool
    team_activity: bool
    framework: str
    seed: int | None
    max_lines: int | None


# ---------------------------
# Global running flag (Ctrl-C)
# ---------------------------

RUNNING = True

def _handle_sigint(_signum, _frame) -> None:
    global RUNNING
    RUNNING = False

signal.signal(signal.SIGINT, _handle_sigint)


# ---------------------------
# Terminal helpers
# ---------------------------

def clear_screen() -> None:
    # Avoid external deps; works on Windows/macOS/Linux
    os.system("cls" if os.name == "nt" else "clear")


def supports_color() -> bool:
    return sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def colorize(s: str, code: str, enabled: bool) -> str:
    if not enabled:
        return s
    return f"\033[{code}m{s}\033[0m"


def c_red(s: str, enabled: bool) -> str:
    return colorize(s, "31;1", enabled)

def c_yellow(s: str, enabled: bool) -> str:
    return colorize(s, "33;1", enabled)

def c_green(s: str, enabled: bool) -> str:
    return colorize(s, "32;1", enabled)

def c_cyan(s: str, enabled: bool) -> str:
    return colorize(s, "36;1", enabled)

def c_magenta(s: str, enabled: bool) -> str:
    return colorize(s, "35;1", enabled)


# ---------------------------
# Fake content generators
# ---------------------------

TOKENS = ["Ω", "λ", "⟂", "∇", "≋", "⊗", "⟦", "⟧", "⋆", "⟁", "⟡", "≀", "≔", "↯", "⊘", "⋄", "⨳", "⧗"]
MODULES = ["core::flux", "std::gloom", "runtime::mirage", "net::haze", "io::crypt",
           "vm::wisp", "compiler::oracle", "scheduler::murmur", "alloc::abyss",
           "linker::dirge", "diagnostics::lament", "kernel::whisper"]

TYPES = ["u13", "i0", "ptr<nullsafe?>", "Maybe<Maybe<T>>", "Σ<T|⊥>", "Ref<'Ω, Mut<T>>",
         "Fn(…)->Never?", "Packet<Δ, 7>", "Matrix<3x∞>", "Tape[::]", "Ghost<λ, 0xDEAD>"]

KEYWORDS = ["transmute", "defer", "reinterpret", "monomorph", "inline@volatile", "unwind?",
            "sealed", "phantom", "atomic-ish", "borrow::mut", "lifetime<'🜁'>",
            "trait≋object", "entropy_hint", "vectorize:maybe", "constexpr-ish",
            "fold_abyss", "seal::void", "detonate_if", "resolve::omens"]

DIAGNOSTICS = [
    "warning: shadowed symbol 'δ' in non-euclidean scope",
    "note: eliding lifetime due to cosmic background noise",
    "hint: please re-run with --offer-soul",
    "info: speculative optimization rolled back (reason: prophecy mismatch)",
    "trace: scheduler heartbeat irregular (arrhythmia detected)",
    "note: panic boundary crossed; reality re-indexed",
]

ERRORS = [
    "E0199: cannot unify type 'Σ<T|⊥>' with 'T' under phase=preheat",
    "E0420: ambiguous overload for operator '≔' (found 7 candidates)",
    "E0991: stack frame collapsed into a Klein bottle",
    "E7777: illegal instruction: attempted to execute ASCII art",
    "E1312: borrow checker demands tribute; value moved into the void",
    "E5001: linker refused: symbol 'hope' not found",
]

STACK = [
    "at core::flux::fold_chaos(line=∞, col=13)",
    "at compiler::oracle::resolve(phase='twilight')",
    "at runtime::mirage::spawn(task='murmur#A3')",
    "at vm::wisp::dispatch(op='⊗', mode='speculative')",
    "at alloc::abyss::reserve(bytes=13_337)",
    "at linker::dirge::bind(symbol='hope', scope='none')",
    "at kernel::whisper::syscall(nr=0x3F, arg='⧗')",
]

TEAM_EVENTS = [
    "PR merged: 'refactor/entropy-router' (approvals: 2, objections: 1)",
    "code review: requested changes (nit: rename 'doom' -> 'doom2')",
    "standup: blocker declared (root cause: metaphysical deadlock)",
    "incident chat: status escalated to SEV-2 (but vibes are stable)",
    "pairing session: driver switched (reason: wrist fatigue + existential dread)",
]

ALERTS = [
    "ALERT: kernel whisper channel saturated; dropping non-critical omens",
    "ALERT: memory pressure rising; heap began to sigh audibly",
    "ALERT: checksum drift detected; reconciling parallel realities",
    "ALERT: clock skew exceeded tolerance; time may be non-linear",
]

def rand_ident(n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(alphabet) for _ in range(n))


def fake_code_line(cfg: SessionConfig) -> str:
    indent = " " * random.choice([0, 2, 4, 8, 12])
    mod = random.choice(MODULES)
    kw = random.choice(KEYWORDS)
    typ = random.choice(TYPES)
    tok = random.choice(TOKENS)
    a = rand_ident(random.randint(3, 10))
    b = rand_ident(random.randint(3, 10))

    framework_hint = f" // fw={cfg.framework}" if cfg.framework else ""
    jargon_spice = {
        JargonLevel.low: "",
        JargonLevel.medium: f" // {random.choice(['cache', 'pipeline', 'hotpath'])}",
        JargonLevel.high: f" // {random.choice(['idempotency', 'backpressure', 'hazard-pointer', 'RCU'])}",
        JargonLevel.extreme: f" // {random.choice(['hypergraph', 'bisimulation', 'monoidal-functor', 'CPS-transform'])}",
    }[cfg.jargon_level]

    patterns = [
        f"{indent}{mod}::{kw}<{typ}>({a} {tok} {b}) ::=> {tok}{typ} {{ … }}{framework_hint}{jargon_spice}",
        f"{indent}let {a}: {typ} {tok}= {kw}({mod}::{b}?) // TODO: appease entropy{framework_hint}",
        f"{indent}match {a} {{ {tok}Ok(v) => {kw}(v), {tok}Err(e) => panic!(e^2) }}{jargon_spice}",
        f"{indent}⟦{a}⟧ := {kw}⟂({b}) ⊗ normalize(Δ={random.randint(1,9999)}){jargon_spice}",
        f"{indent}impl {tok}{typ} for {a} where {b}: trait≋object {{ /* … */ }}{framework_hint}",
        f"{indent}#pragma lament({a}::{b}, dirge={random.randint(0,9)}){jargon_spice}",
    ]
    return random.choice(patterns)


def status_bar(progress: float, width: int, cfg: SessionConfig) -> str:
    progress = max(0.0, min(1.0, progress))
    filled = int(round(progress * width))
    bar = "█" * filled + "░" * (width - filled)

    phase = random.choice(["BOOT", "LINK", "JIT", "SYNC", "SCAN", "TRACE", "FLUSH", "GC", "RITUAL"])
    thr = random.randint(1, 64)
    q = random.randint(0, 9999)

    base = f"[{phase}] [{bar}] {int(progress*100):3d}% | thr={thr:02d} | q={q:04d}"
    if cfg.dev_type == DevelopmentType.devops:
        base += f" | nodes={random.randint(3,99):02d}"
    elif cfg.dev_type == DevelopmentType.ml:
        base += f" | gpu={random.randint(0,7)} | loss≈{random.random():.4f}"
    elif cfg.dev_type == DevelopmentType.security:
        base += f" | sig={random.choice(TOKENS)} | risk={random.randint(1,10)}/10"
    return base


def error_scene(cfg: SessionConfig) -> list[str]:
    err = random.choice(ERRORS)
    diag = random.choice(DIAGNOSTICS)
    mod = random.choice(MODULES)

    # More “dramatic” depending on complexity
    frames_n = {
        Complexity.low: (3, 5),
        Complexity.medium: (5, 8),
        Complexity.high: (7, 10),
        Complexity.extreme: (9, 13),
    }[cfg.complexity]

    header = [
        "",
        "══════════════════════════════════════════════════════════════",
        f"FATAL[{random.randint(100,999)}] :: {err}",
        f"  {diag}",
        f"  --> {mod}::{rand_ident(6)}:{random.randint(1,9999)}:{random.randint(1,200)}",
        "  stacktrace (most recent call last):",
    ]

    frames = ["    " + random.choice(STACK) for _ in range(random.randint(*frames_n))]

    footer = [
        f"  context: project='{cfg.project_name}' | phase={random.choice(['preheat','twilight','aftertaste','null'])} | entropy={random.randint(9000,99999)}",
        f"  evidence: checksum=0x{random.randint(0, 16**8 - 1):08X} | shard={random.randint(1,17)} | omen={random.choice(TOKENS)}",
        "  action: isolating anomaly…",
        "  action: draining queues…",
        "  action: requesting forgiveness from scheduler…",
    ]

    if random.random() < 0.55:
        footer += [
            f"  recover: applied patchset '{rand_ident(4)}-{random.randint(10,99)}' (hot) ✓",
            "  recover: subsystem restarted; continuing with scars",
        ]
    else:
        footer += [
            f"  recover: FAILED (reason: {random.choice(['prophecy unresolved', 'heap wept', 'watchdog asleep'])})",
            "  recover: switching to limp-mode; performance will be symbolic",
        ]

    footer += ["══════════════════════════════════════════════════════════════", ""]
    return header + frames + footer


# ---------------------------
# Display / Activities
# ---------------------------

def boot_sequence(cfg: SessionConfig, emit: Callable[[str], None]) -> None:
    color = supports_color() and (not cfg.minimal_output)

    emit(c_cyan(f"[BOOT] initializing project '{cfg.project_name}' …", color))
    emit(c_cyan(f"[BOOT] dev_type={cfg.dev_type.value} | jargon={cfg.jargon_level.value} | complexity={cfg.complexity.value}", color))
    if cfg.framework:
        emit(c_cyan(f"[BOOT] framework binding: {cfg.framework}", color))
    emit(c_yellow("[BOOT] loading symbols: Σ, ⊥, Ω, and the rest of the unspeakables", color))
    emit("[BOOT] warming caches: █░░░░░░░░░░░░░░░░░░░░░░░░░  3%")
    emit("[BOOT] warming caches: ███████░░░░░░░░░░░░░░░░░░ 27%")
    emit("[BOOT] warming caches: ███████████████░░░░░░░░░ 58%")
    emit("[BOOT] warming caches: ████████████████████████░ 96%")
    emit(c_green("[BOOT] ready. generating convincing output stream.", color))
    emit("")


def run_code_analysis(cfg: SessionConfig, emit: Callable[[str], None]) -> None:
    emit(f"[ANALYZE] {random.choice(MODULES)}::{rand_ident(8)} :: pass={random.randint(1,9)} :: mode='speculative'")
    for _ in range(random.randint(2, 6)):
        emit(fake_code_line(cfg))


def run_performance_metrics(cfg: SessionConfig, emit: Callable[[str], None], progress: float) -> float:
    progress += random.uniform(0.01, 0.06)
    if progress >= 1.0:
        progress = random.uniform(0.0, 0.15)
        emit("[INFO] epoch rollover -> token=" + random.choice(TOKENS) + str(random.randint(1000, 9999)))
        emit("[WARN] cache coherence ritual incomplete; proceeding anyway")
    emit(status_bar(progress, 34 if cfg.complexity != Complexity.low else 28, cfg))
    return progress


def run_system_monitoring(cfg: SessionConfig, emit: Callable[[str], None]) -> None:
    emit(f"[MON] cpu={random.randint(3,99)}% mem={random.randint(12,96)}% io={random.randint(0,999)}MB/s "
         f"fd={random.randint(50,4096)} ctx={random.randint(1000,99999)}")
    if random.random() < 0.25:
        emit(f"[TRACE] {random.choice(MODULES)}::{rand_ident(8)} :: Δt={random.randint(1,900)}ms :: jitter={random.randint(0,999)}ppm")


def run_data_processing(cfg: SessionConfig, emit: Callable[[str], None]) -> None:
    emit(f"[PIPE] ingest -> normalize -> shard -> reconcile :: batch={random.randint(1,9999)}")
    for _ in range(random.randint(1, 4)):
        emit(fake_code_line(cfg))


def run_network_activity(cfg: SessionConfig, emit: Callable[[str], None]) -> None:
    emit(f"[NET] {random.choice(['gRPC', 'QUIC', 'mTLS', 'SSE', 'WS'])} :: "
         f"lat={random.randint(2,320)}ms p95={random.randint(10,900)}ms "
         f"drops={random.randint(0,7)} retries={random.randint(0,12)}")
    if cfg.dev_type == DevelopmentType.security and random.random() < 0.35:
        emit("[SEC] anomaly signature=" + random.choice(TOKENS) + " :: policy='deny-by-vibes' :: action='quarantine'")


def display_random_alert(cfg: SessionConfig, emit: Callable[[str], None]) -> None:
    color = supports_color() and (not cfg.minimal_output)
    emit(c_red("[ALERT] " + random.choice(ALERTS), color))


def display_team_activity(cfg: SessionConfig, emit: Callable[[str], None]) -> None:
    color = supports_color() and (not cfg.minimal_output)
    emit(c_magenta("[TEAM] " + random.choice(TEAM_EVENTS), color))


# ---------------------------
# Core loop
# ---------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate impressive-looking terminal output when stakeholders walk by."
    )
    parser.add_argument("--dev-type", choices=[e.value for e in DevelopmentType], default=DevelopmentType.backend.value)
    parser.add_argument("--jargon", choices=[e.value for e in JargonLevel], default=JargonLevel.medium.value)
    parser.add_argument("--complexity", choices=[e.value for e in Complexity], default=Complexity.medium.value)
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds to run (0 = run until stop token / Ctrl-C).")
    parser.add_argument("--alerts", action="store_true", help="Show critical system alerts.")
    parser.add_argument("--project", default="distributed-cluster", help="Project name.")
    parser.add_argument("--minimal", action="store_true", help="Use less colorful output.")
    parser.add_argument("--team", action="store_true", help="Show team collaboration activity.")
    parser.add_argument("--framework", default="", help="Simulate a specific framework usage.")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic output.")
    parser.add_argument("--max-lines", type=int, default=None, help="Hard cap lines (will print 'x' then exit).")
    parser.add_argument("--stop-prob", type=float, default=0.010, help="Probability per iteration to print 'x' and stop.")

    args = parser.parse_args()

    cfg = SessionConfig(
        dev_type=DevelopmentType(args.dev_type),
        jargon_level=JargonLevel(args.jargon),
        complexity=Complexity(args.complexity),
        duration_s=args.duration,
        alerts_enabled=args.alerts,
        project_name=args.project,
        minimal_output=args.minimal,
        team_activity=args.team,
        framework=args.framework,
        seed=args.seed,
        max_lines=args.max_lines,
    )

    if cfg.seed is not None:
        random.seed(cfg.seed)
    else:
        random.seed()

    # Complexity influences pacing + amount of output
    delay_min, delay_max = {
        Complexity.low: (0.10, 0.45),
        Complexity.medium: (0.08, 0.38),
        Complexity.high: (0.06, 0.32),
        Complexity.extreme: (0.05, 0.28),
    }[cfg.complexity]

    # Increase drama by biasing toward error scenes when alerts are enabled
    error_probability = {
        Complexity.low: 0.16,
        Complexity.medium: 0.22,
        Complexity.high: 0.28,
        Complexity.extreme: 0.34,
    }[cfg.complexity]
    if cfg.alerts_enabled:
        error_probability = min(0.55, error_probability + 0.08)

    status_probability = {
        Complexity.low: 0.26,
        Complexity.medium: 0.28,
        Complexity.high: 0.30,
        Complexity.extreme: 0.32,
    }[cfg.complexity]

    stop_probability = max(0.0, min(1.0, args.stop_prob))

    clear_screen()
    lines_printed = 0
    start = time.time()

    def emit(line: str) -> None:
        nonlocal lines_printed
        print(line)
        lines_printed += 1
        time.sleep(random.uniform(delay_min, delay_max))

        if cfg.max_lines is not None and lines_printed >= cfg.max_lines:
            print("x")
            raise SystemExit(0)

    boot_sequence(cfg, emit)

    activities: list[Callable[[], None]] = []
    progress = 0.0

    def act_code() -> None:
        run_code_analysis(cfg, emit)

    def act_metrics() -> None:
        nonlocal progress
        progress = run_performance_metrics(cfg, emit, progress)

    def act_monitor() -> None:
        run_system_monitoring(cfg, emit)

    def act_pipe() -> None:
        run_data_processing(cfg, emit)

    def act_net() -> None:
        run_network_activity(cfg, emit)

    base_activities = [act_code, act_metrics, act_monitor, act_pipe, act_net]

    # Determine concurrent "activity count" like your Rust mapping
    activities_count = {
        Complexity.low: 1,
        Complexity.medium: 2,
        Complexity.high: 3,
        Complexity.extreme: 4,
    }[cfg.complexity]

    while RUNNING:
        # Optional duration cap
        if cfg.duration_s > 0 and (time.time() - start) >= cfg.duration_s:
            break

        # Stop condition requested: print x and exit
        if random.random() < stop_probability:
            emit("x")
            return 0

        # Shuffle and run N activities
        random.shuffle(base_activities)
        for fn in base_activities[:activities_count]:
            fn()

            # Between-activity pause
            time.sleep(random.uniform(0.10, 0.55))

            if not RUNNING:
                break
            if cfg.duration_s > 0 and (time.time() - start) >= cfg.duration_s:
                break

        # Interleave dramatic error scenes
        if random.random() < error_probability:
            for line in error_scene(cfg):
                emit(line)

        # Optional alert/team events
        if cfg.alerts_enabled and random.random() < 0.12:
            display_random_alert(cfg, emit)

        if cfg.team_activity and random.random() < 0.20:
            display_team_activity(cfg, emit)

    clear_screen()
    color = supports_color() and (not cfg.minimal_output)
    print(c_green("Session terminated.", color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
