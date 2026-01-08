#!/usr/bin/env python3
"""
Slow, dramatic terminal stream that prints pseudo-code, progress bars,
and frequent multi-line "error scenes" until a literal 'x' is printed.

Usage:
  python dramatic_stream.py
  python dramatic_stream.py --seed 123          # deterministic run
  python dramatic_stream.py --max-lines 600     # hard cap (still ends with 'x')
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass


# ---------------------------
# Configuration (slow + dramatic)
# ---------------------------

@dataclass
class Config:
    # Slow pacing
    min_delay_s: float = 0.08
    max_delay_s: float = 0.35

    # Ends when we print a standalone line: x
    stop_probability: float = 0.010  # per iteration

    # More dramatic: more frequent + longer error scenes
    error_probability: float = 0.22
    status_probability: float = 0.28

    # Visuals
    bar_width: int = 34

    # Optional: safety cap (still prints x before exiting)
    max_lines: int | None = None


CFG = Config()


# ---------------------------
# Fake generators
# ---------------------------

TOKENS = [
    "Ω", "λ", "⟂", "∇", "≋", "⊗", "⟦", "⟧", "⋆", "⟁", "⟡", "≀", "≔", "↯", "⊘", "⋄", "⨳", "⧗"
]

FAKE_KEYWORDS = [
    "transmute", "defer", "reinterpret", "monomorph", "inline@volatile",
    "unwind?", "sealed", "phantom", "atomic-ish", "borrow::mut", "lifetime<'🜁'>",
    "trait≋object", "entropy_hint", "vectorize:maybe", "constexpr-ish",
    "fold_abyss", "seal::void", "detonate_if", "resolve::omens",
]

FAKE_TYPES = [
    "u13", "i0", "ptr<nullsafe?>", "Maybe<Maybe<T>>", "Σ<T|⊥>", "Ref<'Ω, Mut<T>>",
    "Fn(…)->Never?", "Packet<Δ, 7>", "Matrix<3x∞>", "Tape[::]", "Ghost<λ, 0xDEAD>",
    "Union<Ok|Err|??? >", "Chrono<phase='twilight'>",
]

FAKE_MODULES = [
    "core::flux", "std::gloom", "runtime::mirage", "net::haze", "io::crypt",
    "vm::wisp", "compiler::oracle", "scheduler::murmur", "alloc::abyss",
    "linker::dirge", "diagnostics::lament", "kernel::whisper",
]

FAKE_DIAGNOSTICS = [
    "warning: shadowed symbol 'δ' in non-euclidean scope",
    "note: eliding lifetime due to cosmic background noise",
    "hint: please re-run with --offer-soul",
    "info: speculative optimization rolled back (reason: prophecy mismatch)",
    "debug: recursion depth stabilized at 1e6±3",
    "trace: scheduler heartbeat irregular (arrhythmia detected)",
    "note: panic boundary crossed; reality re-indexed",
]

FAKE_ERRORS = [
    "E0199: cannot unify type 'Σ<T|⊥>' with 'T' under phase=preheat",
    "E0420: ambiguous overload for operator '≔' (found 7 candidates)",
    "E0991: stack frame collapsed into a Klein bottle",
    "E0007: segmentation fault avoided (prediction engine intervened)",
    "E7777: illegal instruction: attempted to execute ASCII art",
    "E1312: borrow checker demands tribute; value moved into the void",
    "E5001: linker refused: symbol 'hope' not found",
]

FAKE_STACK = [
    "at core::flux::fold_chaos(line=∞, col=13)",
    "at compiler::oracle::resolve(phase='twilight')",
    "at runtime::mirage::spawn(task='murmur#A3')",
    "at vm::wisp::dispatch(op='⊗', mode='speculative')",
    "at alloc::abyss::reserve(bytes=13_337)",
    "at linker::dirge::bind(symbol='hope', scope='none')",
    "at kernel::whisper::syscall(nr=0x3F, arg='⧗')",
]


def rand_ident(n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(alphabet) for _ in range(n))


def slow_print(line: str, cfg: Config) -> None:
    print(line)
    time.sleep(random.uniform(cfg.min_delay_s, cfg.max_delay_s))


def fake_code_line() -> str:
    indent = " " * random.choice([0, 2, 4, 8, 12])
    mod = random.choice(FAKE_MODULES)
    kw = random.choice(FAKE_KEYWORDS)
    typ = random.choice(FAKE_TYPES)
    tok = random.choice(TOKENS)
    ident = rand_ident(random.randint(3, 10))
    ident2 = rand_ident(random.randint(3, 10))

    patterns = [
        f"{indent}{mod}::{kw}<{typ}>({ident} {tok} {ident2}) ::=> {tok}{typ} {{ … }}",
        f"{indent}let {ident}: {typ} {tok}= {kw}({mod}::{ident2}?) // TODO: appease entropy",
        f"{indent}match {ident} {{ {tok}Ok(v) => {kw}(v), {tok}Err(e) => panic!(e^2) }}",
        f"{indent}⟦{ident}⟧ := {kw}⟂({ident2}) ⊗ normalize(Δ={random.randint(1,9999)})",
        f"{indent}impl {tok}{typ} for {ident} where {ident2}: trait≋object {{ /* … */ }}",
        f"{indent}#pragma lament({ident}::{ident2}, dirge={random.randint(0,9)})",
        f"{indent}assert::{kw}({ident} -> {tok}truth?) // if false: awaken the watchdog",
    ]
    return random.choice(patterns)


def status_bar(progress: float, width: int) -> str:
    progress = max(0.0, min(1.0, progress))
    filled = int(round(progress * width))
    bar = "█" * filled + "░" * (width - filled)
    percent = int(progress * 100)
    phase = random.choice(["BOOT", "LINK", "JIT", "SYNC", "SCAN", "TRACE", "FLUSH", "GC", "RITUAL"])
    return f"[{phase}] [{bar}] {percent:3d}% | thr={random.randint(1,64):02d} | q={random.randint(0,9999):04d}"


def error_scene() -> list[str]:
    err = random.choice(FAKE_ERRORS)
    diag = random.choice(FAKE_DIAGNOSTICS)
    mod = random.choice(FAKE_MODULES)

    header = [
        "",
        "══════════════════════════════════════════════════════════════",
        f"FATAL[{random.randint(100,999)}] :: {err}",
        f"  {diag}",
        f"  --> {mod}::{rand_ident(6)}:{random.randint(1,9999)}:{random.randint(1,200)}",
        "  stacktrace (most recent call last):",
    ]

    frames = ["    " + random.choice(FAKE_STACK) for _ in range(random.randint(5, 9))]

    footer = [
        f"  context: phase={random.choice(['preheat','twilight','aftertaste','null'])} | entropy={random.randint(9000,99999)}",
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
# Main loop
# ---------------------------

def main() -> int:
    random.seed()

    # CLI options: --seed N, --max-lines N
    args = sys.argv[1:]
    if "--seed" in args:
        i = args.index("--seed")
        if i + 1 < len(args):
            try:
                random.seed(int(args[i + 1]))
            except ValueError:
                pass

    if "--max-lines" in args:
        i = args.index("--max-lines")
        if i + 1 < len(args):
            try:
                CFG.max_lines = int(args[i + 1])
            except ValueError:
                CFG.max_lines = None

    progress = 0.0
    lines_printed = 0

    def emit(line: str) -> None:
        nonlocal lines_printed
        slow_print(line, CFG)
        lines_printed += 1
        if CFG.max_lines is not None and lines_printed >= CFG.max_lines:
            slow_print("x", CFG)  # guarantee the required stop token
            raise SystemExit(0)

    while True:
        # Primary stop: random chance each iteration
        if random.random() < CFG.stop_probability:
            emit("x")
            return 0

        roll = random.random()

        if roll < CFG.error_probability:
            for line in error_scene():
                emit(line)

        elif roll < CFG.error_probability + CFG.status_probability:
            # Dramatic, slow progress with occasional rollovers
            progress += random.uniform(0.01, 0.06)
            if progress >= 1.0:
                progress = random.uniform(0.0, 0.12)
                emit(f"[INFO] epoch rollover -> token={random.choice(TOKENS)}{random.randint(1000,9999)}")
                emit("[WARN] cache coherence ritual incomplete; proceeding anyway")
            emit(status_bar(progress, CFG.bar_width))

            # Extra "noise" between bars
            if random.random() < 0.35:
                emit(f"[TRACE] {random.choice(FAKE_MODULES)}::{rand_ident(8)} :: Δt={random.randint(1,900)}ms :: jitter={random.randint(0,999)}ppm")

        else:
            # Slow "pseudo-code" bursts
            burst = random.randint(1, 3)
            for _ in range(burst):
                emit(fake_code_line())

            # Insert sporadic ominous commentary
            if random.random() < 0.18:
                emit(f"[NOTE] invariant '{rand_ident(5)}' observed to be {random.choice(['fractured', 'recursive', 'nonlocal', 'haunted'])}")

        # A small extra pause to keep it "slow and dramatic"
        time.sleep(random.uniform(0.12, 0.45))


if __name__ == "__main__":
    raise SystemExit(main())
