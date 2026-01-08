#!/usr/bin/env python3
"""
Fake "complicated code" terminal stream that prints until a literal 'x' is printed.

- Output is intentionally NOT real code.
- Includes status bars, pseudo stack traces, warnings, and cryptic "compiler" logs.
- Stops only after printing a standalone line: x
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass

# ---------------------------
# Configuration
# ---------------------------

@dataclass
class Config:
    min_delay_s: float = 0.015
    max_delay_s: float = 0.09
    # Probability per loop iteration to end by printing 'x'
    stop_probability: float = 0.012
    # Probability to print an "error block" in an iteration
    error_probability: float = 0.10
    # Probability to print a "status bar" line in an iteration
    status_probability: float = 0.35
    # How wide the status bar should be
    bar_width: int = 28


CFG = Config()


# ---------------------------
# Fake generators
# ---------------------------

TOKENS = [
    "Ω", "λ", "⟂", "∇", "≋", "⊗", "⟦", "⟧", "⋆", "⟁", "⟡", "≀", "≔", "↯", "⊘", "⋄"
]

FAKE_KEYWORDS = [
    "transmute", "defer", "reinterpret", "monomorph", "inline@volatile",
    "unwind?", "sealed", "phantom", "atomic-ish", "borrow::mut", "lifetime<'🜁'>",
    "trait≋object", "entropy_hint", "vectorize:maybe", "constexpr-ish",
]

FAKE_TYPES = [
    "u13", "i0", "ptr<nullsafe?>", "Maybe<Maybe<T>>", "Σ<T|⊥>", "Ref<'Ω, Mut<T>>",
    "Fn(…)->Never?", "Packet<Δ, 7>", "Matrix<3x∞>", "Tape[::]",
]

FAKE_MODULES = [
    "core::flux", "std::gloom", "runtime::mirage", "net::haze", "io::crypt",
    "vm::wisp", "compiler::oracle", "scheduler::murmur", "alloc::abyss",
]

FAKE_DIAGNOSTICS = [
    "warning: shadowed symbol 'δ' in non-euclidean scope",
    "note: eliding lifetime due to cosmic background noise",
    "hint: try sacrificing a goat to the borrow checker",
    "info: speculative optimization rolled back (reason: vibes mismatch)",
    "debug: recursion depth stabilized at 1e6±3",
]

FAKE_ERRORS = [
    "E0199: cannot unify type 'Σ<T|⊥>' with 'T' under phase=preheat",
    "E0420: ambiguous overload for operator '≔' (found 7 candidates)",
    "E0991: stack frame collapsed into a Klein bottle",
    "E0007: segmentation fault avoided (prediction engine intervened)",
    "E7777: illegal instruction: attempted to execute ASCII art",
]

FAKE_STACK = [
    "at core::flux::fold_chaos(line=∞, col=13)",
    "at compiler::oracle::resolve(phase='twilight')",
    "at runtime::mirage::spawn(task='murmur#A3')",
    "at vm::wisp::dispatch(op='⊗', mode='speculative')",
    "at alloc::abyss::reserve(bytes=13_337)",
]


def rand_ident(n: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(alphabet) for _ in range(n))


def fake_code_line() -> str:
    indent = " " * random.choice([0, 0, 2, 4, 8, 12])
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
        f"{indent}#pragma mystic({ident}::{ident2}, level={random.randint(0,9)})",
    ]
    return random.choice(patterns)


def status_bar(progress: float, width: int) -> str:
    progress = max(0.0, min(1.0, progress))
    filled = int(round(progress * width))
    bar = "█" * filled + "░" * (width - filled)
    percent = int(progress * 100)
    phase = random.choice(["BOOT", "LINK", "JIT", "SYNC", "SCAN", "TRACE", "FLUSH", "GC"])
    return f"[{phase}] [{bar}] {percent:3d}% | thr={random.randint(1,64):02d} | q={random.randint(0,9999):04d}"


def error_block() -> list[str]:
    err = random.choice(FAKE_ERRORS)
    diag = random.choice(FAKE_DIAGNOSTICS)
    lines = [
        f"error[{random.randint(100,999)}]: {err}",
        f"  {diag}",
        f"  --> {random.choice(FAKE_MODULES)}::{rand_ident(6)}:{random.randint(1,9999)}:{random.randint(1,200)}",
        "  stacktrace (most recent call last):",
    ]
    for _ in range(random.randint(2, 5)):
        lines.append("    " + random.choice(FAKE_STACK))
    if random.random() < 0.5:
        lines.append(f"  recover: applied patchset '{rand_ident(4)}-{random.randint(10,99)}' (hot) ✓")
    else:
        lines.append(f"  recover: deferred (reason: {random.choice(['phase shift', 'checksum drift', 'undefined mood'])})")
    return lines


def sleep_jitter(cfg: Config) -> None:
    time.sleep(random.uniform(cfg.min_delay_s, cfg.max_delay_s))


# ---------------------------
# Main loop
# ---------------------------

def main() -> int:
    random.seed()  # non-deterministic by default
    progress = 0.0

    # Optional: allow user to force a deterministic run for debugging
    # via: python script.py --seed 123
    if len(sys.argv) >= 3 and sys.argv[1] == "--seed":
        try:
            random.seed(int(sys.argv[2]))
        except ValueError:
            pass

    while True:
        # Stop condition: print a literal 'x' line, then exit.
        if random.random() < CFG.stop_probability:
            print("x")
            return 0

        roll = random.random()

        if roll < CFG.error_probability:
            for line in error_block():
                print(line)
                sleep_jitter(CFG)

        elif roll < CFG.error_probability + CFG.status_probability:
            # Advance progress in a jittery way and wrap around
            progress += random.uniform(0.01, 0.12)
            if progress >= 1.0:
                progress = random.uniform(0.0, 0.15)
                print(f"[INFO] epoch rollover -> token={random.choice(TOKENS)}{random.randint(1000,9999)}")
            print(status_bar(progress, CFG.bar_width))

        else:
            # Print a burst of fake lines to feel "busy"
            burst = random.randint(1, 4)
            for _ in range(burst):
                print(fake_code_line())

        sleep_jitter(CFG)


if __name__ == "__main__":
    raise SystemExit(main())

