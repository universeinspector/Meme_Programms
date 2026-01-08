from __future__ import annotations

import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text

# ---------------------------
# Keypress (press 'x' to exit, no Enter)
# ---------------------------

class KeyWatcher:
    def __init__(self, exit_key: str = "x") -> None:
        self.exit_key = exit_key
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def pressed_exit(self) -> bool:
        return self._stop.is_set()

    def _run(self) -> None:
        if os.name == "nt":
            self._run_windows()
        else:
            self._run_posix()

    def _run_windows(self) -> None:
        import msvcrt  # type: ignore
        while not self._stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch.lower() == self.exit_key:
                    self._stop.set()
                    return
            time.sleep(0.02)

    def _run_posix(self) -> None:
        import termios
        import tty
        import select

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # read single chars
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if ch.lower() == self.exit_key:
                        self._stop.set()
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------
# Config + fake content
# ---------------------------

@dataclass
class Config:
    project: str = "Cipher/Keyed/Caesar/Tokenizer/onion"
    environment: str = "Backend Development"
    bar_width: int = 44
    max_log_lines: int = 14  # visible log area height
    refresh_hz: int = 20


TOKENS = ["Ω", "λ", "⟂", "∇", "≋", "⊗", "⟦", "⟧", "⋆", "⟁", "⟡", "≀", "≔", "↯", "⊘", "⋄", "⨳", "⧗"]
MODULES = ["core::flux", "runtime::mirage", "compiler::oracle", "scheduler::murmur", "alloc::abyss", "net::haze", "vm::wisp"]
PHASES = ["BOOT", "SYNC", "SCAN", "TRACE", "LINK", "JIT", "GC", "RITUAL"]
WARNINGS = [
    "File descriptor usage approaching limit – review resource management",
    "Cache coherence ritual incomplete; proceeding anyway",
    "Clock skew exceeded tolerance; time may be non-linear",
    "Entropy budget exceeded; switching to symbolic execution",
    "Deprecated opcode '⟁' encountered; consider refactoring",
    "Memory page faults rising; investigate potential leaks",
    "High system load detected; performance may degrade",
]
ERRORS = [
    "E5001: linker refused: symbol 'hope' not found",
    "E1312: borrow checker demands tribute; value moved into the void",
    "E0991: stack frame collapsed into a Klein bottle",
    "E0420: ambiguous overload for operator '≔' (found 7 candidates)",
    "E0199: cannot unify type 'Σ<T|⊥>' with 'T' under phase=preheat",
    "E0007: segmentation fault avoided (prediction engine intervened)",
    "E7777: illegal instruction: attempted to execute ASCII art",
    "E3141: infinite recursion detected; call stack spiraled into fractal dimension",
    "E6666: runtime exorcism required; ghost in the machine detected",
]


def make_monitor_line() -> str:
    cpu = random.randint(45, 78)
    ram = random.randint(30, 58)
    net = random.randint(10, 28)
    disk = random.randint(18, 44)
    procs = random.randint(80, 220)
    return f"CPU: {cpu:2d}%  |  RAM: {ram:2d}%  |  Network: {net:2d} MB/s  |  Disk I/O: {disk:2d} MB/s  |  Processes: {procs:3d}"


def make_random_log() -> str:
    kind = random.random()
    if kind < 0.40:
        return f"[{random.choice(PHASES)}] {make_monitor_line()}"
    if kind < 0.68:
        rec = random.randint(120, 2200)
        gb = random.randint(3, 120)
        return f"[PIPE] Processing API Data Streams :: validating integrity {rec} records ({gb} GB)"
    if kind < 0.86:
        mod = random.choice(MODULES)
        tok = random.choice(TOKENS)
        return f"[TRACE] {mod}::{''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(8))} :: token={tok}{random.randint(1000,9999)}"
    if kind < 0.94:
        return f"[WARN] {random.choice(WARNINGS)}"
    return f"[ERROR] {random.choice(ERRORS)}"


# ---------------------------
# UI builder (Rust-like layout)
# ---------------------------

def build_view(cfg: Config, progress: Progress, logs: List[str], hint: str) -> Group:
    # Header region (top-left), then a blank spacer, then the live progress + logs.
    header = [
        Text("INITIALIZING DEVELOPMENT ENVIRONMENT", style="bold"),
        Text(f"Project: {cfg.project}"),
        Text.assemble("Environment: ", (cfg.environment, "bold green")),
        Text(""),
        Text(hint, style="dim"),
        progress,
        Text(""),
        Text("DEVELOPMENT ENVIRONMENT INITIALIZED", style="bold green"),
        Text(""),
        Text("System Resource Monitoring", style="bold green"),
    ]

    # Log region (fixed height -> looks like a dashboard, no endless scroll)
    log_text = Text()
    for line in logs[-cfg.max_log_lines :]:
        # subtle coloring by prefix
        if line.startswith("[ERROR]"):
            log_text.append(line + "\n", style="bold red")
        elif line.startswith("[WARN]"):
            log_text.append(line + "\n", style="bold yellow")
        elif line.startswith("[TRACE]"):
            log_text.append(line + "\n", style="cyan")
        elif line.startswith("[PIPE]"):
            log_text.append(line + "\n", style="blue")
        else:
            log_text.append(line + "\n")

    footer = Text("\nPress 'x' to terminate.", style="dim")
    return Group(*header, log_text, footer)


# ---------------------------
# Main loop
# ---------------------------

def main() -> int:
    cfg = Config()

    console = Console()
    console.clear()

    # Live progress bar line like your Rust screenshot: [00:00:00] [====] 2/100 (7s)
    progress = Progress(
        TextColumn("[bold][{task.fields[timestamp]}][/bold]"),
        BarColumn(bar_width=cfg.bar_width),
        TextColumn("{task.completed}/{task.total} ({task.fields[eta]}s)"),
        expand=False,
    )
    task = progress.add_task("", total=100, timestamp="00:00:00", eta="7")

    # Boot hint line
    hint = "Loading configuration files..."

    logs: List[str] = []

    watcher = KeyWatcher(exit_key="x")
    watcher.start()

    start = time.time()
    completed = 0

    # Full-screen redraw to keep it “Rust-like” (no scrolling)
    with Live(
        build_view(cfg, progress, logs, hint),
        console=console,
        refresh_per_second=cfg.refresh_hz,
        screen=True,
    ) as live:
        # Phase 1: Boot progress to 100%
        while completed < 100 and not watcher.pressed_exit():
            elapsed = int(time.time() - start)
            timestamp = f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"

            completed = min(100, completed + random.randint(1, 4))
            remaining = max(0, 100 - completed)
            eta = max(0, int(remaining / random.uniform(10, 16)))

            progress.update(task, completed=completed, timestamp=timestamp, eta=str(eta))

            if random.random() < 0.35:
                logs.append(make_random_log())

            live.update(build_view(cfg, progress, logs, hint))
            time.sleep(random.uniform(0.06, 0.16))

        # Phase 2: Run indefinitely (dashboard mode) until 'x'
        hint = "Environment ready. Streaming activity..."
        while not watcher.pressed_exit():
            elapsed = int(time.time() - start)
            timestamp = f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"

            # Keep progress "alive": oscillate 90..100 like a busy indicator
            # (looks active without needing a real task)
            wobble = 90 + int((time.time() * 7) % 11)  # 90..100
            progress.update(task, completed=wobble, timestamp=timestamp, eta=str(random.randint(1, 9)))

            # Emit random activity lines
            burst = 1 if random.random() < 0.75 else 2
            for _ in range(burst):
                logs.append(make_random_log())

            live.update(build_view(cfg, progress, logs, hint))
            time.sleep(random.uniform(0.18, 0.55))

    # Cleanup
    watcher.stop()
    console.clear()
    console.print("[bold green]Session terminated.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
