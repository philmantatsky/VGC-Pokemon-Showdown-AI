#!/usr/bin/env python
"""Unattended overnight self-play training on a laptop (made for Apple Silicon MacBooks).

  bash scripts/overnight.sh                        # set up if needed, then train for 8 hours
  bash scripts/overnight.sh --hours 10             # longer night
  python scripts/overnight.py --report-only        # rewrite the morning report

1. Checks the machine: memory, cores, torch / MPS, power, free disk, and whether the
   antonius1 bot's heavy jobs (batteries, mirrors, training) are running.
2. Picks settings: the `small` model with a minibatch that fits in memory.
3. Probes a few updates on CPU and on the Mac GPU (MPS) and keeps the faster device.
4. Trains with `caffeinate` holding the Mac awake, evaluating against the greedy
   baseline about every 30 minutes. A watchdog restarts the trainer from the last
   checkpoint if it dies or hangs, and moves it to the CPU if MPS misbehaves.
5. Writes REPORT.md in the run directory: games played, speed, the win-rate curve,
   a final 1,000-game evaluation, the result against the model the night started
   with, and the teams that did best.

Ctrl-C (or `kill <pid>`) stops cleanly: the trainer saves, then the report is written
(Ctrl-C again during the report skips the remaining evaluations). Closing the Terminal
window does not stop the run. Running it again with the same --run-dir continues where
it stopped, so the model keeps improving night after night.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Scripts that would fight this run for the CPU/GPU (and skew their own results): the
# antonius1 batteries, mirrors, training and data generation, and other vgczero runs.
# Matched only in commands run by an interpreter, so an editor with train.py open is ignored.
HEAVY = ["evaluation/", "training/", "datagen/", "_chain.sh", "run_overnight.sh", "vgc_bench.train", "vgc_bench/train.py",
         "vgc_bench.pretrain", "vgc_bench/pretrain.py", "vgc_bench.eval", "vgc_bench/eval.py",
         "scripts/train.py", "scripts/evolve.py", "scripts/overnight.py"]
LIGHT = ["ladder_ourteam.py", "exhibition_mode.sh", "challenges_deployed.sh", "ladder_deployed.sh", "scripts/play.py"]
INTERPRETERS = ("python", "bash", "sh", "zsh")
WRAPPERS = ("tmux", "screen")  # their command lines contain the command they started

# ---- output -----------------------------------------------------------------------------
# Terminal output goes through a queue and a thread, so a paused (Ctrl-S) or closed terminal
# can never block the runner or, through the pipe, the trainer. Lines are dropped if it backs up.
_ECHO: queue.Queue = queue.Queue(maxsize=5000)


def _echo_worker() -> None:
    while True:
        s = _ECHO.get()
        try:
            print(s, flush=True)
        except (OSError, ValueError):
            pass  # terminal gone: keep draining


threading.Thread(target=_echo_worker, daemon=True).start()


def echo(text: str) -> None:
    try:
        _ECHO.put_nowait(text)
    except queue.Full:
        pass


def flush_echo(timeout: float = 5.0) -> None:
    t = time.time()
    while not _ECHO.empty() and time.time() - t < timeout:
        time.sleep(0.05)
    time.sleep(0.1)


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# ---- machine ---------------------------------------------------------------------------


def memory_gb() -> float:
    if sys.platform == "darwin":
        v = sh(["sysctl", "-n", "hw.memsize"])
        return int(v) / 2**30 if v.isdigit() else 8.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return 8.0


def perf_cores() -> int:
    if sys.platform == "darwin":
        for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
            v = sh(["sysctl", "-n", key])
            if v.isdigit():
                return int(v)
    return os.cpu_count() or 4


def chip() -> str:
    if sys.platform == "darwin":
        return sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def on_battery() -> bool:
    return sys.platform == "darwin" and "Battery Power" in sh(["pmset", "-g", "batt"])


def icloud_synced(path: Path) -> bool:
    """True if `path` is under ~/Desktop or ~/Documents and iCloud 'Desktop & Documents' looks enabled."""
    home = Path.home()
    cloud = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    p = path.resolve()
    for name in ("Desktop", "Documents"):
        try:
            p.relative_to((home / name).resolve())
        except ValueError:
            continue
        if (cloud / name).exists():
            return True
    return False


def classify(cmd: str) -> str | None:
    """'heavy', 'light' or None for one process command line."""
    toks = cmd.split()
    if not toks or Path(toks[0]).name.lower() in WRAPPERS:
        return None
    seen_interp = False
    for t in toks:
        if Path(t).name.lower().startswith(INTERPRETERS):
            seen_interp = True
            continue
        if seen_interp:
            if any(p in t for p in HEAVY):
                return "heavy"
            if any(p in t for p in LIGHT):
                return "light"
    return None


def other_jobs() -> tuple[list[str], list[str]]:
    me = {os.getpid(), os.getppid()}
    heavy, light = [], []
    for line in sh(["ps", "-A", "-o", "pid=", "-o", "command="]).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) in me:
            continue
        kind = classify(parts[1])
        if kind == "heavy":
            heavy.append(line.strip()[:160])
        elif kind == "light":
            light.append(line.strip()[:160])
    return heavy, light


def torch_info() -> tuple[str, bool]:
    out = sh([PY, "-c", "import torch; print(torch.__version__, int(bool(getattr(torch.backends, 'mps', None) "
                        "and torch.backends.mps.is_available())))"])
    parts = out.split()
    return (parts[0], parts[1] == "1") if len(parts) == 2 else ("?", False)


def torch_at_least(version: str, major: int, minor: int) -> bool:
    m = re.match(r"(\d+)\.(\d+)", version)
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (major, minor)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    """After a crash the trainer redoes the updates since its last save; keep the last row per update."""
    by = {}
    for r in rows:
        by[r.get("update")] = r
    return sorted(by.values(), key=lambda r: r.get("update", 0))


# ---- training processes -------------------------------------------------------------------


def train_cmd(run_dir: Path, s: dict, device: str, extra: list[str]) -> list[str]:
    return [PY, "-u", str(ROOT / "scripts" / "train.py"), "--run-dir", str(run_dir), "--model", s["model"],
            "--device", device, "--n-envs", str(s["n_envs"]), "--rollout-steps", str(s["rollout_steps"]),
            "--minibatch", str(s["minibatch"]), *extra]


def child_env(device: str, threads: int) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "python")
    if device == "cpu":
        env.setdefault("OMP_NUM_THREADS", str(threads))
    else:
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


class Runner:
    def __init__(self, args):
        self.args = args
        rd = Path(args.run_dir)
        self.run_dir = rd if rd.is_absolute() else ROOT / rd
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.run_dir / "overnight.log", "a")
        self.proc: subprocess.Popen | None = None
        self.signals = 0  # Ctrl-C / SIGTERM count; the handler only counts and forwards
        self.seen_signals = 0
        self.report_signals = 0
        self.phase = "setup"
        self.session = time.strftime("%Y%m%d_%H%M%S")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        echo(line)
        self.f.write(line + "\n")
        self.f.flush()

    # The handler must not log or take locks (it can interrupt the main thread anywhere): it counts
    # the signal and forwards it to the current child, which runs in its own session and so never
    # sees the terminal's Ctrl-C directly. The main loop reports it.
    def on_signal(self, signum, _frame) -> None:
        self.signals += 1
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                if self.phase == "train" and self.signals < 3:
                    p.send_signal(signal.SIGINT)  # 1st: finish the update and save; 2nd: stop sooner, still save
                else:
                    p.kill()
            except OSError:
                pass
        if self.signals >= 6:
            os._exit(130)  # escape hatch

    def note_signals(self) -> None:
        if self.signals > self.seen_signals:
            self.seen_signals = self.signals
            what = {"train": "the trainer is saving and stopping" if self.signals < 3 else "stopping the trainer now",
                    "report": "skipping the remaining evaluations"}.get(self.phase, "stopping")
            self.log(f"Ctrl-C/SIGTERM ({self.signals}): {what}")

    def keep_awake(self) -> None:
        if shutil.which("caffeinate"):
            # -i: no idle sleep, -s: no system sleep on AC power; -w: until this process exits.
            # Own session, so closing the Terminal window does not end it.
            subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def finish_early(self, code: int, reason: str) -> int:
        """Stop before training and leave a REPORT.md that says why (not last night's)."""
        self.log(reason)
        (self.run_dir / "REPORT.md").write_text(
            f"# vgczero overnight report ({time.strftime('%Y-%m-%d %H:%M')})\n\nThis session did not train: {reason}\n"
            f"\nThe previous report, if any, is REPORT.prev.md. Log: `{self.run_dir / 'overnight.log'}`.\n")
        return code

    def run_child(self, cmd: list[str], env: dict, timeout: float) -> tuple[int, str]:
        """Run a helper process in its own session (Ctrl-C reaches it only through on_signal)."""
        try:
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         env=env, start_new_session=True)
            if self.signals > self.report_signals and self.phase == "report" or self.signals and self.phase == "probe":
                self.proc.kill()
            try:
                text, _ = self.proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                text, _ = self.proc.communicate()
                return -999, (text or "") + "\n(timed out)"
            return self.proc.returncode, text or ""
        finally:
            self.proc = None

    def probe(self, s: dict, device: str, cores: int) -> dict | None:
        """Train a few updates from scratch on `device`; return its speed, or None if it failed."""
        a = self.args
        d = self.run_dir / f"probe_{device}"
        shutil.rmtree(d, ignore_errors=True)
        cmd = train_cmd(d, s, device, ["--updates", str(a.probe_updates), "--eval-every", "0", "--save-every", "100000"])
        self.log(f"probe: {a.probe_updates} updates of '{s['model']}' on {device} ...")
        rc, text = self.run_child(cmd, child_env(device, cores), timeout=1800)
        rows = read_jsonl(d / "metrics.jsonl")
        shutil.rmtree(d, ignore_errors=True)
        if self.signals:
            return None
        if rc != 0 or len(rows) < 2:
            self.log(f"probe on {device} failed (exit {rc}):\n  " + "\n  ".join(text.splitlines()[-15:]))
            return None
        if any(row.get("illegal", 0) for row in rows):
            self.log(f"probe on {device} produced illegal actions; not using it")
            return None
        rows = rows[1:]  # the first update includes start-up costs
        secs = sum(row["collect_s"] + row["update_s"] for row in rows)
        out = {"device": device, "games_per_s": sum(row["games"] for row in rows) / max(secs, 1e-9),
               "s_per_update": secs / len(rows)}
        self.log(f"probe on {device}: {out['games_per_s']:.1f} games/s, {out['s_per_update']:.1f} s per update")
        return out

    def run(self) -> int:
        a = self.args
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)  # closing the Terminal window does not stop the night
        self.keep_awake()  # first thing: setup and probes must not let the Mac sleep either
        if a.report_only:
            self.phase = "report"
            self.report(final_eval=not a.no_eval)
            return 0
        report = self.run_dir / "REPORT.md"
        if report.exists():
            report.replace(self.run_dir / "REPORT.prev.md")
        start = time.time()
        deadline = start + a.hours * 3600
        self.log(f"=== overnight session {self.session}: {a.hours:g} h into {self.run_dir}")
        if shutil.which("caffeinate"):
            self.log("caffeinate keeps the Mac awake while this runs (the display may sleep; keep the lid open)")

        # 1. Machine.
        mem, cores = memory_gb(), perf_cores()
        tv, mps = torch_info()
        self.log(f"machine: {chip()}, {mem:.0f} GB memory, {cores} performance cores, {platform.system()} {platform.release()}")
        self.log(f"python {platform.python_version()}, torch {tv}, MPS {'available' if mps else 'not available'}")
        free_gb = shutil.disk_usage(self.run_dir).free / 2**30
        if free_gb < 2:
            return self.finish_early(2, f"only {free_gb:.1f} GB of disk free; a night needs about 1 GB. Free some space.")
        if free_gb < 5:
            self.log(f"note: {free_gb:.1f} GB of disk free (a night needs about 1 GB)")
        if on_battery():
            self.log("WARNING: on battery. Plug in: caffeinate cannot hold off sleep on battery.")
        if icloud_synced(self.run_dir):
            self.log("note: the run directory is in an iCloud-synced Desktop/Documents folder; checkpoints (~30 MB "
                     "every ~10 min) will upload all night. Use --run-dir outside it (e.g. ~/vgczero-runs/night) to avoid that.")
        heavy, light = other_jobs()
        if heavy and not a.force:
            return self.finish_early(3, "these jobs are already running and would compete for the CPU/GPU:\n  "
                                     + "\n  ".join(heavy) + "\nWait for them to finish, or pass --force to run anyway.")
        if light:
            self.log("note: running alongside (fine, it will just be a little slower):\n  " + "\n  ".join(light))

        # 2. Settings.
        latest = self.run_dir / "checkpoints" / "latest.pt"
        prev_model = read_json(self.run_dir / "cfg.json").get("model") if latest.exists() else None
        model = prev_model or a.model or "small"
        if prev_model and a.model and a.model != prev_model:
            self.log(f"this run already holds a '{prev_model}' model and will keep training it "
                     f"(use a new --run-dir for '{a.model}')")
        if a.minibatch:
            minibatch = a.minibatch
        elif model == "base":
            minibatch = 1024 if mem < 24 else 2048
        else:
            minibatch = 1024 if mem < 12 else 2048
        s = {"model": model, "n_envs": a.n_envs, "rollout_steps": a.rollout_steps, "minibatch": minibatch}
        self.log(f"settings: model {model}, {a.n_envs} envs x {a.rollout_steps} steps, minibatch {minibatch}"
                 + (" (resuming)" if latest.exists() else " (new run)"))

        # 3. Device.
        if a.device == "mps" and not mps:
            return self.finish_early(4, "--device mps, but MPS is not available on this machine")
        if a.device in ("cpu", "mps"):
            devices = [a.device]
        else:
            devices = ["cpu"]
            if mps and torch_at_least(tv, 2, 5):
                devices.append("mps")
            elif mps:
                self.log(f"torch {tv} is too old for vgczero on MPS (needs 2.5+); using the CPU")
        self.phase = "probe"
        speeds = {}
        if not a.no_probe:
            for d in devices:
                if self.signals:
                    break
                p = self.probe(s, d, cores)
                if p:
                    speeds[d] = p
        self.note_signals()
        if self.signals:
            return self.finish_early(130, "stopped (Ctrl-C) before training started")
        if speeds:
            device = max(speeds, key=lambda d: speeds[d]["games_per_s"])
        elif a.no_probe:
            device = devices[-1]
        else:
            return self.finish_early(5, "every speed probe failed; see overnight.log")
        spu = speeds.get(device, {}).get("s_per_update")
        eval_every = a.eval_every or (max(5, min(200, round(1800 / spu))) if spu else 25)
        save_every = max(5, min(50, round(600 / spu))) if spu else 10
        if spu:
            hours_left = (deadline - time.time()) / 3600
            self.log(f"using {device}: about {speeds[device]['games_per_s'] * 3600 * hours_left / 1e3:,.0f}k games "
                     f"in the {hours_left:.1f} h left; eval vs greedy every {eval_every} updates "
                     f"(~{eval_every * spu / 60:.0f} min)")
        else:
            self.log(f"using {device}; eval vs greedy every {eval_every} updates")

        # Keep the starting model so the report can show tonight's improvement.
        if latest.exists():
            shutil.copy2(latest, self.run_dir / "checkpoints" / f"start_{self.session}.pt")

        # 4. Train, with a watchdog.
        self.phase = "train"
        extra = ["--eval-every", str(eval_every), "--eval-games", str(a.eval_games), "--save-every", str(save_every)]
        failures = {"cpu": 0, "mps": 0}
        restarts = 0
        metrics_path = self.run_dir / "metrics.jsonl"
        lines_before = len(read_jsonl(metrics_path))
        # A hang: no output for this long (the trainer prints every update; evals add a little).
        stall_s = max(1800.0, 20 * (spu or 60))
        while not self.signals:
            left = deadline - time.time()
            if left < 120:
                break
            cmd = train_cmd(self.run_dir, s, device, extra + ["--hours", f"{left / 3600:.4f}"])
            self.log(f"training on {device} for {left / 3600:.2f} h (log: {self.run_dir / 'train.log'})")
            rc, tail, why = self.train_once(cmd, child_env(device, cores), deadline, stall_s, device, metrics_path)
            self.note_signals()
            if self.signals:
                break
            if why == "mps-illegal":
                device = "cpu"
                self.log("MPS produced illegal actions; continuing on the CPU")
                continue
            if rc == 0:
                break
            failures[device] += 1
            restarts += 1
            self.log(f"trainer {'hung' if why == 'stall' else f'exited with code {rc}'} (restart {restarts}); last lines:\n  "
                     + "\n  ".join(tail[-12:]))
            if restarts > a.max_restarts:
                self.log("too many restarts; giving up for tonight")
                break
            if device == "mps" and failures["mps"] >= 2:
                device = "cpu"
                self.log("MPS failed twice; continuing on the CPU")
            for _ in range(min(60, 10 * restarts)):
                if self.signals:
                    break
                time.sleep(1)
        self.note_signals()
        session_rows = dedupe(read_jsonl(metrics_path)[lines_before:])
        self.log(f"training done: {len(session_rows)} updates, {sum(r.get('games', 0) for r in session_rows):,} games "
                 f"in {(time.time() - start) / 3600:.2f} h")

        # 5. Report.
        self.phase = "report"
        self.report(final_eval=not a.no_eval, session_rows=session_rows, device=device, speeds=speeds,
                    hours=(time.time() - start) / 3600, restarts=restarts)
        return 0

    def train_once(self, cmd, env, deadline, stall_s, device, metrics_path) -> tuple[int, list[str], str]:
        """Run the trainer once. Returns (exit code, last lines, reason: '' / 'stall' / 'mps-illegal' / 'deadline')."""
        lines: queue.Queue = queue.Queue()
        tail: list[str] = []
        why = ""
        with open(self.run_dir / "train.log", "a") as tl:
            tl.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(cmd)}\n")
            tl.flush()
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         env=env, start_new_session=True)
            proc = self.proc
            if self.signals:
                proc.send_signal(signal.SIGINT)

            def reader():
                for line in proc.stdout:
                    lines.put(line)
                lines.put(None)

            threading.Thread(target=reader, daemon=True).start()
            last_out = prev_tick = time.time()
            m_off = metrics_path.stat().st_size if metrics_path.exists() else 0
            next_check = time.time() + 60
            stopping_at = None
            done = False
            while not done:
                try:
                    line = lines.get(timeout=1.0)
                except queue.Empty:
                    line = ""
                while line:
                    tl.write(line)
                    tail = (tail + [line.rstrip()])[-25:]
                    if not self.args.quiet:
                        echo(line.rstrip("\n"))
                    last_out = time.time()
                    try:
                        line = lines.get_nowait()
                    except queue.Empty:
                        line = ""
                if line is None:
                    done = True
                tl.flush()
                self.note_signals()
                now = time.time()
                if now - prev_tick > 30:
                    last_out = now  # the machine slept (lid closed?): not a hang
                prev_tick = now
                if done or self.signals:
                    continue
                if stopping_at is not None:
                    if now > stopping_at + 300:
                        proc.kill()
                    continue
                # Deadline backstop: the trainer stops itself at the deadline (between updates);
                # if it has not after a grace period, ask it, then kill it.
                if now > deadline + max(600.0, 3 * stall_s / 20):
                    self.log("past the deadline; asking the trainer to stop")
                    proc.send_signal(signal.SIGINT)
                    stopping_at, why = now, "deadline"
                elif now - last_out > stall_s:
                    self.log(f"no trainer output for {stall_s / 60:.0f} min; restarting it")
                    proc.kill()
                    stopping_at, why = now, "stall"
                elif now >= next_check:
                    next_check = now + 60
                    if device == "mps" and metrics_path.exists():
                        with open(metrics_path, "rb") as mf:
                            mf.seek(m_off)
                            chunk = mf.read()
                        cut = chunk.rfind(b"\n") + 1  # whole lines only
                        m_off += cut
                        ill = 0
                        for ln in chunk[:cut].decode(errors="replace").splitlines():
                            try:
                                ill += json.loads(ln).get("illegal", 0)
                            except ValueError:
                                pass
                        if ill:
                            self.log(f"{ill} illegal actions on MPS; stopping the trainer to switch to the CPU")
                            proc.send_signal(signal.SIGINT)
                            stopping_at, why = now, "mps-illegal"
            rc = proc.wait()
        self.proc = None
        return rc, tail, why

    # ---- morning report ---------------------------------------------------------------

    def evaluate(self, a: str, b: str, games: int) -> str:
        if self.signals > self.report_signals:
            return "skipped (Ctrl-C)"
        rc, out = self.run_child([PY, "-m", "vgczero.evaluate", a, b, "--games", str(games), "--device", "cpu"],
                                 child_env("cpu", perf_cores()), timeout=1800)
        self.note_signals()
        if self.signals > self.report_signals:
            return "skipped (Ctrl-C)"
        lines = out.strip().splitlines() or ["?"]
        # The result looks like "0.612 [0.581, 0.642] over 1000 games (3 draws, 8.1 turns)";
        # match it directly because the paths before it can contain spaces.
        for ln in reversed(lines):
            m = re.search(r"\d\.\d+ \[[\d.]+, [\d.]+\] over \d+ games \([^)]*\)", ln)
            if m:
                return m.group(0)
        return f"failed (exit {rc}): {lines[-1][:200]}"

    def report(self, final_eval: bool = True, session_rows=None, device=None, speeds=None, hours=None, restarts=0) -> None:
        a, rd = self.args, self.run_dir
        self.report_signals = self.signals
        latest = rd / "checkpoints" / "latest.pt"
        if not latest.exists():
            self.finish_early(0, "no checkpoint was saved, so there is nothing to report")
            return
        metrics = dedupe(read_jsonl(rd / "metrics.jsonl"))
        evals = dedupe(read_jsonl(rd / "eval.jsonl"))
        cfg = read_json(rd / "cfg.json")
        head = [f"# vgczero overnight report ({time.strftime('%Y-%m-%d %H:%M')})", "",
                f"Run: `{rd}`, model `{cfg.get('model', '?')}`, minibatch {cfg.get('minibatch', '?')}, "
                f"machine: {chip()}, {memory_gb():.0f} GB."]
        body = []
        if session_rows is not None:
            g = sum(r.get("games", 0) for r in session_rows)
            secs = sum(r.get("collect_s", 0) + r.get("update_s", 0) for r in session_rows)
            body += ["", "## Tonight", "",
                     f"* {hours:.2f} h on {device}, {len(session_rows)} updates, {g:,} self-play games "
                     f"({g / max(secs, 1e-9):.1f} games/s while training)"
                     + (f", {restarts} restart(s) (see overnight.log)" if restarts else "") + "."]
            if speeds:
                body.append("* Probe: " + ", ".join(f"{d} {p['games_per_s']:.1f} games/s" for d, p in speeds.items()) + ".")
            ill = sum(r.get("illegal", 0) for r in session_rows)
            if ill:
                body.append(f"* WARNING: {ill} illegal actions (those games were forfeited); worth investigating.")
        if metrics:
            last = metrics[-1]
            body += ["", "## Whole run", "",
                     f"* {last.get('update', '?')} updates, {last.get('games_total', 0):,} self-play games in total."]
        if evals:
            body += ["", "## Win rate against the greedy baseline during training", "",
                     "| update | games played | win rate vs greedy |", "|---:|---:|---:|"]
            step = max(1, len(evals) // 20)
            shown = evals[::step] + ([evals[-1]] if (len(evals) - 1) % step else [])
            for e in shown:
                body.append(f"| {e['update']} | {e.get('games_total', 0):,} | {e['vs_greedy']:.3f} |")
        q = shlex.quote
        tail = ["", "## Next", "",
                "* Run the same command again tonight: it resumes this run and keeps improving it.",
                f"* Try the model: `python scripts/matchup.py --checkpoint {q(str(latest))} --a MC2001 --b MC1000`.",
                "* Full logs: `train.log`, `metrics.jsonl`, `eval.jsonl`, `overnight.log` in the run directory."]

        def write(extra: list[str]) -> str:
            text = "\n".join(head + body + extra + tail) + "\n"
            tmp = rd / "REPORT.md.tmp"
            tmp.write_text(text)
            tmp.replace(rd / "REPORT.md")
            return text

        write(["", "## Final evaluation", "", "* running ..."] if final_eval else [])  # a report exists from here on
        extra = []
        if final_eval:
            self.log("final evaluation (a few minutes on the CPU; Ctrl-C skips it) ...")
            extra += ["", "## Final evaluation", ""]
            extra.append(f"* vs greedy: {self.evaluate(str(latest), 'greedy', 1000)}")
            extra.append(f"* vs random: {self.evaluate(str(latest), 'random', 400)}")
            start = rd / "checkpoints" / f"start_{self.session}.pt"
            if start.exists():
                extra.append(f"* vs the model this session started from: {self.evaluate(str(latest), str(start), 1000)}")
        if self.signals <= self.report_signals:
            rc, out = self.run_child([PY, str(ROOT / "scripts" / "top_teams.py"), "--run-dir", str(rd), "--k", "10",
                                      "--min-games", str(a.min_team_games)], child_env("cpu", 1), timeout=600)
            rows = [ln for ln in out.splitlines() if " lb " in ln]
            if rows:
                extra += ["", f"## Best teams so far (Wilson lower bound of win rate, >= {a.min_team_games} games)", "",
                          "```"] + rows + ["```", "", f"Saved to `{rd / 'top_teams.json'}` for `play.py --team-file`."]
        text = write(extra)
        echo("\n" + text)
        self.log(f"report written to {rd / 'REPORT.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--run-dir", default="runs/overnight")
    ap.add_argument("--model", choices=["tiny", "small", "base"], default=None, help="default: small (or the run's model)")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto", help="auto: probe CPU and MPS, keep the faster")
    ap.add_argument("--minibatch", type=int, default=None, help="default: 1024 under 12 GB of memory, else 2048")
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--rollout-steps", type=int, default=32)
    ap.add_argument("--probe-updates", type=int, default=4)
    ap.add_argument("--no-probe", action="store_true", help="skip the speed probe (with --device auto: use MPS if present)")
    ap.add_argument("--eval-every", type=int, default=0, help="updates between greedy evals (default: ~30 min)")
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--min-team-games", type=int, default=30)
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="run even if antonius1's heavy jobs are running")
    ap.add_argument("--quiet", action="store_true", help="only write trainer output to train.log")
    ap.add_argument("--no-eval", action="store_true", help="skip the final evaluation in the report")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    try:
        return Runner(args).run()
    finally:
        flush_echo()


if __name__ == "__main__":
    sys.exit(main())
