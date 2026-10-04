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
   baseline about every 30 minutes. If the trainer dies it restarts from the last
   checkpoint (and drops to CPU if MPS keeps failing).
5. Writes REPORT.md in the run directory: games played, speed, the win-rate curve,
   a final 1,000-game evaluation, the result against the model the night started
   with, and the teams that did best.

Ctrl-C (or `kill <pid>`) stops cleanly: the trainer saves first, then the report is
written. Running it again with the same --run-dir continues where it stopped, so the
model keeps improving night after night.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# antonius1 jobs that would fight this run for the CPU / GPU (and skew their own results).
HEAVY = ["run_guard_ab", "mirror_guard_ab", "_trial.py", "_chain.sh", "vgc_bench.train", "vgc_bench/train.py",
         "vgc_bench.pretrain", "pretrain.py", "train_mac.sh", "pretrain_mac.sh", "run_overnight.sh", "eval_mac.sh",
         "vgc_bench.eval", "scripts/train.py", "scripts/evolve.py", "scripts/overnight.py"]
# Light jobs: fine to run alongside, but worth a note.
LIGHT = ["ladder_ourteam.py", "exhibition_mode.sh", "challenges_deployed.sh", "ladder_deployed.sh", "scripts/play.py"]


_TTY = [True]  # cleared when the terminal goes away (window closed): keep running, stop printing


def echo(text: str) -> None:
    if _TTY[0]:
        try:
            print(text, flush=True)
        except OSError:
            _TTY[0] = False


def log(msg: str, f=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    echo(line)
    if f is not None:
        f.write(line + "\n")
        f.flush()


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


def other_jobs() -> tuple[list[str], list[str]]:
    me = {os.getpid(), os.getppid()}
    heavy, light = [], []
    for line in sh(["ps", "-A", "-o", "pid=", "-o", "command="]).splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) in me:
            continue
        cmd = parts[1]
        if any(p in cmd for p in HEAVY):
            heavy.append(line.strip()[:160])
        elif any(p in cmd for p in LIGHT):
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


# ---- training processes -------------------------------------------------------------------


def train_cmd(run_dir: Path, s: dict, device: str, extra: list[str]) -> list[str]:
    return [PY, "-u", str(ROOT / "scripts" / "train.py"), "--run-dir", str(run_dir), "--model", s["model"],
            "--device", device, "--n-envs", str(s["n_envs"]), "--rollout-steps", str(s["rollout_steps"]),
            "--minibatch", str(s["minibatch"]), *extra]


def child_env(device: str, threads: int) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if device == "cpu":
        env.setdefault("OMP_NUM_THREADS", str(threads))
    else:
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


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


def probe(run_dir: Path, s: dict, device: str, updates: int, threads: int, f) -> dict | None:
    """Train a few updates from scratch on `device`; return speed, or None if it failed."""
    d = run_dir / f"probe_{device}"
    shutil.rmtree(d, ignore_errors=True)
    cmd = train_cmd(d, s, device, ["--updates", str(updates), "--eval-every", "0", "--save-every", "100000"])
    log(f"probe: {updates} updates of '{s['model']}' on {device} ...", f)
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800, env=child_env(device, threads))
    except subprocess.TimeoutExpired:
        log(f"probe on {device} took over 30 minutes; skipping it", f)
        shutil.rmtree(d, ignore_errors=True)
        return None
    rows = read_jsonl(d / "metrics.jsonl")
    shutil.rmtree(d, ignore_errors=True)
    if r.returncode != 0 or len(rows) < 2:
        log(f"probe on {device} failed (exit {r.returncode}):\n" + "\n".join((r.stdout + r.stderr).splitlines()[-15:]), f)
        return None
    if any(row.get("illegal", 0) for row in rows):
        log(f"probe on {device} produced illegal actions; not using it", f)
        return None
    rows = rows[1:]  # the first update includes start-up costs
    secs = sum(row["collect_s"] + row["update_s"] for row in rows)
    games = sum(row["games"] for row in rows)
    out = {"device": device, "games_per_s": games / max(secs, 1e-9), "s_per_update": secs / len(rows),
           "wall_s": time.time() - t}
    log(f"probe on {device}: {out['games_per_s']:.1f} games/s, {out['s_per_update']:.1f} s per update", f)
    return out


class Runner:
    def __init__(self, args):
        self.args = args
        self.run_dir = (ROOT / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.run_dir / "overnight.log", "a")
        self.proc: subprocess.Popen | None = None
        self.stop_requested = 0
        self.session = time.strftime("%Y%m%d_%H%M%S")

    # Ctrl-C / SIGTERM: ask the trainer (in its own session, so it does not get the
    # terminal's Ctrl-C directly) to save and stop; a second one makes it stop at once.
    def on_signal(self, signum, _frame) -> None:
        self.stop_requested += 1
        name = signal.Signals(signum).name
        if self.proc is not None and self.proc.poll() is None:
            log(f"{name}: asking the trainer to save and stop" + (" now" if self.stop_requested > 1 else ""), self.f)
            self.proc.send_signal(signal.SIGINT)
        else:
            log(f"{name}: stopping", self.f)
        if self.stop_requested > 2:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.kill()
            raise KeyboardInterrupt

    def keep_awake(self) -> None:
        if shutil.which("caffeinate"):
            # -i: no idle sleep, -s: no system sleep on AC power; -w: until this process exits.
            subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log("caffeinate is keeping the Mac awake while this runs (display may sleep; keep the lid open)", self.f)

    def run(self) -> int:
        a, f = self.args, self.f
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)  # closing the terminal window does not stop the night
        if a.report_only:
            self.report(final_eval=not a.no_eval)
            return 0
        start = time.time()
        deadline = start + a.hours * 3600
        log(f"=== overnight session {self.session}: {a.hours:g} h into {self.run_dir}", f)

        # 1. Machine.
        mem, cores = memory_gb(), perf_cores()
        tv, mps = torch_info()
        log(f"machine: {chip()}, {mem:.0f} GB memory, {cores} performance cores, {platform.system()} {platform.release()}", f)
        log(f"python {platform.python_version()}, torch {tv}, MPS {'available' if mps else 'not available'}", f)
        free_gb = shutil.disk_usage(self.run_dir).free / 2**30
        if free_gb < 5:
            log(f"only {free_gb:.1f} GB of disk free; checkpoints need ~1 GB over a night. Free some space first.", f)
            return 2
        if on_battery():
            log("WARNING: on battery. Plug in: caffeinate cannot hold off sleep on battery and a long run drains it.", f)
        heavy, light = other_jobs()
        if heavy and not a.force:
            log("these jobs are already running and would compete for the CPU/GPU:\n  " + "\n  ".join(heavy), f)
            log("wait for them to finish, or pass --force to run anyway", f)
            return 3
        if light:
            log("note: running alongside (fine, it will just be a little slower):\n  " + "\n  ".join(light), f)

        # 2. Settings.
        latest = self.run_dir / "checkpoints" / "latest.pt"
        prev_cfg = json.loads((self.run_dir / "cfg.json").read_text()) if (self.run_dir / "cfg.json").exists() else {}
        model = a.model or prev_cfg.get("model") or "small"
        if latest.exists() and prev_cfg.get("model") and a.model and a.model != prev_cfg["model"]:
            log(f"{self.run_dir} already holds a '{prev_cfg['model']}' model; it will keep training that "
                f"(use a new --run-dir for '{a.model}')", f)
            model = prev_cfg["model"]
        if a.minibatch:
            minibatch = a.minibatch
        elif model == "base":
            minibatch = 1024 if mem < 24 else 2048
        else:
            minibatch = 1024 if mem < 12 else 2048
        s = {"model": model, "n_envs": a.n_envs, "rollout_steps": a.rollout_steps, "minibatch": minibatch}
        log(f"settings: model {model}, {a.n_envs} envs x {a.rollout_steps} steps, minibatch {minibatch}"
            + (f" (resuming {latest})" if latest.exists() else " (new run)"), f)

        # 3. Device.
        devices = []
        if a.device in ("cpu", "mps"):
            devices = [a.device]
        else:
            devices = ["cpu"]
            if mps and torch_at_least(tv, 2, 5):
                devices.append("mps")
            elif mps:
                log(f"torch {tv} is too old for vgczero on MPS (needs 2.5+); using the CPU", f)
        if a.device == "mps" and not mps:
            log("--device mps but MPS is not available here", f)
            return 4
        speeds = {}
        if len(devices) > 1 or not a.no_probe:
            for d in devices:
                if self.stop_requested:
                    break
                p = probe(self.run_dir, s, d, a.probe_updates, cores, f)
                if p:
                    speeds[d] = p
        if self.stop_requested:
            log("stopped before training started", f)
            return 130
        if speeds:
            device = max(speeds, key=lambda d: speeds[d]["games_per_s"])
        elif len(devices) == 1 and a.no_probe:
            device = devices[0]
        else:
            log("every probe failed; see the output above", f)
            return 5
        spu = speeds.get(device, {}).get("s_per_update")
        eval_every = a.eval_every or (max(5, min(200, round(1800 / spu))) if spu else 25)
        save_every = max(5, min(50, round(600 / spu))) if spu else 10
        if spu:
            gps = speeds[device]["games_per_s"]
            hours_left = (deadline - time.time()) / 3600
            log(f"using {device}: about {gps * 3600 * hours_left / 1e3:,.0f}k games in the {hours_left:.1f} h left; "
                f"eval vs greedy every {eval_every} updates (~{eval_every * spu / 60:.0f} min)", f)
        else:
            log(f"using {device}; eval vs greedy every {eval_every} updates", f)

        # Keep the starting model so the morning report can show tonight's improvement.
        if latest.exists():
            shutil.copy2(latest, self.run_dir / "checkpoints" / f"start_{self.session}.pt")

        # 4. Train, restarting on crashes.
        self.keep_awake()
        extra = ["--eval-every", str(eval_every), "--eval-games", str(a.eval_games), "--save-every", str(save_every)]
        failures = {"cpu": 0, "mps": 0}
        restarts = 0
        lines_before = len(read_jsonl(self.run_dir / "metrics.jsonl"))
        while not self.stop_requested:
            left = deadline - time.time()
            if left < 120:
                break
            cmd = train_cmd(self.run_dir, s, device, extra + ["--hours", f"{left / 3600:.4f}"])
            log(f"training on {device} for {left / 3600:.2f} h (log: {self.run_dir / 'train.log'})", f)
            with open(self.run_dir / "train.log", "a") as tl:
                tl.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
                self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                             env=child_env(device, cores), start_new_session=True)
                tail: list[str] = []
                for line in self.proc.stdout:
                    tl.write(line)
                    tl.flush()
                    tail = (tail + [line.rstrip()])[-25:]
                    if not a.quiet:
                        echo(line.rstrip("\n"))
                rc = self.proc.wait()
            self.proc = None
            if rc == 0 or self.stop_requested:
                break
            failures[device] += 1
            restarts += 1
            log(f"trainer exited with code {rc} (restart {restarts}); last lines:\n  " + "\n  ".join(tail[-12:]), f)
            if restarts > a.max_restarts:
                log("too many restarts; giving up for tonight", f)
                break
            if device == "mps" and failures["mps"] >= 2:
                device = "cpu"
                log("MPS failed twice; continuing on the CPU", f)
            time.sleep(min(60, 10 * restarts))
        session_rows = read_jsonl(self.run_dir / "metrics.jsonl")[lines_before:]
        log(f"training done: {len(session_rows)} updates, {sum(r.get('games', 0) for r in session_rows):,} games "
            f"in {(time.time() - start) / 3600:.2f} h", f)
        self.report(final_eval=not a.no_eval, session_rows=session_rows, device=device, speeds=speeds,
                    hours=(time.time() - start) / 3600, restarts=restarts)
        return 0

    # ---- morning report ---------------------------------------------------------------

    def evaluate(self, a: str, b: str, games: int) -> str:
        env = child_env("cpu", perf_cores())
        r = subprocess.run([PY, "-m", "vgczero.evaluate", a, b, "--games", str(games), "--device", "cpu"],
                           cwd=ROOT, capture_output=True, text=True, env={**env, "PYTHONPATH": str(ROOT / "python")})
        out = (r.stdout.strip().splitlines() or ["?"])[-1]
        if r.returncode != 0:
            return "failed: " + ((r.stderr.strip().splitlines() or ["?"])[-1])
        # The result looks like "0.612 [0.581, 0.642] over 1000 games (3 draws, 8.1 turns)";
        # match it directly because the paths before it can contain spaces.
        m = re.search(r"\d\.\d+ \[[\d.]+, [\d.]+\] over \d+ games \([^)]*\)", out)
        return m.group(0) if m else out

    def report(self, final_eval: bool = True, session_rows=None, device=None, speeds=None, hours=None, restarts=0) -> None:
        a, rd = self.args, self.run_dir
        latest = rd / "checkpoints" / "latest.pt"
        if not latest.exists():
            log("no checkpoint yet; nothing to report", self.f)
            return
        metrics = read_jsonl(rd / "metrics.jsonl")
        evals = read_jsonl(rd / "eval.jsonl")
        cfg = json.loads((rd / "cfg.json").read_text()) if (rd / "cfg.json").exists() else {}
        lines = [f"# vgczero overnight report ({time.strftime('%Y-%m-%d %H:%M')})", ""]
        lines.append(f"Run: `{rd}`, model `{cfg.get('model', '?')}`, minibatch {cfg.get('minibatch', '?')}, "
                     f"machine: {chip()}, {memory_gb():.0f} GB.")
        if session_rows is not None:
            g = sum(r.get("games", 0) for r in session_rows)
            secs = sum(r.get("collect_s", 0) + r.get("update_s", 0) for r in session_rows)
            lines += ["", "## Tonight", "",
                      f"* {hours:.2f} h on {device}, {len(session_rows)} updates, {g:,} self-play games "
                      f"({g / max(secs, 1e-9):.1f} games/s while training)"
                      + (f", {restarts} restart(s)" if restarts else "") + "."]
            if speeds:
                lines.append("* Probe: " + ", ".join(f"{d} {p['games_per_s']:.1f} games/s" for d, p in speeds.items()) + ".")
            ill = sum(r.get("illegal", 0) for r in session_rows)
            if ill:
                lines.append(f"* WARNING: {ill} illegal actions (games forfeited); worth investigating.")
        if metrics:
            last = metrics[-1]
            lines += ["", "## Whole run", "",
                      f"* {last.get('update', '?')} updates, {last.get('games_total', 0):,} self-play games in total."]
        if evals:
            lines += ["", "## Win rate against the greedy baseline during training", "",
                      "| update | games played | win rate vs greedy |", "|---:|---:|---:|"]
            step = max(1, len(evals) // 20)
            shown = evals[::step] + ([evals[-1]] if (len(evals) - 1) % step else [])
            for e in shown:
                lines.append(f"| {e['update']} | {e.get('games_total', 0):,} | {e['vs_greedy']:.3f} |")
        if final_eval:
            log("final evaluation (a few minutes on the CPU) ...", self.f)
            lines += ["", "## Final evaluation", ""]
            lines.append(f"* vs greedy: {self.evaluate(str(latest), 'greedy', 1000)}")
            lines.append(f"* vs random: {self.evaluate(str(latest), 'random', 400)}")
            start = rd / "checkpoints" / f"start_{self.session}.pt"
            if start.exists():
                lines.append(f"* vs the model this session started from: {self.evaluate(str(latest), str(start), 1000)}")
        r = subprocess.run([PY, str(ROOT / "scripts" / "top_teams.py"), "--run-dir", str(rd), "--k", "10",
                            "--min-games", str(a.min_team_games)], cwd=ROOT, capture_output=True, text=True)
        rows = [ln for ln in r.stdout.splitlines() if " lb " in ln]
        if rows:
            lines += ["", f"## Best teams so far (Wilson lower bound of win rate, >= {a.min_team_games} games)", "", "```"]
            lines += rows + ["```", "", f"Saved to `{rd / 'top_teams.json'}` for `play.py --team-file`."]
        lines += ["", "## Next", "",
                  "* Run the same command again tonight: it resumes this run and keeps improving it.",
                  f"* Try the model: `python scripts/matchup.py --checkpoint {latest} --a MC2001 --b MC1000`.",
                  "* Full logs: `train.log`, `metrics.jsonl`, `eval.jsonl`, `overnight.log` in the run directory."]
        text = "\n".join(lines) + "\n"
        (rd / "REPORT.md").write_text(text)
        echo("\n" + text)
        log(f"report written to {rd / 'REPORT.md'}", self.f)


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
    ap.add_argument("--no-probe", action="store_true", help="with --device cpu/mps: skip the speed probe")
    ap.add_argument("--eval-every", type=int, default=0, help="updates between greedy evals (default: ~30 min)")
    ap.add_argument("--eval-games", type=int, default=400)
    ap.add_argument("--min-team-games", type=int, default=30)
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="run even if antonius1's heavy jobs are running")
    ap.add_argument("--quiet", action="store_true", help="only write trainer output to train.log")
    ap.add_argument("--no-eval", action="store_true", help="skip the final evaluation in the report")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    return Runner(args).run()


if __name__ == "__main__":
    sys.exit(main())
