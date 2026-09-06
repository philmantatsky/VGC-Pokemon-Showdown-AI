"""Meta-game payoff matrix and Nash opponent weights for a league round.

Why (2026-09-06): league rounds 3 and 3b trained against a hand-weighted pool
and overfit to the newest adversary (the classic PSRO failure). The textbook
fix measures the population against itself and trains against the
equilibrium mixture of that meta-game instead of a guess.

Rows: our-side policies piloting the fixed team (teams/reg_mb/our_team.txt).
Cols: opponent policies piloting the pool teams (the league team weights).
Cell (i, j): win rate of row i vs column j over --n battles, hidden sheets,
via ``evaluation/eval_counterfactual.py --baseline-only`` (one policy vs one
opponent). Columns listed under ``stochastic_cols`` (behavior clones) are
sampled rather than played deterministically.

The zero-sum solve returns the COLUMN player's equilibrium ``y*``: the
opponent mixture that is hardest for the row population as a whole, which
is the PSRO meta-strategy the next best response should train against. An
FP league samples pool files uniformly, so multiplicities are
``round(y* * copies)``; the tool prints the suggested copy table.

Config JSON:
  {"rows": {"deployed": "results_league/league_champion.zip", ...},
   "cols": {"exploiter": "results_exploiter/.../17694720.zip", ...},
   "stochastic_cols": ["bc_mix_A"], "n": 300, "seed": 83,
   "out": "results_meta_game/round4", "copies": 12}

Usage (from the repo root; restarts the eval server on --port itself):
  .venv/bin/python training/meta_game.py --config <config.json> --port 7600
  .venv/bin/python training/meta_game.py --config <config.json> --solve-only
      (re-solve from the cells already on disk)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys as _sys
import time
from pathlib import Path as _Path

import numpy as np

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

ROOT = _Path(__file__).resolve().parents[1]


def solve_zero_sum(matrix: np.ndarray) -> dict:
    """Equilibrium of the zero-sum game whose ROW player maximises ``matrix``.

    Returns the row strategy, the column strategy and the game value. Uses two
    linear programs (scipy HiGHS); a constant matrix yields uniform strategies.
    """
    from scipy.optimize import linprog

    m = np.asarray(matrix, dtype=float)
    rows, cols = m.shape
    # Row player: max v  s.t.  m^T x >= v, sum x = 1, x >= 0.
    c = np.zeros(rows + 1)
    c[-1] = -1.0
    a_ub = np.hstack([-m.T, np.ones((cols, 1))])
    b_ub = np.zeros(cols)
    a_eq = np.hstack([np.ones((1, rows)), np.zeros((1, 1))])
    res_row = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=[1.0],
        bounds=[(0, None)] * rows + [(None, None)],
        method="highs",
    )
    # Column player: min v  s.t.  m y <= v, sum y = 1, y >= 0.
    c2 = np.zeros(cols + 1)
    c2[-1] = 1.0
    a_ub2 = np.hstack([m, -np.ones((rows, 1))])
    b_ub2 = np.zeros(rows)
    a_eq2 = np.hstack([np.ones((1, cols)), np.zeros((1, 1))])
    res_col = linprog(
        c2,
        A_ub=a_ub2,
        b_ub=b_ub2,
        A_eq=a_eq2,
        b_eq=[1.0],
        bounds=[(0, None)] * cols + [(None, None)],
        method="highs",
    )
    if not (res_row.success and res_col.success):
        raise RuntimeError(f"LP failed: {res_row.message} / {res_col.message}")
    x = np.clip(res_row.x[:rows], 0, None)
    y = np.clip(res_col.x[:cols], 0, None)
    x = x / x.sum()
    y = y / y.sum()
    return {"row": x.tolist(), "col": y.tolist(), "value": float(res_row.x[-1])}


def multiplicities(weights: list[float], copies: int) -> list[int]:
    """Integer copy counts that sum to ``copies`` (largest-remainder rounding)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / len(w))
    raw = w * copies
    base = np.floor(raw).astype(int)
    remainder = copies - int(base.sum())
    order = np.argsort(-(raw - base))
    for i in order[:remainder]:
        base[i] += 1
    return base.tolist()


def ensure_server(port: int) -> None:
    listeners = subprocess.run(
        ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
    ).stdout.split()
    for pid in listeners:
        subprocess.run(["kill", pid], check=False)
    if listeners:
        time.sleep(3)
    subprocess.Popen(
        ["node", "pokemon-showdown", "start", str(port), "--no-security"],
        cwd=ROOT / "pokemon-showdown",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        up = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True
        )
        if up.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError(f"eval server on {port} did not start")


def cell_path(out: _Path, row: str, col: str) -> _Path:
    return out / "cells" / f"{row}__vs__{col}.json"


def run_cell(
    row_ckpt: str,
    col_ckpt: str,
    stochastic: bool,
    n: int,
    seed: int,
    port: int,
    workers: int,
    output: _Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _sys.executable,
        str(ROOT / "evaluation" / "eval_counterfactual.py"),
        "--baseline",
        row_ckpt,
        "--candidate",
        row_ckpt,
        "--baseline-only",
        "--opponent-checkpoint",
        col_ckpt,
        *(["--opponent-stochastic"] if stochastic else []),
        "--n-battles",
        str(n),
        "--hidden-sheets",
        "--seed",
        str(seed),
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--output",
        str(output),
    ]
    print("  " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(f"cell failed: {output}")


def read_cell(path: _Path) -> float:
    data = json.loads(path.read_text())
    arm = data["arms"]["champion_policy"]
    return arm["wins"] / max(int(arm["battles"]), 1)


def assemble(config: dict, out: _Path) -> tuple[np.ndarray, list[str], list[str]]:
    rows, cols = list(config["rows"]), list(config["cols"])
    m = np.full((len(rows), len(cols)), np.nan)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            p = cell_path(out, r, c)
            if p.exists():
                m[i, j] = read_cell(p)
    return m, rows, cols


def solve_and_write(config: dict, out: _Path) -> dict:
    m, rows, cols = assemble(config, out)
    if np.isnan(m).any():
        missing = [
            (rows[i], cols[j]) for i, j in zip(*np.where(np.isnan(m)), strict=True)
        ]
        raise RuntimeError(f"missing cells: {missing}")
    eq = solve_zero_sum(m)
    copies = int(config.get("copies", 12))
    report = {
        "rows": rows,
        "cols": cols,
        "n_per_cell": config.get("n"),
        "payoff_row_win_rate": m.tolist(),
        "row_equilibrium": dict(zip(rows, eq["row"], strict=True)),
        "col_equilibrium": dict(zip(cols, eq["col"], strict=True)),
        "value_row_win_rate": eq["value"],
        "copies": copies,
        "suggested_multiplicity": dict(
            zip(cols, multiplicities(eq["col"], copies), strict=True)
        ),
    }
    (out / "meta_game.json").write_text(json.dumps(report, indent=2))
    return report


def render(report: dict) -> str:
    rows, cols, m = report["rows"], report["cols"], report["payoff_row_win_rate"]
    width = max(len(c) for c in cols + rows) + 2
    lines = ["row-player win rate (our side pilots the fixed team):"]
    lines.append(" " * width + "".join(f"{c:>{width}}" for c in cols))
    for r, vals in zip(rows, m, strict=True):
        lines.append(f"{r:<{width}}" + "".join(f"{100 * v:>{width}.1f}" for v in vals))
    value = 100 * report["value_row_win_rate"]
    lines.append(f"game value (row win rate at equilibrium): {value:.1f}%")
    lines.append("column equilibrium y* = opponent mixture to train against:")
    for c in cols:
        lines.append(
            f"  {c:<{width}} {100 * report['col_equilibrium'][c]:5.1f}%  "
            f"-> {report['suggested_multiplicity'][c]} of {report['copies']} copies"
        )
    lines.append("row equilibrium x* (which of our policies the equilibrium mixes):")
    for r in rows:
        lines.append(f"  {r:<{width}} {100 * report['row_equilibrium'][r]:5.1f}%")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, type=_Path)
    ap.add_argument("--port", type=int, default=7600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--solve-only", action="store_true")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    out = ROOT / config["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    if not args.solve_only:
        ensure_server(args.port)
        stochastic = set(config.get("stochastic_cols", []))
        n, seed = int(config.get("n", 300)), int(config.get("seed", 83))
        for r, r_ckpt in config["rows"].items():
            for c, c_ckpt in config["cols"].items():
                p = cell_path(out, r, c)
                if p.exists():
                    print(f"[cell] {r} vs {c}: on disk, skipping", flush=True)
                    continue
                print(f"[cell] {r} vs {c}: {n} battles", flush=True)
                run_cell(
                    r_ckpt, c_ckpt, c in stochastic, n, seed, args.port, args.workers, p
                )
                print(
                    f"CELL_DONE {r} vs {c} row_win_rate={read_cell(p):.3f}", flush=True
                )
    report = solve_and_write(config, out)
    print(render(report))
    print(f"META_GAME_COMPLETE {out / 'meta_game.json'}")


if __name__ == "__main__":
    main()
