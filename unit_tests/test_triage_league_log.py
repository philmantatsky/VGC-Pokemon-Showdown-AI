"""training/triage_league_log.py: the SB3-table parse, the stem mapping with
the finalist pick, and the tensorboard read that covers a paused-and-resumed
run (two events files in one run directory, keyed by step)."""

from __future__ import annotations

from pathlib import Path

from training.triage_league_log import (
    SAVE_INTERVAL,
    finalists,
    parse_probes,
    probes_from_tensorboard,
)

RESUME = 12779520


def _table(heuristic: float, bc: float) -> str:
    return (
        "-----------------------------------\n"
        f"| eval/bc            | {bc:<8} |\n"
        f"| eval/heuristic     | {heuristic:<8} |\n"
        "| rollout/ep_rew_mean | 0.5     |\n"
        "-----------------------------------\n"
    )


def _write_events(
    run_dir: Path, rows: list[tuple[int, float, float]], suffix: str
) -> None:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(run_dir), filename_suffix=suffix)
    for step, heuristic, bc in rows:
        writer.add_scalar("eval/heuristic", heuristic, step)
        writer.add_scalar("eval/bc", bc, step)
    writer.close()


def _touch_saves(save_dir: Path, count: int) -> None:
    save_dir.mkdir(parents=True)
    for i in range(count):
        (save_dir / f"{RESUME + (i + 1) * SAVE_INTERVAL}.zip").write_bytes(b"")


def test_parse_probes_pairs_tables_in_order() -> None:
    text = _table(0.61, 0.70) + "noise\n" + _table(0.66, 0.64)
    assert parse_probes(text) == [(0.61, 0.70), (0.66, 0.64)]


def test_finalists_best_probe_sum_and_final(tmp_path: Path) -> None:
    saves = tmp_path / "saves"
    _touch_saves(saves, 3)
    probes = [(0.61, 0.70), (0.66, 0.67), (0.59, 0.72)]
    picks = finalists(probes, RESUME, saves)
    assert [p.name for p in picks] == [
        f"{RESUME + 2 * SAVE_INTERVAL}.zip",
        f"{RESUME + 3 * SAVE_INTERVAL}.zip",
    ]
    # the final save is also the best: one finalist, not two copies
    assert len(finalists([(0.5, 0.5), (0.6, 0.6), (0.7, 0.7)], RESUME, saves)) == 1


def test_finalists_skip_missing_saves(tmp_path: Path) -> None:
    saves = tmp_path / "saves"
    _touch_saves(saves, 2)
    probes = [(0.61, 0.70), (0.66, 0.67), (0.90, 0.90)]  # third probe, no save yet
    picks = finalists(probes, RESUME, saves)
    assert [p.name for p in picks] == [f"{RESUME + 2 * SAVE_INTERVAL}.zip"]


def test_tensorboard_probes_span_a_resumed_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "logs" / "seed1_0"
    before = [(RESUME + (i + 1) * SAVE_INTERVAL, 0.6 + i / 100, 0.7) for i in range(4)]
    after = [(RESUME + (i + 1) * SAVE_INTERVAL, 0.8, 0.65) for i in range(4, 6)]
    _write_events(run_dir, before, ".first")
    _write_events(run_dir, after, ".resume")
    probes = probes_from_tensorboard(tmp_path)
    assert len(probes) == 6
    assert [round(h, 2) for h, _ in probes] == [0.6, 0.61, 0.62, 0.63, 0.8, 0.8]
    assert [round(b, 2) for _, b in probes] == [0.7] * 4 + [0.65] * 2


def test_tensorboard_probes_dedupe_a_replayed_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "logs" / "seed1_0"
    step = RESUME + SAVE_INTERVAL
    _write_events(run_dir, [(step, 0.61, 0.70)], ".first")
    replay = [(step, 0.61, 0.70), (step + SAVE_INTERVAL, 0.66, 0.64)]
    _write_events(run_dir, replay, ".resume")
    probes = [(round(h, 2), round(b, 2)) for h, b in probes_from_tensorboard(tmp_path)]
    assert probes == [(0.61, 0.70), (0.66, 0.64)]


def test_tensorboard_probes_empty_without_events(tmp_path: Path) -> None:
    assert probes_from_tensorboard(tmp_path) == []
