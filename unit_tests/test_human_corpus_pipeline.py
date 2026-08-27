"""Stage-B: log prefiltering and the eval-only quarantine.

The bo1 top-500 file looked convertible while 94% of it silently failed deep in
move parsing (no team sheets); prefilter_reason names and counts those skips.
The quarantine keeps the held-out human-imitation arm out of every training
mix, which is the property that makes its gate meaningful.
"""

import json
from pathlib import Path

import pytest

from vgc_bench.logs2trajs import prefilter_reason
from vgc_bench.src.utils import refuse_eval_only_checkpoint

SHEETED = (
    "|player|p1|Alice|169|1200\n|player|p2|Bob|265|1180\n"
    "|showteam|p1|Pikachu||LightBall|static|thunderbolt|...\n"
    "|showteam|p2|Eevee||Everstone|runaway|tackle|...\n"
    "|turn|1\n|win|Alice\n"
)


class TestPrefilterReason:
    def test_sheeted_complete_log_passes(self):
        assert prefilter_reason(SHEETED) is None

    def test_missing_showteam_named(self):
        assert prefilter_reason(SHEETED.replace("|showteam|p2|", "|x|")) == (
            "no_showteam"
        )

    def test_missing_win_named(self):
        assert prefilter_reason(SHEETED.replace("|win|Alice\n", "")) == "no_win"

    def test_missing_turn_named(self):
        assert prefilter_reason(SHEETED.replace("|turn|1\n", "")) == "no_turn_1"

    def test_zoroark_named(self):
        assert prefilter_reason(SHEETED + "|poke|p1|Zoroark, L50|\n") == "zoroark"


class TestEvalOnlyQuarantine:
    def test_eval_only_sidecar_refused(self, tmp_path: Path):
        ckpt = tmp_path / "arm.zip"
        ckpt.write_bytes(b"weights")
        Path(str(ckpt) + ".metadata.json").write_text(
            json.dumps({"role": "eval_only"})
        )
        with pytest.raises(SystemExit, match="eval_only"):
            refuse_eval_only_checkpoint(ckpt)

    def test_other_roles_and_missing_sidecar_pass(self, tmp_path: Path):
        ckpt = tmp_path / "ok.zip"
        ckpt.write_bytes(b"weights")
        refuse_eval_only_checkpoint(ckpt)  # no sidecar
        Path(str(ckpt) + ".metadata.json").write_text(
            json.dumps({"role": "training_opponent"})
        )
        refuse_eval_only_checkpoint(ckpt)
