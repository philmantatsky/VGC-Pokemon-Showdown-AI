"""Oracle ceiling of the existing opponent reranker (design reading R4).

Question: if the bot had known the opponent's action PERFECTLY, how many of its
logged ladder move decisions would the existing consumer of opponent predictions
(``opponent_reranker.rerank_candidates``) have changed, and were the changed
picks better against what the opponent really did? Perfect foresight is an upper
bound on any learned predictor through that consumer, so this test can only
falsify (design: OPPONENT_PREDICTOR.md, "PRE-REGISTERED readings", R4).

Method. Every move decision in ``ladder_replays_mc*/decisions.jsonl`` (team
preview and forced-replacement records removed) is rebuilt from the saved
player-view replay of its game: a poke-env ``DoubleBattle`` driven line by line
to the ``|turn|N`` of the decision, our six registered first from the team file
named in that directory's ``run_config.json``, with the Champions stat formula.
Nothing heavier than the CPU damage calculator and the three small prediction
nets the bot itself runs on CPU is used: no brain, no simulator bridge, no
server, no MPS. The logged candidates (the audit keeps the first eight after
the whole pipeline) are put back in the order the reranker received them and
the reranker is called again:

  A  logged      the bot's own predictions, recomputed by its own feeding code
                 (``PolicyPlayer.opponent_move_predictions`` /
                 ``opponent_switch_predictions`` on a bare instance). Sanity arm:
                 it must reproduce the played pick and the logged reranker
                 report. The reading is refused below ``MIN_AGREEMENT``.
  B  oracle      for each opposing slot whose true action is known, a CERTAIN
                 prediction: probability 1, reliability 1.0, which also
                 satisfies the reranker's switch-evidence gate. Slots whose
                 action is not known keep the logged prediction at its logged
                 weight. The reranker's own constants (0.30 eligibility ratio,
                 0.65 weight, 0.65 switch cap) are untouched. THE R4 ARM.
  C  gated       the same true actions, but with every gate at its normal
                 value: the logged reliability (revealed moves / 4) on each
                 slot, and no prediction at all where the bot's feeding code
                 gave none. B minus C = what the gates block.
  D  wide        B with the eligibility ratio opened to 0 as well (every logged
                 live candidate may move). Sensitivity only.
  N  control     (post hoc) B's gates, but a known slot tells the reranker
                 nothing. A flip N makes too is the opening, not the foresight.

Definitions fixed on 2026-10-04 BEFORE any oracle arm (B, C, D) was run; only
arm A's agreement with the log had been looked at:

* Move decision: a record with candidates, turn > 0, and at least one candidate
  order that is neither a switch nor a pass (the critic's rule; 2,631 in 403
  battles on the 17 directories of that day). All of them stay in the
  denominator. A record that the server refused and the bot replaced in the same
  turn is ``superseded``: counted, flagged, and reported both ways.
* True action: ``oppmodel.events.read_turn_actions`` for the turn. Known = a
  voluntary switch, or a move line (for an aimed damaging move, with its
  target). The target is where the move LANDED. A slot with no visible action
  (fainted first, flinch, sleep, forfeit, overridden click, blank target) is
  unknown.
* Flip: the arm's final pick (reranker, then the directory's own
  sticky-correction setting) differs from the played pair.
* R4 flip share: arm B flips / ALL move decisions. A decision that cannot be
  rebuilt counts as not flipped; the share among rebuilt decisions and an upper
  bound that counts every such decision as a flip are reported beside it.
* Hindsight score of a pair (PRIMARY, the one the reading uses): the reranker's
  own two terms evaluated against the true actions,
  ``opponent_reranker._outgoing_damage(battle, pair, true switches)`` minus
  ``opponent_reranker._incoming_damage(battle, pair, true moves)``, with a
  certain prediction on every known slot and an EMPTY one on every unknown slot
  (so a censored slot contributes zero to both pairs). Paired difference =
  score(oracle pick) - score(played pick), on flips only; mean with a
  game-clustered bootstrap 95% percentile interval; share better / equal / worse.
* SECONDARY score, reported because the primary is the very quantity arm B
  optimises (a positive primary mean is expected by construction):
  ``exchange`` = expected damage our pair deals to the Pokemon that truly stood
  in each opposing slot (the switch-in after a true voluntary switch, nothing
  into a true Protect, capped at the HP left; ``opponent_reranker._damage``)
  minus the same ``_incoming_damage`` term.
* Reading: this consumer CANNOT use any predictor if the R4 flip share is below
  5%, or if the primary mean difference has a lower interval bound <= 0. The
  secondary score and arms C / D only qualify the statement.

Added after the first full run, and labelled post hoc wherever they appear:
control arm N; the split of arm B's flips into "arm A does not reproduce the log
there", "N makes the same pick" and "needs the true action"; both arms again
with sticky guard corrections on in every directory (today's pipeline); the
share where both opposing actions are visible. One correction was made to the
secondary score at the same time: a pair that attacks a switch-in the bot had
never seen is left out (the first version counted that attack as zero damage).
None of this enters the reading.

What the hindsight score cannot see (also written to the README): turn order and
knock-outs before a move is made, Fake Out, redirection that our own pick would
have changed, the value of status / speed control / set-up, our own positioning
after a switch, spread reduction, items and abilities not yet revealed (the
opponent's stats are the bot's synthetic spread), anything after the turn, and
therefore win probability. A censored slot is dropped, not imputed.

Outputs under ``results_oppmodel/oracle/``: ``summary.json``,
``decisions.jsonl`` (one row per move decision) and ``README.md`` (rendered from
the summary). Usage, from the repo root, one core, about fifteen seconds:

    nice -n 19 .venv/bin/python evaluation/oppmodel_oracle.py
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from poke_env.battle import DoubleBattle, Move, MoveCategory, Pokemon
from poke_env.data import GenData, to_id_str
from poke_env.teambuilder import Teambuilder

from vgc_bench.src import opponent_reranker as R
from vgc_bench.src import pokeenv_patches
from vgc_bench.src import vgc_knowledge as K
from vgc_bench.src.guards import Candidate
from vgc_bench.src.oppmodel import events as E
from vgc_bench.src.opponent_preview import (
    BattlePlanState,
    OpponentBelief,
    PreviewPredictor,
)
from vgc_bench.src.opponent_tactics import MovePrediction, SwitchPrediction
from vgc_bench.src.policy_player import PolicyPlayer, keep_guard_correction
from vgc_bench.src.tempo_reranker import score_candidates

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_oppmodel" / "oracle"
DEFAULT_GLOB = "ladder_replays_mc*"

RECORD_MOVE = "move"
RECORD_PREVIEW = "team_preview"
RECORD_FORCED = "forced_replacement"
RECORD_OTHER = "no_candidates"

ARM_LOGGED = "A"
ARM_ORACLE = "B"
ARM_GATED = "C"
ARM_WIDE = "D"
ARM_NULL = "N"
ORACLE_ARMS: tuple[str, ...] = (ARM_ORACLE, ARM_GATED, ARM_WIDE, ARM_NULL)

SHEET_OPEN = "open"
SHEET_OPEN_INFERRED = "open_inferred"
SHEET_CLOSED = "closed"
SHEET_UNKNOWN = "unknown"

TRUTH_MOVE_ONLY = "move_only"
TRUTH_SWITCH = "switch_involved"
TRUTH_NONE = "none_known"

FLIP_THRESHOLD = 0.05  # R4: fewer flips than this and the consumer is closed
MIN_AGREEMENT = 0.95  # arm A must reproduce the played pick this often
SWITCH_EVIDENCE_GATE = 0.999  # opponent_reranker.rerank_candidates, ``sets_known``
ELIGIBILITY_RATIO = 0.30  # the same function's default ``min_policy_ratio``
SCORE_TOLERANCE = 1e-6
EQUAL_TOLERANCE = 1e-9
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261004

# The critic's count on the 17 directories of 2026-10-04 (scratch
# critic/audit_and_probes.py part A); the summary states any difference.
CRITIC_COUNT = {
    "records_with_candidates": 3433,
    RECORD_PREVIEW: 250,
    RECORD_FORCED: 552,
    RECORD_MOVE: 2631,
    "battles": 403,
}

# Environment switches the reranker and its feeding code read at call time.
ENV_SWITCHES: tuple[str, ...] = (
    "VGC_OPPONENT_RERANK_WEIGHT",
    "VGC_TEMPO_RERANK_WEIGHT",
    "VGC_OPPONENT_RERANK_MIN_RATIO",
    "VGC_PREDICTED_SURVIVAL",
    "VGC_PRIOR_RELIABILITY_FLOOR",
    "VGC_STRATEGIC_CANDIDATES",
)

# poke-env's Player consumes these before ``parse_message`` ever sees them.
_PLAYER_LEVEL_KINDS = frozenset(
    {
        "t:",
        "expire",
        "uhtmlchange",
        "tempnotify",
        "tempnotifyoff",
        "uhtml",
        "request",
        "error",
        "bigerror",
        "showteam",
    }
)
_REPLAY_NAME = re.compile(r"^(?P<player>.+?) - (?P<room>battle-[a-z0-9-]+)$")
_PUBLIC_ID = re.compile(r"battle-([a-z0-9]+-\d+)")
_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
# events' target classes -> the names MovePredictor.TARGET_NAMES uses. ``foe_a``
# means the same in both: slot a of the side the mover faces, i.e. OUR slot a.
_TARGET_NAMES = {
    E.TARGET_FOE_A: "foe_a",
    E.TARGET_FOE_B: "foe_b",
    E.TARGET_ALLY: "ally",
    E.TARGET_SELF: "self",
    E.TARGET_AUTO: "field",
}
_LOGGER = logging.getLogger("oppmodel_oracle")
_LOGGER.addHandler(logging.NullHandler())
_LOGGER.propagate = False

EMPTY_MOVES = MovePrediction((), (), (), reliability=1.0)
NO_SWITCH = SwitchPrediction(0.0, ())


# --- decision records ---------------------------------------------------------


def classify_record(row: Mapping[str, Any]) -> str:
    """Which kind of audit record ``row`` is (one of the ``RECORD_*`` names).

    Same rule as the critic's recount: no candidates -> not a decision; turn 0 ->
    team preview; every candidate order a switch or a pass -> a forced
    replacement request; anything else is a real move decision.
    """
    candidates = row.get("candidates")
    if not candidates or not isinstance(candidates, list):
        return RECORD_OTHER
    try:
        turn = int(row.get("turn") or 0)
    except (TypeError, ValueError):
        return RECORD_OTHER
    if turn == 0:
        return RECORD_PREVIEW
    kinds: set[Any] = set()
    for candidate in candidates:
        orders = candidate.get("orders") if isinstance(candidate, Mapping) else None
        for order in orders or ():
            kinds.add(order.get("kind") if isinstance(order, Mapping) else None)
    if kinds <= {"switch", "pass", None}:
        return RECORD_FORCED
    return RECORD_MOVE


@dataclass
class Decision:
    """One logged move decision. ``line`` is its 0-based line in decisions.jsonl."""

    folder: str
    line: int
    room: str
    turn: int
    row: dict[str, Any]
    superseded: bool = False


def public_battle_id(room: str) -> str:
    """``format-number`` of a room tag, without a private room's access suffix."""
    match = _PUBLIC_ID.search(room)
    return match.group(1) if match else room


def read_decisions(
    lines: Iterable[str], folder: str = ""
) -> tuple[list[Decision], Counter[str], dict[str, bool]]:
    """Move decisions of one audit file, the record census, and sheet flags.

    The census counts every line by ``classify_record`` (plus ``bad_json``).
    Sheet flags are what the bot recorded at team preview, keyed by room: True
    for an open sheet (a ``preview_shadow.open_sheet`` flag or a stored
    ``their_sheet``), False for a recorded closed one. A move decision followed
    by another move decision of the same battle and turn is ``superseded``.
    """
    decisions: list[Decision] = []
    census: Counter[str] = Counter()
    sheets: dict[str, bool] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            census["bad_json"] += 1
            continue
        if not isinstance(row, dict):
            census["bad_json"] += 1
            continue
        room = str(row.get("battle", ""))
        shadow = row.get("preview_shadow")
        if isinstance(shadow, dict) and isinstance(shadow.get("open_sheet"), bool):
            sheets[room] = sheets.get(room, False) or shadow["open_sheet"]
        if row.get("their_sheet"):
            sheets[room] = True
        kind = classify_record(row)
        census[kind] += 1
        if kind == RECORD_MOVE:
            decisions.append(Decision(folder, index, room, int(row["turn"]), row))
    last: dict[tuple[str, int], int] = {}
    for decision in decisions:
        last[(decision.room, decision.turn)] = decision.line
    for decision in decisions:
        decision.superseded = last[(decision.room, decision.turn)] != decision.line
    return decisions, census, sheets


# --- the opponent's true action as a certain prediction ------------------------


def oracle_known(action: E.SlotAction | None) -> bool:
    """Whether the log shows this slot's action well enough to state it as fact.

    A voluntary switch always is. A move is, unless it is an aimed damaging
    move whose target was never logged (the consumer needs the slot it hit).
    """
    if action is None:
        return False
    if action.kind == E.KIND_SWITCH:
        return bool(action.switch_to)
    if action.kind != E.KIND_MOVE or not action.move:
        return False
    if action.target is not None:
        return True
    return not (E.is_damaging(action.move) and E.is_choosable_target(action.move))


def certain_predictions(
    action: E.SlotAction | None, reliability: float = 1.0
) -> tuple[MovePrediction, SwitchPrediction] | None:
    """The slot's true action in the shapes the reranker consumes, or None.

    A move becomes one move with probability 1 on the slot it landed on and
    switch probability 0; a voluntary switch becomes no move at all and switch
    probability 1 to the species that came in. ``reliability`` is 1.0 for the
    oracle arm and the logged value for the gated arm.
    """
    if action is None or not oracle_known(action):
        return None
    if action.kind == E.KIND_SWITCH:
        target = str(action.switch_to)
        return (
            MovePrediction((), (), (), reliability=reliability),
            SwitchPrediction(1.0, ((target, 1.0),)),
        )
    move = str(action.move)
    name = _TARGET_NAMES.get(action.target or "")
    if name is None:
        moves = MovePrediction(((move, 1.0),), (), (), reliability=reliability)
    else:
        moves = MovePrediction(
            ((move, 1.0),), ((name, 1.0),), ((move, name, 1.0),), reliability
        )
    return moves, NO_SWITCH


def truth_class(actions: Sequence[E.SlotAction | None]) -> str:
    """``switch_involved`` / ``move_only`` / ``none_known`` for a decision."""
    known = [
        action for action in actions if action is not None and oracle_known(action)
    ]
    if any(action.kind == E.KIND_SWITCH for action in known):
        return TRUTH_SWITCH
    return TRUTH_MOVE_ONLY if known else TRUTH_NONE


def prescaled(prediction: MovePrediction) -> MovePrediction:
    """The same incoming-damage evidence stated at reliability 1.0.

    ``_incoming_damage`` weighs a move by probability x reliability, so moving
    the reliability into the probabilities changes nothing there; it only stops
    this slot from closing the reranker's all-slots switch-evidence gate.
    """
    weight = min(1.0, max(0.0, prediction.reliability))
    return MovePrediction(
        tuple((move, probability * weight) for move, probability in prediction.moves),
        prediction.targets,
        tuple(
            (move, target, probability * weight)
            for move, target, probability in prediction.actions
        ),
        reliability=1.0,
    )


@dataclass
class OracleInputs:
    """Predictions for one arm, plus what had to be done to build them."""

    moves: tuple[MovePrediction, MovePrediction] | None
    switches: tuple[SwitchPrediction, SwitchPrediction] | None
    prescaled: bool = False


def oracle_inputs(
    truth: Sequence[E.SlotAction | None],
    logged_moves: tuple[MovePrediction, MovePrediction] | None,
    logged_switches: tuple[SwitchPrediction, SwitchPrediction] | None,
    *,
    gated: bool,
    content: bool = True,
) -> OracleInputs:
    """Arm B (``gated`` False) or arm C (``gated`` True) inputs for one decision.

    ``truth`` holds the true action of their slot a and slot b (None = nobody
    stood there or nothing could be read). Known slots get the certain
    prediction; unknown slots keep the logged one. ``content=False`` builds the
    control arm N: every gate as open as in arm B, but a known slot says
    nothing (no move, no switch), so only the opening itself is left.
    """
    if gated and logged_moves is None:
        return OracleInputs(None, None)
    known_switch = any(
        action is not None and oracle_known(action) and action.kind == E.KIND_SWITCH
        for action in truth
    )
    moves: list[MovePrediction] = []
    switches: list[SwitchPrediction] = []
    did_prescale = False
    for index in range(2):
        action = truth[index] if index < len(truth) else None
        logged = logged_moves[index] if logged_moves is not None else None
        reliability = logged.reliability if gated and logged is not None else 1.0
        certain = certain_predictions(action, reliability)
        if certain is not None:
            moves.append(certain[0] if content else EMPTY_MOVES)
            switches.append(certain[1] if content else NO_SWITCH)
            continue
        if logged is None:
            moves.append(EMPTY_MOVES)
            switches.append(NO_SWITCH)
            continue
        if not gated and known_switch and logged.reliability < SWITCH_EVIDENCE_GATE:
            # Arm B: an unknown slot must not close the gate on a switch the
            # other slot is known to have made. Its own evidence keeps its
            # logged weight, and its switch guess stays as inactive as it was.
            moves.append(prescaled(logged))
            switches.append(NO_SWITCH)
            did_prescale = True
            continue
        moves.append(logged)
        switches.append(
            logged_switches[index] if logged_switches is not None else NO_SWITCH
        )
    return OracleInputs((moves[0], moves[1]), (switches[0], switches[1]), did_prescale)


def truth_inputs(
    truth: Sequence[E.SlotAction | None],
) -> tuple[
    tuple[MovePrediction, MovePrediction], tuple[SwitchPrediction, SwitchPrediction]
]:
    """The true actions alone: certain where known, EMPTY where censored."""
    moves: list[MovePrediction] = []
    switches: list[SwitchPrediction] = []
    for index in range(2):
        certain = certain_predictions(truth[index] if index < len(truth) else None)
        moves.append(certain[0] if certain is not None else EMPTY_MOVES)
        switches.append(certain[1] if certain is not None else NO_SWITCH)
    return (moves[0], moves[1]), (switches[0], switches[1])


# --- game-clustered bootstrap ---------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A ratio of sums over clusters with its percentile bootstrap interval."""

    point: float | None
    low: float | None
    high: float | None
    clusters: int
    total: float
    resamples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "clusters": self.clusters,
            "n": self.total,
            "resamples": self.resamples,
        }


def cluster_bootstrap(
    numerators: Sequence[float],
    denominators: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> Interval:
    """sum(numerators) / sum(denominators), clusters resampled with replacement.

    One entry per cluster (game). A mean over observations is the case
    numerator = the cluster's sum, denominator = its count. Resamples whose
    denominator is zero are dropped and not counted in ``resamples``.
    """
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    if num.shape != den.shape or num.ndim != 1:
        raise ValueError("one numerator and one denominator per cluster")
    count = int(num.size)
    total = float(den.sum()) if count else 0.0
    if count == 0 or total <= 0:
        return Interval(None, None, None, count, total, 0)
    point = float(num.sum() / total)
    if resamples <= 0:
        return Interval(point, None, None, count, total, 0)
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    done = 0
    while done < resamples:
        batch = min(1000, resamples - done)
        picks = rng.integers(0, count, size=(batch, count))
        sums = num[picks].sum(axis=1)
        weights = den[picks].sum(axis=1)
        valid = weights > 0
        values.append(sums[valid] / weights[valid])
        done += batch
    drawn = np.concatenate(values) if values else np.empty(0)
    if drawn.size == 0:
        return Interval(point, None, None, count, total, 0)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(drawn, [tail, 1.0 - tail])
    return Interval(point, float(low), float(high), count, total, int(drawn.size))


def by_cluster(pairs: Iterable[tuple[str, float]]) -> tuple[list[float], list[float]]:
    """(per-cluster sums, per-cluster counts) of ``(cluster, value)`` pairs."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, float] = defaultdict(float)
    for cluster, value in pairs:
        sums[cluster] += float(value)
        counts[cluster] += 1.0
    keys = sorted(sums)
    return [sums[key] for key in keys], [counts[key] for key in keys]


# --- one directory's configuration ----------------------------------------------


@dataclass
class DirectoryConfig:
    """What a replay directory's run_config.json says the bot ran with."""

    name: str
    team: Path
    move_model: Path | None
    switch_model: Path | None
    preview_model: Path | None
    use_opponent: bool
    use_tempo: bool
    sticky: bool
    moveset_prior: bool
    set_prior_reg: str
    mixing: str
    varies: bool = False


_CONFIG_KEYS = (
    "our_team",
    "move_model",
    "switch_model",
    "preview_model",
    "opponent_aware",
    "tempo_aware",
    "sticky_corrections",
    "moveset_prior",
    "set_prior_reg",
    "mixing",
)


def read_run_config(folder: Path) -> DirectoryConfig | None:
    """The directory's recorded configuration (its last run), or None."""
    try:
        runs = json.loads((folder / "run_config.json").read_text())["runs"]
        material = dict(runs[-1]["material"])
        team = material["our_team"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None

    def path(key: str) -> Path | None:
        value = material.get(key)
        return ROOT / str(value) if value else None

    varies = any(
        {key: run.get("material", {}).get(key) for key in _CONFIG_KEYS}
        != {key: material.get(key) for key in _CONFIG_KEYS}
        for run in runs
    )
    return DirectoryConfig(
        name=folder.name,
        team=ROOT / str(team),
        move_model=path("move_model"),
        switch_model=path("switch_model"),
        preview_model=path("preview_model"),
        use_opponent=bool(material.get("opponent_aware", True)),
        use_tempo=bool(material.get("tempo_aware", True)),
        sticky=bool(material.get("sticky_corrections", False)),
        moveset_prior=bool(material.get("moveset_prior", True)),
        # Directories from before the key was recorded used the Reg M-B set
        # prior for every format: with it the logged opponent scores of those
        # two directories reproduce (50 of 58 and 69 of 114 reports), with the
        # Reg M-C file they do not (about 18 and 28).
        set_prior_reg=str(material.get("set_prior_reg") or "mb"),
        mixing=str(material.get("mixing", "off")),
        varies=varies,
    )


# --- rebuilding a position --------------------------------------------------------


def champions_stats(mon: Pokemon, sheet: Any) -> dict[str, int | None]:
    """Champions' formula: HP = base + EVs + 75, others (base + EVs + 20) x nature.

    This is what the server sends in the request; poke-env's teambuilder would
    apply the main-series formula instead.
    """
    nature = GenData.from_gen(9).natures.get(to_id_str(sheet.nature or ""), {})
    evs = list(sheet.evs or [0] * len(_STATS))
    stats: dict[str, int | None] = {}
    for stat, ev in zip(_STATS, evs):
        base = mon.base_stats[stat]
        if stat == "hp":
            stats[stat] = base + ev + 75
        else:
            stats[stat] = int((base + ev + 20) * nature.get(stat, 1))
    return stats


class Position:
    """One saved game rebuilt turn by turn, as far as the bot's log allows.

    Our six are registered first, in team-file order as the live request lists
    them, so switch actions 1-6 decode to the Pokemon they did on ladder. The
    opponent's side holds what the protocol revealed (percent HP, revealed
    moves): a saved page never carries a team sheet.
    """

    def __init__(self, tag: str, player: str, role: str, sets: Sequence[Any]) -> None:
        pokeenv_patches.install()
        self.battle = DoubleBattle(tag, player, _LOGGER, gen=9)
        self.battle._player_role = role
        self.own: list[tuple[Pokemon, Any]] = []
        for sheet in sets:  # sets first: the protocol's Mega formes must survive them
            species = sheet.species or sheet.nickname
            mon = self.battle.get_pokemon(
                f"{role}: {species}", details=f"{species}, L50"
            )
            mon._update_from_teambuilder(sheet)
            self.own.append((mon, sheet))

    def feed(self, event: Sequence[str], counters: Counter[str]) -> str:
        """Apply one event; returns its kind. Parse failures are counted."""
        kind = event[1] if len(event) > 1 else ""
        if kind in _PLAYER_LEVEL_KINDS:
            return kind
        try:
            if kind == "win":
                self.battle.won_by(event[2])
            elif kind == "tie":
                self.battle.tied()
            else:
                self.battle.parse_message(list(event))
        except Exception as exc:
            counters[f"parse_error:{kind}:{type(exc).__name__}"] += 1
        return kind

    def at_turn_start(self) -> None:
        """Bring the stats to what the live bot held when it chose this turn."""
        for mon, sheet in self.own:
            if mon.species != to_id_str(mon.base_species):  # a Mega: its own ability
                abilities = mon.possible_abilities
                if abilities:
                    mon._ability = to_id_str(abilities[0])
            mon.stats = champions_stats(mon, sheet)
        # The live observation runs the damage calculator against every active
        # foe at every decision, and the first call freezes a synthetic spread on
        # that Pokemon: a foe that Mega-evolves later keeps its pre-Mega numbers.
        # Doing the same here is what makes the logged scores reproduce.
        for foe in self.battle.opponent_active_pokemon:
            if foe is not None and not foe.fainted:
                K.ensure_stats(foe)


class LoggedPredictions:
    """The bot's own opponent predictions, recomputed by its own feeding code.

    A bare ``PolicyPlayer`` (no network client, no policy) carries just the
    state the two prediction hooks read, so the logged arm goes through the
    deployed code path: plan belief, move pool from revealed moves plus the set
    prior, reliability = revealed moves / 4, both-sides-two-actives gate.
    """

    def __init__(self, config: DirectoryConfig, counters: Counter[str]) -> None:
        self.counters = counters
        self.player = PolicyPlayer.__new__(PolicyPlayer)
        self.player.move_model_path = config.move_model
        self.player.switch_model_path = config.switch_model
        self.player._move_predictor = None
        self.player._switch_predictor = None
        self.player._move_prediction_cache = {}
        self.player._switch_prediction_cache = {}
        self.player._battle_plans = {}
        self.preview: PreviewPredictor | None = None
        if config.preview_model is not None:
            try:
                self.preview = PreviewPredictor.load(config.preview_model)
            except Exception as exc:
                counters[f"preview_model_error:{type(exc).__name__}"] += 1

    def at_preview(self, battle: DoubleBattle) -> None:
        """The opponent-plan belief the bot built at team preview."""
        if self.preview is None:
            return
        try:
            ours = PolicyPlayer._preview_roster(battle.team)
            theirs = PolicyPlayer._preview_roster(battle.opponent_team)
            plans = self.preview.predict_plans(theirs, ours, top_k=90)
            self.player._battle_plans[battle.battle_tag] = BattlePlanState(
                own_plan=None, opponent_belief=OpponentBelief(theirs, plans)
            )
        except Exception as exc:
            self.counters[f"plan_belief_error:{type(exc).__name__}"] += 1

    def observe(self, battle: DoubleBattle) -> None:
        """Condition the plan belief on the opposing actives, as each choice did."""
        self.player._update_battle_plan(battle)

    def predict(
        self, battle: DoubleBattle
    ) -> tuple[
        tuple[MovePrediction, MovePrediction] | None,
        tuple[SwitchPrediction, SwitchPrediction] | None,
    ]:
        return (
            self.player.opponent_move_predictions(battle),
            self.player.opponent_switch_predictions(battle),
        )

    def done(self, battle: DoubleBattle) -> None:
        self.player._battle_plans.pop(battle.battle_tag, None)
        self.player._move_prediction_cache.clear()
        self.player._switch_prediction_cache.clear()


def bot_role(events: Sequence[Sequence[str]], player: str) -> str | None:
    """The side (``p1`` / ``p2``) of the account the replay file is named after."""
    wanted = E.user_id(player)
    for event in events:
        if len(event) > 3 and event[1] == "player" and E.user_id(event[3]) == wanted:
            return event[2] if event[2] in E.SIDES else None
    return None


def find_replays(folder: Path) -> dict[str, tuple[Path, str]]:
    """Room tag -> (saved page, the bot's account name from the file name)."""
    found: dict[str, tuple[Path, str]] = {}
    for path in sorted(folder.glob("*.html")):
        match = _REPLAY_NAME.match(path.stem)
        if match is not None:
            found.setdefault(match.group("room"), (path, match.group("player")))
    return found


# --- replaying the consumer --------------------------------------------------------


def rebuild_candidates(row: Mapping[str, Any]) -> list[Candidate]:
    """The logged candidates in the order the reranker received them.

    The audit keeps the first eight AFTER the reranker. Without a reranker
    report nothing was reordered. With one, the report's ``before`` pair was
    first and the eligible candidates were sorted by score; the rest of the
    live prefix is restored to policy order (how ``build_candidates`` sorts and
    guards leave it, apart from the promoted head), vetoed pairs stay last.
    ``strategic_only`` is not logged and is taken as False.
    """
    candidates = [
        Candidate(
            actions=(int(entry["actions"][0]), int(entry["actions"][1])),
            prob=float(entry["policy_probability"]),
            demoted_by=entry.get("demoted_by"),
        )
        for entry in row["candidates"]
    ]
    report = row.get("reranker")
    if not isinstance(report, Mapping):
        return candidates
    before = tuple(report.get("before") or ())
    live = [candidate for candidate in candidates if candidate.demoted_by is None]
    dead = [candidate for candidate in candidates if candidate.demoted_by is not None]
    head = [candidate for candidate in live if candidate.actions == before]
    if not head:
        return candidates
    rest = sorted(
        (candidate for candidate in live if candidate.actions != before),
        key=lambda candidate: -candidate.prob,
    )
    return head[:1] + rest + dead


def decode_matches(battle: DoubleBattle, row: Mapping[str, Any]) -> bool:
    """Every logged action decodes on the rebuilt position to the logged order."""
    for entry in row["candidates"]:
        for position, (action, logged) in enumerate(
            zip(entry["actions"], entry["orders"])
        ):
            if PolicyPlayer._audit_order(battle, int(action), position) != logged:
                return False
    return True


@dataclass
class ArmResult:
    """The consumer's answer for one arm of one decision."""

    pick: tuple[int, int]
    report: R.OpponentRerankReport | None
    sticky_kept: bool

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "pick": list(self.pick),
            "report": self.report is not None,
            "sticky_kept": self.sticky_kept,
        }
        if self.report is not None:
            out.update(
                after=list(self.report.after),
                evaluated=self.report.evaluated,
                opponent_score_before=self.report.tactical_score_before,
                opponent_score_after=self.report.tactical_score_after,
                tempo_score_before=self.report.tempo_score_before,
                tempo_score_after=self.report.tempo_score_after,
            )
        return out


def run_consumer(
    battle: DoubleBattle,
    candidates: list[Candidate],
    moves: tuple[MovePrediction, MovePrediction] | None,
    switches: tuple[SwitchPrediction, SwitchPrediction] | None,
    config: DirectoryConfig,
    guard_fired: bool,
    *,
    min_policy_ratio: float | None = None,
) -> ArmResult:
    """The reranker, then the directory's sticky-correction stage.

    The live legality filter and mixing that follow in ``_guarded_action`` are
    not replayed: every logged candidate already passed the filter, and a
    record that mixing touched carries a ``mixing`` entry (counted in the
    summary; none on the directories of 2026-10-04).
    """
    ranked, report = R.rerank_candidates(
        battle,
        candidates,
        moves,
        switches,
        min_policy_ratio=min_policy_ratio,
        use_opponent=config.use_opponent,
        use_tempo=config.use_tempo,
    )
    kept = False
    if config.sticky and guard_fired and ranked:
        # raw_top is the policy's own favourite; among the logged candidates
        # that is the most probable one (exact unless it was vetoed AND fell
        # outside the eight logged, which the caller counts).
        raw_top = max(candidates, key=lambda candidate: candidate.prob).actions
        ranked, kept = keep_guard_correction(
            ranked,
            raw_top,
            candidates[0].actions,
            report.special_reason if report is not None else None,
        )
    return ArmResult(ranked[0].actions, report, kept)


def primary_score(
    battle: DoubleBattle,
    candidate: Candidate,
    moves: tuple[MovePrediction, MovePrediction],
    switches: tuple[SwitchPrediction, SwitchPrediction],
) -> tuple[float, float]:
    """(outgoing, incoming): the reranker's own two terms against the truth."""
    cache: dict[tuple[int, int, str], float] = {}
    return (
        R._outgoing_damage(battle, candidate, switches, cache),
        R._incoming_damage(battle, candidate, moves, cache),
    )


def _hit_slots(battle: DoubleBattle, move: Move, order: object) -> list[int]:
    """Their slots our move reaches: the reranker's rule plus the lone-foe retarget."""
    foes = battle.opponent_active_pokemon
    standing = [
        slot for slot, foe in enumerate(foes[:2]) if foe is not None and not foe.fainted
    ]
    slots = R._target_slots(move, order)
    if len(slots) == 1 and slots[0] not in standing and len(standing) == 1:
        return standing  # Showdown sends a single-target move to the only foe left
    return [slot for slot in slots if slot in standing]


def dealt_damage(
    battle: DoubleBattle, candidate: Candidate, truth: Sequence[E.SlotAction | None]
) -> tuple[float, int]:
    """Expected damage into the Pokemon that truly stood in each opposing slot.

    Returns (damage as a share of max HP, summed over their two slots and
    capped at the HP left; number of our damaging moves that cannot be scored).
    A slot whose true action was a voluntary switch is scored against the
    Pokemon that came in; when the bot had never seen that Pokemon there is
    nothing to compute against, and a move aimed there is counted as unscored
    (the caller then leaves the pair out). Nothing lands on a slot that truly
    used a Protect-family move.
    """
    foes = battle.opponent_active_pokemon
    bench = R._bench_by_species(battle)
    cache: dict[tuple[int, int, str], float] = {}
    defenders: list[Pokemon | None] = [None, None]
    unseen: set[int] = set()
    for slot in range(2):
        action = truth[slot] if slot < len(truth) else None
        current = foes[slot] if slot < len(foes) else None
        if action is not None and oracle_known(action):
            if action.kind == E.KIND_SWITCH:
                defenders[slot] = bench.get(str(action.switch_to))
                if defenders[slot] is None:
                    unseen.add(slot)
                continue
            if action.is_protect:
                continue
        if current is not None and not current.fainted:
            defenders[slot] = current
    unscored = 0
    per_slot = [0.0, 0.0]
    for position, order in enumerate(R._orders(battle, candidate)):
        move = getattr(order, "order", None)
        attacker = battle.active_pokemon[position]
        if (
            not isinstance(move, Move)
            or attacker is None
            or move.category == MoveCategory.STATUS
            or move.base_power <= 0
        ):
            continue
        for slot in _hit_slots(battle, move, order):
            defender = defenders[slot]
            if slot in unseen:
                unscored += 1
            elif defender is not None:
                per_slot[slot] += R._damage(battle, attacker, defender, move, cache)
    total = 0.0
    for slot, defender in enumerate(defenders):
        if defender is not None:
            left = float(defender.current_hp_fraction or 0.0)
            total += min(left, per_slot[slot])
    return total, unscored


# --- per-decision work ----------------------------------------------------------------


def _close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= SCORE_TOLERANCE
    except (TypeError, ValueError):
        return False


def _slot_truth(
    battle: DoubleBattle,
    actions: Mapping[str, E.SlotAction],
    opponent_role: str,
    counters: Counter[str],
) -> list[E.SlotAction | None]:
    """True actions of their slots a and b, checked against the rebuilt board."""
    foes = battle.opponent_active_pokemon
    truth: list[E.SlotAction | None] = []
    for index, letter in enumerate(E.SLOT_LETTERS):
        action = actions.get(opponent_role + letter)
        foe = foes[index] if index < len(foes) else None
        if action is not None:
            standing = E.base_species_id(foe.base_species) if foe is not None else ""
            if standing != action.species:
                counters["truth_species_mismatch"] += 1
                action = None
        truth.append(action)
    return truth


def _truth_row(
    battle: DoubleBattle, truth: Sequence[E.SlotAction | None]
) -> list[dict[str, Any]]:
    bench = R._bench_by_species(battle)
    rows: list[dict[str, Any]] = []
    for letter, action in zip(E.SLOT_LETTERS, truth):
        if action is None:
            rows.append({"slot": letter, "kind": None, "known": False})
            continue
        row: dict[str, Any] = {
            "slot": letter,
            "kind": action.kind,
            "move": action.move,
            "target": action.target,
            "switch_to": action.switch_to,
            "reason": action.reason,
            "known": oracle_known(action),
            "target_trusted": action.target_trusted,
            "protect": action.is_protect,
        }
        if action.kind == E.KIND_SWITCH:
            row["switch_in_seen"] = str(action.switch_to) in bench
        rows.append(row)
    return rows


def _own_action_matches(
    row: Mapping[str, Any], actions: Mapping[str, E.SlotAction], role: str
) -> bool | None:
    """Whether the log shows the bot doing what the record says it chose.

    None when neither of our slots shows a move or a voluntary switch.
    """
    orders = (row.get("chosen") or {}).get("orders") or []
    seen = False
    for letter, order in zip(E.SLOT_LETTERS, orders):
        action = actions.get(role + letter)
        if action is None or not isinstance(order, Mapping):
            continue
        if action.kind == E.KIND_MOVE and not action.locked:
            seen = True
            if order.get("kind") != "move" or order.get("id") != action.move:
                return False
        elif action.kind == E.KIND_SWITCH:
            seen = True
            if order.get("kind") != "switch":
                return False
    return True if seen else None


def analyse_decision(
    position: Position,
    decision: Decision,
    actions: Mapping[str, E.SlotAction],
    role: str,
    config: DirectoryConfig,
    logged: LoggedPredictions,
    counters: Counter[str],
) -> dict[str, Any]:
    """Everything this test says about one rebuilt move decision."""
    battle = position.battle
    row = decision.row
    out: dict[str, Any] = {}
    if not decode_matches(battle, row):
        out["status"] = "decode_mismatch"
        return out
    opponent_role = "p2" if role == "p1" else "p1"
    candidates = rebuild_candidates(row)
    by_actions = {candidate.actions: candidate for candidate in candidates}
    chosen = row["chosen"]["actions"]
    played = (int(chosen[0]), int(chosen[1]))
    if played not in by_actions:
        out["status"] = "played_not_in_candidates"
        return out
    guard_fired = bool((row.get("guards") or {}).get("stages"))
    live = [candidate for candidate in candidates if candidate.demoted_by is None]
    truth = _slot_truth(battle, actions, opponent_role, counters)
    logged_moves, logged_switches = logged.predict(battle)

    certain = oracle_inputs(truth, logged_moves, logged_switches, gated=False)
    gated = oracle_inputs(truth, logged_moves, logged_switches, gated=True)
    null = oracle_inputs(
        truth, logged_moves, logged_switches, gated=False, content=False
    )
    as_logged = OracleInputs(logged_moves, logged_switches)

    def consume(
        inputs: OracleInputs, sticky: bool | None = None, ratio: float | None = None
    ) -> ArmResult:
        return run_consumer(
            battle,
            candidates,
            inputs.moves,
            inputs.switches,
            config if sticky is None else replace(config, sticky=sticky),
            guard_fired,
            min_policy_ratio=ratio,
        )

    arms: dict[str, ArmResult] = {
        ARM_LOGGED: consume(as_logged),
        ARM_ORACLE: consume(certain),
        ARM_GATED: consume(gated),
        ARM_WIDE: consume(certain, ratio=0.0),
        ARM_NULL: consume(null),
    }
    # Post hoc: both arms again under the pipeline deployed today (sticky on).
    sticky_logged = consume(as_logged, sticky=True)
    sticky_oracle = consume(certain, sticky=True)

    flags: list[str] = []
    if certain.prescaled and logged_moves is not None and certain.moves is not None:
        flags.append("prescaled_unknown_slot")
        # The tempo term reads reliability and the switch guess directly: did
        # restating the unknown slot move it for any live candidate?
        known = [certain_predictions(action) is not None for action in truth]
        plain_moves = (
            certain.moves[0] if known[0] else logged_moves[0],
            certain.moves[1] if known[1] else logged_moves[1],
        )
        plain_switches = certain.switches
        if logged_switches is not None and certain.switches is not None:
            plain_switches = (
                certain.switches[0] if known[0] else logged_switches[0],
                certain.switches[1] if known[1] else logged_switches[1],
            )
        restated = score_candidates(battle, live, certain.moves, certain.switches)
        plain = score_candidates(battle, live, plain_moves, plain_switches)
        if restated.utility != plain.utility:
            flags.append("prescale_changed_tempo")
    vetoed = {tuple(pair) for pair in (row.get("guards") or {}).get("vetoed") or []}
    if any(arm.sticky_kept for arm in arms.values()) and vetoed - set(by_actions):
        flags.append("sticky_raw_top_uncertain")

    truth_moves, truth_switches = truth_inputs(truth)
    played_parts = primary_score(
        battle, by_actions[played], truth_moves, truth_switches
    )
    played_dealt, played_unscored = dealt_damage(battle, by_actions[played], truth)

    def hindsight(pick: tuple[int, int]) -> dict[str, Any]:
        parts = primary_score(battle, by_actions[pick], truth_moves, truth_switches)
        dealt, unscored = dealt_damage(battle, by_actions[pick], truth)
        primary = (parts[0] - parts[1]) - (played_parts[0] - played_parts[1])
        exchange = (dealt - parts[1]) - (played_dealt - played_parts[1])
        return {
            "primary_diff": primary,
            "outgoing": [played_parts[0], parts[0]],
            "incoming": [played_parts[1], parts[1]],
            # None: one of the two pairs attacks a switch-in never seen before.
            "exchange_diff": exchange if unscored + played_unscored == 0 else None,
            "dealt": [played_dealt, dealt],
            "unscored_hits": [played_unscored, unscored],
        }

    readable = {
        (int(entry["actions"][0]), int(entry["actions"][1])): entry.get("orders")
        for entry in row["candidates"]
    }
    arm_rows: dict[str, Any] = {}
    for name, arm in arms.items():
        entry = arm.to_dict()
        entry["flip"] = arm.pick != played
        if arm.pick != played:
            entry["orders"] = readable.get(arm.pick)
            if name != ARM_LOGGED:
                entry["hindsight"] = hindsight(arm.pick)
        arm_rows[name] = entry
    arm_rows["sticky_on"] = {
        "logged_pick": list(sticky_logged.pick),
        "oracle_pick": list(sticky_oracle.pick),
        "oracle_changes_pick": sticky_oracle.pick != sticky_logged.pick,
    }

    logged_report = row.get("reranker")
    replayed = arms[ARM_LOGGED].report
    agree: dict[str, Any] = {
        "pick": arms[ARM_LOGGED].pick == played,
        "report": (logged_report is not None) == (replayed is not None),
    }
    if isinstance(logged_report, Mapping) and replayed is not None:
        agree["after"] = tuple(logged_report.get("after") or ()) == replayed.after
        agree["opponent_score"] = _close(
            logged_report.get("opponent_score_before"), replayed.tactical_score_before
        ) and _close(
            logged_report.get("opponent_score_after"), replayed.tactical_score_after
        )
        agree["tempo_score"] = _close(
            logged_report.get("tempo_score_before"), replayed.tempo_score_before
        ) and _close(logged_report.get("tempo_score_after"), replayed.tempo_score_after)
    # A logged opponent term where a closed sheet leaves no evidence at all: the
    # live bot must have held an open team sheet.
    no_evidence = logged_moves is None or all(
        prediction.reliability <= 0 for prediction in logged_moves
    )
    open_signature = (
        isinstance(logged_report, Mapping)
        and no_evidence
        and (
            not _close(logged_report.get("opponent_score_before"), 0.0)
            or not _close(logged_report.get("opponent_score_after"), 0.0)
        )
    )
    top_probability = max(live[0].prob, 1e-12) if live else 0.0
    out.update(
        status="ok",
        played={
            "actions": list(played),
            "orders": row["chosen"].get("orders"),
            "policy_probability": row["chosen"].get("policy_probability"),
        },
        n_candidates=len(candidates),
        n_live=len(live),
        n_eligible=sum(
            candidate.prob >= top_probability * ELIGIBILITY_RATIO for candidate in live
        ),
        truth=_truth_row(battle, truth),
        truth_class=truth_class(truth),
        known_slots=sum(oracle_known(action) for action in truth),
        logged={
            "predictions": logged_moves is not None,
            "reliability": [prediction.reliability for prediction in logged_moves]
            if logged_moves is not None
            else None,
            "reranker": {
                key: logged_report.get(key)
                for key in (
                    "before",
                    "after",
                    "opponent_score_before",
                    "opponent_score_after",
                    "tempo_score_before",
                    "tempo_score_after",
                )
            }
            if isinstance(logged_report, Mapping)
            else None,
        },
        arms=arm_rows,
        agree=agree,
        own_action_matches_log=_own_action_matches(row, actions, role),
        open_sheet_signature=bool(open_signature),
        flags=flags,
    )
    return out


def process_directory(
    folder: Path, counters: Counter[str]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Every move decision of one replay directory, as output rows."""
    census: Counter[str] = Counter()
    audit = folder / "decisions.jsonl"
    try:
        text = audit.read_text(encoding="utf-8", errors="replace")
    except OSError:
        counters["directory_without_audit"] += 1
        return [], census
    decisions, census, sheets = read_decisions(text.split("\n"), folder.name)
    config = read_run_config(folder)
    rows: list[dict[str, Any]] = []

    def base(decision: Decision) -> dict[str, Any]:
        flag = sheets.get(decision.room)
        return {
            "dir": folder.name,
            "line": decision.line,
            "battle": public_battle_id(decision.room),
            "turn": decision.turn,
            "superseded": decision.superseded,
            "sheet": SHEET_UNKNOWN
            if flag is None
            else (SHEET_OPEN if flag else SHEET_CLOSED),
            "guard_stages": list(
                (decision.row.get("guards") or {}).get("stages") or []
            ),
            "sticky": bool(config.sticky) if config is not None else None,
            "mixing_logged": "mixing" in decision.row,
        }

    if config is None:
        counters["directory_without_run_config"] += 1
        return [dict(base(d), status="no_run_config") for d in decisions], census
    if config.varies:
        counters["run_config_varies_within_directory"] += 1
    try:
        sets = Teambuilder.parse_showdown_team(config.team.read_text())
    except Exception as exc:
        counters[f"team_file_error:{type(exc).__name__}"] += 1
        return [dict(base(d), status="no_team_file") for d in decisions], census

    previous = (PolicyPlayer.use_moveset_prior, _os.environ.get("VGC_SET_PRIOR_REG"))
    PolicyPlayer.use_moveset_prior = config.moveset_prior
    _os.environ["VGC_SET_PRIOR_REG"] = config.set_prior_reg
    try:
        logged = LoggedPredictions(config, counters)
        replays = find_replays(folder)
        by_room: dict[str, list[Decision]] = defaultdict(list)
        for decision in decisions:
            by_room[decision.room].append(decision)
        for room, room_decisions in by_room.items():
            done = _process_battle(
                room, room_decisions, replays, sets, config, logged, counters
            )
            inferred = any(row.get("open_sheet_signature") for row in done.values())
            for decision in room_decisions:
                row = dict(base(decision), **done.get(decision.line, {}))
                if inferred and row["sheet"] == SHEET_UNKNOWN:
                    row["sheet"] = SHEET_OPEN_INFERRED
                rows.append(row)
    finally:
        PolicyPlayer.use_moveset_prior = previous[0]
        if previous[1] is None:
            _os.environ.pop("VGC_SET_PRIOR_REG", None)
        else:
            _os.environ["VGC_SET_PRIOR_REG"] = previous[1]
    rows.sort(key=lambda row: row["line"])
    return rows, census


def _process_battle(
    room: str,
    decisions: Sequence[Decision],
    replays: Mapping[str, tuple[Path, str]],
    sets: Sequence[Any],
    config: DirectoryConfig,
    logged: LoggedPredictions,
    counters: Counter[str],
) -> dict[int, dict[str, Any]]:
    """Output fields of one battle's decisions, keyed by audit line."""

    def all_status(status: str) -> dict[int, dict[str, Any]]:
        return {decision.line: {"status": status} for decision in decisions}

    if room not in replays:
        return all_status("no_replay")
    path, player = replays[room]
    try:
        log = E.extract_log_from_html(
            path.read_text(encoding="utf-8", errors="replace")
        )
    except OSError:
        log = None
    if not log:
        return all_status("no_log_in_replay")
    events = E.split_log(log)
    role = bot_role(events, player)
    if role is None:
        return all_status("bot_side_not_found")
    reader: Counter[str] = Counter()
    actions_by_turn: dict[int, Mapping[str, E.SlotAction]] = {}
    for segment, actions in E.read_log_actions(events, reader):
        actions_by_turn.setdefault(segment.turn, actions)
    for name, count in reader.items():
        if name.startswith("reader_error"):
            counters[name] += count
    pending: dict[int, list[Decision]] = defaultdict(list)
    for decision in decisions:
        pending[decision.turn].append(decision)
    out: dict[int, dict[str, Any]] = {}
    try:
        position = Position("battle-" + public_battle_id(room), player, role, sets)
    except Exception as exc:
        return all_status(f"rebuild_error:{type(exc).__name__}")
    battle = position.battle
    for event in events:
        kind = position.feed(event, counters)
        if kind == "teampreview":
            logged.at_preview(battle)
        if kind != "turn":
            continue
        try:
            turn = int(event[2])
        except (IndexError, ValueError):
            continue
        try:
            position.at_turn_start()
            logged.observe(battle)
        except Exception as exc:
            counters[f"turn_start_error:{type(exc).__name__}"] += 1
        for decision in pending.pop(turn, []):
            try:
                out[decision.line] = analyse_decision(
                    position,
                    decision,
                    actions_by_turn.get(turn, {}),
                    role,
                    config,
                    logged,
                    counters,
                )
            except Exception as exc:
                out[decision.line] = {"status": f"error:{type(exc).__name__}"}
    for leftovers in pending.values():
        for decision in leftovers:
            out[decision.line] = {"status": "turn_not_in_replay"}
    logged.done(battle)
    return out


# --- summary -------------------------------------------------------------------------


def _share(count: float, total: float) -> float | None:
    return count / total if total else None


def _game(row: Mapping[str, Any]) -> str:
    return f"{row['dir']}|{row['battle']}"


def _split(
    rows: Sequence[Mapping[str, Any]], arm: str, key: Callable[[Mapping[str, Any]], Any]
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        group = groups[str(key(row))]
        group[0] += 1
        group[1] += bool(row["arms"][arm]["flip"])
    return {
        name: {"decisions": n, "flips": flips, "share": _share(flips, n)}
        for name, (n, flips) in sorted(groups.items())
    }


def _hindsight(
    flips: Sequence[Mapping[str, Any]], arm: str, field_name: str, resamples: int
) -> dict[str, Any]:
    scores = [row["arms"][arm]["hindsight"][field_name] for row in flips]
    values = [
        (_game(row), float(score))
        for row, score in zip(flips, scores)
        if score is not None
    ]
    sums, counts = by_cluster(values)
    interval = cluster_bootstrap(sums, counts, resamples=resamples)
    better = sum(value > EQUAL_TOLERANCE for _, value in values)
    worse = sum(value < -EQUAL_TOLERANCE for _, value in values)
    n = len(values)
    return {
        "n": n,
        "left_out": len(scores) - n,
        "mean_diff": interval.point,
        "low": interval.low,
        "high": interval.high,
        "games": interval.clusters,
        "better": better,
        "equal": n - better - worse,
        "worse": worse,
        "share_better": _share(better, n),
        "share_equal": _share(n - better - worse, n),
        "share_worse": _share(worse, n),
    }


def change_kind(
    played: Sequence[Mapping[str, Any]], pick: Sequence[Mapping[str, Any]]
) -> str:
    """What a flip changes, per slot of ours: protect / switch / retarget / move."""
    kinds: list[str] = []
    for before, after in zip(played, pick):
        if before == after:
            continue
        if after.get("kind") == "switch":
            kinds.append("switch")
        elif E.is_protect_family(str(after.get("id") or "")):
            kinds.append("protect")
        elif after.get("kind") == "move" and after.get("id") == before.get("id"):
            kinds.append("retarget")
        else:
            kinds.append("move")
    return "+".join(sorted(kinds)) or "none"


def summarise_arm(
    rows: Sequence[Mapping[str, Any]],
    all_rows: Sequence[Mapping[str, Any]],
    arm: str,
    resamples: int,
) -> dict[str, Any]:
    """Flip counts, splits and hindsight of one oracle arm.

    ``rows`` are the rebuilt decisions, ``all_rows`` every move decision.
    """
    flips = [row for row in rows if row["arms"][arm]["flip"]]
    clean = [row for row in flips if row["agree"]["pick"]]
    per_game: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in all_rows:
        entry = per_game[_game(row)]
        entry[1] += 1.0
        if row.get("status") == "ok" and row["arms"][arm]["flip"]:
            entry[0] += 1.0
    keys = sorted(per_game)
    share = cluster_bootstrap(
        [per_game[key][0] for key in keys],
        [per_game[key][1] for key in keys],
        resamples=resamples,
    )
    kept = [row for row in all_rows if not row.get("superseded")]
    kept_flips = [row for row in flips if not row.get("superseded")]
    visible: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        if row["known_slots"] == 2:
            visible[_game(row)][1] += 1.0
            visible[_game(row)][0] += bool(row["arms"][arm]["flip"])
    both_visible = cluster_bootstrap(
        [entry[0] for entry in visible.values()],
        [entry[1] for entry in visible.values()],
        resamples=resamples,
    )
    not_rebuilt = len(all_rows) - len(rows)
    not_reproduced = sum(not row["agree"]["pick"] for row in rows)
    return {
        "flips": len(flips),
        "share_of_all_move_decisions": share.to_dict(),
        "share_of_rebuilt": _share(len(flips), len(rows)),
        "upper_bound_share": _share(len(flips) + not_rebuilt, len(all_rows)),
        "flips_where_arm_A_reproduces_the_played_pick": len(clean),
        "share_excluding_unreproduced": _share(len(clean), len(rows) - not_reproduced),
        "share_excluding_superseded": _share(len(kept_flips), len(kept)),
        "flips_with_report": sum(bool(row["arms"][arm]["report"]) for row in flips),
        "by_turn": _split(
            rows, arm, lambda row: "turn_1" if row["turn"] == 1 else "later"
        ),
        "by_truth": _split(rows, arm, lambda row: row["truth_class"]),
        "by_known_slots": _split(rows, arm, lambda row: row["known_slots"]),
        "share_where_both_opposing_actions_are_visible": both_visible.to_dict(),
        "by_guard_fired": _split(rows, arm, lambda row: bool(row["guard_stages"])),
        "by_sticky": _split(rows, arm, lambda row: bool(row["sticky"])),
        "by_sheet": _split(rows, arm, lambda row: row["sheet"]),
        "hindsight_primary": _hindsight(flips, arm, "primary_diff", resamples),
        "hindsight_exchange": _hindsight(flips, arm, "exchange_diff", resamples),
        "hindsight_primary_clean": _hindsight(clean, arm, "primary_diff", resamples),
        "hindsight_exchange_clean": _hindsight(clean, arm, "exchange_diff", resamples),
    }


def _agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    both = [row for row in rows if "after" in row["agree"]]
    return {
        "decisions": len(rows),
        "pick_reproduced": sum(row["agree"]["pick"] for row in rows),
        "pick_rate": _share(sum(row["agree"]["pick"] for row in rows), len(rows)),
        "logged_reports": sum(row["logged"]["reranker"] is not None for row in rows),
        "replayed_reports": sum(
            bool(row["arms"][ARM_LOGGED]["report"]) for row in rows
        ),
        "report_presence_same": sum(row["agree"]["report"] for row in rows),
        "both_report": len(both),
        "after_same": sum(row["agree"]["after"] for row in both),
        "opponent_score_same": sum(row["agree"]["opponent_score"] for row in both),
        "tempo_score_same": sum(row["agree"]["tempo_score"] for row in both),
    }


def summarise(
    all_rows: Sequence[Mapping[str, Any]],
    census: Mapping[str, int],
    counters: Mapping[str, int],
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """The whole reading from the per-decision rows (pure: no files, no models)."""
    rows = [row for row in all_rows if row.get("status") == "ok"]
    battles = {_game(row) for row in all_rows}
    enumeration: dict[str, Any] = {
        "records_with_candidates": census.get(RECORD_MOVE, 0)
        + census.get(RECORD_PREVIEW, 0)
        + census.get(RECORD_FORCED, 0),
        RECORD_PREVIEW: census.get(RECORD_PREVIEW, 0),
        RECORD_FORCED: census.get(RECORD_FORCED, 0),
        RECORD_MOVE: census.get(RECORD_MOVE, 0),
        "battles": len(battles),
        "superseded_in_the_same_turn": sum(bool(r.get("superseded")) for r in all_rows),
        "records_without_candidates": census.get(RECORD_OTHER, 0),
        "bad_json_lines": census.get("bad_json", 0),
    }
    enumeration["matches_critic_count"] = all(
        enumeration[key] == value for key, value in CRITIC_COUNT.items()
    )
    enumeration["critic_count"] = dict(CRITIC_COUNT)

    agreement = _agreement(rows)
    agreement["by_sheet"] = {
        sheet: _agreement([row for row in rows if row["sheet"] == sheet])
        for sheet in sorted({row["sheet"] for row in rows})
    }
    agreement["by_directory"] = {
        name: _agreement([row for row in rows if row["dir"] == name])
        for name in sorted({row["dir"] for row in rows})
    }
    own = [row for row in rows if row["own_action_matches_log"] is not None]
    agreement["played_pair_seen_in_log"] = {
        "checked": len(own),
        "match": sum(bool(row["own_action_matches_log"]) for row in own),
        "match_excluding_superseded": sum(
            bool(row["own_action_matches_log"]) for row in own if not row["superseded"]
        ),
        "checked_excluding_superseded": sum(not row["superseded"] for row in own),
    }

    slots: Counter[str] = Counter()
    for row in rows:
        for slot in row["truth"]:
            if slot["kind"] is None:
                slots["empty_or_unreadable"] += 1
            elif slot["known"]:
                slots[f"known:{slot['kind']}"] += 1
                if slot["kind"] == E.KIND_SWITCH:
                    slots[f"switch_in_seen:{bool(slot.get('switch_in_seen'))}"] += 1
                elif slot.get("protect"):
                    slots["known:protect"] += 1
                if slot["kind"] == E.KIND_MOVE and not slot["target_trusted"]:
                    slots["known:move_target_not_trusted"] += 1
            else:
                slots[f"unknown:{slot['kind']}:{slot.get('reason')}"] += 1
    truth = {
        "decisions_by_class": dict(Counter(row["truth_class"] for row in rows)),
        "decisions_by_known_slots": dict(
            Counter(str(row["known_slots"]) for row in rows)
        ),
        "slots": dict(sorted(slots.items())),
    }

    arms = {arm: summarise_arm(rows, all_rows, arm, resamples) for arm in ORACLE_ARMS}

    # Post hoc (added after the first full run): what arm B's flips are made of.
    # Three disjoint parts: the replay does not reproduce the log there at all;
    # the control arm N (gates open, no content) makes the same pick; or the
    # pick needs the true action.
    oracle_flips = [row for row in rows if row["arms"][ARM_ORACLE]["flip"]]
    unreproduced = [row for row in oracle_flips if not row["agree"]["pick"]]
    reproduced = [row for row in oracle_flips if row["agree"]["pick"]]
    need_truth = [
        row
        for row in reproduced
        if row["arms"][ARM_ORACLE]["pick"] != row["arms"][ARM_NULL]["pick"]
    ]
    opening = [row for row in reproduced if row not in need_truth]
    sticky_changes = sum(
        bool(row["arms"]["sticky_on"]["oracle_changes_pick"]) for row in rows
    )
    post_hoc = {
        "B_pick_differs_from_arm_A_pick": sum(
            row["arms"][ARM_ORACLE]["pick"] != row["arms"][ARM_LOGGED]["pick"]
            for row in rows
        ),
        "B_flips": len(oracle_flips),
        "B_flips_where_arm_A_does_not_reproduce_the_log": len(unreproduced),
        "B_flips_the_control_arm_N_makes_too": len(opening),
        "of_those_with_a_guard_fired": sum(
            bool(row["guard_stages"]) for row in opening
        ),
        "B_flips_that_need_the_true_action": len(need_truth),
        "share_that_need_the_true_action": _share(len(need_truth), len(all_rows)),
        "what_those_flips_change": dict(
            Counter(
                change_kind(
                    row["played"]["orders"] or [],
                    row["arms"][ARM_ORACLE].get("orders") or [],
                )
                for row in need_truth
            ).most_common()
        ),
        "hindsight_primary_need_truth": _hindsight(
            need_truth, ARM_ORACLE, "primary_diff", resamples
        ),
        "hindsight_exchange_need_truth": _hindsight(
            need_truth, ARM_ORACLE, "exchange_diff", resamples
        ),
        "sticky_on_everywhere": {
            "oracle_pick_differs_from_logged_pick": sticky_changes,
            "share_of_all_move_decisions": _share(sticky_changes, len(all_rows)),
        },
    }

    def flipped(arm: str) -> set[tuple[str, int]]:
        return {(row["dir"], row["line"]) for row in rows if row["arms"][arm]["flip"]}

    b, c, d = flipped(ARM_ORACLE), flipped(ARM_GATED), flipped(ARM_WIDE)
    gates = {
        "flips_B_oracle": len(b),
        "flips_C_gates_normal": len(c),
        "flips_D_eligibility_open": len(d),
        "B_and_C": len(b & c),
        "B_only_blocked_by_the_gates": len(b - c),
        "C_only": len(c - b),
        "D_only_blocked_by_the_eligibility_ratio": len(d - b),
        "decisions_with_two_or_more_live_candidates": sum(
            row["n_live"] >= 2 for row in rows
        ),
        "decisions_with_two_or_more_eligible": sum(
            row["n_eligible"] >= 2 for row in rows
        ),
        "decisions_with_a_known_slot": sum(row["known_slots"] > 0 for row in rows),
        "no_flip_despite_known_slot_and_two_eligible": sum(
            row["known_slots"] > 0
            and row["n_eligible"] >= 2
            and not row["arms"][ARM_ORACLE]["flip"]
            for row in rows
        ),
    }

    primary = arms[ARM_ORACLE]["hindsight_primary"]
    exchange = arms[ARM_ORACLE]["hindsight_exchange"]
    share = arms[ARM_ORACLE]["share_of_all_move_decisions"]
    pick_rate = agreement["pick_rate"]
    reading: dict[str, Any] = {
        "arm": ARM_ORACLE,
        "flip_share": share["point"],
        "flip_share_interval": [share["low"], share["high"]],
        "flip_threshold": FLIP_THRESHOLD,
        "hindsight_mean_diff": primary["mean_diff"],
        "hindsight_interval": [primary["low"], primary["high"]],
        "secondary_mean_diff": exchange["mean_diff"],
        "secondary_interval": [exchange["low"], exchange["high"]],
        "secondary_better": exchange["low"] is not None and exchange["low"] > 0,
        "arm_A_pick_rate": pick_rate,
        "min_agreement": MIN_AGREEMENT,
    }
    if pick_rate is None or pick_rate < MIN_AGREEMENT:
        reading["verdict"] = "not_read"
        reading["why"] = "arm A does not reproduce the played picks often enough"
    else:
        below = share["point"] is None or share["point"] < FLIP_THRESHOLD
        better = primary["low"] is not None and primary["low"] > 0
        reading["flips_below_threshold"] = below
        reading["flips_better_in_hindsight"] = better
        reading["verdict"] = (
            "consumer_cannot_use_a_predictor"
            if below or not better
            else "not_falsified"
        )
        reading["why"] = (
            "perfect foresight flips fewer than 5% of the move decisions"
            if below
            else (
                "the flipped picks are not better against the true action"
                if not better
                else "flips reach 5% and are better on the reranker's own terms"
            )
        )
    flags: Counter[str] = Counter()
    for row in rows:
        flags.update(row["flags"])
    return {
        "enumeration": enumeration,
        "rebuild": {
            "move_decisions": len(all_rows),
            "rebuilt": len(rows),
            "status": dict(Counter(str(row.get("status")) for row in all_rows)),
            "mixing_records": sum(bool(row.get("mixing_logged")) for row in all_rows),
        },
        "sheets": {
            "decisions": dict(Counter(row["sheet"] for row in all_rows)),
            "battles": dict(
                Counter({_game(row): row["sheet"] for row in all_rows}.values())
            ),
        },
        "agreement_arm_A": agreement,
        "truth": truth,
        "arms": arms,
        "gates": gates,
        "post_hoc": post_hoc,
        "reading_R4": reading,
        "flags": dict(flags),
        "counters": dict(sorted(counters.items())),
    }


# --- README ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _cells(split: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{name} {cell['decisions']} / {cell['flips']} ({_pct(cell['share'])})"
        for name, cell in split.items()
    )


def _counts(table: Mapping[str, Any]) -> str:
    return ", ".join(f"{name} {count}" for name, count in table.items()) or "none"


def render_readme(summary: Mapping[str, Any]) -> str:
    """The short reading that sits beside summary.json (every number from it)."""
    e = summary["enumeration"]
    r = summary["rebuild"]
    a = summary["agreement_arm_A"]
    arms = summary["arms"]
    b = arms[ARM_ORACLE]
    gated = arms[ARM_GATED]
    g = summary["gates"]
    post = summary["post_hoc"]
    reading = summary["reading_R4"]
    share = b["share_of_all_move_decisions"]
    seen = b["share_where_both_opposing_actions_are_visible"]
    own = a["played_pair_seen_in_log"]
    slots = summary["truth"]["slots"]
    lines = [
        "# Oracle ceiling of the opponent reranker (reading R4)",
        "",
        "Written by `evaluation/oppmodel_oracle.py`; every number is in"
        " `summary.json`, one row per move decision in `decisions.jsonl`."
        " Definitions were fixed before any oracle arm was run (script docstring);"
        " the section marked post hoc was added after the first run.",
        "",
        "## Reading",
        "",
        f"Verdict: **{reading['verdict']}** ({reading['why']}).",
        "",
        f"- Move decisions: {e[RECORD_MOVE]} in {e['battles']} battles"
        f" ({e['records_with_candidates']} audited records, {e[RECORD_PREVIEW]} team"
        f" preview, {e[RECORD_FORCED]} forced replacement); same as the critic's"
        f" count: {e['matches_critic_count']}. Rebuilt: {r['rebuilt']}"
        f" (by status: {_counts(r['status'])}).",
        f"- Arm B (the true action as a certain prediction, gates satisfied) flips"
        f" {b['flips']}: {_pct(share['point'])} of all move decisions, game-clustered"
        f" 95% interval {_pct(share['low'])} to {_pct(share['high'])}. Among rebuilt"
        f" decisions {_pct(b['share_of_rebuilt'])}; counting every decision that"
        f" could not be rebuilt as a flip {_pct(b['upper_bound_share'])}; where arm A"
        f" reproduces the played pick"
        f" {b['flips_where_arm_A_reproduces_the_played_pick']} flips"
        f" ({_pct(b['share_excluding_unreproduced'])}). R4 threshold: 5%.",
        f"- Where both opposing actions are visible in the log ({int(seen['n'])}"
        f" decisions) the share is {_pct(seen['point'])} ({_pct(seen['low'])} to"
        f" {_pct(seen['high'])}): the oracle is only as complete as the log.",
    ]
    for label, key in (
        ("the reranker's own terms (primary)", "hindsight_primary"),
        ("the damage exchange (secondary)", "hindsight_exchange"),
    ):
        h = b[key]
        left = (
            f" ({h['left_out']} left out: a pair attacks a switch-in never seen)"
            if h["left_out"]
            else ""
        )
        lines.append(
            f"- Hindsight of {h['n']} flips{left} on {label}: mean paired difference"
            f" {_num(h['mean_diff'])}, interval {_num(h['low'])} to {_num(h['high'])}"
            f" over {h['games']} games; better {_pct(h['share_better'])}, equal"
            f" {_pct(h['share_equal'])}, worse {_pct(h['share_worse'])}."
        )
    lines += [
        "",
        "## Gates",
        "",
        f"- Arm C (same true actions, every gate at its normal value) flips"
        f" {g['flips_C_gates_normal']}, of which"
        f" {gated['flips_where_arm_A_reproduces_the_played_pick']} where arm"
        f" A reproduces the played pick. {g['B_only_blocked_by_the_gates']} of arm"
        f" B's flips exist only because the gates were opened.",
        f"- Arm C by turn: {_cells(gated['by_turn'])}; by the opponent's true"
        f" action: {_cells(gated['by_truth'])}; by a guard had fired:"
        f" {_cells(gated['by_guard_fired'])}.",
        f"- Arm D (eligibility ratio opened as well) flips"
        f" {g['flips_D_eligibility_open']};"
        f" {g['D_only_blocked_by_the_eligibility_ratio']} of them are not in arm B.",
        f"- {g['decisions_with_two_or_more_eligible']} rebuilt decisions have two or"
        f" more eligible candidates and {g['decisions_with_a_known_slot']} have a known"
        f" opposing action; in {g['no_flip_despite_known_slot_and_two_eligible']} with"
        f" both, the scoring left the pick where it was.",
        f"- What the consumer cannot use at all: {slots.get('known:protect', 0)} true"
        f" Protects (no term for them) and {slots.get('switch_in_seen:False', 0)} of"
        f" {slots.get('known:switch', 0)} true switches whose incoming Pokemon had"
        f" not been seen (nothing to compute damage against).",
        "",
        "## Splits of arm B (decisions / flips)",
        "",
    ]
    for title, key in (
        ("turn", "by_turn"),
        ("opponent's true action", "by_truth"),
        ("a guard had fired", "by_guard_fired"),
        ("sticky corrections in that directory", "by_sticky"),
    ):
        lines.append(f"- By {title}: {_cells(b[key])}.")
    need = post["hindsight_primary_need_truth"]
    need_exchange = post["hindsight_exchange_need_truth"]
    sticky = post["sticky_on_everywhere"]
    lines += [
        "",
        "## Post hoc: what the flips are made of",
        "",
        f"- Arm B's {post['B_flips']} flips are three things. In"
        f" {post['B_flips_where_arm_A_does_not_reproduce_the_log']} the replay does"
        f" not reproduce the log to begin with (arm A differs from the played pick)."
        f" {post['B_flips_the_control_arm_N_makes_too']} are also made by control arm"
        f" N, which opens the same gates but tells the reranker nothing about a"
        f" known slot ({post['of_those_with_a_guard_fired']} of them after a guard"
        f" had fired: once any evidence is present the reranker re-sorts by policy"
        f" probability and undoes the guard's promotion)."
        f" {post['B_flips_that_need_the_true_action']} flips"
        f" ({_pct(post['share_that_need_the_true_action'])} of all move decisions)"
        f" need the true action.",
        f"- Those {need['n']} flips: primary {_num(need['mean_diff'])}"
        f" ({_num(need['low'])} to {_num(need['high'])}), secondary on"
        f" {need_exchange['n']} of them {_num(need_exchange['mean_diff'])}"
        f" ({_num(need_exchange['low'])} to {_num(need_exchange['high'])})."
        f" What they change, per slot of ours:"
        f" {_counts(post['what_those_flips_change'])}.",
        f"- With sticky guard corrections on in every directory (today's pipeline)"
        f" the oracle changes the pipeline's own pick in"
        f" {sticky['oracle_pick_differs_from_logged_pick']} decisions"
        f" ({_pct(sticky['share_of_all_move_decisions'])}).",
        "",
        "## Sanity (arm A: the bot's own predictions, recomputed)",
        "",
        f"- Reproduces the played pick in {a['pick_reproduced']} of {a['decisions']}"
        f" rebuilt decisions ({_pct(a['pick_rate'])}; the reading is refused below"
        f" {_pct(reading['min_agreement'])}).",
        f"- Where both the log and the replay have a reranker report"
        f" ({a['both_report']}): same pick after reranking {a['after_same']}, same"
        f" opponent score {a['opponent_score_same']}, same tempo score"
        f" {a['tempo_score_same']}.",
        f"- The log shows the bot making the played pair in"
        f" {own['match_excluding_superseded']} of"
        f" {own['checked_excluding_superseded']} decisions where its own move is"
        f" visible ({e['superseded_in_the_same_turn']} records were refused by the"
        f" server and replaced in the same turn; they are kept and flagged).",
        f"- Sheet state of the decisions: {_counts(summary['sheets']['decisions'])}."
        " A saved page never carries a team sheet, so in open-sheet games the bot"
        " knew more than the rebuilt position does; `open_inferred` = the log shows"
        " an opponent term where a closed sheet leaves no evidence.",
        "",
        "## Limits",
        "",
        "- The primary hindsight score is the quantity arm B optimises, so a positive"
        " mean is expected by construction; only a non-positive one would inform."
        " The secondary score adds the damage our own pair deals; after the first"
        " run it was corrected to leave out a pair that attacks a switch-in never"
        " seen (first counted as zero damage).",
        "- Neither score sees turn order, knock-outs before a move, Fake Out,"
        " redirection our own pick would have changed, status, speed control,"
        " set-up, positioning after a switch, hidden items and abilities (the"
        " opponent's stats are the bot's synthetic spread), or anything after the"
        " turn. It is a one-turn damage ledger, not a win probability.",
        "- The true target is where the move landed. Slots whose action the log does"
        " not show (knocked out first, flinch, sleep, forfeit) are dropped from both"
        " picks; in arm B they keep the bot's logged prediction.",
        "- Only the first eight candidates after the pipeline were logged, without"
        " the strategic-only flag. Not run, and needed to go further: the policy"
        " (candidates beyond those eight), the exact simulator (to play a flip"
        " out), and the team sheets of open-sheet games (never saved).",
        "- Flips are counted, not played out: no game was replayed, so this says"
        " nothing about win rate.",
        "",
    ]
    return "\n".join(lines)


# --- entry point ----------------------------------------------------------------------


def run(
    folders: Sequence[Path], resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows and summary for the given replay directories."""
    torch.set_num_threads(1)
    counters: Counter[str] = Counter()
    census: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    before = Counter(PolicyPlayer.guard_fire_counts)
    for folder in folders:
        folder_rows, folder_census = process_directory(folder, counters)
        rows.extend(folder_rows)
        census.update(folder_census)
    for name, count in (Counter(PolicyPlayer.guard_fire_counts) - before).items():
        if "prediction_error" in name:
            counters[f"bot:{name}"] += count
    for name, count in pokeenv_patches.PARSE_ERRORS.items():
        counters[f"pokeenv_patch:{name}"] += count
    return rows, summarise(rows, census, counters, resamples)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--dirs", default=DEFAULT_GLOB, help="glob of replay directories (repo root)"
    )
    parser.add_argument("--out", default=str(OUT_DIR), help="output directory")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    started = time.time()
    folders = sorted(path for path in ROOT.glob(args.dirs) if path.is_dir())
    rows, summary = run(folders, args.resamples)
    summary = {
        "script": "evaluation/oppmodel_oracle.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": round(time.time() - started, 1),
        "directories": [folder.name for folder in folders],
        "bootstrap": {"resamples": args.resamples, "seed": BOOTSTRAP_SEED},
        "environment": {name: _os.environ.get(name) for name in ENV_SWITCHES},
        "hindsight_functions": {
            "primary": "opponent_reranker._outgoing_damage(true switches)"
            " - opponent_reranker._incoming_damage(true moves)",
            "secondary": "dealt_damage (opponent_reranker._damage on the true"
            " board, capped at HP left) - opponent_reranker._incoming_damage",
        },
        "dex": E.dex_signature().get("dex_available"),
        **summary,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "README.md").write_text(render_readme(summary))
    reading = summary["reading_R4"]
    print(
        f"move decisions {summary['enumeration'][RECORD_MOVE]}, rebuilt"
        f" {summary['rebuild']['rebuilt']}, arm A pick rate"
        f" {_pct(summary['agreement_arm_A']['pick_rate'])}"
    )
    for arm in ORACLE_ARMS:
        print(f"  arm {arm}: flips {summary['arms'][arm]['flips']}")
    print(
        f"R4: flip share {_pct(reading['flip_share'])}, hindsight"
        f" {_num(reading['hindsight_mean_diff'])} -> {reading['verdict']}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
