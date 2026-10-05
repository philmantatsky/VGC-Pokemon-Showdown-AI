"""Scorecard of the opponent predictor: the pre-registered readings R1, R2, R3.

Design: OPPONENT_PREDICTOR.md, "PRE-REGISTERED readings". Every predictor given
(count-table and neural artifacts, and with ``--legacy`` the bot's shipped
opponent models through ``oppmodel.legacy``) goes through one harness on three
evaluation sets of a built dataset:

  (a) ladder_holdout    the opponents in the bot's own saved ladder games
  (b) test              held-out human players
  (c) test_time_slice   (b), only the games in the latest 10% by upload time.
                        Training covers the same period, so (c) is NOT an
                        out-of-time test: it shows the late games of unseen
                        players, not a model meeting a later metagame.

What the harness guarantees:

* A predictor gets FEATURES only. Label (``y_*``) and bookkeeping (``m_*``)
  arrays are removed before ``predict`` and the sheet code "unknown" is read as
  "closed" (``features.sheet_unknown_as_closed``), for every predictor alike.
* A predictor that failed inside ``predict`` and fell back to the uniform
  prediction (a ``predict_error:*`` counter), or returned something that is not
  a finite prediction of the right shape, stops the run. Nothing is scored
  quietly.
* Only the evaluation sets are read. The training split is read only with
  ``--legacy``, for two base rates of the legacy adapter.

Unit: one slot-turn = one of a player's two active Pokemon at the start of one
turn. All probabilities are floored at ``features.PROBABILITY_FLOOR``.

R1, quality (the gate). Mean NLL per labelled slot-turn of the FINE action,
``features.slot_nll``: ``-log`` of the probability of the action taken (a move,
or a switch and its destination; for a slot whose action the log hides, the
probability of the set of actions the log still allows), plus ``-log`` of the
target's probability given the move where the target is known. Reported with
its action part and its target part, and on visible and on hidden slot-turns
separately. A predictor "passes" when its paired difference to the reference
predictor has a 95% interval entirely below zero on (a) AND on (b). Top-1 /
top-3 of the fine action, 7-class intent accuracy and NLL and action top-1 are
reported and not gated.

R2, Elo. The same predictor is scored with the ratings shuffled between the
examples of the set (several seeds; the per-slot NLL is averaged over them),
blanked, and swapped between the two players. The gain of the rating is NLL
shuffled minus NLL kept. With ``--elo-blind`` a retrained model without the
rating is compared to its rating-reading twin (``--elo-model``): gain = NLL
blind minus NLL kept. "Elo stays" only if every gain measured is at least 0.02
nats on (a) and on (b); with the retrain missing a passing shuffle test is
"not decided".

R3, decision relevance (descriptive). For the events "the slot switches", "the
slot uses a Protect move" and "the slot attacks a given opposing slot"
(``features.event_probs``): a 10-bin reliability table, the Brier score and its
skill, and for each threshold the share of slot-turns at or above it with the
precision and the mean prediction there. Denominators: every slot-turn where
the log tells whether the event happened (``y_flag`` switch known / Protect
known, hidden actions included). Skill is given twice: against the set's own
rate, and, since a slot with no legal switch (or no Protect candidate) is
predicted 0 by every predictor for free, among the slot-turns where the event
can happen, against their own rate. The attack label (``y_attack``) exists only
where the click is visible, and it holds where a move LANDED: a slot whose aimed
attack was redirected or retargeted (visible move, target not certain) is left
out. The attack event is reported under two conventions: ``attack`` counts the
slot-turns with a visible click, and ``attack_arrived`` also counts the
slot-turns the log proves were stopped before acting (fainted first, flinch,
sleep, ...) as "no attack arrived". Each also as "a slot is attacked by at
least one of the two predicted slots", taking the two slots' predictions as
independent; a row is known there only when EVERY attacker on the field is
known, whatever the answer (keeping a row with a hidden attacker when the
other one hit would keep it only when the answer is yes). ``attack`` and
``attacked_by_either`` are also split by whether two opposing Pokemon stand
(``*_two_targets``: a real choice of target, the row a guard should read) or
one (``*_one_target``: the target is given, and most confident calls of a
weak target head are these).

Intervals. 95% percentile intervals of a bootstrap that resamples GAMES
(``m_battle``) with replacement; every statistic is a ratio of sums over the
resampled games, and one set of resampled games serves every predictor and
every column of an evaluation set, so differences are paired. That is the
pre-registered interval and the one the verdicts read. The headline
differences are also given with ACCOUNTS (``m_actor``) as the resampling unit:
(b) is split by account and a few accounts hold many of its games, so a small
difference there should be judged by the wider account interval. Slices (sheet
state, bo1 / bo3, turn 1 / later, the actor's rating band, four active Pokemon
or fewer) are descriptive, never a gate.

Joint reply coverage (descriptive; ``joint_coverage`` in the card, sets (a) and
(b); ``--no-joint`` leaves it out). The measure a search asks for: how often
the reply a player made with BOTH slots is among the K joint replies a
predictor ranks highest (K = 1, 2, 4, 8, 16, 32), the mean probability on its
first 8, and the mean log-probability of the true joint reply. The joint is
built by ``oppmodel.joint`` as the product of the two slots' predictions, the
model's own factorisation, with what the rules forbid removed and the rest
renormalised; without the Mega bit (the headline) and with it. Counted are the
examples where every slot on the field shows its whole action (the move and a
certain target, or the switch and its destination); an example with a hidden
action in either slot is left out. A true reply that used a move outside the
candidates (the OTHER bucket) is counted as NOT covered at every K, and as
covered through the bucket in a second column. Top-8 has a game-clustered
interval from a bootstrap of its own; nothing else in the card is touched.
Next to the three reply classes among hidden sheets the section gives their
pooled complement, "no move not shown before" (a switch or every move shown):
with "a move not shown before" it is the two-way split a search session
reports. ``dependence`` sizes what the product of the two slots loses: for a
few pair events (both Protect, both switch, both attack the same slot, ...)
the observed rate on the counted two-slot turns against the product of the
observed marginal rates and against each predictor's own product.

Outputs: ``<out>/scorecard.json`` (everything) and ``<out>/README.md``, which
is rendered from the JSON only (``--render-only`` rewrites it). The card's
``warnings`` list names anything that weakens the reading (an artifact fitted
on another dataset build, a dex difference, a prediction that needed
renormalising). The last line printed is ``SCORECARD_DONE`` (exit code 0) or
``SCORECARD_FAILED <reason>`` (exit code 1, no card written), whatever went
wrong. A directory that already holds a card is not written over without
``--overwrite``.

    nice -n 19 .venv/bin/python evaluation/oppmodel_scorecard.py \\
        --artifact results_oppmodel/tables_v1/species_table.pt \\
        --artifact results_oppmodel/tables_v1/flags_table.pt \\
        --artifact results_oppmodel/tables_v1/elo_table.pt \\
        --legacy --out results_oppmodel/scorecard_tables_v1
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # numpy's BLAS on macOS
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel.artifact import load_predictor
from vgc_bench.src.oppmodel.events import (
    INTENT_ATTACK_FOE_A,
    INTENT_ATTACK_FOE_B,
    INTENT_CLASSES,
    INTENT_PROTECT,
    INTENT_SWITCH,
    REASON_OVERRIDDEN,
)

ROOT = Path(__file__).resolve().parents[1]
LEGACY_MOVE = "data/opponent_move_top500_regmc.pt"  # relative to the repo root
LEGACY_SWITCH = "data/opponent_switch_top500_regmc.pt"
FORMAT = "oppmodel-scorecard"
VERSION = 1
CARD_JSON = "scorecard.json"
CARD_MD = "README.md"

SET_LADDER = "ladder_holdout"
SET_TEST = "test"
SET_TIME = "test_time_slice"
SETS: tuple[str, ...] = (SET_LADDER, SET_TEST, SET_TIME)
GATED_SETS: tuple[str, ...] = (SET_LADDER, SET_TEST)
SET_LABELS = {
    SET_LADDER: "(a) ladder holdout: the opponents in the bot's own ladder games",
    SET_TEST: "(b) test players: held-out human players",
    SET_TIME: "(c) test players, only games in the latest 10% by time "
    "(training covers the same period: not an out-of-time test)",
}
SET_SHORT = {SET_LADDER: "(a) ladder", SET_TEST: "(b) test", SET_TIME: "(c) late test"}
FLAG_TIME_SLICE = 1  # m_flag bit, as the dataset builder documents it
CLUSTER_GAME = "game"
CLUSTER_ACCOUNT = "account"

DEFAULT_REFERENCE = "flags_table"
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261004
DEFAULT_SHUFFLES = 5
ELO_GAIN_BAR = 0.02
LEVEL = 0.95
BINS = 10
UNCHANGED_TOLERANCE = 1e-5
ELO_BAND_EDGES: tuple[int, ...] = (1100, 1200, 1300, 1400)
SLICE_ALL = "all"

EVENT_SWITCH = INTENT_SWITCH
EVENT_PROTECT = INTENT_PROTECT
EVENT_ATTACK = "attack"
EVENT_ATTACK_TWO = "attack_two_targets"
EVENT_ATTACK_ONE = "attack_one_target"
EVENT_ATTACK_ARRIVED = "attack_arrived"
EVENT_EITHER = "attacked_by_either"
EVENT_EITHER_TWO = "attacked_by_either_two_targets"
EVENT_EITHER_ONE = "attacked_by_either_one_target"
EVENT_EITHER_ARRIVED = "attacked_by_either_arrived"
EVENTS: tuple[str, ...] = (
    EVENT_SWITCH,
    EVENT_PROTECT,
    EVENT_ATTACK,
    EVENT_ATTACK_TWO,
    EVENT_ATTACK_ONE,
    EVENT_ATTACK_ARRIVED,
    EVENT_EITHER,
    EVENT_EITHER_TWO,
    EVENT_EITHER_ONE,
    EVENT_EITHER_ARRIVED,
)
EVENT_TEXT = {
    EVENT_SWITCH: "the slot switches out by choice",
    EVENT_PROTECT: "the slot uses a Protect-family move",
    EVENT_ATTACK: "the slot attacks a given opposing slot (click visible)",
    EVENT_ATTACK_TWO: "the slot attacks a given opposing slot, two opposing "
    "Pokemon on the field (click visible)",
    EVENT_ATTACK_ONE: "the slot attacks the only opposing Pokemon on the field "
    "(click visible)",
    EVENT_ATTACK_ARRIVED: "the slot attacks a given opposing slot (stopped slots "
    "count as no attack)",
    EVENT_EITHER: "an opposing slot is attacked by at least one of the two slots "
    "(every click visible)",
    EVENT_EITHER_TWO: "an opposing slot is attacked by at least one of the two "
    "slots, two opposing Pokemon on the field (every click visible)",
    EVENT_EITHER_ONE: "the only opposing Pokemon is attacked by at least one of "
    "the two slots (every click visible)",
    EVENT_EITHER_ARRIVED: "an opposing slot is attacked by at least one of the two "
    "slots (stopped slots count as no attack)",
}
PRE_REGISTERED = {
    EVENT_SWITCH: 0.35,
    EVENT_PROTECT: 0.5,
    EVENT_ATTACK: 0.7,
    EVENT_ATTACK_TWO: 0.7,
    EVENT_ATTACK_ONE: 0.7,
    EVENT_ATTACK_ARRIVED: 0.7,
    EVENT_EITHER: 0.7,
    EVENT_EITHER_TWO: 0.7,
    EVENT_EITHER_ONE: 0.7,
    EVENT_EITHER_ARRIVED: 0.7,
}
EXTRA_THRESHOLDS: tuple[float, ...] = (0.6, 0.8, 0.9)
README_EVENTS: tuple[str, ...] = (
    EVENT_SWITCH,
    EVENT_PROTECT,
    EVENT_ATTACK,
    EVENT_ATTACK_TWO,
)
# The events a predictor answers 0 for free where they cannot happen.
MASKED_EVENTS: tuple[str, ...] = (EVENT_SWITCH, EVENT_PROTECT)

DEFINITIONS = {
    "slot_turn": "one of a player's two active Pokemon at the start of one turn",
    "fine_nll": "mean over labelled slot-turns of -log P(action taken) - log "
    "P(target | move); a hidden action is scored by the probability of the set "
    "of actions the log still allows; value with a 95% interval",
    "parts": "action = the action term and target = the target term, each per "
    "labelled slot-turn (they add up to fine_nll); visible / censored = fine "
    "NLL over the slot-turns with a visible / a hidden action",
    "difference": "this predictor minus the reference on the same slot-turns; "
    "diff with a 95% interval (low, high); negative is better",
    "interval": "percentile interval over bootstrap resamples of whole games; "
    "every statistic is a ratio of sums over the resampled games",
    "fine_top1 / fine_top3": "share of fully visible actions (with target) "
    "among the predictor's first / first three choices",
    "action_top1 / action_top3": "the same for the action alone",
    "intent_accuracy / intent_nll": "over the seven intent classes of the "
    "design; the OTHER bucket's mass counts for no class",
    "elo": "fine NLL under changed ratings; keep_minus_mode is negative when "
    "the rating helps; shuffle is averaged over the seeds",
    "events": "per event: slot-turns where the log tells (slots), observed "
    "rate, mean predicted, Brier score, skill against the set's own rate, a "
    "10-bin reliability table, and per threshold the slot-turns at or above "
    "it, their share, the precision and the mean prediction there and the "
    "share of occurrences caught",
    "events.where_possible": "the same over the slot-turns where the event can "
    "happen (a legal switch target; a valid Protect candidate): impossible = "
    "the known slot-turns left out, skill against the rate of the rest, and "
    "per threshold share_where_possible",
    "events.attack": "the attack label is where a move landed; a visible aimed "
    "attack whose target is not certain (redirected or retargeted) is unknown. "
    "'attacked by either' is known only when every attacker on the field is "
    "known. *_two_targets / *_one_target split the rows by the number of "
    "opposing Pokemon on the field",
    "difference_by_account": "the same paired difference with accounts "
    "(m_actor) as the resampling unit instead of games",
    "sanity": "max_abs_change = largest change normalising makes; floor_hits = "
    "labels whose probability was at or below the floor",
    "slices.games": "games with at least one labelled slot-turn in the slice",
}

# Joint reply coverage: a section of its own, with its own bootstrap.
JOINT_KEY = "joint_coverage"
JOINT_KS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
JOINT_K = 8  # the list length a search keeps: the interval and the mass are for it
JOINT_PLAIN = "without_mega"
JOINT_MEGA = "with_mega"
JOINT_VARIANTS: tuple[str, ...] = (JOINT_PLAIN, JOINT_MEGA)
JOINT_VARIANT_TEXT = {
    JOINT_PLAIN: {
        "short": "without the Mega bit",
        "text": "Without the Mega bit (the headline): a joint reply is the two "
        "slots' actions.",
    },
    JOINT_MEGA: {
        "short": "with the Mega bit",
        "text": "With the Mega bit: a joint reply must also name who "
        "Mega-evolves (nobody, slot a or slot b).",
    },
}
JOINT_SETS: tuple[str, ...] = GATED_SETS
SLICE_SHEETS: tuple[str, ...] = ("sheet closed", "sheet open", "sheet unknown")
SLICE_TURNS: tuple[str, ...] = ("turn 1", "turn 2 and later")
SLOTS_TWO = "slots: two acting"
SLOTS_ONE = "slots: one acting"
REPLY_UNSHOWN = "reply: a move not shown before"
REPLY_SWITCH = "reply: a switch, no move not shown before"
REPLY_SHOWN = "reply: every move shown or on the sheet"
REPLY_CLASSES: tuple[str, ...] = (REPLY_UNSHOWN, REPLY_SWITCH, REPLY_SHOWN)
# The last two classes together: with REPLY_UNSHOWN the two-way split of a
# search session's hidden-sheet readings (a reply that needs a move nobody has
# seen, or one that does not).
REPLY_NO_UNSHOWN = "reply: no move not shown before (a switch, or every move shown)"
SHEETS_NOT_OPEN = "sheets not open, "
# Pair events of a turn with two acting slots, as intent classes of (slot a,
# slot b): what the product of the two slots' predictions gets wrong when the
# two choices go together. Descriptive.
_I = {name: index for index, name in enumerate(INTENT_CLASSES)}
PAIR_EVENTS: dict[str, tuple[tuple[int, int], ...]] = {
    "both use a Protect-family move": ((_I[INTENT_PROTECT], _I[INTENT_PROTECT]),),
    "both switch": ((_I[INTENT_SWITCH], _I[INTENT_SWITCH]),),
    "one switches, the other uses a Protect-family move": (
        (_I[INTENT_SWITCH], _I[INTENT_PROTECT]),
        (_I[INTENT_PROTECT], _I[INTENT_SWITCH]),
    ),
    "both attack the same opposing slot": (
        (_I[INTENT_ATTACK_FOE_A], _I[INTENT_ATTACK_FOE_A]),
        (_I[INTENT_ATTACK_FOE_B], _I[INTENT_ATTACK_FOE_B]),
    ),
    "they attack different opposing slots": (
        (_I[INTENT_ATTACK_FOE_A], _I[INTENT_ATTACK_FOE_B]),
        (_I[INTENT_ATTACK_FOE_B], _I[INTENT_ATTACK_FOE_A]),
    ),
    "one uses a Protect-family move, the other attacks one opposing slot": (
        (_I[INTENT_PROTECT], _I[INTENT_ATTACK_FOE_A]),
        (_I[INTENT_PROTECT], _I[INTENT_ATTACK_FOE_B]),
        (_I[INTENT_ATTACK_FOE_A], _I[INTENT_PROTECT]),
        (_I[INTENT_ATTACK_FOE_B], _I[INTENT_PROTECT]),
    ),
}
JOINT_DEFINITIONS = {
    "example": "One example is one player's turn; the reply is what that player "
    "chose for every Pokemon they had on the field (two slots, or one).",
    "joint": "The predictors answer for each slot on its own. The probability "
    "of a joint reply is the product of the two slots' probabilities: this is "
    "the model's own factorisation, it holds no dependence between the two "
    "choices. What the rules forbid is removed (both slots switching to the "
    "same bench Pokemon; with the Mega bit also two Mega Evolutions, a switch "
    "together with a Mega Evolution, and a Mega Evolution the public state "
    "rules out) and the rest is renormalised.",
    "space": "A slot's reply is a candidate move with its target class, a "
    "switch with its destination, or OTHER: some move outside the candidates, "
    "as one bucket. OTHER holds a place in the ranking, but a true reply that "
    "involves OTHER is counted as NOT covered at every K, because nobody can "
    "play 'some other move'. The column 'OTHER as a hit' counts the bucket "
    "instead; what a consumer gets lies between the two.",
    "counted": "Counted are the examples where every slot on the field shows "
    "its whole action: the move and, where it is aimed, a target that is "
    "certain, or the switch and its destination. An example with a hidden "
    "action in either slot is left out. Those are not a random part of the "
    "turns (a slot that fainted or flinched before it moved), so the numbers "
    "describe turns where both actions were seen.",
    "top": "Top-K: share of counted examples whose true reply is among the K "
    "most probable joint replies. Replies with exactly the same probability "
    "keep a fixed order (Mega state, slot a, slot b).",
    "mass": f"Mass of the top {JOINT_K}: the mean probability a predictor puts "
    f"on its own first {JOINT_K} joint replies.",
    "log_prob": "Mean log p: the mean natural log of the probability of the "
    "true joint reply (where the truth is OTHER, of the bucket), floored at "
    "the card's probability floor.",
    "interval": f"The bracket after top-{JOINT_K} is a 95% percentile interval "
    "from resampling whole games.",
    "slices": "Sheet state is the recorded one; a predictor reads an unknown "
    "sheet as closed. 'A move not shown before': at least one slot used a "
    "move that Pokemon had not shown in this battle and that is not on an "
    "open sheet (a guessed candidate, or OTHER). 'A switch': no such move, "
    "and at least one slot switched. 'Every move shown or on the sheet': the "
    "rest. 'Sheets not open' are those three among the examples whose sheet "
    "is closed or unknown, and 'no move not shown before' is the last two of "
    "them together.",
    "dependence": "What the factorisation loses: for pair events of the "
    "counted turns with two acting slots whose two intent classes are known, "
    "the observed rate, the product of the two slots' observed marginal rates "
    "(the data's own dependence: a ratio of 1 would be independence) and each "
    "predictor's own product (the mean of p_a x p_b over the same turns). The "
    "counted turns are those where both actions were seen, which hold more "
    "switches and Protect moves than turn starts do, so part of every ratio "
    "against a predictor is that selection and not dependence.",
    "search": "This is not the search session's measure. There the unit is a "
    "legal Showdown choice string (move, target and Mega in one) inside one "
    "world whose hidden sets were sampled, and the opponent model is the "
    "bot's own brain in mirror games. Here it is what human players did, "
    "over the public action space: a move the predictor did not list is the "
    "OTHER bucket and can never be covered ('involves OTHER' says how often "
    "that is), and a move the player does not aim has the one target class "
    "'auto'. The search session splits its hidden-sheet decisions in two: the "
    "reply held a move never shown before, or it did not. The rows that "
    "correspond here are 'sheets not open, a move not shown before' and "
    "'sheets not open, no move not shown before'; the second pools switches "
    "with replies whose every move was shown, as theirs does.",
}

VERDICT_PASS = "passes"
VERDICT_FAIL = "fails"
ELO_STAYS = "Elo stays"
ELO_GOES = "Elo does not stay"
ELO_OPEN = "not decided"
MODE_SHUFFLE = F.ELO_SHUFFLE
ELO_MODES: tuple[str, ...] = (F.ELO_SHUFFLE, F.ELO_BLANK, F.ELO_SWAP)
KIND_LEGACY = "legacy"
KIND_OPPNET = "oppnet"

Prediction = dict[str, np.ndarray]
Log = Callable[[str], None]


class ScorecardError(RuntimeError):
    """The run cannot give a valid reading: a failed predictor or a bad input."""


def say(text: str) -> None:
    """Print a progress line at once (stdout may be a file)."""
    print(text, flush=True)


# --- the harness around predict -------------------------------------------------


def features_only(batch: Mapping[str, np.ndarray]) -> F.Batch:
    """A batch without its label (``y_*``) and bookkeeping (``m_*``) arrays."""
    return {
        name: array
        for name, array in batch.items()
        if not name.startswith(("y_", "m_"))
    }


def predict_errors(predictor: Any) -> int:
    """How often the predictor's ``predict`` has failed and fallen back so far."""
    counters = getattr(predictor, "counters", None)
    if not isinstance(counters, Mapping):
        return 0
    return int(
        sum(
            int(count)
            for key, count in counters.items()
            if str(key).startswith("predict_error")
        )
    )


def guarded_predict(
    predictor: Any,
    batch: Mapping[str, np.ndarray],
    *,
    name: str = "predictor",
    elo_mode: str = F.ELO_KEEP,
    rng: np.random.Generator | None = None,
) -> tuple[Prediction, float]:
    """``predictor.predict`` as the scorecard calls it: (prediction, seconds).

    The predictor sees the features only, with unknown sheets read as closed
    and, for an Elo probe, the ratings changed by ``features.apply_elo_mode``.
    Raises ``ScorecardError`` when the predictor raised, counted a
    ``predict_error``, or returned something that cannot be scored.
    """
    features = F.sheet_unknown_as_closed(features_only(batch))
    if elo_mode != F.ELO_KEEP:
        features = F.apply_elo_mode(features, elo_mode, rng)
    before = predict_errors(predictor)
    started = time.perf_counter()
    try:
        made = predictor.predict(features)
    except Exception as exc:
        raise ScorecardError(f"{name}: predict raised {exc!r}") from exc
    seconds = time.perf_counter() - started
    if predict_errors(predictor) > before:
        counters = dict(getattr(predictor, "counters", {}))
        raise ScorecardError(
            f"{name}: predict failed and fell back to the uniform prediction "
            f"(counters {counters}); it is not scored"
        )
    if not isinstance(made, Mapping) or any(
        key not in made for key in ("action", "target", "mega")
    ):
        raise ScorecardError(f"{name}: predict did not return action / target / mega")
    pred = {
        key: np.asarray(made[key], dtype=np.float64)
        for key in ("action", "target", "mega")
    }
    if not all(np.isfinite(value).all() for value in pred.values()):
        raise ScorecardError(f"{name}: the prediction holds a non-finite number")
    try:
        F.normalize_prediction(pred, batch)
    except ValueError as exc:
        raise ScorecardError(f"{name}: the prediction cannot be scored: {exc}") from exc
    return pred, seconds


def sanity_block(
    pred: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    seconds: float | None = None,
    floor: float = F.PROBABILITY_FLOOR,
) -> dict[str, Any]:
    """Whether a prediction is already a clean probability table.

    ``max_abs_change``: the largest change ``features.normalize_prediction``
    makes to each array (0 = mass only on legal entries and rows summing to
    1). ``floor_hits``: scored labels whose probability is at or below the
    floor, so that their NLL is the capped ``-log(floor)``.
    """
    norm = F.normalize_prediction(pred, batch)
    change = {
        key: float(np.abs(norm[key] - np.asarray(pred[key], dtype=np.float64)).max())
        if norm[key].size
        else 0.0
        for key in ("action", "target", "mega")
    }
    nll = F.slot_nll(pred, batch, floor)
    limit = -math.log(floor) - 1e-9
    n = int(np.asarray(batch["action_mask"]).shape[0])
    out: dict[str, Any] = {
        "max_abs_change": change,
        "unchanged": bool(max(change.values()) <= UNCHANGED_TOLERANCE),
        "non_finite": int(
            sum(int((~np.isfinite(np.asarray(v))).sum()) for v in pred.values())
        ),
        "floor_hits": {
            "action": int((nll["action_scored"] & (nll["action"] >= limit)).sum()),
            "target": int((nll["target_scored"] & (nll["target"] >= limit)).sum()),
        },
    }
    if seconds is not None:
        out["predict_seconds"] = float(seconds)
        out["microseconds_per_example"] = 1e6 * float(seconds) / n if n else None
    return out


# --- game-clustered bootstrap ---------------------------------------------------


def _interval(samples: np.ndarray, level: float = LEVEL) -> tuple[float, float] | None:
    kept = samples[np.isfinite(samples)]
    if kept.size == 0:
        return None
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(kept, [tail, 1.0 - tail])
    return float(low), float(high)


class Draws:
    """Column totals of the real games and of each bootstrap resample of them."""

    def __init__(
        self,
        names: Mapping[Hashable, int],
        totals: np.ndarray,
        samples: np.ndarray,
        n_games: int,
    ) -> None:
        self.names = dict(names)
        self.totals = totals
        self.samples = samples
        self.n_games = n_games

    @property
    def resamples(self) -> int:
        return int(self.samples.shape[0])

    def total(self, name: Hashable) -> float:
        return float(self.totals[self.names[name]])

    def apply(
        self, function: Callable[..., Any], *names: Hashable
    ) -> dict[str, float | None]:
        """``function`` of the named column totals: value and 95% interval.

        ``function`` gets one float per name for the value and one array per
        name for the resamples; a resample where it is not finite (an empty
        denominator) is dropped.
        """
        columns = [self.names[name] for name in names]
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.asarray(
                function(*(np.float64(self.totals[c]) for c in columns)),
                dtype=np.float64,
            )
            drawn = (
                np.asarray(
                    function(*(self.samples[:, c] for c in columns)), dtype=np.float64
                )
                if self.resamples
                else np.empty(0)
            )
        point = float(value) if np.isfinite(value) else None
        bounds = _interval(drawn) if point is not None and self.n_games >= 2 else None
        return {
            "value": point,
            "low": bounds[0] if bounds else None,
            "high": bounds[1] if bounds else None,
        }

    def ratio(self, numerator: Hashable, denominator: Hashable) -> dict[str, Any]:
        """Sum of one column over sum of another: value and 95% interval."""
        return self.apply(lambda top, bottom: top / bottom, numerator, denominator)


class GameSums:
    """Per-game sums of per-slot quantities: the columns of the bootstrap.

    ``battle`` is the game of every example. ``add(name, values, mask)`` sums
    ``values`` (leading axis = examples, any trailing shape) over the entries
    where ``mask`` holds, per game. ``draws`` resamples the games.
    """

    def __init__(self, battle: np.ndarray) -> None:
        games = np.asarray(battle).reshape(-1)
        if games.size:
            _, index = np.unique(games, return_inverse=True)
            self.index = np.asarray(index, dtype=np.int64).reshape(-1)
            self.n_games = int(self.index.max()) + 1
        else:
            self.index = np.zeros(0, dtype=np.int64)
            self.n_games = 0
        self._names: dict[Hashable, int] = {}
        self._columns: list[np.ndarray] = []

    def add(
        self, name: Hashable, values: np.ndarray, mask: np.ndarray | None = None
    ) -> None:
        if name in self._names:
            raise ValueError(f"column {name!r} was added twice")
        data = np.asarray(values, dtype=np.float64)
        if data.shape[:1] != self.index.shape:
            raise ValueError(f"column {name!r}: one row per example is needed")
        shape = (-1, *([1] * (data.ndim - 1)))
        games = np.broadcast_to(self.index.reshape(shape), data.shape)
        if mask is None:
            picked, weights = games.reshape(-1), data.reshape(-1)
        else:
            keep = np.broadcast_to(np.asarray(mask, dtype=bool), data.shape)
            picked, weights = games[keep], data[keep]
        column = np.bincount(picked, weights=weights, minlength=self.n_games)
        self._names[name] = len(self._columns)
        self._columns.append(column.astype(np.float64))

    def count(self, name: Hashable, mask: np.ndarray) -> None:
        """Number of entries of ``mask`` (full per-slot shape) per game."""
        keep = np.asarray(mask, dtype=bool)
        self.add(name, np.ones(keep.shape, dtype=np.float64), keep)

    def column(self, name: Hashable) -> np.ndarray:
        return self._columns[self._names[name]]

    def draws(self, resamples: int, seed: int, chunk: int = 200) -> Draws:
        """Totals over the real games and over ``resamples`` resamples of them."""
        width = len(self._columns)
        if width == 0 or self.n_games == 0:
            return Draws(
                self._names, np.zeros(width), np.zeros((0, width)), self.n_games
            )
        matrix = np.stack(self._columns, axis=1)  # [games, columns]
        totals = matrix.sum(axis=0)
        count = max(0, int(resamples)) if self.n_games >= 2 else 0
        samples = np.zeros((count, width), dtype=np.float64)
        rng = np.random.default_rng(seed)
        share = np.full(self.n_games, 1.0 / self.n_games)
        for start in range(0, count, chunk):
            size = min(chunk, count - start)
            picks = rng.multinomial(self.n_games, share, size=size)
            samples[start : start + size] = picks.astype(np.float64) @ matrix
        return Draws(self._names, totals, samples, self.n_games)


def _as_difference(found: Mapping[str, Any]) -> dict[str, Any]:
    return {"diff": found["value"], "low": found["low"], "high": found["high"]}


def paired_difference(
    first: np.ndarray,
    second: np.ndarray,
    scored: np.ndarray,
    battle: np.ndarray,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Mean of ``first - second`` over scored slots, games resampled.

    The estimator is the sum of the per-slot differences over the number of
    scored slots (a ratio of sums over games), and both predictors are read on
    the same resampled games. Returns ``diff``, ``low``, ``high`` (95%),
    ``slots`` and ``games``; no interval with fewer than two games.
    """
    keep = np.asarray(scored, dtype=bool)
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    sums = GameSums(battle)
    sums.add("difference", delta, keep)
    sums.count("slots", keep)
    out = _as_difference(sums.draws(resamples, seed).ratio("difference", "slots"))
    out["slots"] = int(keep.sum())
    out["games"] = int((sums.column("slots") > 0).sum())
    return out


# --- events ---------------------------------------------------------------------


def thresholds_for(event: str) -> tuple[float, ...]:
    """The pre-registered threshold of an event and the common ones, ascending."""
    return tuple(sorted({PRE_REGISTERED[event], *EXTRA_THRESHOLDS}))


def event_labels(
    batch: Mapping[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each event: (the log tells whether it happened, it happened).

    ``switch`` / ``protect`` are ``[N, 2]`` over the actor's slots and are known
    wherever ``y_flag`` says so, hidden actions included. ``attack`` is
    ``[N, 2, 2]`` (actor slot, opposing slot) and known where ``y_attack`` is
    not -1, which needs a visible click, and where the target is not merely
    where the move landed: a visible move that attacked a slot while its
    target is not certain (``y_target`` -1: a redirector stood, or a foe had
    fainted and the move was retargeted) is unknown for both opposing slots.
    ``attack_arrived`` adds the slots the log proves chose a move and were
    stopped before making it (a hidden action with a set loss, except a click
    another move overrode) as "not attacked". The ``attacked_by_either``
    events are ``[N, 2]`` over the OPPOSING slots and are known only when
    every actor slot on the field is known for that opposing slot, whatever
    the answer. ``*_two_targets`` / ``*_one_target`` are the same rows split
    by whether both opposing slots hold a Pokemon.
    """
    active = np.asarray(batch["act_mon"]) >= 0
    foes = np.asarray(batch["foe_mon"]) >= 0
    foe_up = foes[:, None, :]
    flags = np.asarray(batch["y_flag"])
    attack = np.asarray(batch["y_attack"]).astype(np.int64)
    kind = np.asarray(batch["y_kind"]).astype(np.int64)
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    n_move = np.asarray(batch["cand_tmask"]).shape[-1]
    overridden = F.REASON_NAMES.index(REASON_OVERRIDDEN)
    stopped = (
        active
        & np.asarray(batch["y_set"]).astype(bool).any(-1)
        & (y_action < 0)
        & ((kind == F.Y_KIND_HIDDEN) | (kind == F.Y_KIND_NONE))
        & (np.asarray(batch["y_reason"]).astype(np.int64) != overridden)
    )
    landed_only = (
        (y_action >= 0)
        & (y_action < n_move)
        & (np.asarray(batch["y_target"]).astype(np.int64) < 0)
        & (attack == 1).any(-1)
    )
    click = active[..., None] & (attack != -1) & ~landed_only[..., None]
    hit = click & (attack == 1)
    arrived = click | (stopped[..., None] & foe_up)
    two = foes.all(-1)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {
        EVENT_SWITCH: (
            active & (flags[..., F.Y_SWITCH_KNOWN] == 1),
            flags[..., F.Y_SWITCHED] == 1,
        ),
        EVENT_PROTECT: (
            active & (flags[..., F.Y_PROTECT_KNOWN] == 1),
            flags[..., F.Y_PROTECTED] == 1,
        ),
        EVENT_ATTACK: (click, hit),
        EVENT_ATTACK_TWO: (click & two[:, None, None], hit & two[:, None, None]),
        EVENT_ATTACK_ONE: (click & ~two[:, None, None], hit & ~two[:, None, None]),
        EVENT_ATTACK_ARRIVED: (arrived, hit),
    }
    for name, known in ((EVENT_EITHER, click), (EVENT_EITHER_ARRIVED, arrived)):
        yes = (known & hit).any(axis=1)
        # Every attacker on the field must be known, whatever the answer: a row
        # with a hidden attacker kept because the other one hit would be kept
        # only when the answer is yes.
        every = (known | ~active[..., None]).all(axis=1)
        out[name] = (every & foes & active.any(-1)[:, None], yes & every)
    either, yes = out[EVENT_EITHER]
    out[EVENT_EITHER_TWO] = (either & two[:, None], yes & two[:, None])
    out[EVENT_EITHER_ONE] = (either & ~two[:, None], yes & ~two[:, None])
    return {name: out[name] for name in EVENTS}


def event_possible(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """For the events of ``MASKED_EVENTS``: whether the event can happen at all.

    ``[N, 2]`` over the actor's slots. A switch needs a legal switch pointer; a
    Protect needs a valid Protect-family candidate (``features.event_probs``
    gives the OTHER bucket to neither). Where it is False every predictor
    answers exactly 0 because of the mask, not because of anything it learned.
    """
    mask = np.asarray(batch["action_mask"]).astype(bool)
    flags = np.asarray(batch["cand_flag"]).astype(np.int64)
    n_cand = flags.shape[-1]
    guard = (
        ((flags & F.CAND_VALID) > 0)
        & ((flags & F.CAND_PROTECT) > 0)
        & mask[..., :n_cand]
    )
    return {EVENT_SWITCH: mask[..., n_cand + 1 :].any(-1), EVENT_PROTECT: guard.any(-1)}


def event_predictions(
    pred: Mapping[str, np.ndarray], batch: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Predicted probability of each event, shaped like ``event_labels``."""
    events = F.event_probs(pred, batch)
    if not events:
        raise ScorecardError("the prediction does not fit the batch (event_probs)")
    attack = np.asarray(events["attack"], dtype=np.float64)
    either = 1.0 - np.prod(1.0 - np.clip(attack, 0.0, 1.0), axis=1)
    return {
        EVENT_SWITCH: np.asarray(events[INTENT_SWITCH], dtype=np.float64),
        EVENT_PROTECT: np.asarray(events[INTENT_PROTECT], dtype=np.float64),
        EVENT_ATTACK: attack,
        EVENT_ATTACK_TWO: attack,
        EVENT_ATTACK_ONE: attack,
        EVENT_ATTACK_ARRIVED: attack,
        EVENT_EITHER: either,
        EVENT_EITHER_TWO: either,
        EVENT_EITHER_ONE: either,
        EVENT_EITHER_ARRIVED: either,
    }


def reliability_table(
    probability: np.ndarray, outcome: np.ndarray, bins: int = BINS
) -> list[dict[str, Any]]:
    """Equal-width bins of the predicted probability: count, mean, observed rate.

    Bin ``i`` holds ``i / bins <= p < (i + 1) / bins``; the last one includes 1.
    """
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    index = np.clip(np.floor(p * bins).astype(np.int64), 0, bins - 1)
    count = np.bincount(index, minlength=bins)
    sum_p = np.bincount(index, weights=p, minlength=bins)
    sum_y = np.bincount(index, weights=y, minlength=bins)
    return [
        {
            "from": i / bins,
            "to": (i + 1) / bins,
            "slots": int(count[i]),
            "predicted": float(sum_p[i] / count[i]) if count[i] else None,
            "observed": float(sum_y[i] / count[i]) if count[i] else None,
        }
        for i in range(bins)
    ]


def brier_summary(probability: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    """Brier score, the base-rate predictor's Brier score, and the skill.

    Skill = 1 - Brier / (rate * (1 - rate)) with ``rate`` the observed rate of
    the same slot-turns: above 0 beats always predicting that rate.
    """
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(outcome, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return {
            "slots": 0,
            "observed": None,
            "predicted": None,
            "brier": None,
            "brier_base_rate": None,
            "skill": None,
        }
    rate = float(y.mean())
    score = float(((p - y) ** 2).mean())
    base = rate * (1.0 - rate)
    return {
        "slots": int(p.size),
        "observed": rate,
        "predicted": float(p.mean()),
        "brier": score,
        "brier_base_rate": base,
        "skill": 1.0 - score / base if base > 0 else None,
    }


def decision_row(
    probability: np.ndarray, outcome: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Share of slot-turns at or above ``threshold``, precision and recall there.

    ``predicted`` is the mean prediction among those slot-turns: next to the
    precision it shows whether the confident calls are calibrated.
    """
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    y = np.asarray(outcome, dtype=bool).reshape(-1)
    fired = p >= threshold
    hits = int((fired & y).sum())
    return {
        "threshold": float(threshold),
        "slots": int(fired.sum()),
        "share": float(fired.mean()) if p.size else None,
        "precision": hits / int(fired.sum()) if fired.any() else None,
        "predicted": float(p[fired].mean()) if fired.any() else None,
        "recall": hits / int(y.sum()) if y.any() else None,
    }


def _skill(sse: Any, hits: Any, slots: Any) -> Any:
    rate = hits / slots
    return 1.0 - (sse / slots) / (rate * (1.0 - rate))


# --- slices ---------------------------------------------------------------------


def slice_masks(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Named per-example masks; ``all`` first. Descriptive, never a gate."""
    n = int(np.asarray(batch["action_mask"]).shape[0])
    flags = np.asarray(batch["game_flag"])
    sheet = np.asarray(batch["m_sheet"] if "m_sheet" in batch else flags[:, 2])
    turn = np.asarray(batch["turn"]).astype(np.int64)
    elo = np.asarray(batch["elo"])[:, 0].astype(np.int64)
    rated = np.asarray(batch["elo_known"])[:, 0] > 0
    full = (np.asarray(batch["act_mon"]) >= 0).all(-1) & (
        np.asarray(batch["foe_mon"]) >= 0
    ).all(-1)
    out: dict[str, np.ndarray] = {SLICE_ALL: np.ones(n, dtype=bool)}
    for code, name in (
        (F.SHEET_CLOSED, "closed"),
        (F.SHEET_OPEN, "open"),
        (F.SHEET_UNKNOWN, "unknown"),
    ):
        out[f"sheet {name}"] = sheet == code
    out["bo1"] = flags[:, F.G_BO3] == 0
    out["bo3"] = flags[:, F.G_BO3] > 0
    out["turn 1"] = turn == 1
    out["turn 2 and later"] = turn > 1
    edges = (-(10**9), *ELO_BAND_EDGES, 10**9)
    for low, high in zip(edges[:-1], edges[1:]):
        if low < 0:
            name = f"actor Elo below {high}"
        elif high > 10**8:
            name = f"actor Elo {low} and above"
        else:
            name = f"actor Elo {low}-{high - 1}"
        out[name] = rated & (elo >= low) & (elo < high)
    out["actor Elo unknown"] = ~rated
    out["four active Pokemon"] = full
    out["fewer than four active"] = ~full
    return out


# --- verdicts -------------------------------------------------------------------


def r1_verdict(differences: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """R1: passes when the interval is entirely below zero on (a) and on (b).

    ``differences[set]`` holds the paired difference to the reference with its
    ``high`` bound. A set without an interval cannot pass.
    """
    below: dict[str, bool] = {}
    for name in GATED_SETS:
        found = differences.get(name) or {}
        high = found.get("high")
        below[name] = high is not None and float(high) < 0.0
    return {
        "interval_below_zero": below,
        "verdict": VERDICT_PASS if all(below.values()) else VERDICT_FAIL,
    }


def r2_verdict(
    shuffle_gain: Mapping[str, float | None],
    blind_gain: Mapping[str, float | None] | None = None,
    bar: float = ELO_GAIN_BAR,
) -> dict[str, Any]:
    """R2: Elo stays only if every measured gain is at least ``bar`` on (a) and (b).

    ``shuffle_gain[set]`` = NLL with the ratings shuffled minus NLL with them
    kept; ``blind_gain[set]`` = NLL of the Elo-blind retrain minus NLL of the
    model, or None when no retrain was scored. A passing shuffle test without
    the retrain is "not decided".
    """

    def reaches(gains: Mapping[str, float | None]) -> bool:
        return all(
            gains.get(name) is not None and float(gains[name] or 0.0) >= bar
            for name in GATED_SETS
        )

    tests = {"shuffle": reaches(shuffle_gain)}
    if blind_gain is not None:
        tests["blind_retrain"] = reaches(blind_gain)
    if not all(tests.values()):
        verdict = ELO_GOES
    elif blind_gain is None:
        verdict = ELO_OPEN
    else:
        verdict = ELO_STAYS
    return {"bar": bar, "reaches_bar": tests, "verdict": verdict}


# --- scoring one evaluation set -------------------------------------------------


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    return float(values[mask].mean()) if mask.any() else None


def _top_metrics(
    pred: Mapping[str, np.ndarray],
    batch: Mapping[str, np.ndarray],
    nll: Mapping[str, np.ndarray],
    tables: F.Tables | None,
    floor: float,
) -> dict[str, Any]:
    """The reported, ungated numbers: top-k, intent, Mega."""
    norm = F.normalize_prediction(pred, batch)
    joint, label = F.fine_probs(pred, batch), F.fine_label(batch)
    y_action = np.asarray(batch["y_action"])
    out: dict[str, Any] = {}
    for k in (1, 3):
        hits, counted = F.topk_hits(joint, label, k)
        out[f"fine_top{k}"] = _mean(hits.astype(np.float64), counted)
        hits, counted = F.topk_hits(norm["action"], y_action, k)
        out[f"action_top{k}"] = _mean(hits.astype(np.float64), counted)
    out["fine_labels"] = int((label >= 0).sum())
    out["action_labels"] = int((np.asarray(y_action) >= 0).sum())
    out["mega_nll"] = _mean(nll["mega"], nll["mega_scored"])
    out["mega_labels"] = int(nll["mega_scored"].sum())
    out["intent_accuracy"] = out["intent_nll"] = None
    out["intent_labels"] = 0
    if tables is not None:
        intents = F.intent_probs(pred, batch, tables)
        if intents is None:
            raise ScorecardError("intent_probs could not read the prediction")
        y_intent = np.asarray(batch["y_intent"]).astype(np.int64)
        known = y_intent >= 0
        classes = intents[..., : F.N_INTENT]
        guess = classes.argmax(-1)
        picked = np.take_along_axis(
            classes, np.clip(y_intent, 0, F.N_INTENT - 1)[..., None], -1
        )[..., 0]
        out["intent_accuracy"] = _mean((guess == y_intent).astype(np.float64), known)
        out["intent_nll"] = _mean(-np.log(np.maximum(picked, floor)), known)
        out["intent_labels"] = int(known.sum())
    return out


def score_set(
    batch: Mapping[str, np.ndarray],
    preds: Mapping[str, Mapping[str, np.ndarray]],
    reference: str,
    *,
    tables: F.Tables | None = None,
    elo: Mapping[str, Mapping[str, Sequence[np.ndarray]]] | None = None,
    pairs: Sequence[tuple[str, str]] = (),
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    floor: float = F.PROBABILITY_FLOOR,
) -> dict[str, Any]:
    """Every number of one evaluation set, for every predictor.

    ``batch`` carries labels and ``m_battle``; ``preds[name]`` is that
    predictor's raw prediction on it. ``elo[name][mode]`` lists the per-slot
    fine NLL arrays of the predictor under an Elo mode (one per shuffle seed;
    they are averaged). ``pairs`` are (model, its Elo-blind retrain) names.
    With ``m_actor`` in the batch the headline differences are also given with
    accounts as the resampling unit (``difference_by_account``).
    """
    if reference not in preds:
        raise ScorecardError(f"reference {reference!r} is not among the predictors")
    n = int(np.asarray(batch["action_mask"]).shape[0])
    sums = GameSums(np.asarray(batch["m_battle"]))
    nlls = {name: F.slot_nll(pred, batch, floor) for name, pred in preds.items()}
    base = nlls[reference]
    scored = base["fine_scored"]
    censored = base["censored"]
    visible = scored & ~censored
    targeted = base["target_scored"]
    slices = {name: scored & mask[:, None] for name, mask in slice_masks(batch).items()}
    for name, mask in slices.items():
        sums.count(("n", name), mask)
    sums.count("n_visible", visible)
    sums.count("n_censored", censored)
    parts = {
        "action": ("action", scored, ("n", SLICE_ALL)),
        "target": ("target", scored, ("n", SLICE_ALL)),
        "visible": ("fine", visible, "n_visible"),
        "censored": ("fine", censored, "n_censored"),
    }
    for name, nll in nlls.items():
        for label, mask in slices.items():
            sums.add(("fine", name, label), nll["fine"], mask)
            if name != reference:
                sums.add(("d", name, label), nll["fine"] - base["fine"], mask)
        for label, (key, mask, _) in parts.items():
            sums.add(("part", name, label), nll[key], mask)
            if name != reference:
                sums.add(("d_part", name, label), nll[key] - base[key], mask)
    elo = elo or {}
    elo_mean: dict[tuple[str, str], np.ndarray] = {}
    for name, modes in elo.items():
        for mode, arrays in modes.items():
            if name not in nlls or not arrays:
                continue
            mean = np.mean(np.stack([np.asarray(a) for a in arrays]), axis=0)
            elo_mean[(name, mode)] = mean
            sums.add(("fine_elo", name, mode), mean, scored)
            sums.add(("d_elo", name, mode), nlls[name]["fine"] - mean, scored)
    for keep, blind in pairs:
        if keep in nlls and blind in nlls:
            sums.add(
                ("d_pair", keep, blind),
                nlls[keep]["fine"] - nlls[blind]["fine"],
                scored,
            )
    labels = event_labels(batch)
    masked = event_possible(batch)
    # Per event: the known slot-turns where the event can happen at all.
    possible = {
        event: known & masked[event] if event in masked else known
        for event, (known, _) in labels.items()
    }
    probabilities = {
        name: event_predictions(pred, batch) for name, pred in preds.items()
    }
    for event, (known, outcome) in labels.items():
        sums.count(("ev_n", event), known)
        sums.add(("ev_y", event), outcome.astype(np.float64), known)
        if event in masked:
            sums.count(("evp_n", event), possible[event])
            sums.add(("evp_y", event), outcome.astype(np.float64), possible[event])
        for name, events in probabilities.items():
            p = events[event]
            sums.add(("ev_sse", name, event), (p - outcome) ** 2, known)
            if event in masked:
                sums.add(("evp_sse", name, event), (p - outcome) ** 2, possible[event])
            for threshold in thresholds_for(event):
                fired = known & (p >= threshold)
                sums.count(("ev_fired", name, event, threshold), fired)
                sums.count(("ev_hit", name, event, threshold), fired & outcome)
    draws = sums.draws(resamples, seed)

    # The same headline differences with ACCOUNTS as the resampling unit.
    by_account: Draws | None = None
    accounts = 0
    if "m_actor" in batch:
        people = GameSums(np.asarray(batch["m_actor"]))
        people.count("n", scored)
        for name, nll in nlls.items():
            if name != reference:
                people.add(("d", name), nll["fine"] - base["fine"], scored)
        for (name, mode), mean in elo_mean.items():
            people.add(("d_elo", name, mode), nlls[name]["fine"] - mean, scored)
        for keep, blind in pairs:
            if keep in nlls and blind in nlls:
                people.add(
                    ("d_pair", keep, blind),
                    nlls[keep]["fine"] - nlls[blind]["fine"],
                    scored,
                )
        by_account = people.draws(resamples, seed)
        accounts = int((people.column("n") > 0).sum())

    def games_in(column: Hashable) -> int:
        return int((sums.column(column) > 0).sum())

    out: dict[str, Any] = {
        "examples": n,
        "games": sums.n_games,
        "slot_turns": int((np.asarray(batch["act_mon"]) >= 0).sum()),
        "slots": {
            "scored": int(scored.sum()),
            "visible": int(visible.sum()),
            "censored": int(censored.sum()),
            "target_labels": int(targeted.sum()),
        },
        "slices": {
            name: {"slots": int(mask.sum()), "games": games_in(("n", name))}
            for name, mask in slices.items()
        },
        "events": {
            event: {
                "known": int(known.sum()),
                "observed": float(outcome[known].mean()) if known.any() else None,
                "impossible": int((known & ~possible[event]).sum()),
            }
            for event, (known, outcome) in labels.items()
        },
        "clusters": {
            CLUSTER_GAME: games_in(("n", SLICE_ALL)),
            CLUSTER_ACCOUNT: accounts if by_account is not None else None,
        },
        "resamples": draws.resamples,
        "predictors": {},
        "pairs": [],
    }
    for name, pred in preds.items():
        nll = nlls[name]
        row: dict[str, Any] = {
            "fine_nll": draws.ratio(("fine", name, SLICE_ALL), ("n", SLICE_ALL)),
            "parts": {
                label: draws.total(("part", name, label)) / draws.total(count)
                if draws.total(count) > 0
                else None
                for label, (_, _, count) in parts.items()
            },
            "target_nll_per_label": _mean(nll["target"], targeted),
            "action_nll_visible": _mean(nll["action"], visible),
        }
        row.update(_top_metrics(pred, batch, nll, tables, floor))
        row["difference_by_account"] = None
        if name == reference:
            row["difference"] = None
        else:
            row["difference"] = {
                "fine": _as_difference(
                    draws.ratio(("d", name, SLICE_ALL), ("n", SLICE_ALL))
                ),
                **{
                    label: _as_difference(draws.ratio(("d_part", name, label), count))
                    for label, (_, _, count) in parts.items()
                },
            }
            if by_account is not None:
                row["difference_by_account"] = _as_difference(
                    by_account.ratio(("d", name), "n")
                )
        row["slices"] = {}
        for label in slices:
            entry: dict[str, Any] = {
                "fine_nll": draws.ratio(("fine", name, label), ("n", label))["value"]
            }
            if name != reference:
                entry["difference"] = _as_difference(
                    draws.ratio(("d", name, label), ("n", label))
                )
            row["slices"][label] = entry
        modes = {mode for (who, mode) in elo_mean if who == name}
        if modes:
            row["elo"] = {}
            for mode in sorted(modes):
                seeds = [
                    _mean(nll["fine"] - np.asarray(a), scored) for a in elo[name][mode]
                ]
                row["elo"][mode] = {
                    "fine_nll": draws.ratio(("fine_elo", name, mode), ("n", SLICE_ALL))[
                        "value"
                    ],
                    "keep_minus_mode": _as_difference(
                        draws.ratio(("d_elo", name, mode), ("n", SLICE_ALL))
                    ),
                    "keep_minus_mode_by_account": None
                    if by_account is None
                    else _as_difference(by_account.ratio(("d_elo", name, mode), "n")),
                    "keep_minus_mode_by_seed": seeds,
                }
        row["events"] = {}
        for event, (known, outcome) in labels.items():
            p = probabilities[name][event]
            summary = brier_summary(p[known], outcome[known])
            summary["skill"] = draws.apply(
                _skill, ("ev_sse", name, event), ("ev_y", event), ("ev_n", event)
            )
            can = possible[event]
            summary["impossible"] = int((known & ~can).sum())
            summary["slots_where_possible"] = int(can.sum())
            summary["observed_where_possible"] = (
                float(outcome[can].mean()) if can.any() else None
            )
            summary["skill_where_possible"] = (
                draws.apply(
                    _skill, ("evp_sse", name, event), ("evp_y", event), ("evp_n", event)
                )
                if event in masked
                else dict(summary["skill"])
            )
            summary["reliability"] = reliability_table(p[known], outcome[known])
            summary["decisions"] = []
            for threshold in thresholds_for(event):
                decision = decision_row(p[known], outcome[known], threshold)
                decision["pre_registered"] = threshold == PRE_REGISTERED[event]
                decision["precision"] = draws.ratio(
                    ("ev_hit", name, event, threshold),
                    ("ev_fired", name, event, threshold),
                )
                decision["share_where_possible"] = (
                    decision["slots"] / int(can.sum()) if can.any() else None
                )
                summary["decisions"].append(decision)
            row["events"][event] = summary
        out["predictors"][name] = row
    for keep, blind in pairs:
        if ("d_pair", keep, blind) in draws.names:
            out["pairs"].append(
                {
                    "model": keep,
                    "blind": blind,
                    "model_minus_blind": _as_difference(
                        draws.ratio(("d_pair", keep, blind), ("n", SLICE_ALL))
                    ),
                    "model_minus_blind_by_account": None
                    if by_account is None
                    else _as_difference(by_account.ratio(("d_pair", keep, blind), "n")),
                }
            )
    return out


# --- joint reply coverage -------------------------------------------------------


def joint_slices(
    batch: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-example masks of the coverage section; ``all`` first. Descriptive.

    ``truth`` is ``joint.true_replies(batch)``. Four families that each
    partition the examples: the recorded sheet state, two acting slots or one,
    turn 1 or later, and the class of the reply (``REPLY_CLASSES``: a move not
    shown before in either slot; else a switch in either slot; else every
    slot used a move it had shown or that is on its open sheet). A fifth
    family holds the same three classes among the examples whose sheet is not
    open (closed or unknown): a partition of those. One more mask, not part
    of a partition: sheets not open and no move not shown before
    (``REPLY_NO_UNSHOWN``, the last two classes pooled).
    """
    base = slice_masks(batch)
    out: dict[str, np.ndarray] = {SLICE_ALL: base[SLICE_ALL]}
    for name in SLICE_SHEETS:
        out[name] = base[name]
    two = np.asarray(truth["active"]).astype(bool).all(-1)
    out[SLOTS_TWO], out[SLOTS_ONE] = two, ~two
    for name in SLICE_TURNS:
        out[name] = base[name]
    unshown = np.asarray(truth["unshown"]).astype(bool).any(-1)
    switch = np.asarray(truth["switch"]).astype(bool).any(-1) & ~unshown
    classes = dict(zip(REPLY_CLASSES, (unshown, switch, ~unshown & ~switch)))
    out.update(classes)
    closed = ~base["sheet open"]
    for name, mask in classes.items():
        out[SHEETS_NOT_OPEN + name] = closed & mask
    out[SHEETS_NOT_OPEN + REPLY_NO_UNSHOWN] = closed & ~unshown
    return out


def pair_dependence(
    batch: Mapping[str, np.ndarray],
    preds: Mapping[str, Mapping[str, np.ndarray]],
    tables: F.Tables,
) -> dict[str, Any]:
    """How far the two slots' choices are from independent, by pair event.

    ``batch`` holds the counted examples only (with labels); ``preds`` the
    predictions on the same rows. Rows: two acting slots whose intent classes
    the labels give. Per event of ``PAIR_EVENTS``: the observed rate, the
    product of the two slots' observed marginal rates and their ratio (the
    data's own dependence), and per predictor the mean of its own product
    over the same rows with the ratio observed / product. Descriptive; a
    predictor whose intent probabilities cannot be read is left out.
    """
    intent = np.asarray(batch["y_intent"]).astype(np.int64)
    rows = (np.asarray(batch["act_mon"]) >= 0).all(-1) & (intent >= 0).all(-1)
    count = int(rows.sum())
    out: dict[str, Any] = {"examples": count, "events": {}, "predictors": {}}
    if count == 0:
        return out
    first, second = intent[rows, 0], intent[rows, 1]
    width = len(INTENT_CLASSES)
    rate_a = np.bincount(first, minlength=width)[:width] / count
    rate_b = np.bincount(second, minlength=width)[:width] / count
    observed: dict[str, float] = {}
    for label, pairs in PAIR_EVENTS.items():
        hit = np.zeros(count, dtype=bool)
        for a, b in pairs:
            hit |= (first == a) & (second == b)
        observed[label] = float(hit.mean())
        apart = float(sum(rate_a[a] * rate_b[b] for a, b in pairs))
        out["events"][label] = {
            "observed": observed[label],
            "product_of_observed_marginals": apart,
            "ratio": observed[label] / apart if apart > 0 else None,
        }
    part = F.take(batch, rows)
    for name, pred in preds.items():
        own = {key: np.asarray(value)[rows] for key, value in pred.items()}
        probs = F.intent_probs(own, part, tables)
        if probs is None:
            continue
        found: dict[str, Any] = {}
        for label, pairs in PAIR_EVENTS.items():
            product = float(
                sum((probs[:, 0, a] * probs[:, 1, b]).mean() for a, b in pairs)
            )
            found[label] = {
                "product": product,
                "ratio": observed[label] / product if product > 0 else None,
            }
        out["predictors"][name] = found
    return out


def _coverage_row(
    mask: np.ndarray,
    rank: np.ndarray,
    other: np.ndarray,
    mass: np.ndarray,
    log_prob: np.ndarray,
) -> dict[str, Any]:
    """The plain means of one slice: top-K both ways, mass, log-probability."""
    size = int(mask.sum())
    found = rank >= 0
    row: dict[str, Any] = {"examples": size, "top": {}, "top_other_as_hit": {}}
    for k in JOINT_KS:
        inside = found & (rank < k)
        row["top"][str(k)] = float((inside & ~other)[mask].mean()) if size else None
        row["top_other_as_hit"][str(k)] = float(inside[mask].mean()) if size else None
    row["mass_top"] = float(mass[mask].mean()) if size else None
    row["log_prob"] = float(log_prob[mask].mean()) if size else None
    return row


def joint_coverage_set(
    batch: Mapping[str, np.ndarray],
    preds: Mapping[str, Mapping[str, np.ndarray]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    floor: float = F.PROBABILITY_FLOOR,
    tables: F.Tables | None = None,
) -> dict[str, Any]:
    """Joint reply coverage of every predictor on one evaluation set.

    ``batch`` carries labels and ``m_battle``; ``preds[name]`` is a predictor's
    raw prediction on it. The labels decide which examples count and what the
    true reply was (``joint.true_replies``); a prediction is only ranked
    (``joint.joint_replies``, on the feature arrays alone). An example whose
    possible replies carry no probability, or whose true reply is not a
    possible one, is a miss and is counted under ``no_distribution`` /
    ``truth_not_possible``. The bootstrap is this function's own: it shares
    nothing with ``score_set``.

    Per predictor and variant (``JOINT_VARIANTS``) and per slice
    (``joint_slices``): ``top[K]`` (a true reply that involves OTHER is a
    miss), ``top_other_as_hit[K]`` (the bucket counts), ``top_interval`` (top
    ``JOINT_K`` with its 95% interval over resampled games; no interval for a
    slice inside one game), ``mass_top`` (mean mass of the first ``JOINT_K``
    replies) and ``log_prob`` (mean log-probability of the true reply, floored).
    ``left_out`` splits the examples that are not counted into those with a
    slot that shows no free choice (``hidden_action``: hidden, stopped before
    acting, or forced) and those where only a target is not certain. With
    ``tables`` (the featurizer's, for the intent class of a move) the result
    also holds ``dependence``: ``pair_dependence`` on the counted examples.
    """
    truth = J.true_replies(batch)
    if not truth:
        raise ScorecardError(
            f"the labels cannot be read as joint replies ({dict(J.COUNTERS)})"
        )
    n = int(np.asarray(batch["action_mask"]).shape[0])
    active = truth["active"]
    unknown = active & (truth["reply"] == J.UNKNOWN)
    hidden = active & (np.asarray(batch["y_action"]).astype(np.int64) < 0)
    visible = truth["visible"]
    count = int(visible.sum())
    part = F.take(batch, visible)
    # The ranking is given the masks only: no label, no bookkeeping array.
    masks = features_only(part)
    told = {name: values[visible] for name, values in truth.items()}
    other = told["other"].any(-1)
    slices = {name: mask[visible] for name, mask in joint_slices(batch, truth).items()}
    sums = GameSums(np.asarray(part["m_battle"]))
    for label, mask in slices.items():
        sums.count(("n", label), mask)

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for name, pred in preds.items():
        own = {key: np.asarray(value)[visible] for key, value in pred.items()}
        for variant in JOINT_VARIANTS:
            made = J.joint_replies(
                own, masks, k=JOINT_K, mega=variant == JOINT_MEGA, truth=told
            )
            if made.n != count:
                raise ScorecardError(
                    f"{name}: the prediction cannot be read as joint replies "
                    f"({dict(J.COUNTERS)})"
                )
            found = made.rank >= 0
            hit = found & (made.rank < JOINT_K) & ~other
            log_prob = np.log(np.maximum(made.prob_true, floor))
            mass = made.mass()
            for label, mask in slices.items():
                sums.count(("hit", name, variant, label), hit & mask)
            rows[(name, variant)] = {
                "no_distribution": int((~made.ok).sum()),
                "truth_not_possible": int((made.ok & ~found).sum()),
                # the truth's place in the first JOINT_K hangs on the order of
                # replies with exactly its probability
                "decided_by_a_tie": int(
                    (
                        found
                        & (made.above < JOINT_K)
                        & (made.above + made.ties >= JOINT_K)
                    ).sum()
                ),
                "replies_possible": float(made.size.mean()) if count else None,
                "mass_kept": float(made.kept.mean()) if count else None,
                "slices": {
                    label: _coverage_row(mask, made.rank, other, mass, log_prob)
                    for label, mask in slices.items()
                },
            }
    draws = sums.draws(resamples, seed)

    def games_in(label: str) -> int:
        return int((sums.column(("n", label)) > 0).sum())

    out: dict[str, Any] = {
        "examples": n,
        "games": GameSums(np.asarray(batch["m_battle"])).n_games,
        "slot_turns": {
            "acting": int(active.sum()),
            "not_fully_visible": int(unknown.sum()),
        },
        "counted": {"examples": count, "games": games_in(SLICE_ALL)},
        "left_out": {
            "examples": n - count,
            "hidden_action": int(hidden.any(-1).sum()),
            "target_not_certain": int((unknown.any(-1) & ~hidden.any(-1)).sum()),
        },
        "truth": {
            "involves_other": int(other.sum()),
            "involves_switch": int(told["switch"].any(-1).sum()),
            "mega": int(np.isin(told["mega"], (J.MEGA_A, J.MEGA_B)).sum()),
            "mega_not_known": int((told["mega"] == J.UNKNOWN).sum()),
        },
        "slices": {
            label: {
                "examples": int(mask.sum()),
                "games": games_in(label),
                "involves_other": int((mask & other).sum()),
            }
            for label, mask in slices.items()
        },
        "resamples": draws.resamples,
        "predictors": {},
    }
    for (name, variant), row in rows.items():
        for label in slices:
            found = draws.ratio(("hit", name, variant, label), ("n", label))
            if games_in(label) < 2:  # one game resampled is that game again
                found["low"] = found["high"] = None
            row["slices"][label]["top_interval"] = found
        out["predictors"].setdefault(name, {})[variant] = row
    if tables is not None:
        counted = {
            name: {key: np.asarray(value)[visible] for key, value in pred.items()}
            for name, pred in preds.items()
        }
        out["dependence"] = pair_dependence(part, counted, tables)
    return out


def joint_coverage(
    batches: Mapping[str, Mapping[str, np.ndarray]],
    preds: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    log: Log = say,
    tables: F.Tables | None = None,
) -> dict[str, Any]:
    """The card's ``joint_coverage`` section: ``joint_coverage_set`` per set."""
    out: dict[str, Any] = {
        "ks": list(JOINT_KS),
        "k": JOINT_K,
        "variants": {key: dict(text) for key, text in JOINT_VARIANT_TEXT.items()},
        "definitions": dict(JOINT_DEFINITIONS),
        "sets": {},
    }
    for set_name, batch in batches.items():
        if len(batch["turn"]) == 0:
            out["sets"][set_name] = {"examples": 0, "games": 0, "predictors": {}}
            continue
        started = time.perf_counter()
        found = joint_coverage_set(
            batch, preds[set_name], resamples=resamples, seed=seed, tables=tables
        )
        out["sets"][set_name] = found
        log(
            f"joint reply coverage on {set_name}: {found['counted']['examples']} of "
            f"{found['examples']} examples counted, "
            f"{time.perf_counter() - started:.1f} s"
        )
    return out


# --- loading --------------------------------------------------------------------


@dataclass
class Entry:
    """One predictor of the run."""

    name: str
    kind: str
    predictor: Any
    info: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(name: str, taken: Sequence[str]) -> str:
    if name not in taken:
        return name
    suffix = 2
    while f"{name}#{suffix}" in taken:
        suffix += 1
    return f"{name}#{suffix}"


def _trained_on(extra: Mapping[str, Any]) -> str | None:
    """The dataset manifest's sha256 an artifact says it was fitted on."""
    found = extra.get("manifest_sha256")
    inner = extra.get("dataset")
    if found is None and isinstance(inner, Mapping):
        found = inner.get("manifest_sha256")
    return str(found) if found else None


def load_artifact_entry(
    spec: str, taken: Sequence[str], manifest_sha: str | None
) -> Entry:
    """An ``--artifact`` argument (``path`` or ``label=path``) as an entry."""
    label, path = "", Path(spec)
    if "=" in spec and not path.is_file():
        label, _, tail = spec.partition("=")
        path = Path(tail)
    try:
        loaded = load_predictor(path)
    except (OSError, ValueError) as exc:
        raise ScorecardError(f"artifact {path}: {exc}") from exc
    name = _unique(label or loaded.name, taken)
    extra = loaded.meta.get("extra") or {}
    trained = _trained_on(extra)
    return Entry(
        name,
        loaded.kind,
        loaded.predictor,
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "artifact_name": loaded.name,
            "created": extra.get("created"),
            "dataset_manifest_sha256": trained,
            "fitted_on_this_dataset": None
            if trained is None or manifest_sha is None
            else trained == manifest_sha,
            "dex_signature_diff": list(loaded.meta.get("dex_signature_diff") or []),
            "elo_mode": getattr(loaded.predictor, "elo_mode", None),
        },
    )


def load_legacy_entries(
    args: argparse.Namespace, featurizer: F.Featurizer, taken: Sequence[str], log: Log
) -> tuple[list[Entry], dict[str, Any]]:
    """The shipped models through ``oppmodel.legacy`` (and its base rates)."""
    from vgc_bench.src.oppmodel import legacy as L

    try:
        base = L.LegacyPredictor.load(args.legacy_move, args.legacy_switch, featurizer)
    except (OSError, ValueError) as exc:
        raise ScorecardError(f"legacy models: {exc}") from exc
    train, _ = F.load_dataset(Path(args.dataset), ["train"])
    rates = L.training_rates(train, base) if train else {}
    log(f"legacy: training-split base rates {rates}")
    files = {
        key: {"path": path, "sha256": sha256_file(Path(path))}
        for key, path in base.files.items()
    }
    entries: list[Entry] = []
    variants: list[tuple[str, dict[str, Any], str]] = [
        (KIND_LEGACY, {}, "the shipped models, as the bot calls them")
    ]
    if args.legacy_tempered:
        try:
            move, destination = (float(x) for x in args.legacy_tempered.split(","))
        except ValueError as exc:
            raise ScorecardError(
                "--legacy-tempered takes two numbers, MOVE,DESTINATION"
            ) from exc
        variants.append(
            (
                f"{KIND_LEGACY}_tempered",
                {"move_temperature": move, "destination_temperature": destination},
                "NOT the shipped behaviour: the move model's scores divided by "
                f"{move:g} and the switch model's destination scores by "
                f"{destination:g} before their softmax (two refitted numbers, to "
                "show how much of the gap is over-confidence)",
            )
        )
    names = list(taken)
    for label, changes, what in variants:
        name = _unique(label, names)
        names.append(name)
        predictor = base.with_config(
            name=name, scope=args.legacy_scope, **rates, **changes
        )
        entries.append(
            Entry(
                name,
                KIND_LEGACY,
                predictor,
                {"what": what, "files": files, "config": predictor.config.to_dict()},
            )
        )
    description = entries[0].predictor.describe()
    description.pop("counters", None)  # per predictor, in card["predictors"]
    description["training_rates"] = rates
    description["files"] = files
    return entries, description


def shared_accounts(
    ladder: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]
) -> dict[str, Any] | None:
    """Accounts that act in both gated sets, and how much of each set they are.

    The dataset keeps a ladder-holdout opponent's other games out of TRAINING
    only, so (a) and (b) can share players: they are then not two independent
    readings of the same model. None when the batches carry no ``m_actor``.
    """
    if "m_actor" not in ladder or "m_actor" not in test:
        return None
    first, second = np.asarray(ladder["m_actor"]), np.asarray(test["m_actor"])
    both = np.intersect1d(first, second)
    return {
        "accounts": int(both.size),
        f"{SET_LADDER}_examples": int(np.isin(first, both).sum()),
        f"{SET_TEST}_examples": int(np.isin(second, both).sum()),
    }


def _split(data: Mapping[str, np.ndarray], code: int, limit: int | None) -> F.Batch:
    part = F.take(data, np.asarray(data["m_split"]) == code)
    if limit is not None:
        part = F.take(part, slice(0, max(0, int(limit))))
    return part


def _rows(pred: Mapping[str, np.ndarray], mask: np.ndarray) -> Prediction:
    return {key: np.asarray(value)[mask] for key, value in pred.items()}


# --- the run --------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """JSON-safe copy: numpy scalars and tuples unpacked, non-finite as None."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def _signed(found: Mapping[str, Any] | None, digits: int = 4) -> str:
    if not found or found.get("diff") is None:
        return "-"
    text = f"{found['diff']:+.{digits}f}"
    if found.get("low") is None or found.get("high") is None:
        return text
    return f"{text} [{found['low']:+.{digits}f}, {found['high']:+.{digits}f}]"


def _percent(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{100.0 * float(value):.{digits}f}%"


def _settings(values: Mapping[str, Any]) -> str:
    """``name value`` pairs of a small dict, numbers to four significant digits."""
    return (
        ", ".join(
            f"{name} {value:.4g}" if isinstance(value, float) else f"{name} {value}"
            for name, value in values.items()
        )
        or "none"
    )


def _elo_gain(row: Mapping[str, Any] | None) -> float | None:
    found = ((row or {}).get("elo") or {}).get(MODE_SHUFFLE)
    diff = (found or {}).get("keep_minus_mode", {}).get("diff")
    return None if diff is None else -float(diff)


def _verdicts(
    card: dict[str, Any], entries: Sequence[Entry], pair: tuple[str, str] | None
) -> None:
    """Fill ``card['r1']``, ``card['r2']`` and ``card['r3']`` from the set rows."""
    sets = card["sets"]
    reference = card["settings"]["reference"]

    def row(set_name: str, name: str) -> Mapping[str, Any] | None:
        return (sets.get(set_name) or {}).get("predictors", {}).get(name)

    r1: dict[str, Any] = {
        "rule": "passes when the 95% interval of fine NLL minus the reference's is "
        "entirely below zero on (a) and on (b); (c) is reported",
        "reference": reference,
        "subject": "the neural model (kind oppnet); the rule is shown for every "
        "predictor",
        "predictors": {},
    }
    for entry in entries:
        if entry.name == reference:
            continue
        differences = {
            name: ((row(name, entry.name) or {}).get("difference") or {}).get("fine")
            for name in SETS
        }
        found = r1_verdict(differences)
        found["difference"] = differences
        by_account = {
            name: (row(name, entry.name) or {}).get("difference_by_account")
            for name in SETS
        }
        found["difference_by_account"] = by_account
        # Reported next to the verdict, never part of it: the pre-registered
        # rule reads the game-clustered interval.
        found["interval_below_zero_by_account"] = {
            name: (by_account.get(name) or {}).get("high") is not None
            and float((by_account.get(name) or {})["high"]) < 0.0
            for name in GATED_SETS
        }
        found["kind"] = entry.kind
        r1["predictors"][entry.name] = found
    card["r1"] = r1

    r2: dict[str, Any] = {
        "rule": f"Elo stays only if every gain measured is at least {ELO_GAIN_BAR} "
        "nats on (a) and on (b). Gain of the shuffle test = fine NLL with the "
        "ratings shuffled minus with them kept; gain of the retrain = fine NLL of "
        "the Elo-blind retrain minus the model's.",
        "bar": ELO_GAIN_BAR,
        "does_not_read_elo": [e.name for e in entries if not e.info.get("reads_elo")],
        "predictors": {},
    }
    for entry in entries:
        if not entry.info.get("reads_elo"):
            continue
        shuffle = {name: _elo_gain(row(name, entry.name)) for name in SETS}
        blind: dict[str, float | None] | None = None
        if pair is not None and pair[0] == entry.name:
            blind = {}
            for name in SETS:
                found_pairs = (sets.get(name) or {}).get("pairs") or []
                diff = next(
                    (
                        item["model_minus_blind"]["diff"]
                        for item in found_pairs
                        if item["model"] == pair[0] and item["blind"] == pair[1]
                    ),
                    None,
                )
                blind[name] = None if diff is None else -float(diff)
        found = r2_verdict(shuffle, blind)
        found["shuffle_gain"] = shuffle
        found["blind_retrain"] = (
            None
            if blind is None
            else {"blind": pair[1] if pair else None, "gain": blind}
        )
        r2["predictors"][entry.name] = found
    card["r2"] = r2

    ladder = (sets.get(SET_LADDER) or {}).get("predictors") or {}
    scores = {
        name: found["fine_nll"]["value"]
        for name, found in ladder.items()
        if found["fine_nll"]["value"] is not None
    }
    best = min(scores, key=lambda name: scores[name]) if scores else None
    card["r3"] = {
        "note": "descriptive: nothing here is a gate; a consumer sets its "
        "threshold from these tables before its own A/B",
        "best": best,
        "best_rule": "lowest fine NLL on (a) among the predictors of this run",
        "pre_registered_thresholds": dict(PRE_REGISTERED),
    }


def warnings_of(card: Mapping[str, Any]) -> list[str]:
    """Plain sentences about anything that weakens the reading of a card."""
    out: list[str] = []
    if card["dataset"].get("dex_signature_diff"):
        out.append(
            "The dataset was built with another dex than the one installed: "
            f"{card['dataset']['dex_signature_diff']}."
        )
    for name, info in card["predictors"].items():
        if info.get("fitted_on_this_dataset") is False:
            out.append(
                f"{name} was fitted on another build of the dataset; its training "
                "split may not match this dataset's evaluation sets."
            )
        if info.get("dex_signature_diff"):
            out.append(
                f"{name} was written with another dex: {info['dex_signature_diff']}."
            )
        if info.get("elo_blind_retrain") and info.get("reads_elo"):
            out.append(
                f"{name} was given as the Elo-blind retrain, but its predictions "
                "change with the ratings."
            )
    shared = (card["dataset"].get("accounts_in_both_gated_sets") or {}).get("accounts")
    if shared:
        found = card["dataset"]["accounts_in_both_gated_sets"]
        out.append(
            f"{shared} accounts act in both (a) and (b) "
            f"({found.get(SET_LADDER + '_examples')} examples of (a), "
            f"{found.get(SET_TEST + '_examples')} of (b)): the dataset keeps a "
            "ladder-holdout opponent's other games out of training only, so the "
            "two gated sets are not fully independent."
        )
    pair = card.get("elo_pair")
    if pair and not card["predictors"].get(pair["model"], {}).get("reads_elo"):
        out.append(
            f"{pair['model']} is paired with an Elo-blind retrain, but its own "
            "predictions do not change with the ratings."
        )
    for set_name, found in card["sets"].items():
        for name, row in (found.get("predictors") or {}).items():
            sanity = row.get("sanity") or {}
            if sanity and not sanity.get("unchanged"):
                out.append(
                    f"{name} on {set_name}: the prediction was not a clean "
                    "probability table (largest change "
                    f"{max(sanity['max_abs_change'].values()):.1e}); it was scored "
                    "after renormalising."
                )
    return out


def run(args: argparse.Namespace, log: Log = say) -> dict[str, Any]:
    """Load, predict, score and write; returns the card (also written).

    Refuses (``ScorecardError``) to write into a directory that already holds
    a card unless ``args.overwrite`` is set, and makes the directory before
    anything is predicted so a bad path fails in the first second. A run that
    fails leaves nothing behind: a directory it made itself is removed again.
    """
    out = Path(args.out)
    if (out / CARD_JSON).exists() and not getattr(args, "overwrite", False):
        raise ScorecardError(
            f"{out / CARD_JSON} exists; give --overwrite to replace that scorecard"
        )
    made = _missing_parents(out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / f".write_test_{_os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        _remove_empty(made)
        raise ScorecardError(f"cannot write to {out}: {exc}") from exc
    try:
        return _run(args, out, log)
    except BaseException:
        _remove_empty(made)
        raise


def _missing_parents(path: Path) -> list[Path]:
    """``path`` and its ancestors that do not exist yet, deepest first."""
    out: list[Path] = []
    current = path
    while not current.exists() and current != current.parent:
        out.append(current)
        current = current.parent
    return out


def _remove_empty(directories: Sequence[Path]) -> None:
    """Remove the directories a failed run made, as far as they are empty."""
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            return


def _run(args: argparse.Namespace, out: Path, log: Log) -> dict[str, Any]:
    started = time.time()
    dataset = Path(args.dataset)
    try:
        data, manifest = F.load_dataset(dataset, [SET_LADDER, SET_TEST])
        featurizer = F.Featurizer.load(dataset)
        manifest_sha = sha256_file(dataset / "manifest.json")
    except (OSError, ValueError, KeyError) as exc:
        raise ScorecardError(f"dataset {dataset}: {exc!r}") from exc
    names = list(manifest.get("splits") or [])
    missing = [name for name in (SET_LADDER, SET_TEST) if name not in names]
    if missing or not data:
        raise ScorecardError(f"dataset {dataset} lacks the splits {missing or names}")
    base = {
        name: _split(data, names.index(name), args.limit)
        for name in (SET_LADDER, SET_TEST)
    }
    del data
    late = (np.asarray(base[SET_TEST]["m_flag"]) & FLAG_TIME_SLICE) > 0
    log(
        f"dataset {dataset}: ladder holdout {len(base[SET_LADDER]['turn'])} examples, "
        f"test {len(base[SET_TEST]['turn'])} ({int(late.sum())} in the time slice)"
    )
    shared = shared_accounts(base[SET_LADDER], base[SET_TEST])

    entries: list[Entry] = []
    paths: dict[str, str] = {}
    for spec in args.artifact:
        entry = load_artifact_entry(spec, [e.name for e in entries], manifest_sha)
        entries.append(entry)
        paths[str(Path(entry.info["path"]).resolve())] = entry.name
    pair: tuple[str, str] | None = None
    if args.elo_blind:
        resolved = str(Path(args.elo_blind).resolve())
        if resolved not in paths:
            entry = load_artifact_entry(
                args.elo_blind, [e.name for e in entries], manifest_sha
            )
            entries.append(entry)
            paths[resolved] = entry.name
        blind = next(e for e in entries if e.name == paths[resolved])
        blind.info["elo_blind_retrain"] = True
        if args.elo_model:
            keep = args.elo_model
        else:
            twins = [e.name for e in entries if e.kind == blind.kind and e is not blind]
            keep = twins[-1] if twins else ""
        if keep not in [e.name for e in entries] or keep == blind.name:
            raise ScorecardError(
                "--elo-blind needs the model it was retrained from: give "
                "--elo-model with one of " + ", ".join(e.name for e in entries)
            )
        pair = (keep, blind.name)
    legacy: dict[str, Any] | None = None
    if args.legacy:
        made, legacy = load_legacy_entries(
            args, featurizer, [e.name for e in entries], log
        )
        entries.extend(made)
    if not entries:
        raise ScorecardError("no predictor: give --artifact and / or --legacy")
    if args.reference not in [e.name for e in entries]:
        raise ScorecardError(
            f"reference {args.reference!r} is not among the predictors "
            f"({', '.join(e.name for e in entries)})"
        )

    preds: dict[str, dict[str, Prediction]] = {name: {} for name in base}
    seconds: dict[str, dict[str, float]] = {name: {} for name in base}
    elo: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {name: {} for name in base}
    for entry in entries:
        for set_name, batch in base.items():
            pred, took = guarded_predict(entry.predictor, batch, name=entry.name)
            preds[set_name][entry.name] = pred
            seconds[set_name][entry.name] = took
        reads = False
        blanked: dict[str, Prediction] = {}
        for set_name, batch in base.items():
            blanked[set_name], _ = guarded_predict(
                entry.predictor, batch, name=entry.name, elo_mode=F.ELO_BLANK
            )
            kept = preds[set_name][entry.name]
            reads = reads or any(
                bool(np.any(blanked[set_name][key] != kept[key])) for key in kept
            )
        entry.info["reads_elo"] = reads
        if reads:
            for set_name, batch in base.items():
                modes: dict[str, list[np.ndarray]] = {
                    F.ELO_BLANK: [F.slot_nll(blanked[set_name], batch)["fine"]]
                }
                swapped, _ = guarded_predict(
                    entry.predictor, batch, name=entry.name, elo_mode=F.ELO_SWAP
                )
                modes[F.ELO_SWAP] = [F.slot_nll(swapped, batch)["fine"]]
                modes[F.ELO_SHUFFLE] = []
                for index in range(max(1, args.shuffles)):
                    rng = np.random.default_rng([args.seed, 7919, index])
                    shuffled, _ = guarded_predict(
                        entry.predictor,
                        batch,
                        name=entry.name,
                        elo_mode=F.ELO_SHUFFLE,
                        rng=rng,
                    )
                    modes[F.ELO_SHUFFLE].append(F.slot_nll(shuffled, batch)["fine"])
                elo[set_name][entry.name] = modes
        log(
            f"predicted {entry.name} ({entry.kind}): "
            f"{seconds[SET_TEST][entry.name]:.2f} s on test, "
            f"reads Elo: {'yes' if reads else 'no'}"
        )

    card: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": args.note,
        "dataset": {
            "directory": str(dataset),
            "tag": manifest.get("tag"),
            "manifest_sha256": manifest_sha,
            "dex_signature_diff": list(featurizer.signature_diff),
            "limit_per_set": args.limit,
            "accounts_in_both_gated_sets": shared,
        },
        "settings": {
            "reference": args.reference,
            "resamples": int(args.resamples),
            "seed": int(args.seed),
            "elo_shuffles": int(max(1, args.shuffles)),
            "level": LEVEL,
            "probability_floor": F.PROBABILITY_FLOOR,
            "elo_gain_bar": ELO_GAIN_BAR,
            "reliability_bins": BINS,
            "unchanged_tolerance": UNCHANGED_TOLERANCE,
        },
        "definitions": dict(DEFINITIONS),
        "order": [entry.name for entry in entries],
        "set_labels": dict(SET_LABELS),
        "event_text": dict(EVENT_TEXT),
        "elo_pair": None if pair is None else {"model": pair[0], "blind": pair[1]},
        "legacy": legacy,
        "sets": {},
    }
    pairs = [pair] if pair is not None else []
    for set_name in SETS:
        if set_name == SET_TIME:
            batch = F.take(base[SET_TEST], late)
            set_preds = {
                name: _rows(pred, late) for name, pred in preds[SET_TEST].items()
            }
            set_elo = {
                name: {
                    mode: [a[late] for a in arrays] for mode, arrays in modes.items()
                }
                for name, modes in elo[SET_TEST].items()
            }
        else:
            batch, set_preds, set_elo = base[set_name], preds[set_name], elo[set_name]
        if len(batch["turn"]) == 0:
            card["sets"][set_name] = {"examples": 0, "games": 0, "predictors": {}}
            continue
        found = score_set(
            batch,
            set_preds,
            args.reference,
            tables=featurizer.tables,
            elo=set_elo,
            pairs=pairs,
            resamples=args.resamples,
            seed=args.seed,
        )
        for name, pred in set_preds.items():
            took = seconds[set_name].get(name) if set_name in seconds else None
            found["predictors"][name]["sanity"] = sanity_block(pred, batch, took)
        card["sets"][set_name] = found
        log(
            f"scored {set_name}: {found['examples']} examples, {found['games']} "
            f"games, {found['slots']['scored']} labelled slot-turns"
        )
    card["predictors"] = {}
    for entry in entries:
        info = dict(entry.info)
        info["kind"] = entry.kind
        info["counters"] = dict(getattr(entry.predictor, "counters", {}) or {})
        card["predictors"][entry.name] = info
    _verdicts(card, entries, pair)
    if not getattr(args, "no_joint", False):
        # A section of its own: it reads the predictions and the labels, and
        # changes no other key of the card.
        card[JOINT_KEY] = joint_coverage(
            {name: base[name] for name in JOINT_SETS},
            {name: preds[name] for name in JOINT_SETS},
            resamples=args.resamples,
            seed=args.seed,
            log=log,
            tables=featurizer.tables,
        )
    card["warnings"] = warnings_of(card)
    card["seconds"] = round(time.time() - started, 1)

    plain = _plain(card)
    (out / CARD_JSON).write_text(
        json.dumps(plain, indent=1, allow_nan=False), encoding="utf-8"
    )
    (out / CARD_MD).write_text(render_readme(plain), encoding="utf-8")
    log(f"wrote {out / CARD_JSON} and {out / CARD_MD} in {card['seconds']} s")
    return plain


# --- markdown -------------------------------------------------------------------


def _line(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _table(header: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        _line(header),
        _line(["---"] * len(header)),
        *(_line(row) for row in rows),
        "",
    ]


def _decision_cells(decision: Mapping[str, Any] | None) -> list[str]:
    """Slot-turns, share, mean predicted, precision [interval], share caught."""
    if not decision:
        return ["-", "-", "-", "-", "-"]
    precision = decision["precision"]
    return [
        f"{decision['slots']:,}",
        _percent(decision["share"], 2),
        _percent(decision.get("predicted")),
        "-"
        if precision["value"] is None
        else f"{_percent(precision['value'])} [{_percent(precision['low'])}, "
        f"{_percent(precision['high'])}]",
        _percent(decision["recall"]),
    ]


def _bracket(found: Mapping[str, Any] | None, digits: int = 3) -> str:
    """``value [low, high]`` of a ``Draws.apply`` result."""
    if not found or found.get("value") is None:
        return "-"
    return (
        f"{_number(found['value'], digits)} [{_number(found.get('low'), digits)}, "
        f"{_number(found.get('high'), digits)}]"
    )


def r1_lines(card: Mapping[str, Any]) -> list[str]:
    """One plain sentence per predictor for the R1 rule."""
    r1 = card["r1"]
    lines: list[str] = []
    for name, found in r1["predictors"].items():
        parts = []
        for set_name in SETS:
            text = f"{SET_SHORT[set_name]} {_signed(found['difference'].get(set_name))}"
            if set_name in found["interval_below_zero"]:
                below = found["interval_below_zero"][set_name]
                text += " (below zero)" if below else " (not below zero)"
            parts.append(text)
        lines.append(
            f"- **{name}** minus {r1['reference']}: {'; '.join(parts)} -> "
            f"**{found['verdict']}**"
        )
    if not any(found["kind"] == KIND_OPPNET for found in r1["predictors"].values()):
        lines.append(
            "- No neural model was scored in this run, so R1 itself is still open; "
            "the lines above apply its rule to the other predictors."
        )
    return lines


def r2_lines(card: Mapping[str, Any]) -> list[str]:
    """One plain sentence per Elo-reading predictor for the R2 rule."""
    r2 = card["r2"]
    lines: list[str] = []
    for name, found in r2["predictors"].items():
        gains = "; ".join(
            f"{SET_SHORT[s]} {_number(found['shuffle_gain'].get(s))}" for s in SETS
        )
        text = f"- **{name}**: gain over shuffled ratings {gains}"
        retrain = found.get("blind_retrain")
        if retrain:
            gains = "; ".join(
                f"{SET_SHORT[s]} {_number(retrain['gain'].get(s))}" for s in SETS
            )
            text += f"; gain over the Elo-blind retrain ({retrain['blind']}) {gains}"
        else:
            text += "; no Elo-blind retrain was scored against it"
        lines.append(f"{text}; bar {r2['bar']} -> **{found['verdict']}**")
    if not r2["predictors"]:
        lines.append("- No predictor in this run reads the rating.")
    if r2["does_not_read_elo"]:
        lines.append(
            "- Predictions do not change when the ratings are blanked: "
            + ", ".join(r2["does_not_read_elo"])
            + "."
        )
    return lines


def _share(found: Mapping[str, Any] | None) -> str:
    """``value [low, high]`` of a ratio, as percentages."""
    if not found or found.get("value") is None:
        return "-"
    if found.get("low") is None or found.get("high") is None:
        return _percent(found["value"])
    return (
        f"{_percent(found['value'])} [{_percent(found['low'])}, "
        f"{_percent(found['high'])}]"
    )


def _of(count: Any, total: Any) -> str:
    """``count (share of total)``."""
    if not total:
        return f"{int(count or 0):,}"
    return f"{int(count or 0):,} ({_percent(int(count or 0) / int(total))})"


def joint_lines(card: Mapping[str, Any]) -> list[str]:
    """The "Joint reply coverage" section; nothing for a card without it."""
    section = card.get(JOINT_KEY)
    if not section:
        return []
    order = list(card["order"])
    ks = [str(k) for k in section["ks"]]
    main = str(section["k"])
    lines = ["## Joint reply coverage", ""]
    lines += [
        "How often the reply a player made with both slots is among the joint "
        "replies a predictor ranks highest. Descriptive: nothing here is a gate.",
        "",
        *(f"- {text}" for text in section["definitions"].values()),
        "",
    ]
    for set_name, found in section["sets"].items():
        lines += [f"### {card['set_labels'].get(set_name, set_name)}", ""]
        if not found.get("examples"):
            lines += ["No examples.", ""]
            continue
        counted, left, told = found["counted"], found["left_out"], found["truth"]
        slots = found["slot_turns"]
        lines += [
            f"Counted: {_of(counted['examples'], found['examples'])} of "
            f"{found['examples']:,} examples, in {counted['games']:,} of "
            f"{found['games']:,} games. Left out: "
            f"{_of(left['examples'], found['examples'])}; "
            f"{left['hidden_action']:,} with a slot that shows no free choice "
            "(hidden, stopped before it acted, or forced) and "
            f"{left['target_not_certain']:,} more with a move whose target is not "
            "certain. Per slot-turn, "
            f"{_of(slots['not_fully_visible'], slots['acting'])} of "
            f"{slots['acting']:,} are not fully visible. Among the counted "
            "examples the true reply involves OTHER in "
            f"{_of(told['involves_other'], counted['examples'])}, a switch in "
            f"{_of(told['involves_switch'], counted['examples'])} and a Mega "
            f"Evolution in {_of(told['mega'], counted['examples'])}.",
            "",
        ]
        notes: list[list[Any]] = []
        for variant, wording in section["variants"].items():
            rows = []
            for name in order:
                made = (found["predictors"].get(name) or {}).get(variant)
                if not made:
                    continue
                row = made["slices"][SLICE_ALL]
                rows.append(
                    [name]
                    + [
                        _share(row["top_interval"])
                        if k == main
                        else _percent(row["top"][k])
                        for k in ks
                    ]
                    + [
                        _percent(row["top_other_as_hit"][main]),
                        _percent(row["mass_top"]),
                        _number(row["log_prob"], 3),
                    ]
                )
                odd = [
                    int(made.get(key) or 0)
                    for key in (
                        "no_distribution",
                        "truth_not_possible",
                        "decided_by_a_tie",
                    )
                ]
                if any(odd):
                    notes.append(
                        [name, wording["short"], *(f"{value:,}" for value in odd)]
                    )
            lines += [wording["text"], ""]
            lines += _table(
                ["predictor"]
                + [f"top-{k} [95%]" if k == main else f"top-{k}" for k in ks]
                + [
                    f"top-{main}, OTHER as a hit",
                    f"mass of the top {main}",
                    "mean log p",
                ],
                rows,
            )
        if notes:
            lines += [
                "Examples counted as misses because the predictor gave no "
                "probability to any possible reply, or because the true reply is "
                "not a possible one; and examples whose place inside or outside "
                f"the top {main} hangs on the fixed order of replies with exactly "
                "the same probability (predictors without a row have none):",
                "",
            ]
            lines += _table(
                [
                    "predictor",
                    "joint",
                    "no probability",
                    "truth not possible",
                    f"a tie decides top-{main}",
                ],
                notes,
            )
        lines += [
            f"Top-{main} by slice, without the Mega bit [95%] (thin slices are "
            f"noise; the other K, the Mega bit and the mass are in `{CARD_JSON}`):",
            "",
        ]
        rows = []
        for label, size in found["slices"].items():
            if not size["examples"]:
                continue
            cells = []
            for name in order:
                made = (found["predictors"].get(name) or {}).get(JOINT_PLAIN) or {}
                row = (made.get("slices") or {}).get(label) or {}
                cells.append(_share(row.get("top_interval")))
            rows.append(
                [
                    label,
                    f"{size['examples']:,}",
                    f"{size['games']:,}",
                    _percent(size["involves_other"] / size["examples"]),
                    *cells,
                ]
            )
        lines += _table(["slice", "examples", "games", "involves OTHER", *order], rows)
        depends = found.get("dependence") or {}
        if depends.get("events"):
            lines += [
                "What the factorisation loses. The joint above is the product of "
                "the two slots' predictions, so it holds none of the dependence "
                f"between them. On the {depends['examples']:,} counted turns with "
                "two acting slots: how often each pair event happened, against "
                "the product of the two slots' observed marginal rates (1.00 "
                "would be independence in the data) and against each "
                "predictor's own product (above 1: the predictor's joint puts "
                "too little on the event). The counted turns hold more switches "
                "and Protect moves than turn starts do, so a ratio against a "
                "predictor is partly that selection.",
                "",
            ]
            rows = []
            for label, row in depends["events"].items():
                cells = []
                for name in order:
                    made = ((depends.get("predictors") or {}).get(name) or {}).get(
                        label
                    ) or {}
                    cells.append(
                        f"{_percent(made.get('product'), 2)} "
                        f"({_number(made.get('ratio'), 2)})"
                        if made
                        else "-"
                    )
                rows.append(
                    [
                        label,
                        _percent(row.get("observed"), 2),
                        f"{_percent(row.get('product_of_observed_marginals'), 2)} "
                        f"({_number(row.get('ratio'), 2)})",
                        *cells,
                    ]
                )
            lines += _table(
                [
                    "pair event",
                    "happened",
                    "product of the observed marginals (ratio)",
                    *(f"{name}: its product (ratio)" for name in order),
                ],
                rows,
            )
    return lines


def render_readme(card: Mapping[str, Any]) -> str:
    """The scorecard as markdown; every number comes from the card."""
    settings, data = card["settings"], card["dataset"]
    order = list(card["order"])
    reference = settings["reference"]
    lines: list[str] = ["# Opponent predictor: scorecard", ""]
    if card.get("note"):
        lines += [f"**{card['note']}**", ""]
    lines += [
        f"Rendered from `{CARD_JSON}` by `evaluation/oppmodel_scorecard.py`. Created "
        f"{card['created']}; dataset `{data['directory']}` (tag {data['tag']}, "
        f"manifest sha256 `{str(data['manifest_sha256'])[:12]}`); "
        f"{settings['resamples']:,} resamples, seed {settings['seed']}; "
        f"{card.get('seconds', '-')} s on one core. Reference predictor: "
        f"**{reference}**.",
        "",
    ]
    if data.get("limit_per_set") is not None:
        lines += [
            f"**Only the first {data['limit_per_set']} examples of each set were "
            "read (a smoke run).**",
            "",
        ]
    found_warnings = list(card.get("warnings") or [])
    lines += ["Warnings: " + ("none." if not found_warnings else ""), ""]
    if found_warnings:
        lines += [*(f"- {text}" for text in found_warnings), ""]
    lines += [
        "## How to read it",
        "",
        "- A slot-turn is one of a player's two active Pokemon at the start of "
        "one turn. The predictor says what that player clicks for it.",
        "- Fine NLL: minus the log of the probability the predictor gave to what "
        "was done (the move and its target, or the switch and its destination; "
        "for a slot whose action the log hides, the set of actions the log still "
        "allows), averaged over labelled slot-turns, in nats. Lower is better; "
        "0.1 lower means the action was given about 10% more probability.",
        "- Minus reference: this predictor's fine NLL minus the reference's on "
        "the same slot-turns. Negative is better than the reference. The bracket "
        "is a 95% interval from resampling whole games, the same resampled games "
        "for both. That interval is the pre-registered one and the one the "
        "verdicts read.",
        "- Accounts resampled: the same difference with players, not games, as "
        "the resampling unit. Set (b) is split by account and a few accounts "
        "hold many of its games, so there the account interval is wider, and a "
        "small difference on (b) should be judged by it.",
        "- Top-1 / top-3: how often the action taken (with its target) is the "
        "predictor's first choice, or among its first three, where the action is "
        "fully visible. Intent: the same for the seven coarse classes of the "
        "design.",
        "- Every predictor is given the public state only, with an unrecorded "
        "team-sheet state read as closed.",
        "",
        "## Verdicts",
        "",
        "R1, quality: " + card["r1"]["rule"] + ".",
        "",
        *r1_lines(card),
        "",
        "R2, Elo: " + card["r2"]["rule"],
        "",
        *r2_lines(card),
        "",
        "R3, decision relevance: " + card["r3"]["note"] + ". Tables below.",
        "",
    ]

    for set_name in SETS:
        found = card["sets"].get(set_name) or {}
        lines += [f"## {card['set_labels'][set_name]}", ""]
        if not found.get("examples"):
            lines += ["No examples.", ""]
            continue
        slots = found["slots"]
        lines += [
            f"{found['examples']:,} examples in {found['games']:,} games "
            f"({found['slices'][SLICE_ALL]['games']:,} of them with a labelled "
            f"slot-turn); "
            f"{slots['scored']:,} labelled slot-turns ({slots['visible']:,} with a "
            f"visible action, {slots['censored']:,} hidden), "
            f"{slots['target_labels']:,} target labels.",
            "",
        ]
        rows = []
        for name in order:
            row = found["predictors"][name]
            fine = row["fine_nll"]
            rows.append(
                [
                    name,
                    f"{_number(fine['value'])} [{_number(fine['low'])}, "
                    f"{_number(fine['high'])}]",
                    "reference"
                    if name == reference
                    else _signed((row["difference"] or {}).get("fine")),
                    _percent(row["fine_top1"]),
                    _percent(row["fine_top3"]),
                    _percent(row["intent_accuracy"]),
                ]
            )
        lines += _table(
            [
                "predictor",
                "fine NLL [95%]",
                "minus reference [95%]",
                "top-1",
                "top-3",
                "intent accuracy",
            ],
            rows,
        )
        clusters = found.get("clusters") or {}
        others = [name for name in order if name != reference]
        if others and clusters.get(CLUSTER_ACCOUNT):
            lines += [
                f"Minus {reference} with a different resampling unit [95%]: "
                f"{clusters[CLUSTER_GAME]:,} games (the pre-registered interval) "
                f"against {clusters[CLUSTER_ACCOUNT]:,} accounts.",
                "",
            ]
            lines += _table(
                ["predictor", "games resampled", "accounts resampled"],
                [
                    [
                        name,
                        _signed(
                            (found["predictors"][name]["difference"] or {}).get("fine")
                        ),
                        _signed(found["predictors"][name].get("difference_by_account")),
                    ]
                    for name in others
                ],
            )
        lines += [
            "Parts of the fine NLL (action part + target part = fine NLL), and the "
            "numbers that are reported but not gated:",
            "",
        ]
        rows = []
        for name in order:
            row = found["predictors"][name]
            parts = row["parts"]
            rows.append(
                [
                    name,
                    _number(parts["action"]),
                    _number(parts["target"]),
                    _number(parts["visible"]),
                    _number(parts["censored"]),
                    _number(row["target_nll_per_label"]),
                    _percent(row["action_top1"]),
                    _number(row["intent_nll"]),
                    _number(row["mega_nll"]),
                ]
            )
        lines += _table(
            [
                "predictor",
                "action part",
                "target part",
                "visible slot-turns",
                "hidden slot-turns",
                "target NLL per target label",
                "action top-1",
                "intent NLL",
                "Mega NLL",
            ],
            rows,
        )
        lines += [
            "By slice (fine NLL; descriptive, not a gate; thin slices are noise):",
            "",
        ]
        rows = []
        for label, size in found["slices"].items():
            if label == SLICE_ALL or not size["slots"]:
                continue
            rows.append(
                [label, f"{size['slots']:,}", f"{size['games']:,}"]
                + [
                    _number(found["predictors"][name]["slices"][label]["fine_nll"], 3)
                    for name in order
                ]
            )
        lines += _table(["slice", "slot-turns", "games", *order], rows)
        others = [name for name in order if name != reference]
        if others:
            lines += [f"The same slices, minus {reference} [95%]:", ""]
            rows = []
            for label, size in found["slices"].items():
                if label == SLICE_ALL or not size["slots"]:
                    continue
                rows.append(
                    [label]
                    + [
                        _signed(
                            found["predictors"][name]["slices"][label].get(
                                "difference"
                            ),
                            3,
                        )
                        for name in others
                    ]
                )
            lines += _table(["slice", *others], rows)

    lines += ["## R2 details: the rating input", ""]
    readers = list(card["r2"]["predictors"])
    if readers:
        lines += [
            "Fine NLL with the ratings kept minus fine NLL with them changed "
            "(negative: the rating helps) [95%]. Shuffled = each example gets "
            f"another example's two ratings ({settings['elo_shuffles']} seeds, "
            "averaged); blanked = both unknown; swapped = the two players' "
            "ratings exchanged.",
            "",
        ]
        rows = []
        for name in readers:
            for set_name in SETS:
                modes = (
                    (card["sets"].get(set_name) or {})
                    .get("predictors", {})
                    .get(name, {})
                    .get("elo")
                )
                if not modes:
                    continue
                rows.append(
                    [name, SET_SHORT[set_name]]
                    + [
                        _signed((modes.get(mode) or {}).get("keep_minus_mode"))
                        for mode in ELO_MODES
                    ]
                    + [
                        _signed(
                            (modes.get(MODE_SHUFFLE) or {}).get(
                                "keep_minus_mode_by_account"
                            )
                        )
                    ]
                )
        lines += _table(
            [
                "predictor",
                "set",
                "kept - shuffled",
                "kept - blanked",
                "kept - swapped",
                "kept - shuffled, accounts resampled",
            ],
            rows,
        )
        pair = card.get("elo_pair")
        if pair:
            rows = []
            for set_name in SETS:
                for item in (card["sets"].get(set_name) or {}).get("pairs") or []:
                    rows.append(
                        [
                            SET_SHORT[set_name],
                            f"{item['model']} - {item['blind']}",
                            _signed(item["model_minus_blind"]),
                            _signed(item.get("model_minus_blind_by_account")),
                        ]
                    )
            lines += [
                f"The Elo-blind retrain: {pair['model']} against {pair['blind']} "
                "(negative: the model that reads the rating is better).",
                "",
            ]
            lines += _table(
                [
                    "set",
                    "difference",
                    "fine NLL difference [95%]",
                    "accounts resampled [95%]",
                ],
                rows,
            )
    else:
        lines += ["No predictor in this run reads the rating.", ""]

    best = card["r3"]["best"]
    lines += ["## R3: events a guard could read", ""]
    if best:
        lines += [
            f"Shown for **{best}** ({card['r3']['best_rule']}); every predictor is "
            f"in `{CARD_JSON}`. A slot-turn is counted wherever the log tells "
            "whether the event happened, hidden actions included. Skill is 1 "
            "minus the Brier score over the Brier score of always predicting the "
            "set's own rate: above 0 is better than that constant. A slot with no "
            "legal switch target cannot switch, and one with no Protect candidate "
            "is given no Protect: every predictor says 0 there because of the "
            "mask. 'Cannot happen' counts those slot-turns, and 'skill where it "
            "can' is the skill over the rest, against their own rate: the part "
            "that is the predictor's own.",
            "",
        ]
        rows = []
        for event in EVENTS:
            for set_name in SETS:
                found = (
                    (card["sets"].get(set_name) or {})
                    .get("predictors", {})
                    .get(best, {})
                    .get("events", {})
                    .get(event)
                )
                if not found or not found["slots"]:
                    continue
                skill = found["skill"]
                rows.append(
                    [
                        card["event_text"][event],
                        SET_SHORT[set_name],
                        f"{found['slots']:,}",
                        _percent(found["observed"]),
                        _percent(found["predicted"]),
                        _number(found["brier"]),
                        _bracket(skill),
                        f"{found.get('impossible', 0):,}",
                        _bracket(found.get("skill_where_possible")),
                    ]
                )
        lines += _table(
            [
                "event",
                "set",
                "slot-turns",
                "happened",
                "mean predicted",
                "Brier",
                "skill [95%]",
                "cannot happen",
                "skill where it can [95%]",
            ],
            rows,
        )
        lines += [
            "The attack event has two readings because the label needs a visible "
            "click. The first counts only slot-turns with a visible click. The "
            "second also counts the slot-turns that the log proves were stopped "
            "before acting (fainted first, flinch, sleep, ...) as 'no attack "
            "arrived'; the predictor still predicts the click, so it reads high "
            "there. The label is where a move landed, so a visible aimed attack "
            "whose target is not certain (a redirector stood, or the move was "
            "retargeted after a faint) is left out of every attack row.",
            "",
            "Read the 'two opposing Pokemon' rows for any decision about WHICH of "
            "two slots is attacked: with one opposing Pokemon the target is given "
            "and the event is just 'the slot uses a damaging move', which happens "
            "most of the time. The undivided attack rows mix the two. 'Attacked "
            "by at least one' is known only when every attacker on the field is "
            "known, whatever the answer.",
            "",
        ]
        for event in README_EVENTS:
            lines += [f"Reliability, {card['event_text'][event]}:", ""]
            tables = [
                (card["sets"].get(set_name) or {})
                .get("predictors", {})
                .get(best, {})
                .get("events", {})
                .get(event, {})
                .get("reliability")
                for set_name in GATED_SETS
            ]
            rows = []
            for index in range(settings["reliability_bins"]):
                cells: list[Any] = []
                for table in tables:
                    entry = table[index] if table else None
                    cells += (
                        ["-", "-", "-"]
                        if not entry
                        else [
                            f"{entry['slots']:,}",
                            _percent(entry["predicted"]),
                            _percent(entry["observed"]),
                        ]
                    )
                first = next((t for t in tables if t), None)
                if first is None:
                    continue
                rows.append(
                    [f"{first[index]['from']:.1f}-{first[index]['to']:.1f}", *cells]
                )
            header = ["predicted"]
            for set_name in GATED_SETS:
                short = SET_SHORT[set_name]
                header += [
                    f"{short} slot-turns",
                    f"{short} mean predicted",
                    f"{short} happened",
                ]
            lines += _table(header, rows)
        lines += [
            "Decision relevance: how often the predictor is confident, and how "
            "often it is right then. Share = slot-turns at or above the threshold "
            "over all slot-turns where the event is known (the share among the "
            "slot-turns where the event can happen is share_where_possible in "
            f"`{CARD_JSON}`); mean predicted = the average prediction among them; "
            "precision = how often the event happened among them [95%]: a "
            "precision below the mean prediction is over-confidence, and a guard "
            "should take its threshold from the precision; caught = share of all "
            "occurrences that were at or above the threshold. * = the threshold "
            "named in the pre-registration. An interval over a handful of "
            "slot-turns says nothing.",
            "",
        ]
        rows = []
        for event in EVENTS:
            per_set = [
                (card["sets"].get(set_name) or {})
                .get("predictors", {})
                .get(best, {})
                .get("events", {})
                .get(event, {})
                .get("decisions")
                or []
                for set_name in GATED_SETS
            ]
            for index, threshold in enumerate(thresholds_for(event)):
                cells: list[Any] = []
                for decisions in per_set:
                    cells += _decision_cells(
                        decisions[index] if index < len(decisions) else None
                    )
                star = "*" if threshold == PRE_REGISTERED[event] else ""
                rows.append(
                    [card["event_text"][event], f"{threshold:.2f}{star}", *cells]
                )
        header = ["event", "P at least"]
        for set_name in GATED_SETS:
            short = SET_SHORT[set_name]
            header += [
                f"{short} slot-turns",
                f"{short} share",
                f"{short} mean predicted",
                f"{short} precision [95%]",
                f"{short} caught",
            ]
        lines += _table(header, rows)

    lines += joint_lines(card)
    lines += [
        "## Sanity of the predictions",
        "",
        "Largest change when illegal mass is removed and rows are renormalised "
        "(0 = already a clean probability table), labels whose probability sat "
        f"at the floor of {settings['probability_floor']:g} (each costs "
        f"{-math.log(settings['probability_floor']):.1f} nats), and the time of "
        "one predict call per example.",
        "",
    ]
    rows = []
    for name in order:
        for set_name in GATED_SETS:
            found = (
                (card["sets"].get(set_name) or {})
                .get("predictors", {})
                .get(name, {})
                .get("sanity")
            )
            if not found:
                continue
            change = found["max_abs_change"]
            rows.append(
                [
                    name,
                    SET_SHORT[set_name],
                    f"{max(change.values()):.1e}",
                    _number(found["unchanged"]),
                    found["non_finite"],
                    found["floor_hits"]["action"],
                    found["floor_hits"]["target"],
                    _number(found.get("microseconds_per_example"), 1),
                ]
            )
    lines += _table(
        [
            "predictor",
            "set",
            "largest change",
            "unchanged",
            "non-finite",
            "action labels at the floor",
            "target labels at the floor",
            "microseconds per example",
        ],
        rows,
    )

    legacy = card.get("legacy")
    if legacy:
        lines += [
            "## The legacy baseline",
            "",
            "The bot's shipped opponent models (`"
            + "`, `".join(item["path"] for item in legacy["files"].values())
            + "`) scored by this harness through `vgc_bench/src/oppmodel/legacy.py`. "
            "How their outputs are mapped:",
            "",
            *(f"- {text}" for text in legacy["mapping"]),
            "",
            "Settings: " + _settings(legacy["config"]) + ". Base rates measured "
            "on the training split: " + _settings(legacy["training_rates"]) + ".",
            "",
            "Where the comparison is not level:",
            "",
            *(f"- {text}" for text in legacy["disadvantages"]),
            "",
        ]

    lines += ["## Predictors", ""]
    twin_of = (card.get("elo_pair") or {}).get("model")
    rows = []
    for name in order:
        info = card["predictors"][name]
        files = (
            [{"path": info["path"], "sha256": info.get("sha256")}]
            if info.get("path")
            else list((info.get("files") or {}).values())
        )
        rows.append(
            [
                name,
                info["kind"],
                ", ".join(f"`{item['path']}`" for item in files),
                ", ".join(f"`{str(item['sha256'])[:12]}`" for item in files),
                _number(info.get("fitted_on_this_dataset")),
                _number(info.get("reads_elo")),
                info.get("what")
                or (
                    f"scored as the Elo-blind twin of {twin_of}"
                    if info.get("elo_blind_retrain")
                    else ""
                ),
            ]
        )
    lines += _table(
        [
            "name",
            "kind",
            "file",
            "sha256",
            "fitted on this dataset",
            "reads Elo",
            "note",
        ],
        rows,
    )
    changed = {
        name: card["predictors"][name].get("dex_signature_diff") for name in order
    }
    changed = {name: found for name, found in changed.items() if found}
    lines += [
        "Dex differences between an artifact and the installed dex: "
        + (
            "; ".join(f"{name}: {found}" for name, found in changed.items())
            if changed
            else "none"
        )
        + ".",
        "",
    ]
    return "\n".join(lines)


# --- command line ---------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", default="results_oppmodel/v1_ondisk")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="an artifact file (repeatable); 'label=path' renames it in the report",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="name of the predictor every other one is compared with",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="also score the bot's shipped opponent models",
    )
    parser.add_argument("--legacy-move", default=LEGACY_MOVE)
    parser.add_argument("--legacy-switch", default=LEGACY_SWITCH)
    parser.add_argument(
        "--legacy-scope",
        default="candidates",
        choices=("candidates", "vocabulary"),
        help="how the move model's scores become a distribution (see legacy.py)",
    )
    parser.add_argument(
        "--legacy-tempered",
        default=None,
        metavar="MOVE,DESTINATION",
        help="add a second legacy row with the move scores and the switch "
        "destination scores divided by these two numbers (not the shipped models)",
    )
    parser.add_argument(
        "--elo-blind", default=None, help="artifact of the Elo-blind retrain (R2)"
    )
    parser.add_argument(
        "--elo-model",
        default=None,
        help="name of the model the Elo-blind artifact was retrained from "
        "(default: the last other artifact of the same kind)",
    )
    parser.add_argument("--out", default="results_oppmodel/scorecard")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--shuffles",
        type=int,
        default=DEFAULT_SHUFFLES,
        help="seeds of the Elo shuffle that are averaged",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="read only the first N examples of each set (a smoke run)",
    )
    parser.add_argument(
        "--note", default=None, help="a line printed at the top of the README"
    )
    parser.add_argument(
        "--no-joint",
        action="store_true",
        help="leave the joint reply coverage section out",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help=f"rewrite {CARD_MD} from an existing {CARD_JSON}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"replace a {CARD_JSON} that --out already holds",
    )
    return parser.parse_args(argv)


def write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` unless the file already holds exactly it.

    Returns whether it wrote. Re-rendering a report whose text is already on
    disk then leaves the file alone, modification time included: a look at an
    old scorecard does not make it look like a new one.
    """
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            return False
    except (OSError, UnicodeDecodeError):
        pass
    path.write_text(text, encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    if args.render_only:
        try:
            card = json.loads((out / CARD_JSON).read_text(encoding="utf-8"))
            wrote = write_if_changed(out / CARD_MD, render_readme(card))
        except Exception as exc:
            reason = " ".join(f"{type(exc).__name__}: {exc}".split())
            say(f"SCORECARD_FAILED cannot render {out / CARD_JSON}: {reason}")
            return 1
        say(
            f"wrote {out / CARD_MD}"
            if wrote
            else f"{out / CARD_MD} is already what the JSON renders to: not rewritten"
        )
        return 0
    try:
        card = run(args)
        lines = r1_lines(card) + r2_lines(card)
    except ScorecardError as exc:
        say(f"SCORECARD_FAILED {' '.join(str(exc).split())}")
        return 1
    except Exception as exc:
        # Anything else as well: a monitor reads the last line, not a traceback.
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        say(f"SCORECARD_FAILED {reason}")
        return 1
    for line in lines:
        say(line)
    say("SCORECARD_DONE")
    return 0


def _one_thread() -> None:
    """Keep a command-line run on one core (the neural predictors use torch)."""
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


if __name__ == "__main__":
    _one_thread()
    raise SystemExit(main())
