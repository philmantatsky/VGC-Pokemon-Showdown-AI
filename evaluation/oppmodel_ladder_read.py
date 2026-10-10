"""How did the opponent predictor do on the bot's own ladder games?

Design: OPPONENT_PREDICTOR.md. The scorecard (``oppmodel_scorecard.py``) reads
a BUILT dataset, so every predictor is scored on one featurizer's examples.
This reads saved replay pages directly, which is what a fresh ladder session
leaves behind, and scores every artifact through ITS OWN featurizer (the
vocabulary and repertoire stored in the artifact), as the bot runs it.

WHICH GAMES. There is no default: name the replay directories (``--dir``,
repeated), or give ``--fresh`` for the SEALED confirmation set, the bot's
ladder games dated on or after ``FRESH_SINCE``. ``--fresh`` alone finds the
directories itself (every ``ladder_replays_mc*`` under the repo root, or under
``--fresh-root``, that holds a page dated on or after that day) and leaves
older pages out; with ``--dir`` it only allows the named directories to hold
sealed games. Without ``--fresh`` a reading that would hold a sealed game is
refused before anything is scored. The refusal is in ``read_games`` itself,
not only in the command line: a script that calls ``read_games`` on a sealed
page without ``fresh=True`` gets ``LadderReadError``, and the page is never
driven (no state, no label is made of it).

ONE DAY OF THE SEALED SET. ``--since D`` keeps the pages dated D or later,
``--until D`` the pages dated BEFORE D (local time; D is ``YYYY-MM-DD`` or
``YYYY-MM-DDTHH:MM``), and both combine with ``--fresh``. The confirmation on
the games of 2026-10-09 alone (a model built before the games of 10-10 were
played did not know those opponents, so its training rows may hold them) is:

    nice -n 19 .venv/bin/python evaluation/oppmodel_ladder_read.py --fresh \\
        --until 2026-10-10 --fresh-root ../vgc-bench \\
        --artifact deployed=results_oppmodel/oppnet_v2_blind/artifact.pt \\
        --artifact new=<path> --reference deployed --out <a new directory>

Without ``--until`` the same line reads every sealed game. The reading stores
both filters (``input.filters``) and the date of every game it kept.

WHOSE SHEETS BUILT THE SET TABLE. A layout-version-2 artifact whose network
reads the set prior carries a set table built from players' open team sheets.
For every such artifact the reading asks the artifact itself
(``Featurizer.set_table_holds``, by a crc32 of the account id) whether the
opponent of each scored game is among the accounts whose sheets built the
table: ``contributors`` (yes: that player's own sets are inside the model's
inputs, so the reading on those games is NOT held out), ``not_contributors``
and ``unknown`` (the table does not record its accounts, or the page names no
opponent). A contributor is a leak: it is counted, its games are listed, a
WARNING line is logged and stands at the top of the report. The reading is
still written: the count is the finding.

What is read, per replay directory of the bot's own games:

  pages    ``<account> - battle-<id>.html``, through the dataset builder's
           own-page path (``datagen.oppmodel_build_dataset.Corpus._own``): the
           bot's exact HP rewritten to the public percent, a cached public
           copy when there is one, ``drive`` over the whole log, and the
           builder's ``drop_reason``. A game the builder would drop is skipped
           and counted by reason; so is a page that cannot be read, a game
           that is not ladder-rated (``--include-unrated`` keeps those) and a
           game already read from an earlier directory. Every page found is
           either kept or counted in ``skipped``. The bot's own side is never
           scored. Labels are ``events.py``'s, through
           ``Featurizer.encode_turn``, exactly as in training.
  the join A page is joined to the decision rows OF ITS OWN DIRECTORY (battle
           id and directory): a local server restarts its battle numbers, so
           two rehearsal directories hold the same ids for different games.
           The same id in two directories is one game only when the two logs
           start alike (players, start time, team preview); then the second
           page is skipped as a duplicate.
  no page  A game with decision rows and no saved page (cut off, unfinished)
           cannot be scored; it is counted, with its logged forecasts.
  sheets   ``decisions*.jsonl`` of the same directory. A game is OPEN when a
           row says so: ``their_sheet`` (the opponent's sets, which are then
           given to the featurizer as the builder does), ``preview_shadow.
           open_sheet``, ``exact_search.open_sheet`` / ``exact_preview.
           open_sheet``, or the logged forecast's ``sheets_reported``. It is
           CLOSED when no row says open and one says closed, UNKNOWN when no
           row of the game says anything about sheets or there is no row (a
           predictor reads unknown as closed, so only the split's name moves).
  own sheet  In an open game the live bot hands the predictor BOTH sheets.
           Its own is in no page, and the builder's own-page path gives the
           opponent's alone. Here the bot's sheet is read from the team file
           the directory's ``run_config.json`` names (``material.our_team``,
           the same file in every run of the directory) and given too; when
           that cannot be done the game is encoded the builder's way and its
           row says so. The file is read as it is TODAY; its path and sha256
           are stored. ``--own-sheet none`` switches it off. ``live_log``
           tells which of the two is what the bot held: it compares with the
           forecast the bot logged.

Per artifact (``label=path``; count tables and networks alike) and per split
(all games; sheet closed / open / unknown / not open; turn 1 / later), with
the scorecard's own functions:

  slot-turns   labelled slot-turns, mean fine NLL with a 95% interval (a
               hidden action is scored by its set loss), on visible and on
               hidden actions, top-1 / top-3 of the fine action, 7-class
               intent accuracy.
  events       "the slot switches" and "the slot uses a Protect move": mean
               predicted against the observed rate, and the precision where
               P(switch) >= 0.35 / P(Protect) >= 0.5.
  joint        over the turns where every acting slot shows its whole action:
               is the real PAIR of actions among the K likeliest joint replies
               (K = 1, 4, 8, 16; ``oppmodel.joint``; a reply that needs a move
               outside the candidates is a miss), by reply class: with a
               switch / with a Protect move and no switch / moves only, and a
               move not shown before / none. The list is built as the bot
               builds it: WITH the artifact's pair coupling when the artifact
               carries one (``artifact.CoupledPredictor``), else as the plain
               product of the two slots. A coupled artifact also gets the
               plain product of the same marginals next to it, so the reading
               shows what the coupling changed; every artifact's row says
               which joint it is.
  search_rule  the same top 8 under the MATCHING RULE of a search session's
               reading (``evaluation/search_ladder_read.py`` section 5: with
               the Mega bit, a slot that showed nothing matching anything),
               on this tool's labels, over two sets of turns: every turn
               where a slot shows a freely chosen action, and the decisions
               the search session itself counts (a row of ``decisions.jsonl``
               that carries the observed reply, after an earlier move
               decision of the game whose forecast was logged). Only the
               second is on the search session's decisions. Next to them:
               "all turns, a partly hidden reply is a miss".
  paired       every artifact against ``--reference`` on the slot-turns (and
               fully visible turns) both have: difference in fine NLL and in
               joint top-8, games resampled (2,000 resamples, fixed seed).
               The slot-turns are the same; the labels need not be (each
               featurizer has its own candidates, so one artifact's label can
               be its OTHER bucket where the other's is a named move): the
               difference is also given where neither labels OTHER.
  live_log     for an artifact whose name is the one the bot logged: its
               P(switch) / P(Protect) and every listed fine action's
               probability here against the logged forecast of the same turn.

Outputs ``<out>/ladder_read.json`` (everything, with one row per game: tag,
directory, date, ratings, sheet state and, per artifact, the sums behind every
headline number: slot-turns and NLL, top-1 / top-3, the two events, joint
coverage by K and by reply class, so a later run can split old and new games
without a re-run) and ``<out>/ladder_read.md``, rendered from the JSON only
(``--render-only``). The reading stores what produced it: the commit, the
sha256 of this script, the scorecard, the builder and every module of
``vgc_bench/src/oppmodel``, of every decision log and team file read, and of
the artifacts. A directory that already holds a reading is not written over
without ``--overwrite``. The last line printed is ``LADDER_READ_DONE`` (exit
0) or ``LADDER_READ_FAILED <reason>`` (exit 1). From the repo root:

    nice -n 19 .venv/bin/python evaluation/oppmodel_ladder_read.py --fresh \\
        --artifact deployed=results_oppmodel/oppnet_v2_blind/artifact.pt \\
        --artifact table=results_oppmodel/tables_v2/flags_table.pt \\
        --reference deployed --out results_oppmodel/ladder_read_fresh
"""

from __future__ import annotations

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from datagen.oppmodel_build_dataset import (
    _AUDIT_BATTLE,
    _OWN_NAME,
    DEFAULT_FORMATS,
    SHEET_STATES,
    Corpus,
    RawBattle,
    _sheet_from_audit,
    drive,
    drop_reason,
)
from evaluation import oppmodel_scorecard as SC
from vgc_bench.src.oppmodel import features as F
from vgc_bench.src.oppmodel import joint as J
from vgc_bench.src.oppmodel.artifact import load_predictor
from vgc_bench.src.oppmodel.events import (
    SIDES,
    TARGET_CLASSES,
    SheetSet,
    parse_showteam,
    user_id,
)
from vgc_bench.src.oppmodel.public_state import RATED_LADDER, DriveResult

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "oppmodel-ladder-read"
VERSION = 2
READ_JSON = "ladder_read.json"
READ_MD = "ladder_read.md"
# The bot's ladder games dated on or after this day are the SEALED confirmation
# set: read only with ``--fresh``. There is no default set of directories.
FRESH_SINCE = "2026-10-09"
FRESH_GLOB = "ladder_replays_mc*"
# What produced a reading besides the artifacts: hashed into every reading.
INSTRUMENT_FILES: tuple[str, ...] = (
    "evaluation/oppmodel_ladder_read.py",
    "evaluation/oppmodel_scorecard.py",
    "datagen/oppmodel_build_dataset.py",
)
INSTRUMENT_GLOB = "vgc_bench/src/oppmodel/*.py"
SEARCH_AUDIT_LOG = "decisions.jsonl"  # the search's audit rows
OWN_DECISION_LOG = "decisions_champion.jsonl"  # the bot's own decision rows
DEFAULT_OUT = "results_oppmodel/ladder_read"
DEFAULT_RESAMPLES = 2_000
DEFAULT_SEED = 20261010
TOP_KS: tuple[int, ...] = (1, 4, 8, 16)
TOP_K = SC.JOINT_K  # the list length a search keeps
EVENTS: tuple[str, ...] = (SC.EVENT_SWITCH, SC.EVENT_PROTECT)
LIVE_TOLERANCE = 1e-3  # the log rounds to four digits
RUN_CONFIG = "run_config.json"
OWN_SHEET_FILE = "team-file"
OWN_SHEET_NONE = "none"

OPEN, CLOSED, UNKNOWN = "open", "closed", "unknown"
ALL = SC.SLICE_ALL
SHEET_CLOSED, SHEET_OPEN, SHEET_UNKNOWN = SC.SLICE_SHEETS
SHEET_NOT_OPEN = "sheet not open"
TURN_FIRST, TURN_LATER = SC.SLICE_TURNS
CONTEXTS: tuple[str, ...] = (
    ALL,
    SHEET_CLOSED,
    SHEET_OPEN,
    SHEET_UNKNOWN,
    SHEET_NOT_OPEN,
    TURN_FIRST,
    TURN_LATER,
)
REPLY_SWITCH = "reply: with a switch"
REPLY_PROTECT = "reply: with a Protect move, no switch"
REPLY_MOVES = "reply: moves only, no Protect move"
REPLY_UNSHOWN = SC.REPLY_UNSHOWN
REPLY_NO_UNSHOWN = "reply: no move not shown before (all shown or on the sheet)"
REPLIES: tuple[str, ...] = (
    ALL,
    REPLY_SWITCH,
    REPLY_PROTECT,
    REPLY_MOVES,
    REPLY_UNSHOWN,
    REPLY_NO_UNSHOWN,
)
PLAIN, MEGA = SC.JOINT_PLAIN, SC.JOINT_MEGA
RULE_VISIBLE = "fully visible turns"
RULE_ALL_MISS = "all turns, a partly hidden reply is a miss"
RULE_ALL_ANY = "turns with an action shown, a slot that showed nothing matches anything"
RULE_SEARCH = "the search session's own counted decisions, the same matching rule"
RULES: tuple[str, ...] = (RULE_VISIBLE, RULE_ALL_MISS, RULE_ALL_ANY, RULE_SEARCH)
JOINT_PRODUCT = "the plain product of the two slots"
# Per-game sums of the reply classes, under short names.
GAME_REPLIES: tuple[tuple[str, str], ...] = (
    ("with_switch", REPLY_SWITCH),
    ("with_protect", REPLY_PROTECT),
    ("moves_only", REPLY_MOVES),
    ("unshown_move", REPLY_UNSHOWN),
)

DEFINITIONS = {
    "slot_turn": "one of the OPPONENT's two active Pokemon at the start of one "
    "turn of one of the bot's own games; the bot's side is never scored",
    "fine_nll": "mean over labelled slot-turns of -log P(action taken) - log "
    "P(target | move); a hidden action is scored by the probability of the set "
    "of actions the log still allows; 95% interval from resampling whole games",
    "top1 / top3": "share of fully visible actions (move with a certain target, "
    "or switch with its destination) among the predictor's first / first three",
    "intent_accuracy": "over the seven intent classes of the design",
    "events": "known = slot-turns where the log tells whether the event "
    "happened (hidden actions included); observed and mean predicted over "
    "those; at the threshold: slot-turns at or above it, the precision there "
    "with its interval, the mean prediction there, the share of occurrences "
    "caught",
    "joint": "counted = turns where every acting slot shows its whole action. "
    "top[K] = share of counted turns whose real pair of actions is among the K "
    "likeliest joint replies (what the rules forbid removed; without the Mega "
    "bit unless said). The list is the artifact's own: the product of the two "
    "slots' predictions, times the artifact's pair coupling when it carries "
    "one (the headline row of every artifact names which; a coupled artifact "
    "also has the plain product of the same marginals beside it). The per-slot "
    "numbers (NLL, top-1, events) are the predictor's marginals and do not "
    "read a coupling. A real reply that used a move outside the candidates is "
    "a miss at every K",
    "reply classes": "with a switch = at least one slot switched; with a "
    "Protect move = no switch and at least one slot used a Protect-family "
    "move; moves only = the rest (the three partition the counted turns). A "
    "move not shown before = at least one slot used a move its Pokemon had not "
    "shown in this battle and that is on no open sheet; the other class is "
    "its complement (switches included)",
    "search_rule": f"top {TOP_K} of the joint list read four ways. "
    f"'{RULE_VISIBLE}' is the joint section's number. '{RULE_ALL_MISS}' "
    "divides by every turn with an acting slot. The two others use the "
    "MATCHING RULE of search_ladder_read.py section 5 on this tool's labels: "
    "a slot whose action the log hides matches any entry, a visible move "
    "whose target is not certain matches on the move, a move outside the "
    "candidates never matches; with the Mega bit the entry must also name who "
    "Mega-evolves (a slot the log cannot tell counts as not Mega-evolving). "
    f"'{RULE_ALL_ANY}' applies it to every turn where at least one slot shows "
    "a freely chosen action: NOT the search session's set of decisions. "
    f"'{RULE_SEARCH}' applies it to the decisions the search session counts: "
    f"a row of {SEARCH_AUDIT_LOG} that carries reply_coverage.observed, taken "
    "as the reply to the game's last earlier move decision, when that "
    f"decision's forecast is in {OWN_DECISION_LOG} (a decision the session "
    "counts twice is counted twice). It is still this tool's reading of the "
    "reply and of the list, not the search script's own reader",
    "paired": "this artifact minus the reference on the slot-turns (fine NLL; "
    "negative is better) and on the fully visible turns (joint top-8; "
    "positive is better) that BOTH featurizers give; 95% interval from "
    "resampling whole games. The slot-turns are identical, the labels need "
    "not be: a move outside one featurizer's candidates is that artifact's "
    "OTHER bucket. 'neither labels OTHER' is the same difference without "
    "those slot-turns. The sign test is exact and two-sided on the turns "
    "where exactly one of the two lists holds the real reply, turns taken as "
    "independent (games are not resampled there). 'resamples not better' is "
    "the share of the game resamples in which this artifact does not beat the "
    "reference (difference at or on the wrong side of zero): read it rather "
    "than an interval's end that sits near zero, which moves with the seed",
    "sheet": "closed / open as the directory's decision logs say: open when a "
    "row of the game says open, closed when none says open and one says "
    "closed, unknown when no row of the game says anything about sheets or "
    "the game has no row. A predictor reads unknown as closed. 'sheet not "
    "open' = closed or unknown. In an open game the predictor is given the "
    "opponent's sheet from the decision log and the bot's own from the team "
    "file of the directory's run_config.json (as the live bot hands over "
    "both; the file as it is on disk today, its sha256 is stored); a game "
    "where the bot's own could not be given is counted",
    "live_log": "for the artifact the bot ran: P(switch), P(Protect) and the "
    "probability of every fine action the log lists (move x target, switch "
    "destination, the OTHER bucket) here against the forecast the bot logged "
    "at the same turn of the same directory (four digits)",
    "set_table": "for an artifact whose featurizer holds a set table (the set "
    "prior of layout version 2): artifact.set_table.games counts the scored "
    "games whose opponent IS among the accounts whose team sheets built the "
    "table (contributors: a leak, the model is given that player's own sets), "
    "is not, or cannot be told (the table records no accounts, or the page "
    "names no opponent). Asked of the artifact itself by a crc32 of the "
    "account id, so a contributor is 'this account or one with its hash'. "
    "games[i].set_table_holds[label] is the answer per game; null for an "
    "artifact without a set table",
    "per-game rows": "games[i].artifacts[label] holds sums, so any share of "
    "the headline can be rebuilt for a subset of games: fine_top1 / "
    "fine_labels, joint_top8 / joint_counted, switch_observed / switch_known, "
    "joint_top8_with_switch / joint_counted_with_switch and so on; *_sum are "
    "sums of probabilities or of nats",
}


class LadderReadError(RuntimeError):
    """The read cannot give a valid result: bad input or a failed predictor."""


Log = Callable[[str], None]


def say(text: str) -> None:
    print(text, flush=True)


# --- what a directory's decision logs say ----------------------------------------


@dataclass
class SheetNote:
    """What the decision logs of one directory say about one game's sheets."""

    says_open: list[str] = field(default_factory=list)
    says_closed: list[str] = field(default_factory=list)
    sets: list[SheetSet] = field(default_factory=list)
    rows: int = 0

    @property
    def state(self) -> str:
        """Open when a row says so, closed when none does and one says closed,
        unknown when no row says anything about sheets (a directory played
        without sheet logging holds open games too)."""
        if self.says_open:
            return OPEN
        return CLOSED if self.says_closed else UNKNOWN

    @property
    def disagree(self) -> bool:
        """A row says open and another says closed (read as open)."""
        return bool(self.says_open and self.says_closed)

    @property
    def evidence(self) -> str:
        names = self.says_open if self.says_open else self.says_closed
        return ",".join(dict.fromkeys(names)) or (
            "rows_say_nothing" if self.rows else ""
        )

    def _say(self, source: str, value: Any) -> None:
        if value is True or value == OPEN:
            self.says_open.append(source)
        elif value is False or value == CLOSED:
            self.says_closed.append(source)


@dataclass
class DirectoryLogs:
    """The decision logs of one replay directory, reduced to what is read here."""

    sheets: dict[str, SheetNote] = field(default_factory=dict)
    # (battle id, turn) -> the first logged forecast of that turn, slimmed.
    forecasts: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    # (battle id, turn) of the forecasts in the bot's own decision log: the
    # file a search session's reading takes its forecasts from.
    own_forecasts: set[tuple[str, int]] = field(default_factory=set)
    # battle id -> the search's audit rows, in the file's order: (turn, the row
    # carries an observed reply, the decision was a forced one).
    search_rows: dict[str, list[tuple[int | None, bool, bool]]] = field(
        default_factory=dict
    )
    files: list[str] = field(default_factory=list)
    # file name -> sha256 and size of exactly the bytes that were read.
    hashes: dict[str, dict[str, Any]] = field(default_factory=dict)
    bad_lines: int = 0


def _slim_forecast(record: Mapping[str, Any]) -> dict[str, Any]:
    slots: list[dict[str, Any] | None] = []
    for slot in list(record.get("slots") or [])[: F.N_SLOT]:
        if not isinstance(slot, Mapping):
            slots.append(None)
            continue
        listed = slot.get("actions")
        slots.append(
            {
                "p_switch": slot.get("p_switch"),
                "p_protect": slot.get("p_protect"),
                "species": slot.get("species"),
                "actions": [
                    {
                        key: entry.get(key)
                        for key in ("kind", "p", "move", "target", "roster_index")
                        if key in entry
                    }
                    for entry in (listed if isinstance(listed, list) else [])
                    if isinstance(entry, Mapping)
                ],
            }
        )
    return {
        "model": record.get("model"),
        "sheets": list(record.get("sheets") or []),
        "sheets_reported": list(record.get("sheets_reported") or []),
        "slots": slots,
    }


def _forced_decision(audit: Mapping[str, Any]) -> bool:
    """``search_ladder_read.py``'s own test of a forced decision (a
    replacement): one of the bot's two actions is "nothing" and both are
    below the move range."""
    try:
        actions = [int(a) for a in (audit.get("champion_actions") or [9, 9])]
    except (TypeError, ValueError):
        return False
    return 0 in actions and all(action < 7 for action in actions)


def read_directory_logs(directory: Path) -> DirectoryLogs:
    """Sheet evidence, logged forecasts and the search's audit rows of
    ``directory/decisions*.jsonl``.

    Never raises: an unreadable file or line is counted and passed over.
    """
    out = DirectoryLogs()
    for path in sorted(directory.glob("decisions*.jsonl")):
        try:
            data = path.read_bytes()
        except OSError:
            out.bad_lines += 1
            continue
        out.files.append(path.name)
        out.hashes[path.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        for line in data.decode("utf-8", errors="replace").split("\n"):
            if '"battle"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                out.bad_lines += 1
                continue
            if not isinstance(row, dict):
                continue
            match = _AUDIT_BATTLE.search(str(row.get("battle", "")))
            if match is None:
                continue
            battle = match.group(1)
            try:
                turn: int | None = int(row.get("turn"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                turn = None
            note = out.sheets.setdefault(battle, SheetNote())
            note.rows += 1
            shadow = row.get("preview_shadow")
            if isinstance(shadow, dict):
                note._say("preview_shadow", shadow.get("open_sheet"))
            their = row.get("their_sheet")
            if isinstance(their, dict) and their:
                note._say("their_sheet", True)
                if not note.sets:
                    note.sets = _sheet_from_audit(their)
            for key in ("exact_search", "exact_preview"):
                audit = row.get(key)
                if isinstance(audit, dict):
                    note._say(key, audit.get("open_sheet"))
            audit = row.get("exact_search")
            if path.name == SEARCH_AUDIT_LOG and "exact_search" in row:
                audit = audit if isinstance(audit, dict) else {}
                coverage = audit.get("reply_coverage")
                observed = isinstance(coverage, dict) and bool(coverage.get("observed"))
                out.search_rows.setdefault(battle, []).append(
                    (turn, observed, _forced_decision(audit))
                )
            forecast = row.get("opponent_forecast")
            if isinstance(forecast, dict) and "slots" in forecast:
                reported = list(forecast.get("sheets_reported") or [])
                if reported:
                    note._say("forecast_log", reported[0])
                if turn is None:
                    continue
                out.forecasts.setdefault((battle, turn), _slim_forecast(forecast))
                if path.name == OWN_DECISION_LOG and turn:
                    out.own_forecasts.add((battle, turn))
    return out


def search_decisions(logs: DirectoryLogs) -> tuple[Counter[tuple[str, int]], int]:
    """The decisions a search session's reading counts, and its replies.

    ``evaluation/search_ladder_read.py`` section 5, row for row: an audit row
    that carries an observed reply is the reply to the game's last earlier
    move decision (a forced decision is no move decision); it is counted when
    that decision's forecast was logged. Returns ((battle id, turn of the
    decision) -> times counted, replies seen with or without a forecast).
    """
    counted: Counter[tuple[str, int]] = Counter()
    replies = 0
    for battle, rows in logs.search_rows.items():
        decided, last = False, None
        for turn, observed, forced in rows:
            if observed and decided:
                replies += 1
                if last is not None and (battle, last) in logs.own_forecasts:
                    counted[(battle, last)] += 1
            if not forced:
                decided, last = True, turn
    return counted, replies


# --- pages ----------------------------------------------------------------------


@dataclass
class TeamFile:
    """The bot's own sheet as a directory's ``run_config.json`` names it."""

    sets: list[SheetSet] = field(default_factory=list)
    why: str = ""  # why there is no sheet; "" when there is one
    path: str = ""
    sha256: str = ""
    run_config_sha256: str = ""

    def row(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "run_config_sha256": self.run_config_sha256,
            "sets": len(self.sets),
            "why_not": self.why,
        }


def read_team_file(root: Path, folder: Path) -> TeamFile:
    """The team file a directory's ``run_config.json`` names, as sheet sets.

    Every run recorded there must name the same ``material.our_team``; the
    file is looked for under ``root`` and next to the directory, and packed by
    poke-env's teambuilder, as the bot packs the team it sends. The file is
    the one on disk NOW (a run records no hash of its team), so its sha256 is
    kept with the reading. Never raises: no sets and a short reason when it
    cannot be read.
    """
    try:
        data = (folder / RUN_CONFIG).read_bytes()
        config = json.loads(data.decode("utf-8"))
    except (OSError, ValueError):
        return TeamFile(why="no_run_config")
    out = TeamFile(run_config_sha256=hashlib.sha256(data).hexdigest())
    runs = config.get("runs") if isinstance(config, dict) else None
    names = {
        str((run.get("material") or {}).get("our_team") or "")
        for run in (runs if isinstance(runs, list) else [])
        if isinstance(run, dict)
    }
    names.discard("")
    if len(names) != 1:
        out.why = "no_team_named" if not names else "runs_name_several_teams"
        return out
    name = names.pop()
    path = next(
        (base / name for base in (root, folder.parent) if (base / name).is_file()),
        Path(name) if Path(name).is_file() else None,
    )
    if path is None:
        out.why = "team_file_missing"
        return out
    out.path = str(path)
    try:
        from poke_env.teambuilder import ConstantTeambuilder

        text = path.read_bytes()
        out.sha256 = hashlib.sha256(text).hexdigest()
        packed = ConstantTeambuilder(text.decode("utf-8")).yield_team()
        out.sets = [entry for entry in parse_showteam(packed) if entry.moves]
    except Exception as exc:
        out.why = f"team_file_unread:{type(exc).__name__}"
        return out
    if not out.sets:
        out.why = "team_file_empty"
    return out


def team_file_sets(root: Path, folder: Path) -> tuple[list[SheetSet], str]:
    """(the bot's own sheet, why not): ``read_team_file`` without the hashes."""
    found = read_team_file(root, folder)
    return found.sets, found.why


def _log_head(log: str) -> str:
    """What tells two games apart whatever was played: the log up to its
    first turn (players, start time, team preview), hashed."""
    head = log.split("\n|turn|", 1)[0]
    return hashlib.sha256(head.encode("utf-8", errors="replace")).hexdigest()


class PageCorpus(Corpus):
    """The dataset builder's own-page reader, one named directory at a time.

    Reading a page, the HP rewrite and the public copy stay the builder's
    (``Corpus._own``). What changes:

    * ``walk`` goes through the directories one by one, and inside one the
      builder sees that directory's pages and that directory's decision rows
      only (``own_pages`` / ``sheet_audit``): a page is joined to its rows by
      battle id AND directory. The same id in a later directory is the same
      game when its log starts alike (``repeated``; skipped by the caller),
      another game otherwise (a local server restarts its numbers).
    * ``sheet_audit``: the sheet state comes from every kind of row
      ``read_directory_logs`` knows, not ``preview_shadow`` alone.
    * An open game whose opponent's sheet the decision log holds is also
      given the bot's own sheet (``read_team_file``), as the live bot gives
      both; ``own_sheet`` names the games where that was done,
      ``own_sheet_missing`` counts why not.
    * ``files`` counts, per directory, every ``*.html`` found and why one is
      no page of a game of these formats.

    A directory is known by ``str(path)`` (``names`` gives the short name to
    show).
    """

    def __init__(
        self,
        root: Path,
        directories: Sequence[Path],
        formats: Sequence[str] = DEFAULT_FORMATS,
        give_own_sheet: bool = True,
    ) -> None:
        super().__init__(root, formats, ("own",))
        self.directories = [Path(d) for d in directories]
        short = Counter(folder.name for folder in self.directories)
        self.names = {
            str(folder): folder.name if short[folder.name] == 1 else str(folder)
            for folder in self.directories
        }
        self.give_own_sheet = bool(give_own_sheet)
        self.logs: dict[str, DirectoryLogs] = {}
        self.files: dict[str, dict[str, int]] = {}
        self.notes_of: dict[tuple[str, str], SheetNote] = {}
        self.repeated: set[tuple[str, str]] = set()
        self.own_sheet: set[tuple[str, str]] = set()
        self.own_sheet_missing: Counter[str] = Counter()
        self.teams: dict[str, TeamFile] = {}
        self._pages: dict[str, list[tuple[Path, str, str]]] = {}
        self._current: Path | None = None

    def pages_in(self, folder: Path) -> list[tuple[Path, str, str]]:
        """(path, account, battle id) of the saved pages of one directory."""
        key = str(folder)
        if key in self._pages:
            return self._pages[key]
        found: list[tuple[Path, str, str]] = []
        count = {"html_files": 0, "not_a_saved_game_page": 0, "another_format": 0}
        for path in sorted(folder.glob("*.html")) if folder.is_dir() else []:
            count["html_files"] += 1
            match = _OWN_NAME.match(path.name)
            if match is None:
                count["not_a_saved_game_page"] += 1
            elif match["id"].rsplit("-", 1)[0] not in self.formats:
                count["another_format"] += 1
            else:
                found.append((path, match["player"].strip(), match["id"]))
        count["pages"] = len(found)
        self.files[key] = count
        self._pages[key] = found
        return found

    def directory_logs(self, folder: Path) -> DirectoryLogs:
        key = str(folder)
        if key not in self.logs:
            self.logs[key] = read_directory_logs(folder)
        return self.logs[key]

    def own_pages(self) -> list[tuple[Path, str, str]]:
        """The current directory's pages while ``walk`` is inside one."""
        if self._current is not None:
            return list(self.pages_in(self._current))
        return [page for folder in self.directories for page in self.pages_in(folder)]

    def bot_accounts(self) -> set[str]:
        return {
            user_id(account)
            for folder in self.directories
            for _, account, _ in self.pages_in(folder)
        }

    def sheet_audit(self) -> dict[str, bool]:
        """The current directory's sheet states; nothing outside ``walk``."""
        if self._audit is not None:
            return self._audit
        audit: dict[str, bool] = {}
        folder = self._current
        if folder is not None:
            for battle, note in self.directory_logs(folder).sheets.items():
                self.notes_of[(str(folder), battle)] = note
                if note.state == UNKNOWN:
                    continue
                audit[battle] = note.state == OPEN
                if note.sets:
                    self._their[battle] = note.sets
        self._audit = audit
        return audit

    def _give_own_sheet(self, folder: Path, raw: RawBattle) -> None:
        side = raw.bot_side
        # Only where the builder took the opponent's sheet from the decision
        # log: a public copy carries both sheets already.
        if raw.sheet_audit != OPEN or raw.sheets_known or side not in SIDES:
            return
        other = SIDES[1 - SIDES.index(side)]
        theirs = (raw.sheets or {}).get(other)
        if not self.give_own_sheet:
            self.own_sheet_missing["switched_off"] += 1
            return
        if not theirs:
            self.own_sheet_missing["no_opponent_sheet_in_the_log"] += 1
            return
        key = str(folder)
        if key not in self.teams:
            self.teams[key] = read_team_file(self.root, folder)
        team = self.teams[key]
        if not team.sets:
            self.own_sheet_missing[team.why] += 1
            return
        raw.sheets = {other: list(theirs), str(side): list(team.sets)}
        raw.sheets_known = True
        self.own_sheet.add((key, raw.battle_id))

    def walk(self) -> Iterator[tuple[str, RawBattle]]:
        """(directory, page as the builder reads it), directory by directory."""
        heads: dict[str, set[str]] = {}
        for folder in self.directories:
            key = str(folder)
            self._current, self._audit, self._their = folder, None, {}
            try:
                for raw in Corpus._own(self):
                    seen = heads.setdefault(raw.battle_id, set())
                    head = _log_head(raw.log)
                    if head in seen:
                        self.repeated.add((key, raw.battle_id))
                    seen.add(head)
                    self._give_own_sheet(folder, raw)
                    yield key, raw
            finally:
                self._current, self._audit, self._their = None, None, {}

    def _own(self) -> Iterator[RawBattle]:
        for _, raw in self.walk():
            yield raw


@dataclass
class Game:
    """One kept game: the builder's view of the page and what was driven."""

    index: int
    battle_id: str
    directory: str  # the name shown
    folder: str  # the directory's key: joins the game to its own decision rows
    path: str
    time: int | None
    time_source: str
    bot_side: str
    opponent: str
    sheet: str
    sheet_evidence: str
    own_sheet: bool  # the bot's own sheet was given from the team file
    both_sheets: bool  # the predictor is given both sheets (either source)
    result: DriveResult

    @property
    def date(self) -> str:
        if self.time is None:
            return ""
        return time.strftime("%Y-%m-%d", time.localtime(self.time))

    def row(self) -> dict[str, Any]:
        result = self.result
        won = None if result.winner is None else result.winner == self.bot_side
        stamp = None
        if self.time is not None:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.time))
        return {
            "index": self.index,
            "battle": f"battle-{self.battle_id}",
            "directory": self.directory,
            "page": self.path,
            "date": self.date,
            "time": stamp,
            "time_source": self.time_source,
            "bot_side": self.bot_side,
            "opponent_rating": result.ratings.get(self.opponent),
            "bot_rating": result.ratings.get(self.bot_side),
            "sheet": self.sheet,
            "sheet_evidence": self.sheet_evidence,
            "own_sheet_given": self.own_sheet,
            "both_sheets": self.both_sheets,
            "turns": len(result.turns),
            "finished": bool(result.finished),
            "bot_won": won,
        }


def _day_start(text: str | None, name: str) -> float | None:
    """Local ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM`` as Unix seconds."""
    if not text:
        return None
    for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"):
        try:
            return time.mktime(time.strptime(text, pattern))
        except ValueError:
            continue
    raise LadderReadError(f"{name} {text!r} is not YYYY-MM-DD or YYYY-MM-DDTHH:MM")


def _game_ids(spec: str | None) -> set[str] | None:
    """``--games``: a comma list of battle ids or tags, or ``@file`` of them."""
    if not spec:
        return None
    text = spec
    if spec.startswith("@"):
        try:
            text = Path(spec[1:]).read_text(encoding="utf-8")
        except OSError as exc:
            raise LadderReadError(f"--games {spec}: {exc}") from exc
    found = {match.group(1) for match in _AUDIT_BATTLE.finditer(text)}
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if part and "battle-" not in part:
            found.add(part)
    return found


def read_games(
    root: Path,
    directories: Sequence[Path],
    *,
    since: str | None = None,
    until: str | None = None,
    games: str | None = None,
    include_unrated: bool = False,
    own_sheet: str = OWN_SHEET_FILE,
    formats: Sequence[str] = DEFAULT_FORMATS,
    fresh: bool = False,
) -> tuple[list[Game], dict[str, Any]]:
    """The games to score, and what was found and left out (by reason).

    The games come back in one order whatever order the directories were
    given in (battle id, then directory). ``info`` accounts for every page
    (``pages`` = ``kept`` + the sum of ``skipped``), counts the games that
    have decision rows and no page (``no_page``) and the forecasts the bot
    logged, and carries two tables for the scoring, both keyed by (directory,
    battle id, turn): ``_forecasts`` (the logged forecasts) and ``_search``
    (how many times a search session's reading counts the decision).

    THE SEALED SET. A page dated on or after ``FRESH_SINCE`` that passes the
    filters is confirmation data. Without ``fresh=True`` it is never driven
    (no state and no label is made of it) and the call raises
    ``LadderReadError`` once every page has been looked at, saying how many
    there are. This function is the guard, so a script that calls it directly
    is refused like the command line. ``until=FRESH_SINCE`` leaves those
    pages out instead: an ordinary reading of the older games.
    """
    if own_sheet not in (OWN_SHEET_FILE, OWN_SHEET_NONE):
        raise LadderReadError(
            f"--own-sheet is {OWN_SHEET_FILE!r} or {OWN_SHEET_NONE!r}, "
            f"got {own_sheet!r}"
        )
    missing = [str(d) for d in directories if not Path(d).is_dir()]
    if missing:
        raise LadderReadError(f"no such replay directory: {', '.join(missing)}")
    places = [Path(d).resolve() for d in directories]
    twice = sorted({str(d) for d in places if places.count(d) > 1})
    if twice:
        raise LadderReadError(f"replay directory given twice: {', '.join(twice)}")
    first, last = _day_start(since, "--since"), _day_start(until, "--until")
    seal = _day_start(FRESH_SINCE, "the sealed set's first day") or 0.0
    sealed_unasked = 0
    wanted = _game_ids(games)
    corpus = PageCorpus(root, directories, formats, own_sheet == OWN_SHEET_FILE)
    bots = corpus.bot_accounts()
    skipped: Counter[str] = Counter()
    kept: list[Game] = []
    for folder, raw in corpus.walk():
        if (folder, raw.battle_id) in corpus.repeated:
            skipped["duplicate:the same game in an earlier directory"] += 1
            continue
        if wanted is not None and raw.battle_id not in wanted:
            skipped["filter:not in --games"] += 1
            continue
        if first is not None and (raw.time is None or raw.time < first):
            skipped["filter:before --since"] += 1
            continue
        if last is not None and (raw.time is None or raw.time >= last):
            skipped["filter:at or after --until"] += 1
            continue
        if not fresh and raw.time is not None and raw.time >= seal:
            # Sealed and not asked for: counted, never driven.
            sealed_unasked += 1
            continue
        result = drive(raw)
        reason = drop_reason(raw, result, formats, bots)
        if reason is not None:
            skipped[f"builder:{reason}"] += 1
            continue
        if result.rated_kind != RATED_LADDER and not include_unrated:
            skipped["not_ladder_rated"] += 1
            continue
        side = str(raw.bot_side)
        note = corpus.notes_of.get((folder, raw.battle_id))
        evidence = raw.sheet_evidence
        if evidence in ("", "audit") and note is not None:
            evidence = note.evidence
        state = raw.sheet_audit or UNKNOWN
        kept.append(
            Game(
                index=len(kept),
                battle_id=raw.battle_id,
                directory=corpus.names[folder],
                folder=folder,
                path=raw.path,
                time=raw.time,
                time_source=raw.time_source,
                bot_side=side,
                opponent=SIDES[1 - SIDES.index(side)],
                sheet=state,
                sheet_evidence=evidence,
                own_sheet=(folder, raw.battle_id) in corpus.own_sheet,
                both_sheets=state == OPEN and bool(raw.sheets_known),
                result=result,
            )
        )
    if sealed_unasked:
        raise LadderReadError(
            f"{sealed_unasked} of the pages to read are dated on or after "
            f"{FRESH_SINCE}: the sealed confirmation set. Give --fresh "
            f"(fresh=True) to read it, or --until {FRESH_SINCE} to leave it out"
        )
    # One order whatever order the directories were given in (the resampling
    # of games depends on it): by battle id, then by directory.
    kept.sort(key=lambda game: (game.battle_id, game.folder))
    for position, game in enumerate(kept):
        game.index = position
    # A page the builder could not turn into a game never reached the loop.
    for name, reason in (
        ("unreadable:own", "page:unreadable"),
        ("own:no_log_block", "page:no_log_block"),
        ("own:account_not_a_player", "page:account_is_not_a_player"),
    ):
        if corpus.notes.get(name):
            skipped[reason] += int(corpus.notes[name])
    names = corpus.names
    pages = {names[key]: count["pages"] for key, count in corpus.files.items()}
    total = int(sum(pages.values()))
    # Games the bot played and no page was saved for: rows, and no way to score.
    logged: dict[tuple[str, str, int], dict[str, Any]] = {}
    counted: Counter[tuple[str, str, int]] = Counter()
    search = {"replies": 0, "decisions": 0}
    kept_keys = {(game.folder, game.battle_id) for game in kept}
    forecasts = {"total": 0, "in_kept_games": 0, "in_games_without_a_page": 0}
    no_page: dict[str, dict[str, Any]] = {}
    disagree = 0
    for folder in corpus.directories:
        key = str(folder)
        logs = corpus.directory_logs(folder)
        with_page = {battle for _, _, battle in corpus.pages_in(folder)}
        absent = sorted(
            battle
            for battle in logs.sheets
            if battle not in with_page and battle.rsplit("-", 1)[0] in formats
        )
        disagree += sum(1 for note in logs.sheets.values() if note.disagree)
        per_game: Counter[str] = Counter(battle for battle, _ in logs.forecasts)
        for (battle, turn), forecast in logs.forecasts.items():
            logged[(key, battle, turn)] = forecast
            forecasts["total"] += 1
            forecasts["in_kept_games"] += int((key, battle) in kept_keys)
            forecasts["in_games_without_a_page"] += int(battle in absent)
        decisions, replies = search_decisions(logs)
        search["replies"] += replies
        search["decisions"] += int(sum(decisions.values()))
        for (battle, turn), times in decisions.items():
            counted[(key, battle, turn)] = times
        if absent:
            no_page[names[key]] = {
                "games": len(absent),
                "decision_rows": sum(logs.sheets[battle].rows for battle in absent),
                "logged_forecasts": sum(per_game[battle] for battle in absent),
                "battles": [f"battle-{battle}" for battle in absent],
            }
    forecasts["in_other_games"] = (
        forecasts["total"]
        - forecasts["in_kept_games"]
        - forecasts["in_games_without_a_page"]
    )
    search["in_kept_games"] = int(
        sum(
            times
            for (key, battle, _), times in counted.items()
            if (key, battle) in kept_keys
        )
    )
    open_kept = [game for game in kept if game.sheet == OPEN]
    info = {
        "directories": [str(d) for d in directories],
        "files": {names[key]: dict(count) for key, count in corpus.files.items()},
        "pages_found": pages,
        "pages": total,
        "kept": len(kept),
        "skipped": dict(sorted(skipped.items())),
        "pages_unaccounted": total - len(kept) - int(sum(skipped.values())),
        "no_page": {
            "games": sum(row["games"] for row in no_page.values()),
            "decision_rows": sum(row["decision_rows"] for row in no_page.values()),
            "logged_forecasts": sum(
                row["logged_forecasts"] for row in no_page.values()
            ),
            "by_directory": no_page,
        },
        "logged_forecasts": forecasts,
        "search_session": search,
        "sheet_rows_disagree": disagree,
        "bot_accounts": sorted(bots),
        "notes": dict(corpus.notes),
        "decision_logs": {
            names[key]: {
                "files": logs.files,
                "bad_lines": logs.bad_lines,
                "read": logs.hashes,
            }
            for key, logs in corpus.logs.items()
        },
        "filters": {"since": since, "until": until, "games": games},
        "include_unrated": bool(include_unrated),
        "own_sheet": {
            "mode": own_sheet,
            "open_games_kept": len(open_kept),
            "both_sheets": sum(1 for game in open_kept if game.both_sheets),
            "given": sum(1 for game in kept if game.own_sheet),
            "from_a_public_copy": sum(
                1 for game in open_kept if game.both_sheets and not game.own_sheet
            ),
            "not_given_all_pages": dict(corpus.own_sheet_missing),
            "team_files": {
                names[key]: team.row() for key, team in corpus.teams.items()
            },
        },
    }
    info["_forecasts"] = logged
    info["_search"] = counted
    return kept, info


# --- one artifact ---------------------------------------------------------------


def encode_games(games: Sequence[Game], featurizer: F.Featurizer) -> F.Batch:
    """The opponent's labelled example of every turn of every game.

    ``m_battle`` is the game's index, ``m_row`` the turn's position in the
    driven log (a key that a repeated turn number cannot break), ``m_sheet``
    the recorded sheet state.
    """
    examples: list[F.Example] = []
    for game in games:
        sheet = SHEET_STATES.index(
            game.sheet if game.sheet in SHEET_STATES else "unknown"
        )
        for position, record in enumerate(game.result.turns):
            example = featurizer.encode_turn(
                record.snapshot, record.actions, game.opponent
            )
            if example is None:
                continue
            example["m_battle"] = np.array(game.index, dtype=np.int32)
            example["m_row"] = np.array(position, dtype=np.int32)
            example["m_turn"] = np.array(min(32767, record.turn), dtype=np.int16)
            example["m_sheet"] = np.array(sheet, dtype=np.uint8)
            examples.append(example)
    return featurizer.collate(examples)


def context_masks(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-example masks of ``CONTEXTS`` (the scorecard's, plus "not open")."""
    base = SC.slice_masks(batch)
    out = {name: base[name] for name in CONTEXTS if name in base}
    out[SHEET_NOT_OPEN] = ~base[SHEET_OPEN]
    return {name: out[name] for name in CONTEXTS}


def reply_masks(
    batch: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Per-example masks of ``REPLIES``; meaningful on fully visible turns."""
    active = np.asarray(truth["active"]).astype(bool)
    switch = np.asarray(truth["switch"]).astype(bool).any(-1)
    flags = np.asarray(batch["y_flag"])
    protect = (active & (flags[..., F.Y_PROTECTED] == 1)).any(-1)
    unshown = np.asarray(truth["unshown"]).astype(bool).any(-1)
    return {
        ALL: np.ones(switch.shape, dtype=bool),
        REPLY_SWITCH: switch,
        REPLY_PROTECT: ~switch & protect,
        REPLY_MOVES: ~switch & ~protect,
        REPLY_UNSHOWN: unshown,
        REPLY_NO_UNSHOWN: ~unshown,
    }


def lenient_hits(
    made: J.JointReplies,
    batch: Mapping[str, np.ndarray],
    truth: Mapping[str, np.ndarray],
) -> np.ndarray:
    """``[N]``: an entry of the list names everything the log shows.

    The search session's matching rule on this module's reply indices: a slot
    whose action is hidden matches any entry, a visible move whose target is
    not certain matches on the move, a move outside the candidates matches
    nothing; with the Mega bit the entry's Mega state must be the observed
    one (a slot the log cannot tell counts as not Mega-evolving).
    """
    active = np.asarray(truth["active"]).astype(bool)
    reply = np.asarray(truth["reply"]).astype(np.int64)
    y_action = np.asarray(batch["y_action"]).astype(np.int64)
    y_target = np.asarray(batch["y_target"]).astype(np.int64)
    n_cand = int(made.n_cand)
    listed = np.asarray(made.reply).astype(np.int64)  # [N, K, 2]
    is_move = (listed >= 0) & (listed < n_cand * F.N_TARGET)
    column = np.where(is_move, listed // F.N_TARGET, -1)
    hidden = active & (y_action < 0)
    outside = active & (y_action == n_cand)
    loose = active & (y_action >= 0) & (y_action < n_cand) & (y_target < 0)
    same = listed == reply[:, None, :]
    by_move = loose[:, None, :] & (column == y_action[:, None, :])
    slot = (same | hidden[:, None, :] | by_move) & ~outside[:, None, :]
    entry = slot.all(-1) & (listed != J.UNKNOWN).all(-1)
    if made.with_mega:
        state = np.asarray(truth["mega"]).astype(np.int64)
        state = np.where(state == J.UNKNOWN, J.MEGA_NONE, state)
        entry &= np.asarray(made.mega).astype(np.int64) == state[:, None]
    return entry.any(-1) & active.any(-1)


@dataclass
class Scored:
    """One artifact's arrays over its own examples."""

    label: str
    batch: F.Batch
    pred: dict[str, np.ndarray]
    tables: F.Tables
    nll: dict[str, np.ndarray]
    truth: dict[str, np.ndarray]
    rank: dict[str, np.ndarray]  # variant -> [N], -1 = not ranked
    other: np.ndarray  # [N]: the real reply needs a move outside the candidates
    lenient: dict[str, np.ndarray]  # variant -> [N]
    seconds: float
    # The name of the artifact's pair coupling the joint lists were built
    # with; "" = the plain product of the two slots.
    coupling: str = ""
    # For a coupled artifact: the truth's rank under the plain product of the
    # same marginals (no Mega bit), to show what the coupling changed.
    rank_plain: np.ndarray | None = None
    # [N]: how many times a search session's reading counts the turn.
    search: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(np.asarray(self.batch["turn"]).shape[0])

    def hit(self, k: int, variant: str = PLAIN) -> np.ndarray:
        rank = self.rank[variant]
        return (rank >= 0) & (rank < k) & ~self.other

    def hit_plain(self, k: int) -> np.ndarray:
        """``hit`` under the plain product; the artifact's own without a coupling."""
        rank = self.rank[PLAIN] if self.rank_plain is None else self.rank_plain
        return (rank >= 0) & (rank < k) & ~self.other

    def keys(self) -> list[tuple[int, int]]:
        games = np.asarray(self.batch["m_battle"]).tolist()
        rows = np.asarray(self.batch["m_row"]).tolist()
        return list(zip(games, rows))


def score_artifact(
    label: str,
    loaded: Any,
    games: Sequence[Game],
    search: Mapping[tuple[str, str, int], int] | None = None,
) -> Scored:
    """Encode the games with the artifact's featurizer, predict, rank.

    The joint lists are the artifact's own: built with its pair coupling when
    it carries one that is not all ones (``loaded.coupling``), as the runtime
    builds them. ``search`` is ``read_games``'s count of the decisions a
    search session's reading counts, by (directory, battle id, turn).
    """
    batch = encode_games(games, loaded.featurizer)
    if not batch or int(np.asarray(batch["turn"]).shape[0]) == 0:
        raise LadderReadError(f"{label}: no example could be encoded")
    try:
        pred, seconds = SC.guarded_predict(loaded.predictor, batch, name=label)
    except SC.ScorecardError as exc:
        raise LadderReadError(str(exc)) from exc
    truth = J.true_replies(batch)
    if not truth:
        raise LadderReadError(f"{label}: the labels cannot be read as joint replies")
    features = SC.features_only(batch)
    n = int(np.asarray(batch["turn"]).shape[0])
    coupling = getattr(loaded, "coupling", None)
    if coupling is not None and bool(coupling.is_identity):
        coupling = None  # all ones: the plain product, on the plain path
    intent = loaded.featurizer.tables.move_intent

    def listed(mega: bool, pair: Any) -> J.JointReplies:
        made = J.joint_replies(
            pred,
            features,
            k=TOP_K,
            mega=mega,
            truth=truth,
            coupling=pair,
            move_intent=intent if pair is not None else None,
        )
        if made.n != n:
            raise LadderReadError(
                f"{label}: the prediction cannot be read as joint replies "
                f"({dict(J.COUNTERS)})"
            )
        if bool(getattr(made, "coupled", False)) != (pair is not None):
            raise LadderReadError(
                f"{label}: the joint list was not built with the artifact's "
                "pair coupling"
            )
        return made

    rank: dict[str, np.ndarray] = {}
    lenient: dict[str, np.ndarray] = {}
    for variant in (PLAIN, MEGA):
        made = listed(variant == MEGA, coupling)
        rank[variant] = np.asarray(made.rank).astype(np.int64)
        lenient[variant] = lenient_hits(made, batch, truth)
    rank_plain = None
    if coupling is not None:
        rank_plain = np.asarray(listed(False, None).rank).astype(np.int64)
    weight = np.zeros(n, dtype=np.int64)
    if search:
        index = np.asarray(batch["m_battle"]).tolist()
        turns = np.asarray(batch["m_turn"]).tolist()
        taken: set[tuple[str, str, int]] = set()
        for row in range(n):
            game = games[int(index[row])]
            key = (game.folder, game.battle_id, int(turns[row]))
            if key not in taken:  # a turn number met twice is counted once
                taken.add(key)
                weight[row] = int(search.get(key, 0))
    return Scored(
        label=label,
        batch=batch,
        pred=pred,
        tables=loaded.featurizer.tables,
        nll=F.slot_nll(pred, batch),
        truth={name: np.asarray(value) for name, value in truth.items()},
        rank=rank,
        other=np.asarray(truth["other"]).astype(bool).any(-1),
        lenient=lenient,
        seconds=float(seconds),
        coupling="" if coupling is None else str(coupling.name or "unnamed"),
        rank_plain=rank_plain,
        search=weight,
    )


def _share(hits: np.ndarray, mask: np.ndarray) -> float | None:
    return float(hits[mask].mean()) if mask.any() else None


def _games(scored: Scored, rows: np.ndarray) -> int:
    """Games with at least one of the examples ``rows`` (a mask)."""
    return int(np.unique(np.asarray(scored.batch["m_battle"])[rows]).size)


def _ratio(
    draws: SC.Draws, top: Any, bottom: Any, games: int
) -> dict[str, float | None]:
    """A ratio of two columns with its interval; no interval inside one game
    (one game resampled is that game again)."""
    found = draws.ratio(top, bottom)
    if games < 2:
        found["low"] = found["high"] = None
    return found


def _slot_block(
    scored: Scored, mask: np.ndarray, draws: SC.Draws, name: str
) -> dict[str, Any]:
    """Per-slot numbers of one context: NLL, top-k, intent."""
    nll = scored.nll
    labelled = nll["fine_scored"] & mask[:, None]
    hidden = nll["censored"] & mask[:, None]
    visible = labelled & ~hidden
    games = _games(scored, mask)
    out: dict[str, Any] = {
        "examples": int(mask.sum()),
        "games": games,
        "slot_turns": int(labelled.sum()),
        "visible": int(visible.sum()),
        "hidden": int(hidden.sum()),
        "fine_nll": _ratio(draws, ("nll", name), ("n", name), games),
        "fine_nll_visible": SC._mean(nll["fine"], visible),
        "fine_nll_hidden": SC._mean(nll["fine"], hidden),
        "action_nll": SC._mean(nll["action"], labelled),
        "target_nll_per_slot_turn": SC._mean(nll["target"], labelled),
    }
    top: dict[str, Any] = {}
    if mask.any():
        part = F.take(scored.batch, mask)
        pred = SC._rows(scored.pred, mask)
        sub = {key: np.asarray(value)[mask] for key, value in nll.items()}
        top = SC._top_metrics(pred, part, sub, scored.tables, F.PROBABILITY_FLOOR)
    for key in (
        "fine_top1",
        "fine_top3",
        "fine_labels",
        "intent_accuracy",
        "intent_labels",
    ):
        out[key] = top.get(key)
    return out


def _event_block(
    scored: Scored,
    mask: np.ndarray,
    draws: SC.Draws,
    name: str,
    labels: Mapping[str, tuple[np.ndarray, np.ndarray]],
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for event in EVENTS:
        known, outcome = labels[event]
        known = known & mask[:, None]
        p = probabilities[event]
        bar = SC.PRE_REGISTERED[event]
        summary = SC.brier_summary(p[known], outcome[known])
        decision = SC.decision_row(p[known], outcome[known], bar)
        fired = known & (p >= bar)
        decision["precision"] = _ratio(
            draws,
            ("ev_hit", event, name),
            ("ev_n", event, name),
            _games(scored, fired.any(-1)),
        )
        out[event] = {
            "known": summary["slots"],
            "observed": summary["observed"],
            "predicted": summary["predicted"],
            "brier_skill": summary["skill"],
            "at_threshold": decision,
        }
    return out


def _joint_block(
    scored: Scored,
    mask: np.ndarray,
    replies: Mapping[str, np.ndarray],
    draws: SC.Draws,
    name: str,
) -> dict[str, Any]:
    visible = np.asarray(scored.truth["visible"]).astype(bool)
    acting = np.asarray(scored.truth["active"]).astype(bool).any(-1) & mask
    counted = visible & mask
    out: dict[str, Any] = {
        "turns": int(acting.sum()),
        "counted": int(counted.sum()),
        "not_fully_visible": int((acting & ~visible).sum()),
        "involves_other": int((counted & scored.other).sum()),
        "replies": {},
    }
    for reply, part in replies.items():
        rows = counted & part
        row: dict[str, Any] = {
            "examples": int(rows.sum()),
            "games": _games(scored, rows),
            "top": {str(k): _share(scored.hit(k), rows) for k in TOP_KS},
            "top_interval": _ratio(
                draws,
                ("j_hit", name, reply),
                ("j_n", name, reply),
                _games(scored, rows),
            ),
            "top_with_mega": _share(scored.hit(TOP_K, MEGA), rows),
        }
        if scored.rank_plain is not None:
            # What the coupling changed: the same marginals, multiplied plainly.
            own, plain = scored.hit(TOP_K), scored.hit_plain(TOP_K)
            row["top_plain_product"] = {
                str(k): _share(scored.hit_plain(k), rows) for k in TOP_KS
            }
            row["only_coupled"] = int((own & ~plain & rows).sum())
            row["only_plain_product"] = int((plain & ~own & rows).sum())
        out["replies"][reply] = row
    # The matching rule of a search session's reading, on two sets of turns:
    # every turn where the opponent showed something, and the decisions the
    # search session itself counts.
    y_action = np.asarray(scored.batch["y_action"]).astype(np.int64)
    active = np.asarray(scored.truth["active"]).astype(bool)
    shown = (active & (y_action >= 0)).any(-1) & mask
    times = (
        np.zeros(mask.shape, dtype=np.int64) if scored.search is None else scored.search
    )
    times = np.where(mask, times, 0)
    out["nothing_shown"] = int((acting & ~shown).sum())
    rule: dict[str, Any] = {}
    for variant in (PLAIN, MEGA):
        strict = scored.hit(TOP_K, variant)
        rule[variant] = {
            RULE_VISIBLE: {
                "turns": int(counted.sum()),
                "hits": int((strict & counted).sum()),
            },
            RULE_ALL_MISS: {
                "turns": int(acting.sum()),
                "hits": int((strict & counted).sum()),
            },
            RULE_ALL_ANY: {
                "turns": int(shown.sum()),
                "hits": int((scored.lenient[variant] & shown).sum()),
            },
            RULE_SEARCH: {
                "turns": int(times.sum()),
                "hits": int((times * scored.lenient[variant]).sum()),
            },
        }
        for cell in rule[variant].values():
            cell["share"] = cell["hits"] / cell["turns"] if cell["turns"] else None
    out["search_rule"] = rule
    return out


def report_artifact(
    scored: Scored, resamples: int, seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Every number of one artifact, and its per-game sums."""
    batch = scored.batch
    contexts = context_masks(batch)
    replies = reply_masks(batch, scored.truth)
    visible = np.asarray(scored.truth["visible"]).astype(bool)
    labels = SC.event_labels(batch)
    probabilities = SC.event_predictions(scored.pred, batch)
    sums = SC.GameSums(np.asarray(batch["m_battle"]))
    for name, mask in contexts.items():
        labelled = scored.nll["fine_scored"] & mask[:, None]
        sums.count(("n", name), labelled)
        sums.add(("nll", name), scored.nll["fine"], labelled)
        for event in EVENTS:
            known, outcome = labels[event]
            fired = (
                known
                & mask[:, None]
                & (probabilities[event] >= SC.PRE_REGISTERED[event])
            )
            sums.count(("ev_n", event, name), fired)
            sums.count(("ev_hit", event, name), fired & outcome)
        for reply, part in replies.items():
            rows = visible & mask & part
            sums.count(("j_n", name, reply), rows)
            sums.count(("j_hit", name, reply), rows & scored.hit(TOP_K))
    draws = sums.draws(resamples, seed)
    out: dict[str, Any] = {
        "examples": scored.n,
        "predict_seconds": round(scored.seconds, 3),
        "sanity": SC.sanity_block(scored.pred, batch),
        "splits": {},
    }
    for name, mask in contexts.items():
        block = _slot_block(scored, mask, draws, name)
        block["events"] = _event_block(scored, mask, draws, name, labels, probabilities)
        block["joint"] = _joint_block(scored, mask, replies, draws, name)
        out["splits"][name] = block
    return out, game_sums(scored, replies, labels, probabilities)


def game_sums(
    scored: Scored,
    replies: Mapping[str, np.ndarray],
    labels: Mapping[str, tuple[np.ndarray, np.ndarray]],
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Per game, the sums behind every headline number of one artifact.

    Counts, except the names ending in ``_sum`` (nats, or probabilities
    added up). A later run can split games by date or directory from these
    without scoring again.
    """
    batch = scored.batch
    games = np.asarray(batch["m_battle"]).astype(np.int64)
    size = int(games.max()) + 1 if games.size else 0

    def per_game(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64)
        return np.bincount(
            games, data.reshape(data.shape[0], -1).sum(-1), minlength=size
        )

    labelled = scored.nll["fine_scored"]
    visible = np.asarray(scored.truth["visible"]).astype(bool)
    fine, label = F.fine_probs(scored.pred, batch), F.fine_label(batch)
    out = {
        "slot_turns": per_game(labelled),
        "nll_sum": per_game(np.where(labelled, scored.nll["fine"], 0.0)),
        "fine_labels": per_game(label >= 0),
    }
    for k in (1, 3):
        out[f"fine_top{k}"] = per_game(F.topk_hits(fine, label, k)[0])
    intents = F.intent_probs(scored.pred, batch, scored.tables)
    if intents is not None:
        y_intent = np.asarray(batch["y_intent"]).astype(np.int64)
        guess = np.asarray(intents)[..., : F.N_INTENT].argmax(-1)
        out["intent_labels"] = per_game(y_intent >= 0)
        out["intent_hits"] = per_game((y_intent >= 0) & (guess == y_intent))
    for event in EVENTS:
        known, outcome = labels[event]
        p = probabilities[event]
        fired = known & (p >= SC.PRE_REGISTERED[event])
        out[f"{event}_known"] = per_game(known)
        out[f"{event}_observed"] = per_game(known & outcome)
        out[f"{event}_predicted_sum"] = per_game(np.where(known, p, 0.0))
        out[f"{event}_at_threshold"] = per_game(fired)
        out[f"{event}_at_threshold_hits"] = per_game(fired & outcome)
    out["joint_counted"] = per_game(visible)
    for k in TOP_KS:
        out[f"joint_top{k}"] = per_game(visible & scored.hit(k))
    for short, reply in GAME_REPLIES:
        rows = visible & replies[reply]
        out[f"joint_counted_{short}"] = per_game(rows)
        out[f"joint_top{TOP_K}_{short}"] = per_game(rows & scored.hit(TOP_K))
    return out


# --- paired comparison ----------------------------------------------------------


def _aligned(first: Scored, second: Scored) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the examples both artifacts have, in ``first``'s order."""
    place = {key: row for row, key in enumerate(second.keys())}
    left: list[int] = []
    right: list[int] = []
    for row, key in enumerate(first.keys()):
        other = place.get(key)
        if other is not None:
            left.append(row)
            right.append(other)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def paired(
    scored: Scored, reference: Scored, resamples: int, seed: int
) -> dict[str, Any]:
    """``scored`` minus ``reference`` on what both have, by context."""
    left, right = _aligned(scored, reference)
    battle = np.asarray(scored.batch["m_battle"])[left]
    contexts = {name: mask[left] for name, mask in context_masks(scored.batch).items()}
    both = scored.nll["fine_scored"][left] & reference.nll["fine_scored"][right]
    seen = (
        np.asarray(scored.truth["visible"]).astype(bool)[left]
        & np.asarray(reference.truth["visible"]).astype(bool)[right]
    )
    own = scored.hit(TOP_K)[left].astype(np.float64)
    base = reference.hit(TOP_K)[right].astype(np.float64)
    # The slot-turns are the same; the label is each featurizer's own.
    here_other = _other_label(scored)[left]
    there_other = _other_label(reference)[right]
    either = here_other | there_other
    acting, differ = _candidate_sets_differ(scored, reference, left, right)
    out: dict[str, Any] = {
        "examples_both": int(left.size),
        "examples_only_here": scored.n - int(left.size),
        "examples_only_reference": reference.n - int(left.size),
        "slot_turns_labelled_by_one_only": int(
            (
                scored.nll["fine_scored"][left] != reference.nll["fine_scored"][right]
            ).sum()
        ),
        "labels": {
            "slot_turns": int(both.sum()),
            "other_here": int((both & here_other).sum()),
            "other_reference": int((both & there_other).sum()),
            "other_either": int((both & either).sum()),
            "slots_acting": acting,
            "slots_with_different_candidates": differ,
        },
        "splits": {},
    }
    for name, mask in contexts.items():
        slots = both & mask[:, None]
        turns = seen & mask
        nll = SC.paired_difference(
            scored.nll["fine"][left],
            reference.nll["fine"][right],
            slots,
            battle,
            resamples=resamples,
            seed=seed,
        )
        named = SC.paired_difference(
            scored.nll["fine"][left],
            reference.nll["fine"][right],
            slots & ~either,
            battle,
            resamples=resamples,
            seed=seed,
        )
        top = SC.paired_difference(
            own, base, turns, battle, resamples=resamples, seed=seed
        )
        top["turns"] = top.pop("slots")
        top["only_here"] = int(((own > base) & turns).sum())
        top["only_reference"] = int(((base > own) & turns).sum())
        top["sign_test_p"] = sign_test(top["only_here"], top["only_reference"])
        # How often a resample of the games does NOT favour this artifact: a
        # steadier reading than an interval's end, which moves with the seed.
        nll["resamples_not_better"] = _not_better(
            reference.nll["fine"][right] - scored.nll["fine"][left],
            slots,
            battle,
            resamples,
            seed,
        )
        top["resamples_not_better"] = _not_better(
            own - base, turns, battle, resamples, seed
        )
        out["splits"][name] = {
            "fine_nll": nll,
            "fine_nll_neither_other": named,
            "joint_top8": top,
        }
    return out


def _not_better(
    gain: np.ndarray, keep: np.ndarray, battle: np.ndarray, resamples: int, seed: int
) -> float | None:
    """Share of the game resamples whose summed ``gain`` is not above zero
    (``gain`` > 0 where the artifact does better); None without resamples.
    The resamples are ``paired_difference``'s own (same games, same seed)."""
    sums = SC.GameSums(battle)
    sums.add("gain", np.asarray(gain, dtype=np.float64), np.asarray(keep, dtype=bool))
    drawn = sums.draws(resamples, seed).samples
    if not drawn.size:
        return None
    return float((drawn[:, 0] <= 0.0).mean())


def _other_label(scored: Scored) -> np.ndarray:
    """``[N, 2]``: the slot's label is the OTHER bucket of this featurizer (a
    visible move outside its candidates)."""
    y_action = np.asarray(scored.batch["y_action"]).astype(np.int64)
    n_cand = int(np.asarray(scored.batch["cand_flag"]).shape[-1])
    return scored.nll["fine_scored"] & (y_action == n_cand)


def _candidate_names(scored: Scored, rows: np.ndarray) -> list[list[frozenset[str]]]:
    """The named candidate moves of both slots of the given examples."""
    names = scored.tables.move_ids
    moves = np.asarray(scored.batch["cand_move"])[rows]
    valid = (np.asarray(scored.batch["cand_flag"])[rows] & F.CAND_VALID) > 0
    out: list[list[frozenset[str]]] = []
    for row in range(moves.shape[0]):
        out.append(
            [
                frozenset(
                    str(names[int(move)])
                    for move, ok in zip(moves[row, slot], valid[row, slot])
                    if ok and 0 < int(move) < len(names)
                )
                for slot in range(F.N_SLOT)
            ]
        )
    return out


def _candidate_sets_differ(
    scored: Scored, reference: Scored, left: np.ndarray, right: np.ndarray
) -> tuple[int, int]:
    """(acting slots both have, those whose candidate moves differ)."""
    active = (np.asarray(scored.batch["act_mon"])[left] >= 0) & (
        np.asarray(reference.batch["act_mon"])[right] >= 0
    )
    here = _candidate_names(scored, left)
    there = _candidate_names(reference, right)
    differ = sum(
        1
        for row in range(len(here))
        for slot in range(F.N_SLOT)
        if active[row, slot] and here[row][slot] != there[row][slot]
    )
    return int(active.sum()), int(differ)


def sign_test(first: int, second: int) -> float | None:
    """Exact two-sided sign test of ``first`` against ``second`` discordant
    cases (each taken as independent); None without any."""
    total = int(first) + int(second)
    if total <= 0:
        return None
    tail = sum(comb(total, i) for i in range(max(first, second), total + 1))
    return float(min(1.0, 2.0 * tail / 2**total))


# --- against the forecast the bot logged ------------------------------------------


def _listed_probability(
    listed: Mapping[str, Any],
    action: np.ndarray,
    target: np.ndarray,
    column_of: Mapping[str, int],
    unnamed: float,
) -> float | None:
    """This artifact's probability of one entry of a logged slot's ranked
    list (``runtime.ActionForecast.to_dict``), or None when the entry names
    something the slot's candidates here do not hold.

    ``action`` ``[A]`` and ``target`` ``[n_cand + 1, 5]`` are the slot's
    normalised prediction; ``unnamed`` is the OTHER bucket as the runtime
    logs it (with the mass of any candidate column it could not name).
    """
    kind = listed.get("kind")
    n_cand = target.shape[0] - 1
    if kind == "switch":
        pointer = listed.get("roster_index")
        if isinstance(pointer, int) and 0 <= pointer < F.N_ROSTER:
            return float(action[n_cand + 1 + pointer])
        return None
    if kind == "other":
        return float(unnamed)
    if kind == "move":
        column = column_of.get(str(listed.get("move")))
        if column is None:
            return None
        aimed = listed.get("target")
        if aimed is None:  # a candidate without a legal target class
            return float(action[column])
        if aimed in TARGET_CLASSES:
            place = TARGET_CLASSES.index(aimed)
            return float(action[column] * target[column, place])
    return None


def live_check(
    scored: Scored,
    games: Sequence[Game],
    logged: Mapping[tuple[str, str, int], Mapping[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    """This artifact's forecast against the one the bot logged at the turn:
    P(switch), P(Protect) and every fine action the log lists.

    ``logged`` is keyed by (directory, battle id, turn): a forecast is only
    ever compared with a game of its own directory.
    """
    batch = scored.batch
    events = F.event_probs(scored.pred, batch)
    if not events:
        return {"status": "skipped", "why": "event_probs could not read the prediction"}
    here = {name: np.asarray(events[name], dtype=np.float64) for name in EVENTS}
    norm = F.normalize_prediction(scored.pred, batch)
    action = np.asarray(norm["action"], dtype=np.float64)
    target = np.asarray(norm["target"], dtype=np.float64)
    n_cand = target.shape[2] - 1
    move_names = scored.tables.move_ids
    cand_move = np.asarray(batch["cand_move"])
    cand_valid = (np.asarray(batch["cand_flag"]) & F.CAND_VALID) > 0
    active = np.asarray(batch["act_mon"]) >= 0
    turns = np.asarray(batch["m_turn"]).tolist()
    index = np.asarray(batch["m_battle"]).tolist()
    names: Counter[str] = Counter()
    cells: dict[str, dict[str, Any]] = {}
    for row in range(scored.n):
        game = games[int(index[row])]
        cell = cells.setdefault(
            game.sheet,
            {
                "turns": 0,
                "with_logged_forecast": 0,
                "slot_presence_differs": 0,
                "sheet_state_differs": 0,
                "slots_compared": 0,
                "slots_differing": 0,
                "max_abs_diff": 0.0,
                "actions_listed": 0,
                "actions_compared": 0,
                "actions_differing": 0,
                "actions_not_among_the_candidates": 0,
                "actions_max_abs_diff": 0.0,
            },
        )
        cell["turns"] += 1
        record = logged.get((game.folder, game.battle_id, int(turns[row])))
        if record is None:
            continue
        cell["with_logged_forecast"] += 1
        names[str(record.get("model"))] += 1
        told = list(record.get("sheets") or [])
        if told and told[0] != (OPEN if game.sheet == OPEN else CLOSED):
            cell["sheet_state_differs"] += 1
        slots = list(record.get("slots") or []) + [None] * F.N_SLOT
        for slot in range(F.N_SLOT):
            entry = slots[slot]
            if (entry is not None) != bool(active[row, slot]):
                cell["slot_presence_differs"] += 1
                continue
            if entry is None:
                continue
            gap = 0.0
            for name, key in (
                (SC.EVENT_SWITCH, "p_switch"),
                (SC.EVENT_PROTECT, "p_protect"),
            ):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    gap = max(gap, abs(float(value) - float(here[name][row, slot])))
            cell["slots_compared"] += 1
            cell["slots_differing"] += int(gap > LIVE_TOLERANCE)
            cell["max_abs_diff"] = max(cell["max_abs_diff"], gap)
            # The ranked list: every fine action the log names, by its move
            # and target (or switch destination), not by its position.
            column_of: dict[str, int] = {}
            unnamed = float(action[row, slot, n_cand])
            for column in range(n_cand):
                move = int(cand_move[row, slot, column])
                if bool(cand_valid[row, slot, column]) and 0 < move < len(move_names):
                    column_of.setdefault(str(move_names[move]), column)
                else:
                    unnamed += float(action[row, slot, column])
            for listed in entry.get("actions") or []:
                value = listed.get("p")
                if not isinstance(value, (int, float)):
                    continue
                cell["actions_listed"] += 1
                mine = _listed_probability(
                    listed, action[row, slot], target[row, slot], column_of, unnamed
                )
                if mine is None:
                    cell["actions_not_among_the_candidates"] += 1
                    continue
                away = abs(float(value) - mine)
                cell["actions_compared"] += 1
                cell["actions_differing"] += int(away > LIVE_TOLERANCE)
                cell["actions_max_abs_diff"] = max(cell["actions_max_abs_diff"], away)
    return {
        "status": "ran",
        "artifact_name": model_name,
        "logged_model_names": dict(names),
        "same_model_name": bool(names) and set(names) == {model_name},
        "tolerance": LIVE_TOLERANCE,
        "by_sheet": cells,
    }


# --- the run --------------------------------------------------------------------


def parse_artifacts(specs: Sequence[str], root: Path) -> list[tuple[str, Path]]:
    """``label=path`` arguments as (label, path); labels must be distinct."""
    out: list[tuple[str, Path]] = []
    for spec in specs:
        label, sep, tail = spec.partition("=")
        if not sep or not label or not tail:
            raise LadderReadError(f"--artifact takes label=path, got {spec!r}")
        if label in [name for name, _ in out]:
            raise LadderReadError(f"artifact label {label!r} is given twice")
        path = Path(tail)
        out.append((label, path if path.is_absolute() else root / path))
    if not out:
        raise LadderReadError("no artifact: give --artifact label=path")
    return out


def fresh_directories(
    root: Path, since: str = FRESH_SINCE, formats: Sequence[str] = DEFAULT_FORMATS
) -> list[Path]:
    """The ladder replay directories under ``root`` that hold a saved page of
    these formats dated (file modification time) on or after ``since``."""
    first = _day_start(since, "the sealed set's first day") or 0.0
    out: list[Path] = []
    for folder in sorted(root.glob(FRESH_GLOB)):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.html"):
            match = _OWN_NAME.match(path.name)
            if match is None or match["id"].rsplit("-", 1)[0] not in formats:
                continue
            try:
                if path.stat().st_mtime >= first:
                    out.append(folder)
                    break
            except OSError:
                continue
    return out


def set_table_check(
    loaded: Any, games: Sequence[Game]
) -> tuple[dict[str, Any] | None, list[bool | None]]:
    """Whether the opponents of the scored games built the artifact's set
    table: (the summary, the answer per game); (None, []) for an artifact
    whose featurizer holds no set table.

    Per game ``Featurizer.set_table_holds(account id of the opponent)``: True
    = that account's team sheets are among those the table was built from (a
    LEAK: the model is given the player's own sets), False = not, None = not
    known (the table does not record its accounts, or the page names no
    opponent account). Counted over games and over distinct opponent
    accounts. Never raises: a featurizer that cannot be asked answers None
    for every game and says why.
    """
    featurizer = getattr(loaded, "featurizer", None)
    if featurizer is None or getattr(featurizer, "set_table", None) is None:
        return None, []
    why = ""
    answers: list[bool | None] = []
    by_account: dict[str, bool | None] = {}
    without_account = 0
    for game in games:
        account = str((game.result.player_ids or {}).get(game.opponent) or "")
        answer: bool | None = None
        if not account:
            without_account += 1
        else:
            try:
                held = featurizer.set_table_holds(account)
                answer = None if held is None else bool(held)
            except Exception as exc:
                why = f"set_table_holds failed: {type(exc).__name__}"
            by_account[account] = answer
        answers.append(answer)
    accounts = getattr(featurizer, "set_accounts", None)
    if accounts is None and not why:
        why = "the set table does not record whose sheets built it"

    def tally(values: Sequence[bool | None]) -> dict[str, int]:
        return {
            "contributors": sum(1 for value in values if value is True),
            "not_contributors": sum(1 for value in values if value is False),
            "unknown": sum(1 for value in values if value is None),
        }

    counted = tally(answers)
    summary = {
        "signature": getattr(featurizer, "set_prior_signature", None),
        "accounts_recorded": None if accounts is None else len(accounts),
        "games": counted,
        "opponent_accounts": tally(list(by_account.values())),
        "games_without_an_opponent_account": without_account,
        "why_unknown": why,
        "leak": bool(counted["contributors"]),
        "contributor_battles": [
            f"battle-{game.battle_id}"
            for game, answer in zip(games, answers)
            if answer is True
        ],
    }
    return summary, answers


def set_table_warning(label: str, found: Mapping[str, Any] | None) -> str:
    """The warning line of an artifact whose set table holds a scored
    opponent's own sheets; '' when there is nothing to warn about."""
    if not found or not (found.get("games") or {}).get("contributors"):
        return ""
    games = found["games"]
    total = sum(int(games.get(name) or 0) for name in games)
    accounts = (found.get("opponent_accounts") or {}).get("contributors")
    return (
        f"WARNING: SET-TABLE LEAK in {label}: the opponents of "
        f"{games['contributors']} of the {total} scored games ({accounts} "
        "accounts) are among the players whose team sheets built this "
        "artifact's set table, so the model is given their own sets. Its "
        "numbers on those games are not a held-out reading"
    )


def code_state(root: Path = ROOT) -> dict[str, Any]:
    """What produced a reading besides the artifacts: the commit, and the
    sha256 of this script, the scorecard, the dataset builder and every
    module of the predictor (``sha256`` is one hash over all of them).

    Never raises: a file that cannot be read hashes to None, a tree that is
    no git checkout has no commit.
    """
    paths = [root / name for name in INSTRUMENT_FILES]
    paths += sorted(root.glob(INSTRUMENT_GLOB))
    files: dict[str, str | None] = {}
    for path in paths:
        try:
            files[str(path.relative_to(root))] = SC.sha256_file(path)
        except OSError:
            files[str(path.relative_to(root))] = None
    commit: str | None = None
    changed: list[str] | None = None
    try:
        git = ["git", "--no-optional-locks", "-C", str(root)]
        commit = (
            subprocess.run(
                [*git, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            ).stdout.strip()
            or None
        )
        status = subprocess.run(
            [*git, "status", "--porcelain", "--", *files],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        changed = sorted(line[3:] for line in status.splitlines() if line.strip())
    except Exception:
        pass
    whole = hashlib.sha256(json.dumps(sorted(files.items())).encode()).hexdigest()
    return {
        "commit": commit,
        "changed_since_commit": changed,
        "files": files,
        "sha256": whole,
    }


def run(
    artifacts: Sequence[str],
    out: Path | str,
    *,
    directories: Sequence[str | Path] | None = None,
    fresh: bool = False,
    fresh_root: Path | str | None = None,
    reference: str | None = None,
    since: str | None = None,
    until: str | None = None,
    games: str | None = None,
    include_unrated: bool = False,
    own_sheet: str = OWN_SHEET_FILE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    note: str = "",
    root: Path = ROOT,
    log: Log = say,
) -> dict[str, Any]:
    """Read, score, write ``ladder_read.json`` / ``.md``; returns the reading.

    ``directories`` names the replay directories; there is no default.
    ``fresh`` opens the sealed confirmation set (games dated on or after
    ``FRESH_SINCE``): without ``directories`` they are found under
    ``fresh_root`` (default ``root``) and older pages are left out; with
    ``directories`` it only allows those to hold sealed games. A reading
    that would hold a sealed game without ``fresh`` is refused before
    anything is scored.

    Raises ``FileExistsError`` before anything is read when ``out`` already
    holds a reading and ``overwrite`` is not set, ``LadderReadError`` for an
    input that cannot be read or scored.
    """
    target = Path(out)
    target = target if target.is_absolute() else root / target
    if (target / READ_JSON).exists() and not overwrite:
        raise FileExistsError(
            f"{target / READ_JSON} exists; give --overwrite to replace that reading"
        )
    started = time.time()
    named = parse_artifacts(artifacts, root)
    labels = [label for label, _ in named]
    base = reference if reference is not None else labels[0]
    if base not in labels:
        raise LadderReadError(
            f"reference {base!r} is not among the artifacts ({', '.join(labels)})"
        )
    found_fresh: list[str] | None = None
    if directories:
        folders = [Path(d) if Path(d).is_absolute() else root / d for d in directories]
    elif fresh:
        where = Path(fresh_root) if fresh_root is not None else root
        where = (where if where.is_absolute() else root / where).resolve()
        folders = fresh_directories(where)
        if not folders:
            raise LadderReadError(
                f"--fresh: no {FRESH_GLOB} directory under {where} holds a page "
                f"dated on or after {FRESH_SINCE}"
            )
        found_fresh = [str(folder) for folder in folders]
        since = since or FRESH_SINCE
        log(
            f"SEALED confirmation set (--fresh): {len(folders)} directories "
            f"under {where}, pages dated on or after {since}"
        )
    else:
        raise LadderReadError(
            "which games? name the replay directories (--dir, repeated), or "
            f"give --fresh for the sealed confirmation set (the ladder games "
            f"dated on or after {FRESH_SINCE})"
        )
    kept, info = read_games(
        root,
        folders,
        since=since,
        until=until,
        games=games,
        include_unrated=include_unrated,
        own_sheet=own_sheet,
        fresh=fresh,
    )
    logged = info.pop("_forecasts")
    search = info.pop("_search")
    log(f"pages {info['pages']}, kept {len(kept)}, skipped {info['skipped'] or 'none'}")
    if info["no_page"]["games"]:
        log(
            f"{info['no_page']['games']} games with decision rows have no saved "
            f"page ({info['no_page']['logged_forecasts']} logged forecasts): not "
            "scored"
        )
    if not kept:
        raise LadderReadError(f"no game to score (skipped: {info['skipped']})")
    seal = _day_start(FRESH_SINCE, "the sealed set's first day") or 0.0
    sealed = sum(1 for game in kept if game.time is not None and game.time >= seal)
    if sealed and not fresh:
        raise LadderReadError(
            f"{sealed} of the {len(kept)} games are dated on or after "
            f"{FRESH_SINCE}: the sealed confirmation set. Give --fresh to read "
            f"it, or --until {FRESH_SINCE} to leave it out"
        )
    info["sealed"] = {
        "since": FRESH_SINCE,
        "games": int(sealed),
        "opened_with_fresh": bool(fresh),
        "directories_found": found_fresh,
    }
    reading: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": note,
        "code": code_state(),
        "settings": {
            "reference": base,
            "resamples": int(resamples),
            "seed": int(seed),
            "level": SC.LEVEL,
            "probability_floor": F.PROBABILITY_FLOOR,
            "top_ks": list(TOP_KS),
            "top_k": TOP_K,
            "thresholds": {event: SC.PRE_REGISTERED[event] for event in EVENTS},
        },
        "definitions": dict(DEFINITIONS),
        "input": info,
        "order": labels,
        # Lines for the top of the report (a set table that holds a scored
        # opponent's own sheets). Empty when there is nothing to warn about.
        "warnings": [],
        "artifacts": {},
        "paired": {},
    }
    rows = [game.row() for game in kept]
    scored: dict[str, Scored] = {}
    for label, path in named:
        try:
            loaded = load_predictor(path)
        except (OSError, ValueError) as exc:
            raise LadderReadError(f"artifact {path}: {exc}") from exc
        made = score_artifact(label, loaded, kept, search)
        scored[label] = made
        found, per_game = report_artifact(made, resamples, seed)
        extra = loaded.meta.get("extra") or {}
        pair = getattr(loaded, "coupling", None)
        found["artifact"] = {
            "path": str(path),
            "sha256": SC.sha256_file(path),
            "name": loaded.name,
            "kind": loaded.kind,
            "elo_mode": getattr(loaded.predictor, "elo_mode", None),
            "n_cand": int(loaded.featurizer.n_cand),
            "dataset_tag": extra.get("dataset_tag") or extra.get("tag"),
            "dex_signature_diff": list(loaded.meta.get("dex_signature_diff") or []),
            "featurizer_counters": dict(loaded.featurizer.counters),
            "predictor_counters": dict(getattr(loaded.predictor, "counters", {}) or {}),
            # Which joint the lists of this reading are: the artifact's own.
            "joint": f"coupled: {made.coupling}" if made.coupling else JOINT_PRODUCT,
            "coupling": None if pair is None else pair.describe(),
        }
        # Are the scored opponents among the players whose sheets built the
        # artifact's set table? None for an artifact without one.
        table_check, held = set_table_check(loaded, kept)
        found["artifact"]["set_table"] = table_check
        warning = set_table_warning(label, table_check)
        if warning:
            reading["warnings"].append(warning)
            log(warning)
        found["search_decisions_joined"] = int(np.asarray(made.search).sum())
        found["live_log"] = live_check(made, kept, logged, loaded.name)
        reading["artifacts"][label] = found
        for row in rows:
            position = int(row["index"])
            cells: dict[str, Any] = {}
            for name, values in per_game.items():
                value = float(values[position]) if position < len(values) else 0.0
                cells[name] = value if name.endswith("_sum") else int(round(value))
            row.setdefault("artifacts", {})[label] = cells
            if table_check is not None and position < len(held):
                row.setdefault("set_table_holds", {})[label] = held[position]
        block = found["splits"][ALL]
        log(
            f"{label} ({loaded.kind} {loaded.name}): {block['slot_turns']} labelled "
            f"slot-turns, fine NLL {SC._number(block['fine_nll']['value'])}, joint "
            f"top-{TOP_K} "
            f"{SC._percent(block['joint']['replies'][ALL]['top'][str(TOP_K)])} of "
            f"{block['joint']['counted']} turns ({found['artifact']['joint']})"
        )
    for label in labels:
        if label != base:
            reading["paired"][label] = paired(
                scored[label], scored[base], resamples, seed
            )
    reading["games"] = rows
    reading["seconds"] = round(time.time() - started, 1)
    plain = SC._plain(reading)
    target.mkdir(parents=True, exist_ok=True)
    (target / READ_JSON).write_text(
        json.dumps(plain, indent=1, allow_nan=False), encoding="utf-8"
    )
    (target / READ_MD).write_text(render_markdown(plain), encoding="utf-8")
    log(f"wrote {target / READ_JSON} and {target / READ_MD} in {reading['seconds']} s")
    return plain


# --- markdown -------------------------------------------------------------------


def _pct(value: Any, digits: int = 1) -> str:
    return SC._percent(value, digits)


def _of(hits: Any, total: Any) -> str:
    if not total:
        return "-"
    return f"{_pct(hits / total)} ({hits} of {total})"


def _interval(found: Mapping[str, Any] | None, percent: bool = False) -> str:
    if not found or found.get("value") is None:
        return "-"
    if percent:
        if found.get("low") is None:
            return _pct(found["value"])
        return f"{_pct(found['value'])} [{_pct(found['low'])}, {_pct(found['high'])}]"
    return SC._bracket(found)


def _set_table_text(found: Mapping[str, Any] | None) -> str:
    """What the set-table check found, for an artifact's line of the report;
    '' for an artifact without a set table (and for an older reading)."""
    if not found:
        return ""
    games = found.get("games") or {}
    text = (
        "; set table: the opponents of "
        f"{games.get('contributors', 0)} scored games are contributors "
        f"(a leak), of {games.get('not_contributors', 0)} are not, "
        f"{games.get('unknown', 0)} unknown"
    )
    if found.get("why_unknown") and games.get("unknown"):
        text += f" ({found['why_unknown']})"
    if found.get("contributor_battles"):
        text += "; contributors' games: " + ", ".join(found["contributor_battles"])
    return text


def _shown_contexts(reading: Mapping[str, Any]) -> list[str]:
    """Contexts with a turn in them; "not open" only when it is not "closed"."""
    first = reading["artifacts"][reading["order"][0]]["splits"]
    shown = [name for name in CONTEXTS if first[name]["examples"]]
    if SHEET_UNKNOWN not in shown and SHEET_NOT_OPEN in shown:
        shown.remove(SHEET_NOT_OPEN)
    return shown


def render_markdown(reading: Mapping[str, Any]) -> str:
    """The reading as markdown; every number comes from ``reading``."""
    settings, info = reading["settings"], reading["input"]
    order = list(reading["order"])
    base = settings["reference"]
    games = list(reading.get("games") or [])
    contexts = _shown_contexts(reading)
    lines = ["# Opponent predictor: read on the bot's own ladder games", ""]
    # A reading of an earlier version has no such entry: nothing was checked.
    for warning in reading.get("warnings") or []:
        lines += [f"**{warning}.**", ""]
    if reading.get("note"):
        lines += [f"**{reading['note']}**", ""]
    lines += [
        f"Rendered from `{READ_JSON}` by `evaluation/oppmodel_ladder_read.py`. "
        f"Created {reading['created']}; {settings['resamples']:,} resamples of "
        f"whole games, seed {settings['seed']}; {reading.get('seconds', '-')} s. "
        f"Reference: **{base}**. Every number is measured on the games below; "
        "each artifact is scored through its own featurizer.",
        "",
        "## Games",
        "",
    ]
    sealed = info.get("sealed") or {}
    if sealed.get("games"):
        lines.append(
            f"- **SEALED confirmation set**: {sealed['games']} of the "
            f"{info['kept']} kept games are dated on or after {sealed['since']} "
            "(read with `--fresh`"
            + (
                f"; directories found: {len(sealed['directories_found'])}"
                if sealed.get("directories_found")
                else ""
            )
            + ")"
        )
    files = info.get("files") or {}
    stray = {
        name: sum(count.get(name, 0) for count in files.values())
        for name in ("html_files", "not_a_saved_game_page", "another_format")
    }
    lines += [
        f"- pages found {info['pages']} ({info['pages_found']}), kept "
        f"{info['kept']}, skipped {info['skipped'] or 'none'}"
        + (
            f"; **{info['pages_unaccounted']} pages neither kept nor counted**"
            if info.get("pages_unaccounted")
            else ""
        ),
        f"- `*.html` files in the directories: {stray['html_files']}; "
        f"{stray['not_a_saved_game_page']} not named as a saved game page, "
        f"{stray['another_format']} of another format",
    ]
    absent = info.get("no_page") or {}
    if absent.get("games"):
        lines.append(
            f"- **games with decision rows and no saved page: {absent['games']}** "
            f"({absent['decision_rows']} rows, {absent['logged_forecasts']} logged "
            "forecasts; cut off or unfinished, cannot be scored): "
            + ", ".join(
                f"{name} {row['games']}"
                for name, row in (absent.get("by_directory") or {}).items()
            )
        )
    else:
        lines.append("- games with decision rows and no saved page: none")
    told = info.get("logged_forecasts") or {}
    if told.get("total"):
        lines.append(
            f"- forecasts the bot logged in these directories: {told['total']}; "
            f"{told['in_kept_games']} in kept games, "
            f"{told['in_games_without_a_page']} in games without a page, "
            f"{told['in_other_games']} in skipped games"
        )
    if info.get("notes"):
        lines.append(f"- reader's notes: {info['notes']}")
    if info.get("sheet_rows_disagree"):
        lines.append(
            f"- games whose rows say both open and closed (read as open): "
            f"{info['sheet_rows_disagree']}"
        )
    lines.append(
        f"- filters: {info['filters']}; unrated games "
        f"{'kept' if info['include_unrated'] else 'left out'}"
    )
    tally: Counter[tuple[str, str]] = Counter()
    sheets: Counter[str] = Counter()
    evidence: Counter[str] = Counter()
    for row in games:
        tally[(row["directory"], row["date"])] += 1
        sheets[row["sheet"]] += 1
        evidence[f"{row['sheet']}: {row['sheet_evidence'] or 'nothing'}"] += 1
    ratings = [row["opponent_rating"] for row in games if row["opponent_rating"]]
    lines.append(
        f"- sheet state of the kept games: {dict(sheets)}; by evidence {dict(evidence)}"
    )
    own = info.get("own_sheet") or {}
    if own.get("open_games_kept"):
        teams = [
            f"{name}: `{team['path']}` sha256 `{str(team['sha256'])[:12]}`"
            for name, team in (own.get("team_files") or {}).items()
            if team.get("path")
        ]
        lines.append(
            f"- open games kept {own.get('open_games_kept')}: the predictor is "
            f"given both sheets in {own.get('both_sheets')} (the bot's own from "
            f"the team file in {own.get('given')}, both from a public copy in "
            f"{own.get('from_a_public_copy')}; mode {own.get('mode')})"
            + (
                f"; not given, over all pages: {own['not_given_all_pages']}"
                if own.get("not_given_all_pages")
                else ""
            )
            + ("; team file as it is on disk now: " + "; ".join(teams) if teams else "")
        )
    code = reading.get("code") or {}
    if code:
        changed = code.get("changed_since_commit")
        lines.append(
            f"- instrument: commit `{str(code.get('commit') or 'unknown')[:10]}`, "
            f"{len(code.get('files') or {})} files hashed, together sha256 "
            f"`{str(code.get('sha256'))[:12]}`"
            + (
                "; git state not read"
                if changed is None
                else f"; {len(changed)} of them differ from the commit"
                + (f" ({', '.join(changed)})" if changed else "")
            )
        )
    logs = [
        f"{name}/{file} `{str(entry['sha256'])[:12]}`"
        for name, found in (info.get("decision_logs") or {}).items()
        for file, entry in (found.get("read") or {}).items()
    ]
    if logs:
        lines.append("- decision logs read (sha256): " + ", ".join(logs))
    if ratings:
        lines.append(
            f"- opponent rating on the page: {len(ratings)} of {len(games)} games, "
            f"lowest {min(ratings)}, median {int(np.median(ratings))}, highest "
            f"{max(ratings)}"
        )
    lines += [
        "",
        *SC._table(
            ["directory", "date of the page", "games kept"],
            [[folder, day, count] for (folder, day), count in sorted(tally.items())],
        ),
    ]

    lines += ["## Headline (all kept games)", ""]
    header = [
        "artifact",
        "labelled slot-turns",
        "fine NLL [95%]",
        "visible / hidden",
        "top-1",
        "top-3",
        "intent",
        f"joint top-{TOP_K} [95%]",
        "counted turns",
    ]
    table = []
    for label in order:
        block = reading["artifacts"][label]["splits"][ALL]
        joint = block["joint"]["replies"][ALL]
        table.append(
            [
                label,
                f"{block['slot_turns']:,}",
                _interval(block["fine_nll"]),
                f"{SC._number(block['fine_nll_visible'], 3)} / "
                f"{SC._number(block['fine_nll_hidden'], 3)}",
                _pct(block["fine_top1"]),
                _pct(block["fine_top3"]),
                _pct(block["intent_accuracy"]),
                _interval(joint["top_interval"], percent=True),
                f"{block['joint']['counted']:,} of {block['joint']['turns']:,}",
            ]
        )
    lines += SC._table(header, table)
    for label in order:
        entry = reading["artifacts"][label]["artifact"]
        lines.append(
            f"- {label}: `{entry['path']}` ({entry['kind']} {entry['name']}, "
            f"sha256 `{str(entry['sha256'])[:12]}`, {entry['n_cand']} candidates"
            + (", dex differs" if entry["dex_signature_diff"] else "")
            + f"); joint lists: {entry.get('joint', JOINT_PRODUCT)}"
            + _set_table_text(entry.get("set_table"))
        )
    lines.append("")
    coupled = [
        label
        for label in order
        if reading["artifacts"][label]["artifact"].get("coupling")
    ]
    if coupled:
        lines += [
            "## What the pair coupling changes",
            "",
            "The joint numbers of a coupled artifact are built with its coupling "
            "(as the bot builds them). Beside them: the plain product of the same "
            "two marginals, on the same counted turns. An artifact whose coupling "
            "is all ones is the plain product and has no row of its own.",
            "",
        ]
        table = []
        for label in coupled:
            splits = reading["artifacts"][label]["splits"]
            for name in contexts:
                for reply in (ALL, REPLY_SWITCH, REPLY_PROTECT, REPLY_MOVES):
                    row = splits[name]["joint"]["replies"][reply]
                    if "top_plain_product" not in row or not row["examples"]:
                        continue
                    table.append(
                        [
                            label,
                            name,
                            reply,
                            row["examples"],
                            " / ".join(_pct(row["top"][str(k)]) for k in TOP_KS),
                            " / ".join(
                                _pct(row["top_plain_product"][str(k)]) for k in TOP_KS
                            ),
                            f"{row['only_coupled']} / {row['only_plain_product']}",
                        ]
                    )
        ks = "/".join(str(k) for k in TOP_KS)
        lines += SC._table(
            [
                "artifact",
                "split",
                "reply class",
                "turns",
                f"top-{ks} with the coupling",
                f"top-{ks} plain product",
                f"top-{TOP_K}: only coupled / only plain",
            ],
            table,
        )

    lines += ["## By sheet state and turn", ""]
    for label in order:
        splits = reading["artifacts"][label]["splits"]
        lines += [f"### {label}", ""]
        table = []
        for name in contexts:
            block = splits[name]
            joint = block["joint"]
            table.append(
                [
                    name,
                    block["games"],
                    f"{block['slot_turns']:,}",
                    _interval(block["fine_nll"]),
                    _pct(block["fine_top1"]),
                    _pct(block["fine_top3"]),
                    _pct(block["intent_accuracy"]),
                    f"{joint['counted']:,}",
                    " / ".join(
                        _pct(joint["replies"][ALL]["top"][str(k)]) for k in TOP_KS
                    ),
                    _interval(joint["replies"][ALL]["top_interval"], percent=True),
                    _pct(joint["replies"][ALL]["top_with_mega"]),
                ]
            )
        lines += SC._table(
            [
                "split",
                "games",
                "labelled slot-turns",
                "fine NLL [95%]",
                "top-1",
                "top-3",
                "intent",
                "counted turns",
                "joint top-" + "/".join(str(k) for k in TOP_KS),
                f"joint top-{TOP_K} [95%]",
                f"top-{TOP_K} with the Mega bit",
            ],
            table,
        )

    lines += [
        f"## Joint reply coverage by reply class (top-{TOP_K}, without the Mega bit)",
        "",
        "Counted turns only (every acting slot shows its whole action). Each "
        "cell: top-8 (counted turns). Thin cells are noise.",
        "",
    ]
    for label in order:
        splits = reading["artifacts"][label]["splits"]
        lines += [f"### {label}", ""]
        table = []
        for reply in REPLIES:
            cells = [reply]
            for name in contexts:
                row = splits[name]["joint"]["replies"][reply]
                cells.append(
                    f"{_pct(row['top'][str(TOP_K)])} ({row['examples']})"
                    if row["examples"]
                    else "-"
                )
            table.append(cells)
        lines += SC._table(["reply class", *contexts], table)
        row = splits[ALL]["joint"]["replies"]
        lines += SC._table(
            [
                "reply class (all games)",
                "turns",
                *[f"top-{k}" for k in TOP_KS],
                f"top-{TOP_K} [95%]",
            ],
            [
                [
                    reply,
                    row[reply]["examples"],
                    *[_pct(row[reply]["top"][str(k)]) for k in TOP_KS],
                    _interval(row[reply]["top_interval"], percent=True),
                ]
                for reply in REPLIES
            ],
        )

    lines += [
        "## Switch and Protect calibration",
        "",
        "Per slot-turn where the log tells whether the event happened. "
        "Threshold columns: slot-turns at or above it, precision [95%], mean "
        "prediction there, share of occurrences caught.",
        "",
    ]
    for label in order:
        splits = reading["artifacts"][label]["splits"]
        lines += [f"### {label}", ""]
        table = []
        for event in EVENTS:
            for name in contexts:
                found = splits[name]["events"][event]
                decision = found["at_threshold"]
                table.append(
                    [
                        event,
                        name,
                        f"{found['known']:,}",
                        _pct(found["observed"]),
                        _pct(found["predicted"]),
                        f"P >= {decision['threshold']:g}",
                        decision["slots"],
                        _interval(decision["precision"], percent=True),
                        _pct(decision["predicted"]),
                        _pct(decision["recall"]),
                    ]
                )
        lines += SC._table(
            [
                "event",
                "split",
                "known slot-turns",
                "observed",
                "mean predicted",
                "threshold",
                "at or above",
                "precision [95%]",
                "mean predicted there",
                "caught",
            ],
            table,
        )

    lines += [f"## Paired against {base}", ""]
    if not reading["paired"]:
        lines += ["Only one artifact: nothing to pair.", ""]
    else:
        lines += [
            "Artifact minus reference on the slot-turns / fully visible turns "
            "both have. Fine NLL: negative is better; 'neither labels OTHER' "
            "leaves out the slot-turns where the real move is outside one "
            "featurizer's candidates (the labels differ there). Joint top-8: "
            "positive is better; 'only one' counts the turns where exactly one "
            "of the two holds the real reply, and the sign test is on those "
            "(exact, two-sided, turns as independent). An interval whose end "
            "sits on zero is not a finding: the end moves with the seed.",
            "",
        ]
        table = []
        for label, found in reading["paired"].items():
            for name in contexts:
                block = found["splits"][name]
                nll, top = block["fine_nll"], block["joint_top8"]
                named = block.get("fine_nll_neither_other") or {}
                p = top.get("sign_test_p")
                table.append(
                    [
                        label,
                        name,
                        f"{nll['slots']:,}",
                        SC._signed(nll),
                        f"{SC._signed(named)} ({named.get('slots', 0):,})"
                        if named
                        else "-",
                        f"{top['turns']:,}",
                        "-"
                        if top["diff"] is None
                        else f"{100 * top['diff']:+.1f} pt"
                        + (
                            ""
                            if top["low"] is None
                            else f" [{100 * top['low']:+.1f}, {100 * top['high']:+.1f}]"
                        ),
                        f"{top['only_here']} / {top['only_reference']}",
                        "-" if p is None else "<0.001" if p < 0.001 else f"{p:.3f}",
                        f"{_pct(nll.get('resamples_not_better'))} / "
                        f"{_pct(top.get('resamples_not_better'))}",
                    ]
                )
        lines += SC._table(
            [
                "artifact",
                "split",
                "slot-turns",
                "fine NLL difference [95%]",
                "neither labels OTHER [95%] (slot-turns)",
                "turns",
                f"joint top-{TOP_K} difference [95%]",
                "only this / only reference",
                "sign test p",
                "resamples not better: NLL / top-8",
            ],
            table,
        )
        for label, found in reading["paired"].items():
            marks = found.get("labels") or {}
            lines.append(
                f"- {label}: {found['examples_both']} turns in both, "
                f"{found['examples_only_here']} only here, "
                f"{found['examples_only_reference']} only in the reference; "
                f"{found['slot_turns_labelled_by_one_only']} slot-turns labelled "
                "by one of the two only (left out); the label is the OTHER bucket "
                f"on {marks.get('other_here')} slot-turns here and "
                f"{marks.get('other_reference')} in the reference (either: "
                f"{marks.get('other_either')} of {marks.get('slot_turns')}); the "
                "candidate moves differ on "
                f"{marks.get('slots_with_different_candidates')} of "
                f"{marks.get('slots_acting')} acting slots"
            )
        lines.append("")

    found_search = info.get("search_session") or {}
    lines += [
        f"## The same top {TOP_K} under a search session's matching rule",
        "",
        DEFINITIONS["search_rule"] + ".",
        "",
        f"The search session's reading counts {found_search.get('decisions', 0)} "
        f"decisions in these directories (of {found_search.get('replies', 0)} "
        f"observed replies; the others have no logged forecast), "
        f"{found_search.get('in_kept_games', 0)} of them in kept games; joined to "
        "a turn here: "
        + ", ".join(
            f"{label} {reading['artifacts'][label].get('search_decisions_joined', 0)}"
            for label in order
        )
        + ".",
        "",
    ]
    table = []
    for label in order:
        splits = reading["artifacts"][label]["splits"]
        for name in contexts:
            rule = splits[name]["joint"]["search_rule"]
            table.append(
                [
                    label,
                    name,
                    *[
                        # A reading of an earlier version lacks the last way.
                        _of(
                            (rule[variant].get(how) or {}).get("hits"),
                            (rule[variant].get(how) or {}).get("turns"),
                        )
                        for how in RULES
                        for variant in (PLAIN, MEGA)
                    ],
                ]
            )
    lines += SC._table(
        [
            "artifact",
            "split",
            *[
                f"{how}, {'with' if variant == MEGA else 'without'} the Mega bit"
                for how in RULES
                for variant in (PLAIN, MEGA)
            ],
        ],
        table,
    )

    lines += ["## Against the forecast the bot logged", ""]
    for label in order:
        found = reading["artifacts"][label]["live_log"]
        if found.get("status") != "ran" or not found.get("logged_model_names"):
            lines.append(f"- {label}: no logged forecast to compare with")
            continue
        if not found["same_model_name"]:
            lines.append(
                f"- {label}: the bot logged {found['logged_model_names']}, this "
                f"artifact is {found['artifact_name']}: not the same model, not "
                "compared"
            )
            continue
        for sheet, cell in sorted(found["by_sheet"].items()):
            lines.append(
                f"- {label}, sheet {sheet}: {cell['with_logged_forecast']} of "
                f"{cell['turns']} turns have a logged forecast; "
                f"{cell['slots_compared']} slots compared, "
                f"{cell['slots_differing']} differ by more than "
                f"{found['tolerance']:g} (largest {cell['max_abs_diff']:.4f}); slot "
                f"presence differs on {cell['slot_presence_differs']}, the sheet "
                f"state given to the model on {cell['sheet_state_differs']} turns; "
                f"listed fine actions: {cell.get('actions_compared', 0)} of "
                f"{cell.get('actions_listed', 0)} compared, "
                f"{cell.get('actions_differing', 0)} differ (largest "
                f"{cell.get('actions_max_abs_diff', 0.0):.4f}), "
                f"{cell.get('actions_not_among_the_candidates', 0)} name something "
                "outside the candidates here"
            )
    lines += ["", "## Definitions", ""]
    lines += [f"- **{key}**: {text}" for key, text in reading["definitions"].items()]
    return "\n".join(lines) + "\n"


# --- command line ---------------------------------------------------------------


def _day_after(day: str) -> str:
    """The local day after ``YYYY-MM-DD`` (noon of the next day, so a clock
    change cannot move it)."""
    start = _day_start(day, "the day") or 0.0
    return time.strftime("%Y-%m-%d", time.localtime(start + 36 * 3600))


# ``--until`` of this day, with ``--fresh``, reads the sealed set's first day alone.
FRESH_FIRST_DAY_UNTIL = _day_after(FRESH_SINCE)
HELP_EPILOG = f"""the sealed set (ladder games dated on or after {FRESH_SINCE}):

  every sealed game:
    evaluation/oppmodel_ladder_read.py --fresh --fresh-root ../vgc-bench \\
        --artifact deployed=results_oppmodel/oppnet_v2_blind/artifact.pt \\
        --artifact new=<path> --reference deployed --out <a new directory>

  the games of {FRESH_SINCE} alone (--until keeps the pages dated BEFORE its day):
    the same line with  --until {FRESH_FIRST_DAY_UNTIL}

  an ordinary reading of older games that share a directory with sealed ones:
    --dir <directory> --until {FRESH_SINCE}     (no --fresh)

Without --fresh a page dated on or after {FRESH_SINCE} is refused, also in a
direct call of read_games (pass fresh=True there).
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="a predictor artifact (count table or network); repeat for several",
    )
    parser.add_argument(
        "--dir",
        action="append",
        default=None,
        help="a replay directory of the bot's own games; repeat for several. "
        "There is no default: give --dir or --fresh",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="read the SEALED confirmation set (ladder games dated on or after "
        f"{FRESH_SINCE}). Without --dir: every {FRESH_GLOB} directory holding "
        "such a page, older pages left out. With --dir: allows the named "
        "directories to hold sealed games",
    )
    parser.add_argument(
        "--fresh-root",
        default=None,
        help="where --fresh looks for the directories (default: the repo root)",
    )
    parser.add_argument(
        "--reference", default=None, help="label to pair against (default: the first)"
    )
    parser.add_argument(
        "--since",
        default=None,
        help="keep pages dated YYYY-MM-DD (or YYYY-MM-DDTHH:MM, local time) or "
        "later; with --fresh and no --since it is the sealed set's first day",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="keep pages dated BEFORE YYYY-MM-DD (or YYYY-MM-DDTHH:MM, local "
        f"time). With --fresh, --until {FRESH_FIRST_DAY_UNTIL} reads the games "
        f"of {FRESH_SINCE} alone; without --fresh, --until {FRESH_SINCE} leaves "
        "the sealed games out",
    )
    parser.add_argument(
        "--games", default=None, help="comma list of battle ids / tags, or @file"
    )
    parser.add_argument("--include-unrated", action="store_true")
    parser.add_argument(
        "--own-sheet",
        default=OWN_SHEET_FILE,
        choices=(OWN_SHEET_FILE, OWN_SHEET_NONE),
        help="in an open game also give the bot's own sheet, read from the team "
        "file run_config.json names (default), or encode it the builder's way",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--note", default="", help="a line for the top of the report")
    parser.add_argument(
        "--render-only", action="store_true", help="rewrite the .md from the .json"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    if args.render_only:
        try:
            reading = json.loads((out / READ_JSON).read_text(encoding="utf-8"))
            SC.write_if_changed(out / READ_MD, render_markdown(reading))
        except Exception as exc:
            reason = " ".join(f"{type(exc).__name__}: {exc}".split())
            say(f"LADDER_READ_FAILED cannot render {out / READ_JSON}: {reason}")
            return 1
        say(f"rendered {out / READ_MD}")
        return 0
    try:
        run(
            args.artifact,
            out,
            directories=args.dir,
            fresh=args.fresh,
            fresh_root=args.fresh_root,
            reference=args.reference,
            since=args.since,
            until=args.until,
            games=args.games,
            include_unrated=args.include_unrated,
            own_sheet=args.own_sheet,
            resamples=args.resamples,
            seed=args.seed,
            overwrite=args.overwrite,
            note=args.note,
        )
    except Exception as exc:
        # A monitor reads the last line, not a traceback.
        reason = " ".join(f"{type(exc).__name__}: {exc}".split())
        say(f"LADDER_READ_FAILED {reason}")
        return 1
    say("LADDER_READ_DONE")
    return 0


if __name__ == "__main__":
    SC._one_thread()
    raise SystemExit(main())
