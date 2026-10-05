"""Build the opponent predictor's dataset: corpus -> examples, splits, manifest.

Design: OPPONENT_PREDICTOR.md ("Splits and hygiene"). One example is one turn
start seen from one ACTOR side; its labels are what that side's two slots then
did. Reads, in this order (first occurrence of a replay id wins):

  top_merged  battle_logs_top_mc_merged_20260920/logs_<fmt>.json
  web_top     battle_logs_web_mc_20260927/top/*.log
  web_low     battle_logs_web_mc_20260927/low/*.log
  t6like      battle_logs_players_t6like/logs_<fmt>.json
  feed        battle_logs_feed_mc/logs/<fmt>/part-*.jsonl.gz (the scraper's shards;
              only whole gzip members are read, so a growing shard is safe)
  own         ladder_replays_mc*/ and challenge_replays_mc*/ (the bot's saved
              pages: player view, the bot's exact HP rewritten to the public
              percent with the rule LiveShadow uses, sheet state taken from the
              decision audit's preview_shadow.open_sheet)

Two passes, one core. Pass 1 drives every log once for metadata and labels,
then fixes the splits and builds the move repertoire from TRAINING actions only.
Pass 2 drives again and encodes. Big JSON files are loaded one at a time.

Dropped and counted by reason: duplicate id, wrong format, no turn, a Pokemon
whose logged identity can be false (Illusion), not doubles, a turn order that
cannot be trusted, the bot's own account in a human log, reader failures. A
human-source copy of one of the bot's own games is dropped without taking its
id, so the saved own page (read last) still enters the ladder holdout.

Splits (per example, by the ACTOR's account id, never the display name):

  ladder_holdout           the opponent's side of the bot's own ladder games. The
                           bot's side is never an example.
  own_unrated              the opponent's side of the bot's other saved games.
  test / val / train       crc32(account id) % 10 -> 0 / 1 / 2-9.
  excluded_holdout_player  would be train or val, but a player of the battle is
                           an opponent of one of the bot's saved games (a model
                           is neither fitted nor selected on such a battle; a
                           test player keeps the test split).
  excluded_clone           would be train, but the battle is one the eval-only
                           human clones were fitted on: both sheets open and
                           crc32(replay id) % 10 >= 5 in top_merged (the bucket
                           function of vgc_bench/logs2trajs.py --buckets).

Flags (m_flag bits): 1 latest 10% of human battles by upload time (a best-of-
three series is flagged as a whole, by its series marker); 2 clone bucket;
4 the battle has a holdout opponent; 8 the actor is a holdout opponent;
16 a ladder-holdout actor who also plays in the human corpus. A series cannot
straddle train / val / test because both of its players keep their account;
the manifest reports the count as a check. m_weight = min(1, 60 / battles of
the actor's account), so no account counts for more than 60 battles.

Output, under results_oppmodel/<tag>/: shard-NNNNN.npz (about 50,000 examples
each; arrays of vgc_bench.src.oppmodel.features.layout() plus the m_* arrays
below), battles.jsonl, vocab.json + tables.npz + repertoire.json (what
Featurizer.load reads), manifest.json.

  m_battle int32   row of battles.jsonl      m_turn   int16  turn number
  m_side   uint8   actor: 0 = p1, 1 = p2     m_actor  uint32 crc32(account id)
  m_source uint8   index into SOURCES        m_split  uint8  index into SPLITS
  m_flag   uint8   bits above                m_weight float32 account-cap weight
  m_time   int64   upload time, 0 unknown    m_sheet  uint8  0 closed, 1 open,
                                             2 unknown: what is KNOWN of the
                                             actor's sheet (own games: audit)

    nice -n 19 .venv/bin/python datagen/oppmodel_build_dataset.py --tag v1_ondisk
    nice -n 19 .venv/bin/python datagen/oppmodel_build_dataset.py --tag smoke \
        --sources web,own --limit 200
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import re
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from datagen.oppmodel_scrape import read_shard, shard_paths
from vgc_bench.src.oppmodel.events import (
    KIND_HIDDEN,
    KIND_MOVE,
    KIND_NONE,
    KIND_SWITCH,
    SIDES,
    TARGET_AUTO,
    SheetSet,
    base_species_id,
    dex_signature,
    extract_log_from_html,
    is_choosable_target,
    parse_showteam,
    split_log,
    to_id,
    user_id,
)
from vgc_bench.src.oppmodel.features import (
    CAND_SHEET,
    CAND_VALID,
    G_ACTOR_SHEET,
    LAYOUT_VERSION,
    N_CAND_DEFAULT,
    REASON_NAMES,
    SHEET_CLOSED,
    SHEET_OPEN,
    Y_KIND_ABSENT,
    Y_KIND_HIDDEN,
    Y_KIND_MOVE,
    Y_KIND_NONE,
    Y_KIND_SWITCH,
    Y_LOCKED,
    Y_OTHER,
    Batch,
    Example,
    Featurizer,
    Repertoire,
    collate,
    has_terrain,
    save_batch,
    set_key,
)
from vgc_bench.src.oppmodel.public_state import (
    RATED_LADDER,
    DriveResult,
    drive_log,
    rewrite_event,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMATS = ("gen9championsvgc2026regmc", "gen9championsvgc2026regmcbo3")
SOURCES: tuple[str, ...] = ("top_merged", "web_top", "web_low", "t6like", "feed", "own")
SOURCE_GROUPS: dict[str, tuple[str, ...]] = {
    "top_merged": ("top_merged",),
    "web": ("web_top", "web_low"),
    "web_top": ("web_top",),
    "web_low": ("web_low",),
    "t6like": ("t6like",),
    "feed": ("feed",),
    "own": ("own",),
}
TOP_MERGED_DIR = "battle_logs_top_mc_merged_20260920"
T6LIKE_DIR = "battle_logs_players_t6like"
WEB_DIR = "battle_logs_web_mc_20260927"
FEED_DIR = "battle_logs_feed_mc"
OWN_GLOBS = ("ladder_replays_mc*", "challenge_replays_mc*")
PUBLIC_CACHE = "results_oppmodel/parity/public"

SPLITS: tuple[str, ...] = (
    "train",
    "val",
    "test",
    "ladder_holdout",
    "own_unrated",
    "excluded_holdout_player",
    "excluded_clone",
)
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_LADDER, SPLIT_OWN_UNRATED = range(5)
SPLIT_EXCLUDED_PLAYER, SPLIT_EXCLUDED_CLONE = 5, 6
FLAG_TIME_SLICE = 1
FLAG_CLONE_BUCKET = 2
FLAG_HOLDOUT_BATTLE = 4
FLAG_ACTOR_IS_HOLDOUT = 8
FLAG_ACTOR_IN_CORPUS = 16

ACCOUNT_CAP = 60
TIME_SLICE_SHARE = 0.10
CLONE_BUCKETS = frozenset({5, 6, 7, 8, 9})
SHARD_SIZE = 50_000
DROP_EXAMPLES = 5

_OWN_NAME = re.compile(r"^(?P<player>.+?) - battle-(?P<id>[a-z0-9]+-\d+)")
_AUDIT_BATTLE = re.compile(r"battle-([a-z0-9]+-\d+)")
_SHEET_NAMES = {SHEET_CLOSED: "closed", SHEET_OPEN: "open"}
# ``m_sheet``: what is KNOWN about the game's sheets, whatever the stream showed.
SHEET_STATES: tuple[str, ...] = ("closed", "open", "unknown")
SCOUT_SOURCES = ("top_merged", "web_top", "web_low", "t6like")


# --- split rules (pure, unit-tested) ------------------------------------------


def account_bucket(account: str) -> int:
    """crc32 of the Showdown account id, modulo 10."""
    return zlib.crc32(account.encode("utf-8")) % 10


def account_hash(account: str) -> int:
    return zlib.crc32(account.encode("utf-8")) & 0xFFFFFFFF


def player_split(account: str) -> int:
    """The player holdout: bucket 0 test, 1 validation, 2-9 train."""
    bucket = account_bucket(account)
    if bucket == 0:
        return SPLIT_TEST
    return SPLIT_VAL if bucket == 1 else SPLIT_TRAIN


def clone_bucket(replay_id: str) -> int:
    """The bucket the human clones were split by: crc32(replay id) % 10.

    Same function as ``vgc_bench/logs2trajs.py --buckets`` applies to the keys
    of the corpus files.
    """
    return zlib.crc32(replay_id.encode()) % 10


def cap_weights(battles: Mapping[str, int], cap: int = ACCOUNT_CAP) -> dict[str, float]:
    """Per-account example weight so no account counts for more than ``cap`` battles."""
    return {
        account: (1.0 if count <= cap else cap / count)
        for account, count in battles.items()
    }


def time_slice_groups(
    times: Mapping[str, int | None],
    groups: Mapping[str, str],
    share: float = TIME_SLICE_SHARE,
) -> tuple[set[str], int | None]:
    """Groups in the latest ``share`` of battles by time, and the cut.

    ``times`` and ``groups`` are keyed by battle. The cut is taken over battles;
    a group (a best-of-three series) is in the slice as a whole when its latest
    game is at or after the cut, so a series never straddles the flag.
    """
    known = sorted(value for value in times.values() if value is not None)
    if not known:
        return set(), None
    position = min(len(known) - 1, int(len(known) * (1.0 - share)))
    cut = known[position]
    latest: dict[str, int] = {}
    for battle, value in times.items():
        if value is not None:
            group = groups.get(battle, battle)
            latest[group] = max(latest.get(group, value), value)
    return {group for group, value in latest.items() if value >= cut}, cut


# --- inputs -------------------------------------------------------------------


@dataclass
class RawBattle:
    """One log as read from disk, before it is driven."""

    battle_id: str
    source: str
    log: str
    time: int | None = None
    time_source: str = ""
    own: bool = False
    bot_side: str | None = None
    sheets: dict[str, list[SheetSet]] | None = None
    sheets_known: bool = True
    sheet_audit: str = ""
    sheet_evidence: str = ""
    path: str = ""


@dataclass
class BattleMeta:
    """What pass 1 keeps of a battle (one row of battles.jsonl)."""

    index: int
    battle_id: str
    source: str
    format_id: str
    players: dict[str, str | None]
    accounts: dict[str, str]
    ratings: dict[str, int | None]
    sheets: dict[str, bool | None]
    rated_kind: str | None
    best_of: int | None
    series_id: str | None
    game_number: int | None
    turns: int
    winner: str | None
    time: int | None
    time_source: str
    own: bool
    bot_side: str | None
    sheet_audit: str
    sheet_evidence: str
    group: str
    clone: bool = False
    holdout_battle: bool = False
    time_slice: bool = False
    split: dict[str, int] = field(default_factory=dict)
    weight: dict[str, float] = field(default_factory=dict)
    # (side, set key, move, auto: -1 unknown / 0 / 1, terrain up) per free move.
    uses: list[tuple[str, str, str, int, bool]] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.battle_id,
            "source": self.source,
            "format": self.format_id,
            "players": self.players,
            "accounts": self.accounts,
            "ratings": self.ratings,
            "sheets": self.sheets,
            "sheet_audit": self.sheet_audit,
            "sheet_evidence": self.sheet_evidence,
            "rated_kind": self.rated_kind,
            "best_of": self.best_of,
            "series": self.series_id,
            "game": self.game_number,
            "group": self.group,
            "turns": self.turns,
            "winner": self.winner,
            "time": self.time,
            "time_source": self.time_source,
            "own": self.own,
            "bot_side": self.bot_side,
            "clone_bucket": self.clone,
            "holdout_battle": self.holdout_battle,
            "time_slice": self.time_slice,
            "split": {side: SPLITS[code] for side, code in self.split.items()},
            "weight": {side: round(value, 6) for side, value in self.weight.items()},
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    """Hash of a folder's contents: its sorted id list."""
    return sha256_bytes("\n".join(sorted(ids)).encode("utf-8"))


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


class Corpus:
    """Readers of every source under ``root``; records what it read in ``inputs``."""

    def __init__(
        self,
        root: Path,
        formats: Sequence[str],
        sources: Sequence[str],
        limit: int | None = None,
    ) -> None:
        self.root = root
        self.formats = tuple(formats)
        self.sources = tuple(name for name in SOURCES if name in set(sources))
        self.limit = limit
        self.inputs: dict[str, dict[str, Any]] = {}
        self.notes: Counter[str] = Counter()
        self.feed_bytes: dict[str, int] = {}
        self._audit: dict[str, bool] | None = None
        self._their: dict[str, list[SheetSet]] = {}

    def _noted(self, path: Path) -> bool:
        return _relative(path, self.root) in self.inputs

    def _note_input(self, path: Path, **info: Any) -> None:
        self.inputs.setdefault(_relative(path, self.root), info)

    # -- human corpora ---------------------------------------------------------

    def _json_file(self, path: Path, source: str) -> Iterator[RawBattle]:
        if not path.exists():
            self.notes[f"missing:{_relative(path, self.root)}"] += 1
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.notes[f"unreadable:{_relative(path, self.root)}"] += 1
            return
        if not self._noted(path):
            self._note_input(path, kind="json", sha256=sha256_file(path), n=len(data))
        for battle_id, row in data.items():
            if not isinstance(row, list) or len(row) < 2 or not isinstance(row[1], str):
                self.notes[f"bad_row:{source}"] += 1
                continue
            when = row[0] if isinstance(row[0], int) else None
            yield RawBattle(str(battle_id), source, row[1], when, "upload")

    def _json_dir(self, folder: str, source: str) -> Iterator[RawBattle]:
        for fmt in self.formats:
            yield from self._json_file(self.root / folder / f"logs_{fmt}.json", source)

    def _web(self, sub: str, source: str) -> Iterator[RawBattle]:
        folder = self.root / WEB_DIR / sub
        paths = sorted(folder.glob("*.log")) if folder.is_dir() else []
        if not paths:
            self.notes[f"missing:{_relative(folder, self.root)}"] += 1
            return
        self._note_input(
            folder,
            kind="folder",
            sha256=sha256_ids([p.stem for p in paths]),
            n=len(paths),
        )
        for path in paths:
            try:
                log = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                self.notes[f"unreadable:{source}"] += 1
                continue
            yield RawBattle(path.stem, source, log, None, "", path=path.name)

    def _feed(self) -> Iterator[RawBattle]:
        out_dir = self.root / FEED_DIR
        found = False
        for fmt in self.formats:
            for path in shard_paths(out_dir, fmt):
                found = True
                key = _relative(path, self.root)
                # A growing shard: pass 2 reads the prefix pass 1 read.
                records, good = read_shard(path)
                if key in self.feed_bytes:
                    good = min(good, self.feed_bytes[key])
                else:
                    self.feed_bytes[key] = good
                if not self._noted(path):
                    with path.open("rb") as handle:
                        prefix = handle.read(good)
                    self._note_input(
                        path,
                        kind="feed_shard",
                        sha256=sha256_bytes(prefix),
                        bytes=good,
                        n=len(records),
                        note="sha256 of the intact prefix that was read",
                    )
                for record in records:
                    log = record.get("log")
                    if not isinstance(log, str):
                        self.notes["bad_row:feed"] += 1
                        continue
                    when = record.get("uploadtime")
                    yield RawBattle(
                        str(record["id"]),
                        "feed",
                        log,
                        when if isinstance(when, int) else None,
                        "upload",
                    )
        if not found:
            self.notes[f"missing:{FEED_DIR}/logs"] += 1

    # -- the bot's own saved games ----------------------------------------------

    def own_pages(self) -> list[tuple[Path, str, str]]:
        """(path, account name, replay id) of every saved own page, once per id."""
        found: dict[str, tuple[Path, str, str]] = {}
        for pattern in OWN_GLOBS:
            for folder in sorted(self.root.glob(pattern)):
                if not folder.is_dir():
                    continue
                for path in sorted(folder.glob("*.html")):
                    match = _OWN_NAME.match(path.name)
                    if match is None:
                        continue
                    fmt = match["id"].rsplit("-", 1)[0]
                    if fmt in self.formats:
                        found.setdefault(
                            match["id"], (path, match["player"].strip(), match["id"])
                        )
        return [found[key] for key in sorted(found)]

    def bot_accounts(self) -> set[str]:
        """Account ids the saved pages were written for (from the file names)."""
        return {user_id(account) for _, account, _ in self.own_pages()}

    def sheet_audit(self) -> dict[str, bool]:
        """Replay id -> the open-sheet flag the bot recorded at team preview."""
        if self._audit is not None:
            return self._audit
        audit: dict[str, bool] = {}
        for pattern in OWN_GLOBS:
            for path in sorted(self.root.glob(pattern + "/decisions.jsonl")):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                self._note_input(path, kind="decision_audit", sha256=sha256_file(path))
                for line in text.split("\n"):
                    if '"open_sheet"' not in line and '"their_sheet"' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    match = _AUDIT_BATTLE.search(str(row.get("battle", "")))
                    if match is None:
                        continue
                    shadow = row.get("preview_shadow")
                    if isinstance(shadow, dict) and isinstance(
                        shadow.get("open_sheet"), bool
                    ):
                        audit[match.group(1)] = shadow["open_sheet"]
                    their = row.get("their_sheet")
                    if isinstance(their, dict):
                        self._their[match.group(1)] = _sheet_from_audit(their)
        self._audit = audit
        return audit

    def _public_copy(
        self, battle_id: str
    ) -> tuple[bool, dict[str, list[SheetSet]], int | None]:
        """(found, open sheets, upload time) of an own game's cached public replay."""
        path = self.root / PUBLIC_CACHE / f"{battle_id}.json"
        if not path.exists():
            return False, {}, None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, {}, None
        if not isinstance(data, dict) or not isinstance(data.get("log"), str):
            return False, {}, None
        sheets: dict[str, list[SheetSet]] = {}
        for event in split_log(str(data.get("log") or "")):
            if len(event) > 3 and event[1] == "showteam" and event[2] in SIDES:
                sheets.setdefault(event[2], parse_showteam("|".join(event[3:])))
        when = data.get("uploadtime")
        if not self._noted(path):
            self._note_input(path, kind="public_copy", sha256=sha256_file(path))
        return True, sheets, when if isinstance(when, int) else None

    def _own(self) -> Iterator[RawBattle]:
        pages = self.own_pages()
        audit = self.sheet_audit()
        by_folder: dict[Path, list[str]] = defaultdict(list)
        for path, _, battle_id in pages:
            by_folder[path.parent].append(battle_id)
        for folder, ids in sorted(by_folder.items()):
            self._note_input(folder, kind="folder", sha256=sha256_ids(ids), n=len(ids))
        for path, account, battle_id in pages:
            try:
                html = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                self.notes["unreadable:own"] += 1
                continue
            log = extract_log_from_html(html)
            if log is None:
                self.notes["own:no_log_block"] += 1
                continue
            events = split_log(log)
            role = next(
                (
                    event[2]
                    for event in events
                    if len(event) > 3
                    and event[1] == "player"
                    and event[2] in SIDES
                    and user_id(event[3]) == user_id(account)
                ),
                None,
            )
            if role is None:
                self.notes["own:account_not_a_player"] += 1
                continue
            # The bot's exact HP becomes the public percent, as LiveShadow does
            # event by event, so the state is the one the runtime would hold.
            public = "\n".join("|".join(rewrite_event(e, role)) for e in events)
            copied, sheets, uploaded = self._public_copy(battle_id)
            recorded = audit.get(battle_id)
            evidence = "audit" if recorded is not None else ""
            if copied:
                # A spectator log of the same game settles it either way.
                state, known, evidence = (
                    ("open", True, "public_copy")
                    if len(sheets) == len(SIDES)
                    else ("closed", True, "public_copy")
                )
                if recorded is not None and recorded != (state == "open"):
                    self.notes["own:sheet_audit_contradicts_public_copy"] += 1
                if state == "closed":
                    sheets = {}
            elif recorded is False:
                state, known, sheets = "closed", True, {}
            elif recorded:
                # Open, but the page never carries the sheet text: only the
                # opponent's sheet, when the audit stored it, can be given.
                state, known, sheets = "open", False, {}
                if battle_id in self._their:
                    sheets = {SIDES[1 - SIDES.index(role)]: self._their[battle_id]}
            else:
                state, known, sheets = "unknown", False, {}
            try:
                modified = int(path.stat().st_mtime)
            except OSError:
                modified = None
            yield RawBattle(
                battle_id,
                "own",
                public,
                uploaded if uploaded is not None else modified,
                "upload" if uploaded is not None else "mtime",
                own=True,
                bot_side=role,
                sheets=sheets or None,
                sheets_known=known,
                sheet_audit=state,
                sheet_evidence=evidence,
                path=_relative(path, self.root),
            )

    def __iter__(self) -> Iterator[RawBattle]:
        readers = {
            "top_merged": lambda: self._json_dir(TOP_MERGED_DIR, "top_merged"),
            "web_top": lambda: self._web("top", "web_top"),
            "web_low": lambda: self._web("low", "web_low"),
            "t6like": lambda: self._json_dir(T6LIKE_DIR, "t6like"),
            "feed": self._feed,
            "own": self._own,
        }
        for source in self.sources:
            for count, raw in enumerate(readers[source]()):
                if self.limit is not None and count >= self.limit:
                    break
                yield raw

    def clone_corpus_ids(self) -> set[str]:
        """Replay ids of the corpus the clones were fitted on (all of top_merged).

        Read even when that source is not selected, so a game that reaches the
        dataset through another folder still carries its hygiene flag.
        """
        ids: set[str] = set()
        for fmt in self.formats:
            path = self.root / TOP_MERGED_DIR / f"logs_{fmt}.json"
            if not path.exists():
                continue
            try:
                ids.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                self.notes[f"unreadable:{_relative(path, self.root)}"] += 1
        return ids


def _sheet_from_audit(record: Mapping[str, Any]) -> list[SheetSet]:
    """The decision audit's ``their_sheet`` record as sheet sets."""
    sets: list[SheetSet] = []
    for name, entry in record.items():
        if not isinstance(entry, dict):
            continue
        moves = entry.get("moves")
        sets.append(
            SheetSet(
                species=base_species_id(str(name)),
                forme=to_id(str(name)),
                item=to_id(str(entry.get("item") or "")) or None,
                ability=to_id(str(entry.get("ability") or "")) or None,
                moves=tuple(to_id(str(m)) for m in moves)
                if isinstance(moves, list)
                else (),
            )
        )
    return sets


def drive(raw: RawBattle) -> DriveResult:
    return drive_log(
        raw.log, raw.battle_id, sheets=raw.sheets, sheets_known=raw.sheets_known
    )


def drop_reason(
    raw: RawBattle, result: DriveResult, formats: Sequence[str], bots: set[str]
) -> str | None:
    """Why a driven battle is not used, or None."""
    if any(name.startswith("drive_error:") for name in result.counters):
        return "reader_failure"
    if result.format_id not in formats:
        return "format"
    if result.skip_reason is not None:
        return result.skip_reason
    if any(name.startswith("feed_error:") for name in result.counters):
        return "reader_failure"
    accounts = {account for account in result.player_ids.values() if account}
    if len(accounts) < len(SIDES):
        return "players_missing"
    if not raw.own and accounts & bots:
        return "bot_account"
    if raw.own and raw.bot_side not in SIDES:
        return "bot_side_unknown"
    if not result.turns:
        return "no_turn_1"
    return None


def free_move_uses(result: DriveResult) -> list[tuple[str, str, str, int, bool]]:
    """Every visible, freely chosen move: (side, set key, move, auto, terrain)."""
    uses: list[tuple[str, str, str, int, bool]] = []
    for record in result.turns:
        terrain = has_terrain(record.snapshot)
        for action in record.actions.values():
            if action.kind != KIND_MOVE or not action.move or action.locked:
                continue
            mon = record.snapshot.mon(action.side, action.slot)
            if mon is None or mon.species != action.species:
                continue
            auto = -1
            if is_choosable_target(action.move) and action.target is not None:
                auto = int(action.target == TARGET_AUTO)
            uses.append(
                (
                    action.side,
                    set_key(mon.forme, mon.species),
                    action.move,
                    auto,
                    terrain,
                )
            )
    return uses


def census_key(kind: str, action: Any) -> str:
    """The scouts' observability classes, for the manifest's headline."""
    if kind == KIND_SWITCH:
        return "switch"
    if kind == KIND_MOVE:
        if action.locked:
            return "forced"
        if action.reason:
            return "move_cant_named"
        return "move_target" if action.target is not None else "move_no_target"
    if kind == KIND_HIDDEN:
        return "hidden"
    if kind == KIND_NONE:
        return "forced" if action.locked else f"none:{action.reason}"
    return "other"


# --- pass 1 -------------------------------------------------------------------


@dataclass
class Pass1:
    metas: list[BattleMeta]
    by_id: dict[str, BattleMeta]
    drops: Counter[str]
    drops_by_source: dict[str, Counter[str]]
    drop_slots: Counter[str]
    drop_examples: dict[str, list[str]]
    read: Counter[str]
    duplicates: Counter[str]
    census_all: dict[str, Counter[str]]
    counters: Counter[str]


def run_pass1(corpus: Corpus, bots: set[str], clone_ids: set[str]) -> Pass1:
    out = Pass1(
        [],
        {},
        Counter(),
        defaultdict(Counter),
        Counter(),
        defaultdict(list),
        Counter(),
        Counter(),
        defaultdict(Counter),
        Counter(),
    )
    seen: set[str] = set()
    for raw in corpus:
        out.read[raw.source] += 1
        if raw.battle_id in seen:
            out.duplicates[raw.source] += 1
            continue
        result = drive(raw)
        reason = drop_reason(raw, result, corpus.formats, bots)
        # A public copy of one of the bot's own games (the feed holds a few) is
        # dropped WITHOUT taking the id: human sources are read before the saved
        # own pages, and the id must stay free for the page, or the game is lost
        # from the ladder holdout as a duplicate.
        if reason != "bot_account":
            seen.add(raw.battle_id)
        # Every slot-turn of every unique log, kept or not: the denominator the
        # scouts used.
        for record in result.turns:
            for action in record.actions.values():
                if raw.own and action.side == raw.bot_side:
                    continue
                out.census_all[raw.source][census_key(action.kind, action)] += 1
        if reason is not None:
            out.drops[reason] += 1
            out.drops_by_source[raw.source][reason] += 1
            out.drop_slots[reason] += sum(len(r.actions) for r in result.turns)
            if len(out.drop_examples[reason]) < DROP_EXAMPLES:
                out.drop_examples[reason].append(raw.battle_id)
            continue
        for name, count in result.counters.items():
            out.counters[name] += count
        when, when_source = raw.time, raw.time_source
        if when is None and result.end_time is not None:
            when, when_source = result.end_time, "log"
        accounts = {side: str(result.player_ids.get(side) or "") for side in SIDES}
        open_both = all(result.sheets.get(side) is True for side in SIDES)
        meta = BattleMeta(
            index=len(out.metas),
            battle_id=raw.battle_id,
            source=raw.source,
            format_id=str(result.format_id),
            players=dict(result.players),
            accounts=accounts,
            ratings=dict(result.ratings),
            sheets=dict(result.sheets),
            rated_kind=result.rated_kind,
            best_of=result.best_of,
            series_id=result.series_id,
            game_number=result.game_number,
            turns=len(result.turns),
            winner=result.winner,
            time=when,
            time_source=when_source,
            own=raw.own,
            bot_side=raw.bot_side,
            sheet_audit=raw.sheet_audit,
            sheet_evidence=raw.sheet_evidence,
            group=result.series_id or raw.battle_id,
            clone=(
                not raw.own
                and open_both
                and raw.battle_id in clone_ids
                and clone_bucket(raw.battle_id) in CLONE_BUCKETS
            ),
            uses=free_move_uses(result),
        )
        out.metas.append(meta)
        out.by_id[meta.battle_id] = meta
    return out


def assign_splits(
    metas: Sequence[BattleMeta], cap: int = ACCOUNT_CAP
) -> dict[str, Any]:
    """Fill ``split`` / ``weight`` / flags of every battle; returns the summary."""
    holdout = {
        meta.accounts[side]
        for meta in metas
        if meta.own
        for side in SIDES
        if side != meta.bot_side
    }
    ladder_opponents = {
        meta.accounts[side]
        for meta in metas
        if meta.own and meta.rated_kind == RATED_LADDER
        for side in SIDES
        if side != meta.bot_side
    }
    human = [meta for meta in metas if not meta.own]
    battles: Counter[str] = Counter()
    by_source: dict[str, set[str]] = defaultdict(set)
    for meta in human:
        for account in set(meta.accounts.values()):
            battles[account] += 1
            by_source[meta.source].add(account)
    weights = cap_weights(battles, cap)
    flagged, cut = time_slice_groups(
        {meta.battle_id: meta.time for meta in human},
        {meta.battle_id: meta.group for meta in human},
    )
    for meta in metas:
        meta.split, meta.weight = {}, {}
        if meta.own:
            code = (
                SPLIT_LADDER if meta.rated_kind == RATED_LADDER else SPLIT_OWN_UNRATED
            )
            for side in SIDES:
                if side != meta.bot_side:
                    meta.split[side], meta.weight[side] = code, 1.0
            continue
        meta.holdout_battle = bool(set(meta.accounts.values()) & holdout)
        meta.time_slice = meta.group in flagged
        for side in SIDES:
            code = player_split(meta.accounts[side])
            # A battle with an own-game opponent neither trains nor selects a
            # model: validation picks table strengths, the stopping epoch and the
            # temperatures. Test players keep their split, so the ladder holdout
            # and the test set can share a few players (the scorecard counts them).
            if code in (SPLIT_TRAIN, SPLIT_VAL) and meta.holdout_battle:
                code = SPLIT_EXCLUDED_PLAYER
            elif code == SPLIT_TRAIN and meta.clone:
                code = SPLIT_EXCLUDED_CLONE
            meta.split[side] = code
            meta.weight[side] = weights.get(meta.accounts[side], 1.0)
    # A series must sit in one of train / val / test per player: check it.
    by_series: dict[tuple[str, str], set[int]] = defaultdict(set)
    for meta in human:
        for side in SIDES:
            by_series[(meta.group, meta.accounts[side])].add(
                player_split(meta.accounts[side])
            )
    corpus_accounts = set(battles)
    top = [
        {
            "account": account,
            "battles": count,
            "weight": round(weights[account], 4),
            "split": SPLITS[player_split(account)],
            "holdout_opponent": account in holdout,
        }
        for account, count in sorted(battles.items(), key=lambda kv: (-kv[1], kv[0]))[
            :30
        ]
    ]
    return {
        "holdout_accounts": holdout,
        "ladder_opponents": ladder_opponents,
        "corpus_accounts": corpus_accounts,
        "weights": weights,
        "time_cut": cut,
        "summary": {
            "distinct_accounts": len(battles),
            "account_cap": cap,
            "accounts_over_cap": sum(1 for n in battles.values() if n > cap),
            "battles_of_accounts_over_cap": sum(n for n in battles.values() if n > cap),
            "top_accounts": top,
            "ladder_holdout_opponents": len(ladder_opponents),
            "own_game_opponents": len(holdout),
            "ladder_opponents_in_human_corpus": len(ladder_opponents & corpus_accounts),
            "ladder_opponents_in_human_corpus_by_source": {
                source: len(ladder_opponents & accounts)
                for source, accounts in sorted(by_source.items())
            },
            "ladder_opponents_in_human_corpus_without_feed": len(
                ladder_opponents
                & set().union(
                    *(a for source, a in by_source.items() if source != "feed")
                )
            ),
            "own_opponents_in_human_corpus": len(holdout & corpus_accounts),
            "series_straddling_player_split": sum(
                1 for codes in by_series.values() if len(codes) > 1
            ),
            "series": len({meta.group for meta in human if meta.series_id}),
            "time_cut": cut,
            "time_slice_battles": sum(1 for meta in human if meta.time_slice),
            "battles_without_time": sum(1 for meta in human if meta.time is None),
        },
    }


def build_repertoire(metas: Sequence[BattleMeta]) -> Repertoire:
    """Move frequencies from the training split only, account-cap weighted."""
    repertoire = Repertoire()
    for meta in metas:
        for side, key, move, auto, terrain in meta.uses:
            if meta.split.get(side) != SPLIT_TRAIN:
                continue
            repertoire.add(
                key,
                move,
                meta.weight.get(side, 1.0),
                auto=None if auto < 0 else bool(auto),
                terrain=terrain,
            )
    return repertoire


# --- pass 2 -------------------------------------------------------------------


def example_flags(meta: BattleMeta, side: str, info: Mapping[str, Any]) -> int:
    account = meta.accounts[side]
    flags = 0
    if meta.time_slice:
        flags |= FLAG_TIME_SLICE
    if meta.clone:
        flags |= FLAG_CLONE_BUCKET
    if meta.holdout_battle:
        flags |= FLAG_HOLDOUT_BATTLE
    if not meta.own and account in info["holdout_accounts"]:
        flags |= FLAG_ACTOR_IS_HOLDOUT
    if meta.own and account in info["corpus_accounts"]:
        flags |= FLAG_ACTOR_IN_CORPUS
    return flags


def encode_battle(
    featurizer: Featurizer,
    meta: BattleMeta,
    result: DriveResult,
    info: Mapping[str, Any],
) -> list[Example]:
    """Every example of one battle, with its m_* arrays."""
    out: list[Example] = []
    for side in SIDES:
        if side not in meta.split:
            continue  # the bot's own side
        actor = np.uint32(account_hash(meta.accounts[side]))
        flags = example_flags(meta, side, info)
        if meta.own:
            sheet = SHEET_STATES.index(meta.sheet_audit or "unknown")
        else:
            state = meta.sheets.get(side)
            sheet = SHEET_STATES.index(
                "unknown" if state is None else ("open" if state else "closed")
            )
        for record in result.turns:
            example = featurizer.encode_turn(record.snapshot, record.actions, side)
            if example is None:
                continue
            example["m_battle"] = np.array(meta.index, dtype=np.int32)
            example["m_turn"] = np.array(min(32767, record.turn), dtype=np.int16)
            example["m_side"] = np.array(SIDES.index(side), dtype=np.uint8)
            example["m_actor"] = np.array(actor, dtype=np.uint32)
            example["m_source"] = np.array(SOURCES.index(meta.source), dtype=np.uint8)
            example["m_split"] = np.array(meta.split[side], dtype=np.uint8)
            example["m_flag"] = np.array(flags, dtype=np.uint8)
            example["m_weight"] = np.array(meta.weight[side], dtype=np.float32)
            example["m_time"] = np.array(meta.time or 0, dtype=np.int64)
            example["m_sheet"] = np.array(sheet, dtype=np.uint8)
            out.append(example)
    return out


class Stats:
    """Counts of the encoded examples, by split and source."""

    def __init__(self) -> None:
        self.examples: dict[str, Counter[str]] = defaultdict(Counter)
        self.kinds: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        self.reasons: dict[str, Counter[str]] = defaultdict(Counter)
        self.other: dict[str, Counter[str]] = defaultdict(Counter)
        self.censor: dict[str, Counter[str]] = defaultdict(Counter)
        self.elo: dict[str, Counter[str]] = defaultdict(Counter)
        self.sheet_examples: dict[str, Counter[str]] = defaultdict(Counter)

    def add(self, batch: Batch) -> None:
        kind_names = {
            Y_KIND_MOVE: "move",
            Y_KIND_SWITCH: "switch",
            Y_KIND_HIDDEN: "hidden",
            Y_KIND_NONE: "none",
        }
        splits = batch["m_split"]
        sources = batch["m_source"]
        sheet = batch["game_flag"][:, G_ACTOR_SHEET]
        elo, known = batch["elo"][:, 0], batch["elo_known"][:, 0]
        y_kind, y_action = batch["y_kind"], batch["y_action"]
        other = batch["y_flag"][..., Y_OTHER]
        locked = batch["y_flag"][..., Y_LOCKED]
        informative = batch["y_set"].any(-1)
        candidates = ((batch["cand_flag"] & CAND_VALID) > 0).sum(-1)
        on_sheet = ((batch["cand_flag"] & CAND_SHEET) > 0).any(-1)
        for row in range(len(splits)):
            split = SPLITS[int(splits[row])]
            source = SOURCES[int(sources[row])]
            mode = _SHEET_NAMES.get(int(sheet[row]), "unknown")
            self.examples[split][source] += 1
            self.sheet_examples[split][mode] += 1
            bucket = f"{int(elo[row]) // 100 * 100}" if known[row] else "unknown"
            self.elo[split][bucket] += 1
            for slot in range(2):
                kind = int(y_kind[row, slot])
                if kind == Y_KIND_ABSENT:
                    continue
                name = kind_names[kind]
                self.kinds[split][source][name] += 1
                self.censor[split]["slots"] += 1
                reason = REASON_NAMES[int(batch["y_reason"][row, slot])]
                if kind in (Y_KIND_HIDDEN, Y_KIND_NONE):
                    self.censor[split]["censored"] += 1
                    self.reasons[split][f"{name}:{reason}"] += 1
                    if informative[row, slot]:
                        self.censor[split]["censored_with_set"] += 1
                elif locked[row, slot]:
                    self.censor[split]["locked"] += 1
                if kind == Y_KIND_MOVE and y_action[row, slot] >= 0:
                    # A sheet-known Pokemon has its real moves as candidates.
                    seen = "sheet" if on_sheet[row, slot] else mode
                    self.other[split][f"{seen}:moves"] += 1
                    self.other[split][f"{seen}:other"] += int(other[row, slot])
                    if not on_sheet[row, slot] and candidates[row, slot] > 4:
                        self.other[split][f"{seen}:moves_incomplete"] += 1
                        self.other[split][f"{seen}:other_incomplete"] += int(
                            other[row, slot]
                        )

    def other_rates(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        total: Counter[str] = Counter()
        for split, counts in sorted(self.other.items()):
            out[split] = _rates(counts)
            total.update(counts)
        out["all"] = _rates(total)
        return out

    def censored(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        total: Counter[str] = Counter()
        for split, counts in sorted(self.censor.items()):
            total.update(counts)
            out[split] = _censor_row(counts)
        out["all"] = _censor_row(total)
        return out


def _rates(counts: Mapping[str, int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in sorted({key.split(":", 1)[0] for key in counts}):
        moves = counts.get(f"{name}:moves", 0)
        misses = counts.get(f"{name}:other", 0)
        row: dict[str, Any] = {
            "moves": moves,
            "outside_candidates": misses,
            "rate": round(misses / moves, 5) if moves else None,
        }
        partial = counts.get(f"{name}:moves_incomplete", 0)
        if partial:
            row["moves_while_moveset_incomplete"] = partial
            row["rate_while_moveset_incomplete"] = round(
                counts.get(f"{name}:other_incomplete", 0) / partial, 5
            )
        out[name] = row
    return out


def _censor_row(counts: Mapping[str, int]) -> dict[str, Any]:
    slots = counts.get("slots", 0)
    return {
        "slot_turns": slots,
        "censored": counts.get("censored", 0),
        "censored_share": round(counts.get("censored", 0) / slots, 5)
        if slots
        else None,
        "censored_with_set_loss": counts.get("censored_with_set", 0),
        "locked_moves": counts.get("locked", 0),
    }


def _nested(counts: Any) -> Any:
    if isinstance(counts, (Counter, dict, defaultdict)):
        return {str(key): _nested(value) for key, value in sorted(counts.items())}
    return counts


def _sum_counters(
    counters: Mapping[str, Counter[str]], names: Sequence[str]
) -> Counter[str]:
    total: Counter[str] = Counter()
    for name in names:
        total.update(counters.get(name, Counter()))
    return total


def census_summary(census: Mapping[str, int]) -> dict[str, Any]:
    """Slot-turn totals in the scouts' terms (fully visible = move+target, switch)."""
    total = sum(census.values())
    visible = census.get("move_target", 0) + census.get("switch", 0)
    return {
        "slot_turns": total,
        "fully_visible": visible,
        "fully_visible_share": round(visible / total, 5) if total else None,
        "classes": dict(sorted(census.items())),
    }


# --- main ---------------------------------------------------------------------


def parse_sources(text: str) -> list[str]:
    names: list[str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "all":
            names.extend(SOURCES)
        elif part in SOURCE_GROUPS:
            names.extend(SOURCE_GROUPS[part])
        else:
            raise SystemExit(f"unknown source {part!r}; known: {sorted(SOURCE_GROUPS)}")
    return [name for name in SOURCES if name in names]


def build(
    tag: str,
    *,
    root: Path = ROOT,
    out_root: Path | None = None,
    sources: Sequence[str] = SOURCES,
    formats: Sequence[str] = DEFAULT_FORMATS,
    limit: int | None = None,
    n_cand: int = N_CAND_DEFAULT,
    shard_size: int = SHARD_SIZE,
    account_cap: int = ACCOUNT_CAP,
    overwrite: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Build the dataset under ``<out_root>/<tag>/`` and return its manifest."""
    started = time.time()
    signature = dex_signature()
    if not signature.get("dex_available"):
        raise SystemExit("dex files not found: nothing can be labelled or encoded")
    out_dir = (out_root if out_root is not None else root / "results_oppmodel") / tag
    if (out_dir / "manifest.json").exists() and not overwrite:
        raise SystemExit(f"{out_dir} already holds a dataset; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("shard-*.npz"):
        stale.unlink()

    corpus = Corpus(root, formats, sources, limit)
    bots = corpus.bot_accounts()
    clone_ids = corpus.clone_corpus_ids()
    first = run_pass1(corpus, bots, clone_ids)
    info = assign_splits(first.metas, account_cap)
    repertoire = build_repertoire(first.metas)
    featurizer = Featurizer.build(repertoire, n_cand)
    featurizer.save(out_dir)
    notes = dict(corpus.notes)  # pass 2 walks the same readers again
    pass1_seconds = time.time() - started
    log(
        f"pass 1: {sum(first.read.values()):,} logs read, {len(first.metas):,} kept, "
        f"{sum(first.drops.values()):,} dropped, "
        f"{sum(first.duplicates.values()):,} duplicate ids ({pass1_seconds:.0f}s)"
    )

    stats = Stats()
    shards: list[dict[str, Any]] = []
    buffer: list[Example] = []
    encode_seconds = 0.0
    encoded = 0
    battles_without_examples = 0

    def flush() -> None:
        if not buffer:
            return
        batch = collate(buffer)
        if not batch:
            raise SystemExit("examples of one shard do not share a layout")
        name = f"shard-{len(shards):05d}.npz"
        save_batch(out_dir / name, batch)
        stats.add(batch)
        shards.append({"file": name, "examples": len(buffer)})
        buffer.clear()

    seen: set[str] = set()
    # The same readers again: a feed shard that grew is read up to the prefix
    # pass 1 saw, and ids pass 1 did not keep are skipped.
    for raw in corpus:
        if raw.battle_id in seen:
            continue
        meta = first.by_id.get(raw.battle_id)
        # Another source's copy of a kept battle (the public copy of an own game
        # that pass 1 dropped) must not be encoded under the kept battle's meta.
        if meta is not None and meta.source != raw.source:
            continue
        seen.add(raw.battle_id)
        if meta is None:
            continue
        result = drive(raw)
        clock = time.perf_counter()
        examples = encode_battle(featurizer, meta, result, info)
        encode_seconds += time.perf_counter() - clock
        encoded += len(examples)
        battles_without_examples += int(not examples)
        buffer.extend(examples)
        if len(buffer) >= shard_size:
            flush()
    flush()

    with (out_dir / "battles.jsonl").open("w", encoding="utf-8") as handle:
        for meta in first.metas:
            handle.write(json.dumps(meta.row(), sort_keys=True) + "\n")

    human = [meta for meta in first.metas if not meta.own]
    own = [meta for meta in first.metas if meta.own]
    ladder = [meta for meta in own if meta.rated_kind == RATED_LADDER]
    own_sheets = Counter(meta.sheet_audit for meta in own)
    slot_kinds = _nested(stats.kinds)
    manifest: dict[str, Any] = {
        "tag": tag,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "layout_version": LAYOUT_VERSION,
        "n_cand": featurizer.n_cand,
        "n_actions": featurizer.n_actions,
        "formats": list(formats),
        "sources_requested": list(corpus.sources),
        "sources": list(SOURCES),
        "splits": list(SPLITS),
        "limit": limit,
        "account_cap": account_cap,
        "shard_size": shard_size,
        "time_slice_share": TIME_SLICE_SHARE,
        "clone_buckets": sorted(CLONE_BUCKETS),
        "dex_signature": signature,
        "headline": {
            "human_logs_unique": sum(
                first.read[s] - first.duplicates[s] for s in SOURCES if s != "own"
            ),
            "human_battles_kept": len(human),
            "own_games_unique": first.read["own"] - first.duplicates["own"],
            "own_games_kept": len(own),
            "ladder_holdout_games": len(ladder),
            "ladder_holdout_opponents": info["summary"]["ladder_holdout_opponents"],
            "ladder_opponents_in_human_corpus": info["summary"][
                "ladder_opponents_in_human_corpus"
            ],
            "examples": encoded,
            "human_all_logs": census_summary(
                _sum_counters(first.census_all, [s for s in SOURCES if s != "own"])
            ),
            "human_all_logs_without_feed": census_summary(
                _sum_counters(first.census_all, SCOUT_SOURCES)
            ),
            "human_logs_unique_without_feed": sum(
                first.read[s] - first.duplicates[s] for s in SCOUT_SOURCES
            ),
            "own_opponents_all_logs": census_summary(first.census_all["own"]),
        },
        "battles": {
            "read": dict(first.read),
            "duplicate_ids": dict(first.duplicates),
            "kept_by_source": dict(Counter(meta.source for meta in first.metas)),
            "kept_by_format": dict(Counter(meta.format_id for meta in first.metas)),
            "dropped": dict(first.drops),
            "dropped_by_source": _nested(first.drops_by_source),
            "dropped_slot_turns": dict(first.drop_slots),
            "dropped_examples": dict(first.drop_examples),
            "kept_without_examples": battles_without_examples,
            "own_sheet_audit": dict(own_sheets),
            "human_open_sheet_both": sum(
                1 for meta in human if all(meta.sheets.get(s) for s in SIDES)
            ),
            "clone_bucket_battles": sum(1 for meta in human if meta.clone),
            "holdout_player_battles": sum(1 for meta in human if meta.holdout_battle),
        },
        "accounts": info["summary"],
        "examples": {
            "by_split_source": _nested(stats.examples),
            "by_split": {
                split: sum(counts.values())
                for split, counts in sorted(stats.examples.items())
            },
            "by_split_actor_sheet": _nested(stats.sheet_examples),
        },
        "slot_turns": {
            "by_split_source_kind": slot_kinds,
            "censored_reasons_by_split": _nested(stats.reasons),
        },
        "censored": stats.censored(),
        "other_rate": stats.other_rates(),
        "elo_histogram_by_split": _nested(stats.elo),
        "repertoire": {
            "set_keys": len(repertoire.counts),
            "weighted_uses": round(
                sum(sum(moves.values()) for moves in repertoire.counts.values()), 2
            ),
            "aimed_moves_seen_as_spread": sorted(
                move for move in repertoire.aim if repertoire.auto_seen(move)
            ),
        },
        "featurizer": {
            "vocab_sizes": featurizer.vocab.sizes(),
            "species_table": list(featurizer.tables.species_num.shape),
            "move_table": list(featurizer.tables.move_num.shape),
            "counters": dict(featurizer.counters),
            "encode_microseconds_per_example": (
                round(1e6 * encode_seconds / encoded, 1) if encoded else None
            ),
        },
        "reader_counters": dict(first.counters.most_common()),
        "notes": notes,
        "inputs": dict(sorted(corpus.inputs.items())),
        "arrays": {
            name: {
                "shape": list(spec.shape),
                "dtype": spec.dtype,
                "group": spec.group,
                "doc": spec.doc,
            }
            for name, spec in featurizer.layout().items()
        },
        "meta_arrays": {
            "m_battle": "int32 row of battles.jsonl",
            "m_turn": "int16 turn number",
            "m_side": "uint8 actor side: 0 = p1, 1 = p2",
            "m_actor": "uint32 crc32 of the actor's account id",
            "m_source": "uint8 index into sources",
            "m_split": "uint8 index into splits",
            "m_flag": "uint8 bits: 1 time slice, 2 clone bucket, 4 battle has a "
            "holdout opponent, 8 actor is a holdout opponent, 16 ladder-holdout "
            "actor also plays in the human corpus",
            "m_weight": "float32 account-cap weight",
            "m_time": "int64 upload time (Unix seconds), 0 unknown",
            "m_sheet": "uint8 what is known of the actor's sheet in this game: "
            "0 closed, 1 open, 2 unknown (own games: the decision audit or a "
            "public copy; it can say open where the stream shows nothing)",
        },
        "shards": shards,
        "seconds": {
            "pass1": round(pass1_seconds, 1),
            "total": round(time.time() - started, 1),
            "encode": round(encode_seconds, 1),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    log(
        f"pass 2: {encoded:,} examples in {len(shards)} shard(s) -> {out_dir} "
        f"({manifest['seconds']['total']:.0f}s total)"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--tag", required=True, help="results_oppmodel/<tag>/")
    parser.add_argument(
        "--sources",
        default="all",
        help="comma list of " + ", ".join(sorted(SOURCE_GROUPS)) + " (default all)",
    )
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    parser.add_argument(
        "--limit", type=int, default=None, help="logs per source, for a smoke run"
    )
    parser.add_argument("--n-cand", type=int, default=N_CAND_DEFAULT)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--out-root", default=None, help="default: results_oppmodel/ in the repo"
    )
    args = parser.parse_args()
    manifest = build(
        args.tag,
        out_root=Path(args.out_root) if args.out_root else None,
        sources=parse_sources(args.sources),
        formats=[part.strip() for part in args.formats.split(",") if part.strip()],
        limit=args.limit,
        n_cand=args.n_cand,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["headline"], indent=1))
    print(json.dumps({"other_rate": manifest["other_rate"]["all"]}, indent=1))


if __name__ == "__main__":
    main()
