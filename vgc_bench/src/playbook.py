"""Playbook planner: OUR game plan at team preview, with its reasons.

The user, 2026-09-27: "our team picker needs to come up with a plan of action as
to why its choosing its specific 4 and why it wants its first two ... this
strategy needs to be already thought of and passed down to the real brain ...
we need to be the human picking and choosing our own strategy not just basing it
off of what others did".

A playbook file (``data/playbook_<team>.json``, the machine form of
``PLAYBOOK_<team>.md``) lists plan cards in priority order: the four to bring,
the two leads, every Pokemon's job, the turn-1 script and the rules for when to
pick the card, written over OPPONENT FEATURES. This module reads the opponent's
six -- species, plus their sets when the sheet is open -- turns them into those
features from the Reg M-C set data and the type chart, and picks the first card
whose rules hold (the default card last). The result carries its reasons, so
every game records WHY it brought these four and led these two.

Team-agnostic code: our species live only in the playbook file; the opponent's
come from their preview; every feature is derived from data (set usage, types,
base stats), never from a species list.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_env.data import GenData, to_id_str

from vgc_bench.src.opponent_preview import PreviewPlan
from vgc_bench.src.preview_rules import species_trick_room_rate
from vgc_bench.src.set_priors import prior_reg

ROOT = Path(__file__).resolve().parents[2]
_SETS_CACHE: dict[str, dict[str, Any]] = {}

# thresholds on set-usage probabilities (share of a species' recorded sets)
SETTER = 0.5  # a weather / terrain setter, a Trick Room setter, a Tailwind user
SOMETIMES = 0.3  # Wide Guard, Imprison, Rock Slide: a real chance is enough
PHYSICAL_ATK = 110  # "heavy physical attacker": base Atk >= this and > base SpA
SLOW_SPEED = 60  # base Speed at or below this plays well under Trick Room


def _sets(formatid: str | None = None) -> dict[str, Any]:
    reg = prior_reg(formatid)
    if reg not in _SETS_CACHE:
        path = ROOT / f"data/joint_sets_reg{reg}.json"
        _SETS_CACHE[reg] = json.loads(path.read_text()) if path.exists() else {}
    return _SETS_CACHE[reg]


def _dex() -> dict[str, Any]:
    return GenData.from_gen(9).pokedex


def _base_species(species: str) -> str:
    """The usage-data key: Megas and cosmetic formes fall back to the base."""
    sid = to_id_str(species)
    entry = _dex().get(sid) or {}
    return to_id_str(entry.get("baseSpecies", sid)) if "mega" in sid else sid


def set_share(
    species: str,
    ability: str | None = None,
    move: str | None = None,
    item: str | None = None,
    formatid: str | None = None,
) -> float:
    """Share of the species' recorded sets with this ability / move / item."""
    entry = _sets(formatid).get(to_id_str(species)) or _sets(formatid).get(
        _base_species(species)
    )
    if not entry:
        return 0.0
    share = 0.0
    for s in entry.get("sets", []):
        # recorded sets can carry "item": null (no item seen), never a crash
        if ability is not None and to_id_str(s.get("ability") or "") != ability:
            continue
        if move is not None and move not in {
            to_id_str(m) for m in s.get("moves") or []
        }:
            continue
        if item is not None and to_id_str(s.get("item") or "") != item:
            continue
        share += float(s.get("prob", 0.0))
    return share


@dataclass(frozen=True)
class OpenSet:
    """What an open team sheet shows for one opposing Pokemon."""

    ability: str = ""
    item: str = ""
    moves: tuple[str, ...] = ()


def _has(
    species: str,
    sheet: OpenSet | None,
    *,
    ability: str | None = None,
    move: str | None = None,
    item: str | None = None,
    formatid: str | None = None,
) -> float:
    """1/0 from an open sheet, else the set-usage share."""
    if sheet is not None:
        if ability is not None:
            return float(to_id_str(sheet.ability) == ability)
        if move is not None:
            return float(move in {to_id_str(m) for m in sheet.moves})
        if item is not None:
            return float(to_id_str(sheet.item) == item)
    return set_share(species, ability=ability, move=move, item=item, formatid=formatid)


def _fire_multiplier(species: str) -> float:
    chart = GenData.from_gen(9).type_chart
    mult = 1.0
    for t in (_dex().get(to_id_str(species)) or {}).get("types", []):
        mult *= float(chart.get(t.upper(), {}).get("FIRE", 1.0))
    return mult


def opponent_features(
    roster: Sequence[str],
    sheets: Mapping[str, OpenSet] | None = None,
    formatid: str | None = None,
) -> dict[str, Any]:
    """Features of the opponent's six the plan cards are written over.

    Each boolean names the species that triggered it (in ``"<feature>_by"``) so
    the reasons can say "sand (Tyranitar)".
    """
    sheets = {to_id_str(k): v for k, v in (sheets or {}).items()}
    feats: dict[str, Any] = {}

    def flag(name: str, threshold: float, **what) -> None:
        who = [
            s
            for s in roster
            if _has(s, sheets.get(to_id_str(s)), formatid=formatid, **what) >= threshold
        ]
        feats[name] = bool(who)
        feats[name + "_by"] = who

    flag("rain_setter", SETTER, ability="drizzle")
    flag("sand_setter", SETTER, ability="sandstream")
    flag("psychic_terrain", SETTER, ability="psychicsurge")
    flag("tailwind", SETTER, move="tailwind")
    flag("wide_guard", SOMETIMES, move="wideguard")
    flag("imprison", SOMETIMES, move="imprison")
    flag("rock_slide", SOMETIMES, move="rockslide")
    fake_out = [
        s
        for s in roster
        if _has(s, sheets.get(to_id_str(s)), move="fakeout", formatid=formatid)
        >= SETTER
    ]
    tr = [
        s
        for s in roster
        if (
            float("trickroom" in {to_id_str(m) for m in sheets[to_id_str(s)].moves})
            if to_id_str(s) in sheets
            else species_trick_room_rate(s, formatid)
        )
        >= SETTER
    ]
    feats["tr_setter"], feats["tr_setter_by"] = bool(tr), tr
    feats["fake_out_users"], feats["fake_out_users_by"] = len(fake_out), fake_out
    physical, slow, fire_weak = [], [], []
    for s in roster:
        stats = (_dex().get(to_id_str(s)) or {}).get("baseStats", {})
        if stats.get("atk", 0) >= PHYSICAL_ATK and stats.get("atk", 0) > stats.get(
            "spa", 0
        ):
            physical.append(s)
        if stats and stats.get("spe", 999) <= SLOW_SPEED:
            slow.append(s)
        if _fire_multiplier(s) >= 2:
            fire_weak.append(s)
    feats["physical_threats"], feats["physical_threats_by"] = len(physical), physical
    feats["slow_count"], feats["slow_count_by"] = len(slow), slow
    feats["fire_weak"], feats["fire_weak_by"] = len(fire_weak), fire_weak
    return feats


_COMPARE = re.compile(r"^([a-z_]+)\s*(>=|<=|==)\s*(\d+)$")


def _holds(condition: str, feats: Mapping[str, Any]) -> bool:
    match = _COMPARE.match(condition.strip())
    if match:
        name, op, value = match.group(1), match.group(2), int(match.group(3))
        have = int(feats.get(name, 0))
        return {">=": have >= value, "<=": have <= value, "==": have == value}[op]
    return bool(feats.get(condition.strip(), False))


def rule_matches(rule: Mapping[str, Sequence[str]], feats: Mapping[str, Any]) -> bool:
    """``{"all": [...], "any": [...], "none": [...]}`` over the features."""
    if not all(_holds(c, feats) for c in rule.get("all", [])):
        return False
    if rule.get("any") and not any(_holds(c, feats) for c in rule["any"]):
        return False
    return not any(_holds(c, feats) for c in rule.get("none", []))


def _describe(condition: str, feats: Mapping[str, Any]) -> str:
    match = _COMPARE.match(condition.strip())
    name = match.group(1) if match else condition.strip()
    who = feats.get(name + "_by") or []
    label = name.replace("_", " ")
    if match:
        return f"{label} {feats.get(name, 0)} ({', '.join(who)})" if who else label
    return f"{label} ({', '.join(who)})" if who else label


@dataclass
class PlaybookChoice:
    card: str
    lead: tuple[str, str]
    back: tuple[str, str]
    plan: PreviewPlan
    roles: dict[str, str]
    turn1: dict[str, Any]
    reasons: list[str]
    features: dict[str, Any] = field(default_factory=dict)

    def audit(self) -> dict[str, Any]:
        """What the decision log records at team preview."""
        return {
            "card": self.card,
            "lead": list(self.lead),
            "back": list(self.back),
            "roles": self.roles,
            "turn1": self.turn1,
            "reasons": self.reasons,
            "features": {
                k: v
                for k, v in self.features.items()
                if not k.endswith("_by") and v not in (False, 0)
            },
        }


class Playbook:
    """Plan cards for one team, chosen by rules over opponent features."""

    def __init__(self, path: str | Path, allow_experimental: bool = False):
        self.path = Path(path)
        # cards marked "experimental" stay in the playbook but are only picked
        # when explicitly allowed (e.g. for their own local test)
        self.allow_experimental = allow_experimental
        spec = json.loads(self.path.read_text())
        self.team = spec.get("team", "")
        self.cards: list[dict[str, Any]] = list(spec["cards"])
        if not self.cards or self.cards[-1].get("when"):
            raise ValueError("the last playbook card must be the unconditional default")
        for card in self.cards:
            for key in ("name", "lead", "back"):
                if key not in card:
                    raise ValueError(f"playbook card without {key}: {card}")
            if len(card["lead"]) != 2 or len(card["back"]) != 2:
                raise ValueError(f"card {card['name']} needs two leads and two back")

    def choose(
        self,
        our_species: Sequence[str],
        their_species: Sequence[str],
        sheets: Mapping[str, OpenSet] | None = None,
        formatid: str | None = None,
    ) -> PlaybookChoice:
        ours = [to_id_str(s) for s in our_species]
        feats = opponent_features(their_species, sheets, formatid)
        passed_over: list[str] = []  # cards whose case held but an avoid rule fired
        for card in self.cards:
            if card.get("experimental") and not self.allow_experimental:
                continue
            rule = card.get("when") or {}
            if not rule_matches(rule, feats):
                if rule_matches({k: v for k, v in rule.items() if k != "none"}, feats):
                    hits = [c for c in rule.get("none", []) if _holds(c, feats)]
                    passed_over.append(
                        f"not {card['name']}: "
                        + ", ".join(_describe(c, feats) for c in hits)
                    )
                continue
            back = list(card["back"])
            swapped = None
            for alt in card.get("back_if", []):
                if rule_matches(alt, feats):
                    back, swapped = list(alt["back"]), alt
                    break
            wanted = [*card["lead"], *back]
            if not set(wanted) <= set(ours):  # the card names a Pokemon we lack
                continue
            reasons = [f"{card['name']}: {card.get('why', '')}".strip()]
            for cond in rule.get("all", []):
                reasons.append("because " + _describe(cond, feats))
            for cond in rule.get("any", []):
                if _holds(cond, feats):
                    reasons.append("because " + _describe(cond, feats))
            if rule.get("none"):
                reasons.append(
                    "and none of: "
                    + ", ".join(c.replace("_", " ") for c in rule["none"])
                )
            if swapped is not None:
                hit = [c for c in swapped.get("any", []) if _holds(c, feats)]
                reasons.append(
                    f"back line {' + '.join(back)} because "
                    + ", ".join(_describe(c, feats) for c in hit)
                )
            reasons.extend(passed_over)
            index = {s: i for i, s in enumerate(ours)}
            lead = (index[wanted[0]], index[wanted[1]])
            bring = tuple(sorted(index[s] for s in wanted))
            plan = PreviewPlan(lead_indices=lead, bring_indices=bring, probability=1.0)  # type: ignore[arg-type]
            return PlaybookChoice(
                card=card["name"],
                lead=(wanted[0], wanted[1]),
                back=(back[0], back[1]),
                plan=plan,
                roles=dict(card.get("roles", {})),
                turn1=dict(card.get("turn1", {})),
                reasons=reasons,
                features=feats,
            )
        raise ValueError("no playbook card fits this team")  # unreachable: default card
