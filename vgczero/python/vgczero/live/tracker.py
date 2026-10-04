"""Track a live Showdown battle into an engine snapshot.

Our side comes from our own team (exact sets) plus the server's request JSON
(HP, status, items, PP, actives). The opponent comes from team preview
(species) and the battle protocol: switches, damage (Champions percentages),
statuses, boosts, revealed moves/items/abilities, Mega Evolution, volatiles,
field and side conditions, and open team sheets. Unknown opponent details are
filled with a placeholder tournament set; the encoder hides what is unrevealed
and search resamples it.
"""

from __future__ import annotations

import builtins
import copy
import json
import re
from dataclasses import dataclass, field

from .sets import Pool, pool, to_id

STATUSES = {"brn", "par", "slp", "frz", "psn", "tox"}
BOOST_NAMES = {"atk", "def", "spa", "spd", "spe", "accuracy", "evasion"}
STALL_MOVES = {"protect", "detect", "spikyshield", "banefulbunker", "kingsshield", "silktrap", "burningbulwark", "endure", "wideguard", "quickguard"}
WEATHER = {"RainDance": "rain", "SunnyDay": "sun", "Sandstorm": "sand", "Snowscape": "snow", "Snow": "snow", "none": ""}


@dataclass
class Mon:
    set: dict | None = None
    species: str = ""  # current species id (forme)
    base: str = ""
    nick: str = ""
    hp: int = 0
    max_hp: int = 0
    pct: float = 1.0
    status: str = ""
    boosts: dict = field(default_factory=dict)
    slot: int = -1
    fainted: bool = False
    brought: bool = False
    seen: bool = False
    is_mega: bool = False
    item: str | None = None  # current item id ('' = none), None = unknown
    item_known: bool = False
    ability: str | None = None
    ability_known: bool = False
    revealed_moves: builtins.set = field(default_factory=builtins.set)
    pp: list | None = None
    vol: dict = field(default_factory=dict)
    move_actions: int = 0
    active_turns: int = 0
    last_move: str = ""
    times_attacked: int = 0
    stall_counter: int = 0
    stall_dur: int = 0
    stalled_this_turn: bool = False
    ate_berry: bool = False

    def reset_switch_out(self) -> None:
        self.boosts = {}
        self.vol = {}
        self.move_actions = 0
        self.active_turns = 0
        self.last_move = ""
        self.times_attacked = 0
        self.stall_counter = 0
        self.stall_dur = 0
        self.slot = -1
        if self.status == "tox":
            pass


@dataclass
class Side:
    id: str = ""
    name: str = ""
    mons: list = field(default_factory=lambda: [Mon() for _ in range(6)])
    conds: dict = field(default_factory=dict)
    mega_used: bool = False
    total_fainted: int = 0
    sheet_open: bool = False
    preview: list = field(default_factory=list)  # species ids from |poke|


def parse_hp(cond: str) -> tuple[int, int, str, bool]:
    """'123/190 brn' -> (123, 190, 'brn', fainted)."""
    cond = cond.strip()
    if cond.endswith("fnt") or cond.startswith("0 "):
        return 0, 0, "", True
    parts = cond.split()
    hp, mx = parts[0].split("/") if "/" in parts[0] else (parts[0], "100")
    st = parts[1] if len(parts) > 1 and parts[1] in STATUSES else ""
    return int(re.sub(r"[^0-9]", "", hp) or 0), int(re.sub(r"[^0-9]", "", mx) or 100), st, False


def ident_side_slot(ident: str) -> tuple[str, int, str]:
    """'p2a: Nick' -> ('p2', 0, 'Nick'); 'p1: Nick' -> ('p1', -1, 'Nick')."""
    head, _, nick = ident.partition(": ")
    sid = head[:2]
    slot = -1
    if len(head) > 2:
        slot = ord(head[2]) - ord("a")
    return sid, slot, nick.strip()


def details_species(details: str) -> str:
    return to_id(details.split(",")[0])


class Tracker:
    def __init__(self, our_team: dict, pool_: Pool | None = None):
        self.pool = pool_ or pool()
        self.team = our_team
        self.sides = {"p1": Side(id="p1"), "p2": Side(id="p2")}
        self.me = ""  # 'p1' or 'p2'
        self.turn = 0
        self.field = {"weather": "", "weather_turns": 0, "terrain": "", "terrain_turns": 0, "trick_room": 0, "gravity": 0}
        self.request: dict | None = None
        self.ended = False
        self.winner: str | None = None
        self.log: list[str] = []

    # ---- helpers ------------------------------------------------------------------------

    @property
    def opp(self) -> str:
        return "p2" if self.me == "p1" else "p1"

    def side(self, sid: str) -> Side:
        return self.sides[sid]

    def _init_our_side(self, sid: str) -> None:
        s = self.sides[sid]
        for i, m in enumerate(self.team["mons"]):
            mon = s.mons[i]
            mon.set = copy.deepcopy(m)
            mon.species = m["species"]
            mon.base = self.pool.base_id(m["species"])
            mon.max_hp = m["stats"][0]
            mon.hp = mon.max_hp
            mon.item = m["item"]
            mon.item_known = True
            mon.ability = m["ability"]
            mon.ability_known = True
            mon.revealed_moves = set(m["moves"])
        s.preview = [m["species"] for m in self.team["mons"]]

    def find_mon(self, sid: str, ident: str, details: str | None = None) -> Mon | None:
        s = self.sides[sid]
        _, _, nick = ident_side_slot(ident)
        for m in s.mons:
            if m.nick and m.nick == nick:
                return m
        sp = details_species(details) if details else to_id(nick)
        base = self.pool.base_id(sp)
        for m in s.mons:
            if m.base == base and (not m.nick or m.nick == nick):
                if not m.nick:
                    m.nick = nick
                return m
        # Fall back to a nickname match on species for our own side.
        for m in s.mons:
            if not m.nick and m.base == self.pool.base_id(nick):
                m.nick = nick
                return m
        return None

    def mon_at(self, ident: str) -> Mon | None:
        sid, _, _ = ident_side_slot(ident)
        if sid not in self.sides:
            return None
        return self.find_mon(sid, ident)

    def _reveal_ability(self, mon: Mon | None, name: str) -> None:
        if mon is None or not name:
            return
        aid = to_id(name)
        mon.ability = aid
        mon.ability_known = True
        if mon.set is not None and not mon.is_mega:
            mon.set["ability"] = aid

    def _reveal_item(self, mon: Mon | None, name: str, consumed: bool = False) -> None:
        if mon is None:
            return
        iid = to_id(name)
        if mon.set is not None and iid:
            mon.set["item"] = iid
        mon.item = "" if consumed else iid
        mon.item_known = True

    def _reveal_move(self, mon: Mon, move_id: str) -> None:
        if move_id in ("struggle", "recharge") or mon.set is None:
            return
        mons = mon.set["moves"]
        if move_id not in mons:
            # Replace the first placeholder move that has not been revealed.
            for k, mv in enumerate(mons):
                if mv not in mon.revealed_moves:
                    mons[k] = move_id
                    break
            else:
                if len(mons) < 4:
                    mons.append(move_id)
        mon.revealed_moves.add(move_id)

    def _tags(self, parts: list[str]) -> dict:
        tags = {}
        for p in parts:
            if p.startswith("[") and "]" in p:
                k, _, v = p[1:].partition("]")
                tags[k] = v.strip()
        return tags

    def _from_effects(self, parts: list[str], target: Mon | None) -> None:
        """Reveal abilities/items named in [from] tags."""
        tags = self._tags(parts)
        frm = tags.get("from", "")
        of = tags.get("of")
        holder = self.mon_at(of) if of else target
        if frm.startswith("ability:"):
            self._reveal_ability(holder, frm.split(":", 1)[1])
        elif frm.startswith("item:"):
            self._reveal_item(holder, frm.split(":", 1)[1])

    # ---- protocol ---------------------------------------------------------------------------

    def feed(self, line: str) -> None:
        if not line.startswith("|"):
            return
        parts = line.split("|")[1:]
        if not parts:
            return
        cmd, args = parts[0], parts[1:]
        if cmd == "request":
            body = "|".join(args)
            if body.strip():
                self.on_request(json.loads(body))
            return
        h = getattr(self, "on_" + cmd.replace("-", "_"), None)
        if h is not None:
            try:
                h(args)
            except Exception as e:  # keep playing on unexpected protocol
                self.log.append(f"tracker error on {line!r}: {e!r}")

    def on_player(self, a):
        if len(a) >= 2 and a[0] in self.sides:
            self.sides[a[0]].name = a[1]

    def on_poke(self, a):
        sid, details = a[0], a[1]
        s = self.sides[sid]
        sp = details_species(details)
        s.preview.append(sp)
        if sid != self.me and self.me:
            i = len(s.preview) - 1
            if i < 6:
                mon = s.mons[i]
                mon.species = sp
                mon.base = self.pool.base_id(sp)
                mon.set = self.pool.placeholder(sp)
                if mon.set is not None:
                    mon.max_hp = mon.set["stats"][0]
                    mon.hp = mon.max_hp

    def on_teampreview(self, a):
        pass

    def on_showteam(self, a):
        # Open team sheets: packed team "Nick|Species|Item|Ability|Moves|..."
        sid, packed = a[0], "|".join(a[1:])
        s = self.sides[sid]
        s.sheet_open = True
        if sid == self.me:
            return
        for chunk in packed.split("]"):
            f = chunk.split("|")
            if len(f) < 5:
                continue
            species = to_id(f[1] or f[0])
            mon = next((m for m in s.mons if m.base == self.pool.base_id(species)), None)
            if mon is None or mon.set is None:
                continue
            item, ability = to_id(f[2]), to_id(f[3])
            moves = [to_id(x) for x in f[4].split(",") if x]
            better = self.pool.placeholder(species)
            for cand in self.pool.sets.get(mon.base, []):
                if cand["item"] == item and cand["ability"] == ability and sorted(cand["moves"]) == sorted(moves):
                    better = copy.deepcopy(cand)
                    break
            if better is not None:
                mon.set = better
            mon.set["item"] = item
            mon.set["ability"] = ability or mon.set["ability"]
            mon.set["moves"] = moves or mon.set["moves"]
            mon.max_hp = mon.set["stats"][0]
            mon.item, mon.item_known = item, True
            mon.ability, mon.ability_known = mon.set["ability"], True
            mon.revealed_moves = set(mon.set["moves"])

    def on_turn(self, a):
        self.turn = int(a[0])
        f = self.field
        for k in ("weather_turns", "terrain_turns", "trick_room", "gravity"):
            if f[k] > 0:
                f[k] -= 1
        if f["weather_turns"] == 0 and f["weather"]:
            f["weather_turns"] = 1  # weather still up: keep at least one turn
        if f["terrain_turns"] == 0:
            f["terrain"] = ""
        for s in self.sides.values():
            for k in list(s.conds):
                if isinstance(s.conds[k], int) and k not in ("spikes", "toxic_spikes"):
                    s.conds[k] = max(0, s.conds[k] - 1)
            for m in s.mons:
                if m.slot >= 0 and not m.fainted:
                    m.active_turns += 1
                    if not m.stalled_this_turn:
                        m.stall_counter = 0
                        m.stall_dur = 0
                    elif m.stall_dur > 0:
                        m.stall_dur -= 1
                    m.stalled_this_turn = False
                    # Showdown announces the end of these (|-end|), so they
                    # never expire here; the count is only a feature.
                    for k in ("taunt", "throat_chop", "yawn", "heal_block", "magnet_rise"):
                        if m.vol.get(k, 0) > 1:
                            m.vol[k] -= 1
                    for k in ("encore", "disable"):
                        if k in m.vol:
                            m.vol[k][0] = max(1, m.vol[k][0] - 1)
                    for k in ("roost", "protect", "helping_hand"):
                        m.vol.pop(k, None)

    def _switch(self, a, drag=False):
        ident, details, cond = a[0], a[1], a[2] if len(a) > 2 else "100/100"
        sid, slot, nick = ident_side_slot(ident)
        s = self.sides[sid]
        # Whoever was in that slot leaves.
        for m in s.mons:
            if m.slot == slot:
                m.reset_switch_out()
        mon = self.find_mon(sid, ident, details)
        if mon is None:
            return
        mon.nick = nick
        mon.slot = slot
        mon.seen = True
        mon.brought = True
        mon.fainted = False
        sp = details_species(details)
        mon.species = sp
        hp, mx, st, fnt = parse_hp(cond)
        mon.status = st
        if sid == self.me:
            mon.hp, mon.max_hp = hp, mx
        else:
            mon.pct = hp / max(1, mx)
            if mon.max_hp:
                mon.hp = max(1, round(mon.pct * mon.max_hp)) if not fnt else 0

    def on_switch(self, a):
        self._switch(a)

    def on_drag(self, a):
        self._switch(a, drag=True)

    def on_replace(self, a):
        self._switch(a)

    def on_detailschange(self, a):
        mon = self.mon_at(a[0])
        if mon is None:
            return
        sp = details_species(a[1])
        mon.species = sp
        if "-mega" in sp or sp.endswith("mega") or "megax" in sp or "megay" in sp or "megaz" in sp:
            self._set_mega(mon, a[0], sp)

    def _set_mega(self, mon: Mon, ident: str, mega_species: str) -> None:
        sid, _, _ = ident_side_slot(ident)
        mon.is_mega = True
        self.sides[sid].mega_used = True
        if sid != self.me and (mon.set is None or not mon.set.get("mega") or mon.set["mega"]["species"] != mega_species):
            new = self.pool.placeholder(mon.base, mega_species=mega_species)
            if new is not None:
                # Keep revealed moves.
                for mv in mon.revealed_moves:
                    if mv not in new["moves"]:
                        for k, x in enumerate(new["moves"]):
                            if x not in mon.revealed_moves:
                                new["moves"][k] = mv
                                break
                mon.set = new
                mon.max_hp = new["stats"][0]
        if mon.set is not None:
            mon.item = mon.set["item"]
            mon.item_known = True
            if mon.set.get("mega"):
                mon.ability = mon.set["mega"]["ability"]

    def on__mega(self, a):
        mon = self.mon_at(a[0])
        if mon is None:
            return
        if len(a) > 2:
            self._reveal_item(mon, a[2])
        sid, _, _ = ident_side_slot(a[0])
        self.sides[sid].mega_used = True

    def on_move(self, a):
        ident, move_name = a[0], a[1]
        mon = self.mon_at(ident)
        if mon is None:
            return
        tags = self._tags(a[2:])
        mid = to_id(move_name)
        if "from" in tags and tags["from"].startswith("lockedmove") is False and tags["from"]:
            # Called by another effect (Magic Bounce, etc.): not this Pokemon's own move choice.
            return
        mon.move_actions += 1
        mon.last_move = mid
        self._reveal_move(mon, mid)
        if mid in STALL_MOVES:
            mon.stalled_this_turn = True
        sid, _, _ = ident_side_slot(ident)
        if sid != self.me and mon.pp is not None and mid in mon.set["moves"]:
            k = mon.set["moves"].index(mid)
            mon.pp[k] = max(0, mon.pp[k] - 1)

    def on_cant(self, a):
        mon = self.mon_at(a[0])
        if mon is not None:
            mon.move_actions += 1
            if len(a) > 1 and a[1] == "recharge":
                mon.vol.pop("must_recharge", None)

    def _hp(self, a):
        mon = self.mon_at(a[0])
        if mon is None or len(a) < 2:
            return
        sid, _, _ = ident_side_slot(a[0])
        hp, mx, st, fnt = parse_hp(a[1])
        if fnt:
            mon.hp = 0
            return
        if sid == self.me:
            mon.hp, mon.max_hp = hp, mx
        else:
            mon.pct = hp / max(1, mx)
            mon.hp = max(1, round(mon.pct * mon.max_hp))
        mon.status = st
        self._from_effects(a[2:], mon)

    def on__damage(self, a):
        self._hp(a)
        tags = self._tags(a[2:])
        mon = self.mon_at(a[0])
        if mon is not None and "from" not in tags:
            mon.times_attacked += 1

    def on__heal(self, a):
        self._hp(a)

    def on__sethp(self, a):
        self._hp(a)

    def on_faint(self, a):
        mon = self.mon_at(a[0])
        if mon is None:
            return
        sid, _, _ = ident_side_slot(a[0])
        mon.fainted = True
        mon.hp = 0
        mon.status = ""
        mon.boosts = {}
        mon.vol = {}
        self.sides[sid].total_fainted += 1

    def on__status(self, a):
        mon = self.mon_at(a[0])
        if mon is not None:
            mon.status = a[1]
            self._from_effects(a[2:], mon)

    def on__curestatus(self, a):
        mon = self.mon_at(a[0])
        if mon is not None:
            mon.status = ""

    def on__cureteam(self, a):
        sid, _, _ = ident_side_slot(a[0])
        for m in self.sides[sid].mons:
            m.status = ""

    def _boost(self, a, sign):
        mon = self.mon_at(a[0])
        if mon is None or len(a) < 3:
            return
        stat = a[1]
        if stat in BOOST_NAMES:
            mon.boosts[stat] = max(-6, min(6, mon.boosts.get(stat, 0) + sign * int(a[2])))
        self._from_effects(a[3:], mon)

    def on__boost(self, a):
        self._boost(a, 1)

    def on__unboost(self, a):
        self._boost(a, -1)

    def on__setboost(self, a):
        mon = self.mon_at(a[0])
        if mon is not None and a[1] in BOOST_NAMES:
            mon.boosts[a[1]] = int(a[2])

    def on__clearboost(self, a):
        mon = self.mon_at(a[0])
        if mon is not None:
            mon.boosts = {}

    def on__clearallboost(self, a):
        for s in self.sides.values():
            for m in s.mons:
                m.boosts = {}

    def on__clearnegativeboost(self, a):
        mon = self.mon_at(a[0])
        if mon is not None:
            mon.boosts = {k: v for k, v in mon.boosts.items() if v > 0}

    def on__invertboost(self, a):
        mon = self.mon_at(a[0])
        if mon is not None:
            mon.boosts = {k: -v for k, v in mon.boosts.items()}

    def on__copyboost(self, a):
        src, dst = self.mon_at(a[0]), self.mon_at(a[1])
        if src is not None and dst is not None:
            dst.boosts = dict(src.boosts)

    def on__weather(self, a):
        w = WEATHER.get(a[0], "")
        tags = self._tags(a[1:])
        if "upkeep" in tags or (len(a) > 1 and a[1] == "[upkeep]"):
            return
        if not w:
            self.field["weather"] = ""
            self.field["weather_turns"] = 0
            return
        self.field["weather"] = w
        self.field["weather_turns"] = 5
        frm, of = tags.get("from", ""), tags.get("of")
        if frm.startswith("ability:") and of:
            holder = self.mon_at(of)
            self._reveal_ability(holder, frm.split(":", 1)[1])

    def on__fieldstart(self, a):
        eff = a[0].replace("move: ", "")
        name = to_id(eff)
        if name == "trickroom":
            self.field["trick_room"] = 5
        elif name == "gravity":
            self.field["gravity"] = 5
        elif name.endswith("terrain"):
            self.field["terrain"] = name.replace("terrain", "")
            self.field["terrain_turns"] = 5
        self._from_effects(a[1:], None)

    def on__fieldend(self, a):
        name = to_id(a[0].replace("move: ", ""))
        if name == "trickroom":
            self.field["trick_room"] = 0
        elif name == "gravity":
            self.field["gravity"] = 0
        elif name.endswith("terrain"):
            self.field["terrain"] = ""
            self.field["terrain_turns"] = 0

    SIDE_CONDS = {"tailwind": 4, "reflect": 5, "lightscreen": 5, "auroraveil": 5, "safeguard": 5, "mist": 5}

    def on__sidestart(self, a):
        sid = a[0][:2]
        name = to_id(a[1].replace("move: ", ""))
        c = self.sides[sid].conds
        if name in self.SIDE_CONDS:
            key = {"lightscreen": "light_screen", "auroraveil": "aurora_veil"}.get(name, name)
            c[key] = self.SIDE_CONDS[name]
        elif name == "stealthrock":
            c["stealth_rock"] = True
        elif name == "spikes":
            c["spikes"] = c.get("spikes", 0) + 1
        elif name == "toxicspikes":
            c["toxic_spikes"] = c.get("toxic_spikes", 0) + 1
        elif name == "stickyweb":
            c["sticky_web"] = True

    def on__sideend(self, a):
        sid = a[0][:2]
        name = to_id(a[1].replace("move: ", ""))
        key = {"lightscreen": "light_screen", "auroraveil": "aurora_veil", "stealthrock": "stealth_rock", "toxicspikes": "toxic_spikes", "stickyweb": "sticky_web"}.get(name, name)
        self.sides[sid].conds.pop(key, None)

    VOL_START = {
        "confusion": ("confusion", 3), "taunt": ("taunt", 3), "substitute": ("substitute", None), "yawn": ("yawn", 2),
        "leechseed": ("leech_seed", True), "saltcure": ("salt_cure", True), "curse": ("curse", True),
        "throatchop": ("throat_chop", 2), "magnetrise": ("magnet_rise", 5), "healblock": ("heal_block", 5),
        "aquaring": ("aqua_ring", True), "ingrain": ("ingrain", True), "smackdown": ("smacked_down", True),
        "torment": ("torment", True), "imprison": ("imprison", True), "charge": ("charge", True),
        "attract": ("attract", True), "noretreat": ("no_retreat", True), "flashfire": ("flash_fire", True),
        "tarshot": ("tar_shot", True), "focusenergy": ("crit_stage", 2), "dragoncheer": ("crit_stage", 1),
        "glaiverush": ("glaive_rush", True), "destinybond": ("destiny_bond", True), "stockpile1": ("stockpile", 1),
        "stockpile2": ("stockpile", 2), "stockpile3": ("stockpile", 3),
    }

    def on__start(self, a):
        mon = self.mon_at(a[0])
        if mon is None:
            return
        eff = a[1].replace("move: ", "").replace("ability: ", "")
        name = to_id(eff)
        if name.startswith("perish"):
            mon.vol["perish"] = int(name[6:] or 3) + 1
        elif name == "encore":
            mon.vol["encore"] = [3, mon.last_move]
        elif name == "disable":
            mv = to_id(a[2]) if len(a) > 2 else mon.last_move
            mon.vol["disable"] = [4, mv]
        elif name == "typechange":
            mon.vol["types"] = [t for t in (a[2] if len(a) > 2 else "").split("/") if t]
        elif name in self.VOL_START:
            key, val = self.VOL_START[name]
            if key == "substitute":
                val = max(1, (mon.max_hp or 4) // 4)
            mon.vol[key] = val
        self._from_effects(a[2:], mon)

    def on__end(self, a):
        mon = self.mon_at(a[0])
        if mon is None:
            return
        name = to_id(a[1].replace("move: ", "").replace("ability: ", ""))
        if name in self.VOL_START:
            mon.vol.pop(self.VOL_START[name][0], None)
        for k in ("encore", "disable", "perish"):
            if name == k:
                mon.vol.pop(k, None)

    def on__singleturn(self, a):
        mon = self.mon_at(a[0])
        if mon is None:
            return
        name = to_id(a[1].replace("move: ", ""))
        if name in ("protect", "spikyshield", "banefulbunker", "kingsshield", "silktrap", "burningbulwark", "endure", "maxguard"):
            mon.stall_counter = 3 if mon.stall_dur == 0 else min(729, mon.stall_counter * 3)
            mon.stall_dur = 2
            mon.stalled_this_turn = True

    def on__mustrecharge(self, a):
        mon = self.mon_at(a[0])
        if mon is not None:
            mon.vol["must_recharge"] = True

    def on__prepare(self, a):
        mon = self.mon_at(a[0])
        if mon is not None:
            mon.vol["charging"] = [to_id(a[1]), 0]

    def on__item(self, a):
        mon = self.mon_at(a[0])
        self._reveal_item(mon, a[1])
        tags = self._tags(a[2:])
        frm, of = tags.get("from", ""), tags.get("of")
        if frm.startswith("ability:") and of:
            self._reveal_ability(self.mon_at(of), frm.split(":", 1)[1])

    def on__enditem(self, a):
        mon = self.mon_at(a[0])
        if mon is None:
            return
        tags = self._tags(a[2:])
        self._reveal_item(mon, a[1], consumed=True)
        if "eat" in tags or "[eat]" in a[2:]:
            mon.ate_berry = True

    def on__ability(self, a):
        mon = self.mon_at(a[0])
        self._reveal_ability(mon, a[1] if len(a) > 1 else "")

    def on__activate(self, a):
        mon = self.mon_at(a[0]) if a and a[0][:2] in self.sides else None
        if mon is not None and len(a) > 1:
            eff = a[1]
            if eff.startswith("ability:"):
                self._reveal_ability(mon, eff.split(":", 1)[1])
            elif eff.startswith("item:"):
                self._reveal_item(mon, eff.split(":", 1)[1])
        self._from_effects(a[2:], mon)

    def on__transform(self, a):
        pass

    def on_win(self, a):
        self.ended = True
        self.winner = a[0] if a else None

    def on_tie(self, a):
        self.ended = True
        self.winner = None

    # ---- request ------------------------------------------------------------------------------

    def on_request(self, req: dict) -> None:
        self.request = req
        side = req.get("side", {})
        sid = side.get("id")
        if sid and not self.me:
            self.me = sid
            self._init_our_side(sid)
            # Opponent preview lines may have arrived before we knew our side.
            opp = self.sides[self.opp]
            for i, sp in enumerate(opp.preview[:6]):
                mon = opp.mons[i]
                if mon.set is None:
                    mon.species = sp
                    mon.base = self.pool.base_id(sp)
                    mon.set = self.pool.placeholder(sp)
                    if mon.set is not None:
                        mon.max_hp = mon.set["stats"][0]
                        mon.hp = mon.max_hp
        if not self.me:
            return
        s = self.sides[self.me]
        for k, p in enumerate(side.get("pokemon", [])):
            mon = self.find_mon(self.me, p["ident"], p.get("details"))
            if mon is None:
                continue
            hp, mx, st, fnt = parse_hp(p.get("condition", "100/100"))
            mon.hp, mon.fainted = (0, True) if fnt else (hp, False)
            if mx:
                mon.max_hp = mx
            mon.status = st
            mon.item = to_id(p.get("item", ""))
            mon.ability = to_id(p.get("ability", "") or p.get("baseAbility", "")) or mon.ability
            if p.get("active"):
                mon.brought = True
            sp = details_species(p.get("details", ""))
            if sp and sp != mon.species:
                mon.species = sp
                if "mega" in sp:
                    mon.is_mega = True
                    s.mega_used = True
        if req.get("teamPreview"):
            return
        # Brought Pokemon are the ones listed after preview (PS drops the rest).
        if not req.get("teamPreview") and side.get("pokemon"):
            listed = {self.pool.base_id(details_species(p["details"])) for p in side["pokemon"]}
            if len(side["pokemon"]) <= 4:
                for m in s.mons:
                    m.brought = m.base in listed
        # Active slots in order.
        actives = [p for p in side.get("pokemon", []) if p.get("active")]
        for m in s.mons:
            if m.slot >= 0 and not any(self.find_mon(self.me, p["ident"], p.get("details")) is m for p in actives):
                pass
        for k, p in enumerate(req.get("active", []) or []):
            if k >= len(actives):
                break
            mon = self.find_mon(self.me, actives[k]["ident"], actives[k].get("details"))
            if mon is None:
                continue
            mon.slot = k
            pps = []
            for mv in p.get("moves", []):
                pps.append(mv.get("pp"))
            if len(pps) == len(mon.set["moves"]):
                mon.pp = [x if x is not None else 0 for x in pps]

    # ---- snapshot ------------------------------------------------------------------------------

    def snapshot(self) -> dict:
        req = self.request or {}
        me = self.me
        sides_out = []
        for sid in ("p1", "p2"):
            s = self.sides[sid]
            mons = []
            for m in s.mons:
                if m.set is None:
                    # Unknown species (not in the pool): borrow any set so the
                    # engine can run; everything stays unrevealed.
                    m.set = copy.deepcopy(self.pool.teams[0]["mons"][0])
                    m.max_hp = m.set["stats"][0]
                own = sid == me
                mask = 0
                for k, mv in enumerate(m.set["moves"]):
                    if own or mv in m.revealed_moves:
                        mask |= 1 << k
                vol = dict(m.vol)
                vol.update({
                    "move_actions": m.move_actions,
                    "active_turns": m.active_turns,
                    "times_attacked": m.times_attacked,
                })
                if m.last_move:
                    vol["last_move"] = m.last_move
                if m.stall_dur > 0:
                    vol["stall"] = [m.stall_counter, m.stall_dur]
                max_hp = m.set["stats"][0] if not own else (m.max_hp or m.set["stats"][0])
                hp = m.hp if own else (0 if m.fainted else max(1, round(m.pct * max_hp)))
                mons.append({
                    "set": m.set,
                    "hp": int(hp),
                    "status": m.status,
                    "boosts": m.boosts,
                    "slot": m.slot,
                    "fainted": m.fainted,
                    "brought": m.brought or (own and m.slot >= 0),
                    "is_mega": m.is_mega,
                    "item": (m.item if (own or m.item_known) and m.item is not None else None),
                    "ability": (m.ability if (own or m.ability_known) else None),
                    "pp": m.pp if own else None,
                    "reveal": {"seen": m.seen or own, "moves": mask, "item": m.item_known or own, "ability": m.ability_known or own},
                    "volatiles": vol,
                    "ate_berry": m.ate_berry,
                })
            sides_out.append({
                "sheet_open": s.sheet_open,
                "mega_used": s.mega_used,
                "total_fainted": s.total_fainted,
                "conds": s.conds,
                "mons": mons,
            })
        viewer = 0 if me == "p1" else 1
        phase = "move"
        switch_slots = [[False, False], [False, False]]
        if req.get("teamPreview"):
            phase = "preview"
        elif req.get("forceSwitch"):
            phase = "switch"
            fs = list(req["forceSwitch"]) + [False, False]
            switch_slots[viewer] = [bool(fs[0]), bool(fs[1])]
        return {"viewer": viewer, "turn": max(1, self.turn), "phase": phase, "switch_slots": switch_slots,
                "field": self.field, "sides": sides_out}

    def snapshot_json(self) -> str:
        return json.dumps(self.snapshot())
