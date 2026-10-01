"""Set-swap audit over our T6 ladder games (2026-09-28, the user's question:
"check our ladder logs of losses / wins where our incineroar wouldve done better
with chople berry and do the rest with the other pokemon too").

For each change proposed in TEAM_REVIEW_T6.md section 4, the moments in our own
ladder games where it would have mattered -- both ways:

* Incineroar, Chople Berry instead of Passho Berry: the first super-effective
  Fighting hit on our Incineroar each game (Chople halves it; a KO is re-scored
  with the damage calculator) and every Passho activation (what the berry saved).
* Torkoal (Earth Power for Heat Wave), Venusaur (Earth Power for Leaf Storm),
  Charizard (Weather Ball for Solar Beam): at every turn it attacked, the best move
  of the current set vs the best move of the proposed set, both scored by the real
  damage calculator on the rebuilt position (guaranteed KOs first, then damage).
  This judges the moveset, not the bot's choice that turn; moves are scored at the
  moment they execute (so an HP-based move is scored at the HP left after earlier
  hits that turn -- hindsight the bot did not have when it chose).
* Farigiraf, Protect instead of Helping Hand (low confidence in the review): every
  Helping Hand whose partner then knocked a foe out, re-scored without the boost,
  and every turn Farigiraf fainted before it acted.

Positions come from unit_tests/ladder_position.py (our six with the Champions stats
the server sends, the bot as p1). Opponents without an open sheet get
vgc_knowledge's standard spread (32 HP, 32 in the better attack, neutral nature).
Our HP is exact in the logs, the opponent's a percentage.
Run from the repo root:
  .venv/bin/python results_analysis/openings_20260927/setswap_audit.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
logging.disable(logging.CRITICAL)

from poke_env.battle import Move  # noqa: E402
from poke_env.data import to_id_str  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402
from unit_tests.ladder_position import position  # noqa: E402
from vgc_bench.src import vgc_knowledge as K  # noqa: E402
from vgc_bench.src.playbook import set_share  # noqa: E402

OUR = "antonius1"
DIRS = [
    "ladder_replays_mc_deployed_T6",
    "ladder_replays_mc_deployed_T6_humanpreview1",
    "ladder_replays_mc_deployed_T6_humanpreview1_attackcheck",
    "ladder_replays_mc_deployed_T6_humanpreview1_throatchop",
    "ladder_replays_mc_deployed_T6_humanpreview1_wideguard",
    "ladder_replays_mc_deployed_T6ctx",
    "ladder_replays_mc_deployed_T6tac",
    "ladder_replays_mc_deployed_T6tac_playbook_t6",
    "ladder_replays_mc_practice1_water_t6",
]
OLD = {
    "Torkoal": ["eruption", "heatwave", "weatherball"],
    "Venusaur": ["sludgebomb", "leafstorm"],
    "Charizard": ["heatwave", "solarbeam", "ancientpower"],
}
NEW = {
    "Torkoal": ["eruption", "earthpower", "weatherball"],
    "Venusaur": ["sludgebomb", "earthpower"],
    "Charizard": ["heatwave", "weatherball", "ancientpower"],
}
MARGIN = 0.20  # damage edge (share of a foe's HP) that counts as "clearly better"
SKIP = {
    "j",
    "l",
    "n",
    "c",
    "raw",
    "html",
    "uhtml",
    "uhtmlchange",
    "t:",
    "player",
    "gametype",
    "gen",
    "tier",
    "rated",
    "rule",
    "clearpoke",
    "poke",
    "teampreview",
    "showteam",
    "teamsize",
    "inactive",
    "inactiveoff",
    "start",
    "title",
    "join",
    "leave",
    "chat",
    "",
    "bigerror",
    "error",
    "win",
    "tie",
    "-message",
    "message",
    "debug",
    "seed",
    "timestamp",
}
ACTION = {"move", "switch", "turn", "cant", "drag", "detailschange", "-mega", "upkeep"}
SWAP = re.compile(r"\bp([12])(?=[ab:|,\]]|$)")


def swap(line: str) -> str:
    return SWAP.sub(lambda m: "p" + ("2" if m.group(1) == "1" else "1"), line)


def battle_lines(log: str, we_are_p2: bool) -> list[str]:
    out, started = [], False
    for raw in log.split("\n"):
        if not raw.startswith("|"):
            continue
        tag = raw.split("|")[1]
        if tag == "start":
            started = True
            continue
        if started and tag not in SKIP:
            out.append(swap(raw) if we_are_p2 else raw)
    return out


MOVES: dict[str, Move] = {}


def mv(move_id: str) -> Move:
    if move_id not in MOVES:
        MOVES[move_id] = Move(move_id, gen=9)
    return MOVES[move_id]


def active(battle, ident: str):
    actives = (
        battle.active_pokemon if ident[:2] == "p1" else battle.opponent_active_pokemon
    )
    idx = {"a": 0, "b": 1}.get(ident[2:3])
    if idx is not None and idx < len(actives):
        return actives[idx]
    return None


def after_move(lines: list[str], i: int) -> list[list[str]]:
    """The effect lines of the move on line i (up to the next action)."""
    out = []
    for line in lines[i + 1 :]:
        parts = line.split("|")
        if parts[1] in ACTION:
            break
        out.append(parts)
    return out


SASH_SHARE = 0.3  # "likely holds a Focus Sash" when the item is not revealed


def sash_likely(b, foe) -> bool:
    """A full-HP foe that probably survives any one hit at 1 HP."""
    if (foe.current_hp_fraction or 0.0) < 0.999:
        return False
    item = to_id_str(foe.item or "")
    if item == "focussash":
        return True
    if item and item != "unknownitem":
        return False  # revealed, not a Sash (or already used)
    return set_share(foe.species, item="focussash", formatid=b.format) >= SASH_SHARE


def move_value(b, me, foes, move_id: str):
    """(guaranteed KOs, possible KOs, damage as a share of foe HP) of one move;
    a likely Focus Sash turns a one-hit KO from full HP into 1 HP left."""
    m = mv(move_id)
    per = []
    for foe in foes:
        fr = K.damage_fraction(b, me, foe, m)
        if fr is None:
            continue
        hp = foe.current_hp_fraction or 0.0
        if sash_likely(b, foe):
            per.append((min((fr[0] + fr[1]) / 2, hp - 0.01), False, False, foe.species))
            continue
        per.append(
            (
                min((fr[0] + fr[1]) / 2, hp),
                fr[0] >= hp > 0,
                fr[1] >= hp > 0,
                foe.species,
            )
        )
    if not per:
        return None
    if m.target.name in ("ALL_ADJACENT_FOES", "ALL_ADJACENT"):
        return (
            sum(p[1] for p in per),
            sum(p[2] for p in per),
            sum(p[0] for p in per),
            "both",
        )
    best = max(per, key=lambda p: (p[1], p[0]))
    return (int(best[1]), int(best[2]), best[0], best[3])


def best_of(b, me, foes, moves: list[str]):
    scored = [(move_value(b, me, foes, m), m) for m in moves]
    scored = [(v, m) for v, m in scored if v is not None]
    if not scored:
        return None
    (ko, maybe, dmg, target), move = max(scored, key=lambda s: (s[0][0], s[0][2]))
    return {
        "move": move,
        "ko": ko,
        "maybe": maybe,
        "dmg": round(dmg, 3),
        "target": target,
    }


def farigiraf_audit(lines: list[str], ctx: dict, report: dict, fail: Counter) -> None:
    """Helping Hand turns whose partner scored a KO (did the KO need the boost?)
    and turns Farigiraf fainted before acting."""
    turn, acted = 0, set()
    for i, line in enumerate(lines):
        parts = line.split("|")
        if parts[1] == "turn":
            turn, acted = int(parts[2]), set()
            continue
        if parts[1] in ("move", "cant") and parts[2].startswith("p1"):
            acted.add(parts[2][:3])
        if (
            parts[1] == "faint"
            and parts[2].startswith("p1")
            and "Farigiraf" in parts[2]
        ):
            if turn > 0 and parts[2][:3] not in acted:
                report["farigiraf_fainted_before_acting"].append(dict(ctx, turn=turn))
        if not (
            parts[1] == "move"
            and parts[2].startswith("p1")
            and "Farigiraf" in parts[2]
            and parts[3] == "Helping Hand"
        ):
            continue
        for j in range(i + 1, len(lines)):
            q = lines[j].split("|")
            if q[1] in ("turn", "upkeep"):
                break
            if not (
                q[1] == "move" and q[2].startswith("p1") and "Farigiraf" not in q[2]
            ):
                continue
            m = mv(to_id_str(q[3]))
            if m.category.name == "STATUS":
                break
            kos = [
                x.split("|")[2][:3]
                for x in lines[j + 1 : j + 10]
                if x.startswith("|-damage|p2") and "|0 fnt" in x and "[from]" not in x
            ]
            needed: list[str] = []
            row = dict(
                ctx, turn=turn, partner_move=q[3], kos=len(kos), needed_hh=needed
            )
            try:
                b = position(lines[:i])  # before Helping Hand: no boost
                me = active(b, q[2][:3])
                for t in kos:
                    foe = active(b, t)
                    fr = K.damage_fraction(b, me, foe, m) if foe is not None else None
                    hp = foe.current_hp_fraction if foe is not None else 1.0
                    needed.append(
                        "unknown"
                        if fr is None
                        else "no"
                        if fr[0] >= hp
                        else "yes"
                        if fr[1] < hp
                        else "maybe"
                    )
            except Exception as e:
                fail[f"farigiraf:{type(e).__name__}"] += 1
            report["farigiraf_helping_hand"].append(row)
            break


def main() -> None:
    report: dict[str, list] = defaultdict(list)
    fail: Counter = Counter()
    games = 0
    for d in DIRS:
        for f in sorted((REPO / d).glob("*.html")):
            log = extract_log(f)
            if not log:
                continue
            games += 1
            we_p2 = f"|player|p2|{OUR}" in log
            won = f"|win|{OUR}" in log
            gid_match = re.search(r"regmc-(\d+)", f.name)
            gid = gid_match.group(1) if gid_match else f.name
            lines = battle_lines(log, we_p2)
            turn = 0
            chople_used = False
            for i, line in enumerate(lines):
                parts = line.split("|")
                if parts[1] == "turn":
                    turn = int(parts[2])
                    continue
                if parts[1] != "move" or "[from]" in line:
                    continue
                user, move_name = parts[2], parts[3]
                move_id = to_id_str(move_name)
                m = mv(move_id)
                if m.category.name == "STATUS":
                    continue
                ctx = {
                    "game": gid,
                    "set": d.replace("ladder_replays_mc_", ""),
                    "won": won,
                    "turn": turn,
                }
                effects = after_move(lines, i)

                # ---------------- A. hits on our Incineroar
                if user.startswith("p2"):
                    dmg = [
                        e
                        for e in effects
                        if e[1] == "-damage"
                        and e[2].startswith("p1")
                        and "Incineroar" in e[2]
                        and len(e) < 5
                    ]
                    passho = any(
                        e[1] == "-enditem" and "Passho" in "|".join(e) for e in effects
                    )
                    fighting = m.type.name == "FIGHTING" and move_id != "finalgambit"
                    if not dmg or not (passho or (fighting and not chople_used)):
                        continue
                    try:
                        b = position(lines[:i])
                        inc = active(b, dmg[0][2][:3])
                        att = active(b, user[:3])
                        before = inc.current_hp or 0
                        maxhp = inc.stats["hp"] or 1
                        est = K.damage_fraction(b, att, inc, m) if fighting else None
                    except Exception as e:  # counted, never fatal
                        fail[f"incineroar:{type(e).__name__}"] += 1
                        continue
                    hp_s = dmg[0][3].split()[0]
                    ko = hp_s.startswith("0")
                    after = 0 if ko else int(hp_s.split("/")[0])
                    row = dict(
                        ctx,
                        attacker=getattr(att, "species", user),
                        move=move_name,
                        before=before,
                        after=after,
                        maxhp=maxhp,
                        ko=ko,
                    )
                    if fighting and not chople_used:
                        chople_used = True
                        if ko and est:
                            lo, hi = est[0] * maxhp / 2, est[1] * maxhp / 2
                            left = f"{int(before - hi)}-{int(before - lo)}"
                            row["with_chople"] = (
                                f"survives (~{left} HP left)"
                                if hi < before
                                else "maybe survives"
                                if lo < before
                                else "still KO'd"
                            )
                            row["est_damage"] = [
                                round(est[0] * maxhp),
                                round(est[1] * maxhp),
                            ]
                        elif ko:
                            row["with_chople"] = "KO (no estimate)"
                        else:
                            row["with_chople"] = f"+{(before - after) // 2} HP"
                        report["incineroar_fighting"].append(row)
                    if passho:
                        halved = before - after
                        row = dict(row)
                        row["without_passho"] = (
                            "KO'd anyway"
                            if ko
                            else "would have been KO'd"
                            if 2 * halved >= before
                            else f"survives at {before - 2 * halved}/{maxhp}"
                        )
                        report["incineroar_passho"].append(row)
                    continue

                # ---------------- B/C/D. our Torkoal / Venusaur / Charizard attacking
                who = next(
                    (sp for sp in OLD if user.startswith("p1") and sp in user), None
                )
                if who is None:
                    continue
                try:
                    b = position(lines[:i])
                    me = active(b, user[:3])
                    foes = [
                        x
                        for x in b.opponent_active_pokemon
                        if x is not None and not x.fainted
                    ]
                    old = best_of(b, me, foes, OLD[who])
                    new = best_of(b, me, foes, NEW[who])
                except Exception as e:
                    fail[f"{who}:{type(e).__name__}"] += 1
                    continue
                if old is None or new is None:
                    fail[f"{who}:no-calc"] += 1
                    continue
                if new["ko"] > old["ko"] or new["dmg"] >= old["dmg"] + MARGIN:
                    verdict = "new set better"
                elif old["ko"] > new["ko"] or old["dmg"] >= new["dmg"] + MARGIN:
                    verdict = "old set better"
                else:
                    verdict = "same"
                weather = next(iter(b.weather), None)
                real = {
                    e[2][:3]: e[3].split()[0]
                    for e in effects
                    if e[1] == "-damage" and e[2].startswith("p2") and len(e) < 5
                }
                report[who.lower()].append(
                    dict(
                        ctx,
                        used=move_name,
                        hp=round(me.current_hp_fraction, 2),
                        weather=weather.name if weather else "none",
                        foes=[x.species for x in foes],
                        old=old,
                        new=new,
                        verdict=verdict,
                        real=real,
                    )
                )
            game_ctx = {
                "game": gid,
                "set": d.replace("ladder_replays_mc_", ""),
                "won": won,
            }
            farigiraf_audit(lines, game_ctx, report, fail)
    out = Path(__file__).with_name("setswap_audit.json")
    json.dump(report, out.open("w"), indent=1, default=str)
    text = summary(report, games, fail)
    out.with_suffix(".txt").write_text(text)
    print(text)


def _row(x: dict, how: str) -> str:
    return (
        f"  {'W' if x['won'] else 'L'} {x['game']} T{x['turn']} {x['attacker']} "
        f"{x['move']}: {x['before']}->{x['after']}/{x['maxhp']} -> {x[how]}"
    )


def summary(report: dict, games: int, fail: Counter) -> str:
    """The readable version: every KO-changing moment, both ways."""
    w = [f"{games} T6-era ladder games; rebuild failures {dict(fail)}"]
    w.append("\n== Incineroar: first super-effective Fighting hit (Chople halves it)")
    w += [_row(x, "with_chople") for x in report["incineroar_fighting"]]
    w.append("== Incineroar: Passho activations (without Passho)")
    w += [_row(x, "without_passho") for x in report["incineroar_passho"]]
    for who in ("torkoal", "venusaur", "charizard"):
        rows = report[who]
        v = Counter(x["verdict"] for x in rows)
        w.append(f"\n== {who}: {len(rows)} attacking turns, {dict(v)}")
        for tag, side, other in (("NEW", "new", "old"), ("OLD", "old", "new")):
            hits = [x for x in rows if x[side]["ko"] > x[other]["ko"]]
            w.append(f"  only the {tag} set had a guaranteed KO: {len(hits)} turns")
            for x in hits:
                w.append(
                    f"    {'W' if x['won'] else 'L'} {x['game']} T{x['turn']} "
                    f"{x['weather']} vs {'/'.join(x['foes'])}: used {x['used']}; "
                    f"old {x['old']['move']} -> {x['old']['target']}, "
                    f"new {x['new']['move']} -> {x['new']['target']}"
                )
    hh = report["farigiraf_helping_hand"]
    need = Counter(n for x in hh for n in x["needed_hh"])
    yes = [x for x in hh if "yes" in x["needed_hh"]]
    w.append(
        f"\n== Farigiraf: {len(hh)} Helping Hands before a partner attack; KOs "
        f"needing the boost {dict(need)}; decisive ones won "
        f"{sum(x['won'] for x in yes)} / lost {sum(not x['won'] for x in yes)}"
    )
    fb = report["farigiraf_fainted_before_acting"]
    w.append(
        f"   fainted before acting: {len(fb)} ({sum(x['won'] for x in fb)} in wins)"
    )
    return "\n".join(w) + "\n"


if __name__ == "__main__":
    main()
