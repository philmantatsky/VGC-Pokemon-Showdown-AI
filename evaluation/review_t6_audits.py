"""Index contextual tactical review signals, NOT automatic move-error verdicts."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict


def scan_battle(record):
    role = record["our_role"]
    turn, weather = 0, "none"
    hp, moves, boosts = {}, defaultdict(list), defaultdict(list)
    signals = []
    last_move = None
    for message in record["protocol"]:
        if len(message) < 2:
            continue
        kind = message[1]
        if kind == "turn":
            turn = int(message[2])
        elif kind == "-weather":
            weather = message[2]
        elif kind in ("switch", "drag", "-damage", "-heal"):
            condition = message[4] if kind in ("switch", "drag") else message[3]
            # Public HP may carry Showdown's color suffix, e.g. 50/100g.
            matched = re.match(r"^(\d+)/(\d+)", condition)
            if matched and int(matched[2]) > 0:
                hp[message[2]] = int(matched[1]) / int(matched[2])
            elif condition.split()[0] == "0":
                hp[message[2]] = 0.0
        elif kind == "move":
            actor, move = message[2:4]
            last_move = (actor, move, turn)
            if actor.startswith(role):
                moves[turn].append(move)
                if move == "Weather Ball" and weather == "none":
                    signals.append(
                        {
                            "turn": turn,
                            "signal": "weather_ball_without_weather",
                            "actor": actor,
                        }
                    )
                if move in ("Water Spout", "Eruption") and hp.get(actor, 1) < 0.4:
                    signals.append(
                        {
                            "turn": turn,
                            "signal": "low_hp_scaling_move",
                            "actor": actor,
                            "hp": hp[actor],
                            "move": move,
                        }
                    )
                if move in ("Water Spout", "Water Pulse") and weather == "SunnyDay":
                    signals.append(
                        {
                            "turn": turn,
                            "signal": "water_attack_in_sun",
                            "actor": actor,
                            "move": move,
                        }
                    )
        elif kind == "-boost" and not message[2].startswith(role):
            if int(message[4]) >= 2:
                boosts[turn].append((message[2], message[3], int(message[4])))
        elif kind == "-fail" and last_move and last_move[0].startswith(role):
            if (
                last_move[1] in ("Tailwind", "Trick Room", "Protect")
                and last_move[2] == turn
            ):
                signals.append(
                    {
                        "turn": turn,
                        "signal": "failed_support_move",
                        "actor": last_move[0],
                        "move": last_move[1],
                    }
                )
    for t, our_moves in moves.items():
        if sum(m in ("Protect", "Detect") for m in our_moves) >= 2 and boosts[t]:
            signals.append(
                {
                    "turn": t,
                    "signal": "double_protect_with_enemy_setup",
                    "enemy_boosts": boosts[t],
                }
            )
    return signals


def review(root):
    groups = {}
    for group in sorted(root.iterdir()) if root.exists() else []:
        if not group.is_dir():
            continue
        counts, flagged, games, losses = Counter(), [], 0, 0
        for path in sorted(group.glob("*/*.json")):
            record = json.loads(path.read_text())
            if "protocol" not in record:
                continue
            games += 1
            losses += not record["won"]
            signals = scan_battle(record)
            counts.update(s["signal"] for s in signals)
            if signals:
                flagged.append(
                    {"path": str(path), "won": record["won"], "signals": signals}
                )
        groups[group.name] = {
            "sampled_games": games,
            "sampled_losses": losses,
            "signal_counts": dict(counts),
            "flagged_games": flagged,
        }
    return {
        "caution": (
            "Signals identify positions for review, not proven mistakes. "
            "Check legal alternatives and opponent threats."
        ),
        "sampling": (
            "First roster per category and sheet mode; "
            "not a random sample or a loss-rate estimate."
        ),
        "groups": groups,
    }
