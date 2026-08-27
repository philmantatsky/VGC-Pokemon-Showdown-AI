"""What moves and abilities actually DO, as features.

The stock encoder gives a move its power, accuracy, category, type, target, priority
and a few numeric knobs -- and nothing about its effect. To the network, Tailwind is a
0-power Flying status move, mechanically indistinguishable from any other 0-power
Flying status move except through a learned ID embedding. Same for Sunny Day, Swords
Dance, Thunder Wave and Protect. That is a large part of why the bot plays the niches
badly.

Showdown's move data IS fully structured (`boosts`, `status`, `weather`, `terrain`,
`sideCondition`, `volatileStatus`, `secondary`, `flags`, ...), so this module turns it
into a fixed-width vector. Every value is static per move id, so the whole table is
built once at import and lookup at encode time is a dict hit -- effectively free.

Abilities are NOT structured: Showdown implements them as JS event handlers, exposing
only `num`, `rating` and a couple of flags. So they are grouped BY EFFECT the way the
Laplace bot does it (value_features.py `_ABILITY_FLAGS`), which means one feature
generalises across abilities the bot has never seen rather than memorising names.
"""

from __future__ import annotations

import numpy as np
from poke_env.data import GenData, to_id_str

_GEN = GenData.from_gen(9)

# --- what we encode -----------------------------------------------------------
_STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")
_WEATHERS = ("sunnyday", "raindance", "sandstorm", "snowscape", "hail")
_TERRAINS = ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain")
# side conditions that matter competitively in VGC doubles
_SIDE_CONDITIONS = (
    "tailwind",
    "reflect",
    "lightscreen",
    "auroraveil",
    "safeguard",
    "mist",
    "stealthrock",
    "spikes",
    "toxicspikes",
    "stickyweb",
    "wideguard",
    "quickguard",
)
_PSEUDO = ("trickroom", "gravity", "magicroom", "wonderroom")
# volatiles worth distinguishing (the long tail is noise)
_VOLATILES = (
    "protect",
    "substitute",
    "taunt",
    "encore",
    "disable",
    "leechseed",
    "confusion",
    "yawn",
    "attract",
    "torment",
    "followme",
    "ragepowder",
    "helpinghand",
    "endure",
    "flinch",
    "curse",
    "aquaring",
    "magnetrise",
)
_STAT_ORDER = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_FLAGS = (
    "contact",
    "protect",
    "sound",
    "powder",
    "punch",
    "bite",
    "pulse",
    "bullet",
    "reflectable",
    "bypasssub",
    "slicing",
    "wind",
    "heal",
)

MOVE_SEM_LEN = (
    len(_STATUSES)
    + len(_WEATHERS)
    + len(_TERRAINS)
    + len(_SIDE_CONDITIONS)
    + len(_PSEUDO)
    + len(_VOLATILES)
    + len(_STAT_ORDER) * 2  # boosts self + target
    + len(_FLAGS)
    + 8  # heal, drain, recoil, multihit, secondary chance, forceSwitch,
    # selfSwitch, stalling
)


def _frac(value) -> float:
    """Showdown writes fractions as [num, den]."""
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[1]:
        return float(value[0]) / float(value[1])
    return 0.0


def _encode_move(entry: dict) -> np.ndarray:
    v = np.zeros(MOVE_SEM_LEN, dtype=np.float32)
    i = 0

    def onehot(value, table):
        nonlocal i
        val = to_id_str(str(value)) if value else None
        for name in table:
            if val == name:
                v[i] = 1.0
            i += 1

    secondary = entry.get("secondary") or {}
    if not isinstance(secondary, dict):
        secondary = {}

    # status: direct or via a 100%-ish secondary
    status = entry.get("status") or secondary.get("status")
    onehot(status, _STATUSES)
    onehot(entry.get("weather"), _WEATHERS)
    onehot(entry.get("terrain"), _TERRAINS)
    onehot(entry.get("sideCondition"), _SIDE_CONDITIONS)
    onehot(entry.get("pseudoWeather"), _PSEUDO)

    vol = entry.get("volatileStatus") or secondary.get("volatileStatus")
    onehot(vol, _VOLATILES)

    # boosts: which stats and by how much, signed, /6 to match the boost scale
    self_boosts = (entry.get("self") or {}).get("boosts") or {}
    target_boosts = entry.get("boosts") or {}
    if entry.get("target") == "self" and target_boosts and not self_boosts:
        self_boosts, target_boosts = target_boosts, {}
    sec_self = (secondary.get("self") or {}).get("boosts") or {}
    for stat in _STAT_ORDER:
        v[i] = float(self_boosts.get(stat, sec_self.get(stat, 0))) / 6.0
        i += 1
    sec_boosts = secondary.get("boosts") or {}
    for stat in _STAT_ORDER:
        v[i] = float(target_boosts.get(stat, sec_boosts.get(stat, 0))) / 6.0
        i += 1

    flags = entry.get("flags") or {}
    for flag in _FLAGS:
        v[i] = float(bool(flags.get(flag)))
        i += 1

    v[i] = _frac(entry.get("heal"))
    i += 1
    v[i] = _frac(entry.get("drain"))
    i += 1
    v[i] = _frac(entry.get("recoil"))
    i += 1
    mh = entry.get("multihit")
    v[i] = (
        float(sum(mh)) / len(mh) / 5.0
        if isinstance(mh, (list, tuple))
        else float(mh) / 5.0
        if mh
        else 0.0
    )
    i += 1
    v[i] = float(secondary.get("chance", 0)) / 100.0
    i += 1
    v[i] = float(bool(entry.get("forceSwitch")))
    i += 1
    v[i] = float(bool(entry.get("selfSwitch")))
    i += 1
    v[i] = float(bool(entry.get("stallingMove")))
    i += 1
    assert i == MOVE_SEM_LEN, (i, MOVE_SEM_LEN)
    return v


MOVE_SEMANTICS: dict[str, np.ndarray] = {
    to_id_str(name): _encode_move(entry) for name, entry in _GEN.moves.items()
}
_ZERO_MOVE = np.zeros(MOVE_SEM_LEN, dtype=np.float32)


def move_semantics(move_id: str) -> np.ndarray:
    return MOVE_SEMANTICS.get(to_id_str(move_id), _ZERO_MOVE)


# --- abilities: grouped by EFFECT, not name -----------------------------------
# Showdown exposes no machine-readable ability semantics, so these are curated the
# way Laplace does it: one flag per mechanical effect, so an unseen ability in a
# known group still lands on the right feature.
ABILITY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("drought", "desolateland", "orichalcumpulse"),  # sets sun
    ("drizzle", "primordialsea"),  # sets rain
    ("sandstream", "sandspit"),  # sets sand
    ("snowwarning",),  # sets snow
    ("electricsurge", "hadronengine"),  # electric terrain
    ("grassysurge", "mistysurge", "psychicsurge"),  # other terrains
    ("intimidate",),  # drops foe attack
    ("levitate", "eartheater"),  # ground immunity
    (
        "waterabsorb",
        "dryskin",
        "stormdrain",
        "voltabsorb",
        "lightningrod",
        "flashfire",
        "sapsipper",
        "wellbakedbody",
        "motordrive",
    ),  # type immunity/absorb
    (
        "thickfat",
        "multiscale",
        "furcoat",
        "icescales",
        "fluffy",
        "heatproof",
    ),  # damage reduction
    (
        "protosynthesis",
        "quarkdrive",
        "speedboost",
        "unburden",
        "chlorophyll",
        "swiftswim",
        "sandrush",
        "slushrush",
    ),  # speed gain
    ("hugepower", "purepower", "gorillatactics", "guts", "hustle"),  # attack boost
    ("regenerator", "naturalcure"),  # switch healing
    ("moldbreaker", "teravolt", "turboblaze"),  # ignores abilities
    ("prankster", "galewings", "triage"),  # priority
    ("magicguard", "overcoat", "immunity", "waterveil"),  # chip immunity
    ("disguise", "iceface", "sturdy"),  # one-hit survival
    ("defiant", "competitive", "justified", "angershell"),  # boosts on trigger
    ("goodasgold", "magicbounce", "clearbody", "whitesmoke"),  # status/boost immunity
    ("commander", "asoneglastrier", "asonespectrier"),  # signature
)
ABILITY_SEM_LEN = len(ABILITY_GROUPS) + 1  # + a rating scalar

_ABILITY_INDEX: dict[str, int] = {}
for _gi, _group in enumerate(ABILITY_GROUPS):
    for _name in _group:
        _ABILITY_INDEX[_name] = _gi

_ZERO_ABILITY = np.zeros(ABILITY_SEM_LEN, dtype=np.float32)

try:
    import json as _json
    from pathlib import Path as _Path

    _ABILITY_RATINGS: dict[str, float] = {
        k: float(v)
        for k, v in _json.loads(
            (
                _Path(__file__).resolve().parents[2] / "data" / "ability_ratings.json"
            ).read_text()
        ).items()
    }
except Exception:
    _ABILITY_RATINGS = {}


def ability_semantics(ability_id: str | None) -> np.ndarray:
    if not ability_id:
        return _ZERO_ABILITY
    key = to_id_str(ability_id)
    v = np.zeros(ABILITY_SEM_LEN, dtype=np.float32)
    idx = _ABILITY_INDEX.get(key)
    if idx is not None:
        v[idx] = 1.0
    # Showdown's own usefulness rating (0-5), extracted to data/ability_ratings.json:
    # a weak but free signal for the long tail of abilities that fit no group.
    # poke-env's GenData has no abilities table, so this cannot come from _GEN.
    v[-1] = _ABILITY_RATINGS.get(key, 0.0) / 5.0
    return v
