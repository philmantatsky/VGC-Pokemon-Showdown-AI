//! Build a `Battle` from an observed-state snapshot (JSON), for live play.
//!
//! The live client tracks the Showdown protocol and our request JSON, and
//! describes the battle as:
//!
//! ```json
//! {"viewer": 0, "turn": 3, "phase": "move" | "switch" | "switch_midturn" | "preview",
//!  "switch_slots": [[false, true], [false, false]],
//!  "field": {"weather": "rain", "weather_turns": 3, "terrain": "grassy", "terrain_turns": 2,
//!            "trick_room": 0, "gravity": 0},
//!  "sides": [{"sheet_open": false, "mega_used": false, "total_fainted": 0,
//!             "conds": {"tailwind": 0, "reflect": 0, ...},
//!             "mons": [{"set": {compiled set, as in teams_regmc.json},
//!                       "hp": 120, "status": "brn", "status_turns": 0,
//!                       "boosts": {"atk": 1}, "slot": 0, "fainted": false,
//!                       "brought": true, "is_mega": false,
//!                       "item": "sitrusberry" | "" | null, "ability": "intimidate" | null,
//!                       "pp": [8, 16, 8, 12],
//!                       "reveal": {"seen": true, "moves": 5, "item": false, "ability": true},
//!                       "volatiles": {"taunt": 2, "encore": [2, "protect"], ...}}, ...]}, ...]}
//! ```
//!
//! For the opponent, `set` is a placeholder (a plausible tournament set,
//! adjusted to revealed moves/items/abilities); the observation encoder hides
//! everything unrevealed, and search resamples it (`worlds::determinize`).

use serde_json::Value;

use crate::battle::{Battle, BattleConfig, Phase};
use crate::dex::{dex, Status, Terrain, Weather, B_ACC, B_ATK, B_DEF, B_EVA, B_SPA, B_SPD, B_SPE};
use crate::state::{Mon, SemiInv, NO_MON};
use crate::teams::{parse_set, set_support, SetSpec, TeamSpec};

fn u(v: &Value, k: &str) -> u64 {
    v.get(k).and_then(|x| x.as_u64()).unwrap_or(0)
}
fn i(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0)
}
fn bo(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}
fn st(v: &Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
}

fn status_of(s: &str) -> Status {
    match s {
        "brn" => Status::Brn,
        "par" => Status::Par,
        "slp" => Status::Slp,
        "frz" => Status::Frz,
        "psn" => Status::Psn,
        "tox" => Status::Tox,
        _ => Status::None,
    }
}

fn weather_of(s: &str) -> Weather {
    match s {
        "rain" | "raindance" | "RainDance" => Weather::Rain,
        "sun" | "sunnyday" | "SunnyDay" => Weather::Sun,
        "sand" | "sandstorm" | "Sandstorm" => Weather::Sand,
        "snow" | "snowscape" | "Snowscape" | "snow_" => Weather::Snow,
        _ => Weather::None,
    }
}

fn terrain_of(s: &str) -> Terrain {
    match s {
        "grassy" | "grassyterrain" => Terrain::Grassy,
        "psychic" | "psychicterrain" => Terrain::Psychic,
        "electric" | "electricterrain" => Terrain::Electric,
        "misty" | "mistyterrain" => Terrain::Misty,
        _ => Terrain::None,
    }
}

fn move_slot_of(m: &Mon, name: &str) -> Option<usize> {
    let id = dex().move_id(name)?;
    m.move_slot(id)
}

fn apply_volatiles(m: &mut Mon, v: &Value) {
    let Some(o) = v.as_object() else { return };
    let num = |x: &Value| x.as_u64().unwrap_or(0) as u8;
    for (k, x) in o {
        let vol = &mut m.vol;
        match k.as_str() {
            "taunt" => vol.taunt = num(x),
            "confusion" => vol.confusion = num(x).max(1),
            "throat_chop" => vol.throat_chop = num(x),
            "yawn" => vol.yawn = num(x),
            "perish" => vol.perish = num(x),
            "substitute" => vol.substitute = x.as_u64().unwrap_or(0) as u16,
            "leech_seed" => vol.leech_seed = x.as_bool().unwrap_or(true),
            "salt_cure" => vol.salt_cure = x.as_bool().unwrap_or(true),
            "curse" => vol.curse = x.as_bool().unwrap_or(true),
            "flash_fire" => vol.flash_fire = x.as_bool().unwrap_or(true),
            "unburden" => vol.unburden = x.as_bool().unwrap_or(true),
            "glaive_rush" => vol.glaive_rush = x.as_bool().unwrap_or(true) as u8,
            "crit_stage" => vol.crit_stage = num(x),
            "tar_shot" => vol.tar_shot = x.as_bool().unwrap_or(true),
            "must_recharge" => vol.must_recharge = x.as_bool().unwrap_or(true),
            "magnet_rise" => vol.magnet_rise = num(x),
            "heal_block" => vol.heal_block = num(x),
            "aqua_ring" => vol.aqua_ring = x.as_bool().unwrap_or(true),
            "ingrain" => vol.ingrain = x.as_bool().unwrap_or(true),
            "smacked_down" => vol.smacked_down = x.as_bool().unwrap_or(true),
            "torment" => vol.torment = x.as_bool().unwrap_or(true),
            "imprison" => vol.imprison = x.as_bool().unwrap_or(true),
            "charge" => vol.charge = x.as_bool().unwrap_or(true),
            "attract" => vol.attract = x.as_bool().unwrap_or(true),
            "no_retreat" => vol.no_retreat = x.as_bool().unwrap_or(true),
            "destiny_bond" => vol.destiny_bond = x.as_bool().unwrap_or(true),
            "active_turns" => vol.active_turns = x.as_u64().unwrap_or(0) as u16,
            "move_actions" => vol.move_actions = x.as_u64().unwrap_or(0) as u16,
            "times_attacked" => vol.times_attacked = num(x),
            "stockpile" => vol.stockpile = num(x),
            "type_changed" => vol.type_changed = x.as_bool().unwrap_or(true),
            "last_move" => {
                if let Some(id) = x.as_str().and_then(|n| dex().move_id(n)) {
                    vol.last_move = id;
                    if let Some(s) = m.move_slot(id) {
                        m.vol.last_move_slot = s as u8;
                    }
                }
            }
            "last_move_failed" => vol.last_move_failed = x.as_bool().unwrap_or(true),
            "stall" => {
                // [counter, turns left]
                if let Some(a) = x.as_array() {
                    vol.stall_counter = a.first().and_then(|y| y.as_u64()).unwrap_or(0) as u16;
                    vol.stall_dur = a.get(1).and_then(|y| y.as_u64()).unwrap_or(0) as u8;
                }
            }
            "encore" | "disable" => {
                if let Some(a) = x.as_array() {
                    let turns = a.first().and_then(|y| y.as_u64()).unwrap_or(0) as u8;
                    let name = a.get(1).and_then(|y| y.as_str()).unwrap_or("");
                    if let Some(s) = move_slot_of(m, name) {
                        if k == "encore" {
                            m.vol.encore = turns;
                            m.vol.encore_slot = s as u8;
                        } else {
                            m.vol.disable = turns;
                            m.vol.disable_slot = s as u8;
                        }
                    }
                }
            }
            "choice_lock" => {
                if let Some(s) = x.as_str().and_then(|n| move_slot_of(m, n)) {
                    m.vol.choice_lock = s as u8 + 1;
                }
            }
            "charging" => {
                if let Some(a) = x.as_array() {
                    let name = a.first().and_then(|y| y.as_str()).unwrap_or("");
                    if let Some(s) = move_slot_of(m, name) {
                        m.vol.charging = s as u8 + 1;
                        m.vol.charge_target = a.get(1).and_then(|y| y.as_i64()).unwrap_or(0) as i8;
                        m.vol.semi_inv = match name {
                            "fly" | "bounce" => SemiInv::Air,
                            "dig" => SemiInv::Underground,
                            "dive" => SemiInv::Underwater,
                            "phantomforce" | "shadowforce" => SemiInv::Shadow,
                            _ => SemiInv::None,
                        };
                    }
                }
            }
            "types" => {
                if let Some(a) = x.as_array() {
                    let t: Vec<_> = a.iter().filter_map(|y| y.as_str().and_then(crate::types::Type::from_name)).collect();
                    if let Some(&t0) = t.first() {
                        m.types = [t0, t.get(1).copied().unwrap_or(crate::types::Type::None)];
                        m.vol.type_changed = true;
                    }
                }
            }
            _ => {}
        }
    }
}

/// Build a battle from a snapshot. `seed` seeds its RNG.
pub fn battle_from_snapshot(v: &Value, seed: u64) -> Result<Battle, String> {
    let d = dex();
    let sides = v.get("sides").and_then(|x| x.as_array()).ok_or("snapshot: no sides")?;
    if sides.len() != 2 {
        return Err("snapshot: need 2 sides".into());
    }
    let mut teams: Vec<TeamSpec> = vec![];
    for s in sides {
        let mons = s.get("mons").and_then(|x| x.as_array()).ok_or("snapshot: side without mons")?;
        if mons.len() != 6 {
            return Err(format!("snapshot: side has {} mons", mons.len()));
        }
        let mut why = vec![];
        let mut sets = [SetSpec::default(); 6];
        for (k, m) in mons.iter().enumerate() {
            sets[k] = parse_set(m.get("set").ok_or("snapshot: mon without set")?, &mut why);
        }
        if !why.is_empty() {
            return Err(format!("snapshot: {}", why.join("; ")));
        }
        let mut unsupported = vec![];
        for set in &sets {
            unsupported.extend(set_support(set));
        }
        teams.push(TeamSpec { name: String::new(), mons: sets, unsupported });
    }
    let sheets = [bo(&sides[0], "sheet_open"), bo(&sides[1], "sheet_open")];
    let cfg = BattleConfig { turn_limit: 1000, ally_damage_targeting: false, log: false, sheets };
    let mut b = Battle::new([&teams[0], &teams[1]], seed, cfg);
    b.turn = u(v, "turn").max(1) as u16;
    // Field.
    if let Some(f) = v.get("field") {
        b.field.weather = weather_of(&st(f, "weather"));
        b.field.weather_turns = if b.field.weather == Weather::None { 0 } else { u(f, "weather_turns").max(1) as u8 };
        b.field.terrain = terrain_of(&st(f, "terrain"));
        b.field.terrain_turns = if b.field.terrain == Terrain::None { 0 } else { u(f, "terrain_turns").max(1) as u8 };
        b.field.trick_room = u(f, "trick_room") as u8;
        b.field.gravity = u(f, "gravity") as u8;
    }
    for (si, s) in sides.iter().enumerate() {
        let side = &mut b.sides[si];
        side.mega_used = bo(s, "mega_used");
        side.total_fainted = u(s, "total_fainted") as u8;
        side.sheet_open = sheets[si];
        if let Some(c) = s.get("conds") {
            side.conds.tailwind = u(c, "tailwind") as u8;
            side.conds.reflect = u(c, "reflect") as u8;
            side.conds.light_screen = u(c, "light_screen") as u8;
            side.conds.aurora_veil = u(c, "aurora_veil") as u8;
            side.conds.safeguard = u(c, "safeguard") as u8;
            side.conds.mist = u(c, "mist") as u8;
            side.conds.stealth_rock = bo(c, "stealth_rock");
            side.conds.spikes = u(c, "spikes") as u8;
            side.conds.toxic_spikes = u(c, "toxic_spikes") as u8;
            side.conds.sticky_web = bo(c, "sticky_web");
        }
        side.active = [NO_MON; 2];
        let mons = s.get("mons").and_then(|x| x.as_array()).unwrap();
        for (k, mv) in mons.iter().enumerate() {
            let m = &mut side.mons[k];
            if bo(mv, "is_mega") && m.can_mega {
                m.species = m.mega_species;
                m.types = m.mega_types;
                m.base_types = m.mega_types;
                for j in 1..6 {
                    m.stats[j] = m.mega_stats[j];
                }
                m.weight_hg = m.mega_weight_hg;
                m.ability = m.mega_ability;
                m.base_ability = m.mega_ability;
                m.is_mega = true;
            }
            if let Some(h) = mv.get("hp").and_then(|x| x.as_u64()) {
                m.hp = (h as u16).min(m.max_hp);
            }
            m.fainted = bo(mv, "fainted") || (mv.get("hp").is_some() && m.hp == 0);
            if m.fainted {
                m.hp = 0;
            }
            m.status = status_of(&st(mv, "status"));
            m.status_turns = u(mv, "status_turns") as u8;
            if let Some(bs) = mv.get("boosts").and_then(|x| x.as_object()) {
                for (bk, bv) in bs {
                    let idx = match bk.as_str() {
                        "atk" => B_ATK,
                        "def" => B_DEF,
                        "spa" => B_SPA,
                        "spd" => B_SPD,
                        "spe" => B_SPE,
                        "accuracy" => B_ACC,
                        "evasion" => B_EVA,
                        _ => continue,
                    };
                    m.boosts[idx] = bv.as_i64().unwrap_or(0).clamp(-6, 6) as i8;
                }
            }
            m.brought = bo(mv, "brought");
            // Current item / ability override (null = keep the set's).
            match mv.get("item") {
                Some(Value::String(it)) => {
                    let id = d.item_id(it).unwrap_or(0);
                    if id == 0 && m.item != 0 {
                        m.last_item = m.item;
                    }
                    m.item = id;
                }
                Some(Value::Null) | None => {}
                _ => {}
            }
            if let Some(ab) = mv.get("ability").and_then(|x| x.as_str()) {
                if let Some(id) = d.ability_id(ab) {
                    m.ability = id;
                }
            }
            if let Some(pp) = mv.get("pp").and_then(|x| x.as_array()) {
                for (j, p) in pp.iter().take(4).enumerate() {
                    if let Some(p) = p.as_u64() {
                        m.pp[j] = (p as u8).min(m.max_pp[j]);
                    }
                }
            }
            if let Some(r) = mv.get("reveal") {
                m.reveal.seen = bo(r, "seen");
                m.reveal.moves = u(r, "moves") as u8;
                m.reveal.item = bo(r, "item");
                m.reveal.ability = bo(r, "ability");
            }
            m.ate_berry = bo(mv, "ate_berry");
            let slot = i(mv, "slot");
            m.slot = -1;
            if (0..2).contains(&slot) && !m.fainted || (0..2).contains(&slot) && bo(mv, "fainted_in_slot") {
                m.slot = slot as i8;
                side.active[slot as usize] = k as u8;
            }
            if let Some(vols) = mv.get("volatiles") {
                apply_volatiles(m, vols);
            }
        }
        // Fainted Pokemon that still occupy a slot (awaiting replacement).
        for (k, mv) in mons.iter().enumerate() {
            let slot = i(mv, "slot");
            if (0..2).contains(&slot) && side.mons[k].fainted && side.active[slot as usize] == NO_MON {
                side.active[slot as usize] = k as u8;
                side.mons[k].slot = slot as i8;
            }
        }
    }
    let phase = st(v, "phase");
    b.phase = match phase.as_str() {
        "preview" => Phase::TeamPreview,
        "switch" => Phase::Switch { midturn: false },
        "switch_midturn" => Phase::Switch { midturn: true },
        _ => Phase::Move,
    };
    if let Some(ss) = v.get("switch_slots").and_then(|x| x.as_array()) {
        for (si, row) in ss.iter().enumerate().take(2) {
            if let Some(r) = row.as_array() {
                for (k, x) in r.iter().enumerate().take(2) {
                    b.switch_slots[si][k] = x.as_bool().unwrap_or(false);
                }
            }
        }
    }
    Ok(b)
}
