//! Battle states exported from Pokemon Showdown, for parity testing.
//!
//! `showdown/parity/lib.js` (`exportState`) writes a Showdown battle's full
//! state as JSON; [`Battle::from_state_json`] rebuilds the equivalent engine
//! battle. [`Battle::outcome_features`] extracts the same comparable features
//! that `lib.js` (`outcome`) extracts from Showdown, and
//! [`Battle::damage_probe`] runs the damage formula alone. Anything in the
//! Showdown state the engine cannot represent is an error, never ignored.

use serde_json::{json, Map, Value};

use crate::actions::{T_ALLY, T_FOE0};
use crate::battle::{Action, ActionKind, Battle, BattleConfig, Phase, Queue};
use crate::dex::{dex, Category, Status, Target, Terrain, Weather};
use crate::kinds::{Ab, It};
use crate::state::{Pos, ProtectKind, SemiInv, NO_MON};
use crate::teams::{parse_team, TeamSpec};
use crate::types::Type;

type R<T> = Result<T, String>;

fn get<'a>(v: &'a Value, k: &str) -> &'a Value {
    v.get(k).unwrap_or(&Value::Null)
}
fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("")
}
fn i(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_f64()).map(|x| x as i64).unwrap_or(0)
}
fn f(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0)
}
fn b(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}

fn weather_from(id: &str) -> R<Weather> {
    Ok(match id {
        "" => Weather::None,
        "raindance" => Weather::Rain,
        "sunnyday" => Weather::Sun,
        "sandstorm" => Weather::Sand,
        "snowscape" | "snow" => Weather::Snow,
        _ => return Err(format!("unsupported weather {id}")),
    })
}
pub fn weather_id(w: Weather) -> &'static str {
    match w {
        Weather::None => "",
        Weather::Rain => "raindance",
        Weather::Sun => "sunnyday",
        Weather::Sand => "sandstorm",
        Weather::Snow => "snowscape",
    }
}
fn terrain_from(id: &str) -> R<Terrain> {
    Ok(match id {
        "" => Terrain::None,
        "grassyterrain" => Terrain::Grassy,
        "psychicterrain" => Terrain::Psychic,
        "electricterrain" => Terrain::Electric,
        "mistyterrain" => Terrain::Misty,
        _ => return Err(format!("unsupported terrain {id}")),
    })
}
pub fn terrain_id(t: Terrain) -> &'static str {
    match t {
        Terrain::None => "",
        Terrain::Grassy => "grassyterrain",
        Terrain::Psychic => "psychicterrain",
        Terrain::Electric => "electricterrain",
        Terrain::Misty => "mistyterrain",
    }
}
fn status_from(id: &str) -> R<Status> {
    Ok(match id {
        "" => Status::None,
        "brn" => Status::Brn,
        "par" => Status::Par,
        "slp" => Status::Slp,
        "frz" => Status::Frz,
        "psn" => Status::Psn,
        "tox" => Status::Tox,
        _ => return Err(format!("unsupported status {id}")),
    })
}
pub fn status_id(st: Status) -> &'static str {
    match st {
        Status::None => "",
        Status::Brn => "brn",
        Status::Par => "par",
        Status::Slp => "slp",
        Status::Frz => "frz",
        Status::Psn => "psn",
        Status::Tox => "tox",
    }
}
/// Showdown slot string code ('p1a' = 0 ... 'p2b' = 3) -> Pos.
fn pos_from_code(c: i64) -> R<Pos> {
    if !(0..4).contains(&c) {
        return Err(format!("bad slot code {c}"));
    }
    Ok(Pos::from_code(c as u8))
}

const BOOST_NAMES: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

impl Battle {
    /// Rebuild a battle from a Showdown state exported by `lib.js`.
    pub fn from_state_json(v: &Value, seed: u64) -> R<Battle> {
        let d = dex();
        let teams_v = v.get("teams").and_then(|x| x.as_array()).ok_or("state: no teams")?;
        if teams_v.len() != 2 {
            return Err("state: need two teams".into());
        }
        let teams: Vec<TeamSpec> = teams_v.iter().map(parse_team).collect();
        for t in &teams {
            if !t.supported() {
                return Err(format!("unsupported team: {:?}", t.unsupported));
            }
        }
        let cfg = BattleConfig { turn_limit: 1000, ally_damage_targeting: true, log: true, sheets: [false, false] };
        let mut bt = Battle::new([&teams[0], &teams[1]], seed, cfg);
        bt.turn = i(v, "turn") as u16;
        let phase = s(v, "phase");
        bt.phase = match phase {
            "teampreview" => return Ok(bt),
            "move" => Phase::Move,
            "switch" => Phase::Switch { midturn: b(v, "mid_turn") },
            "ended" | "" => return Err(format!("state: unsupported phase '{phase}'")),
            _ => return Err(format!("state: unknown phase '{phase}'")),
        };

        // Field.
        let fv = get(v, "field");
        bt.field.weather = weather_from(s(fv, "weather"))?;
        bt.field.weather_turns = i(fv, "weather_turns") as u8;
        bt.field.terrain = terrain_from(s(fv, "terrain"))?;
        bt.field.terrain_turns = i(fv, "terrain_turns") as u8;
        if let Some(pw) = fv.get("pseudo_weather").and_then(|x| x.as_object()) {
            for (id, st) in pw {
                match id.as_str() {
                    "trickroom" => bt.field.trick_room = i(st, "duration") as u8,
                    "gravity" => bt.field.gravity = i(st, "duration") as u8,
                    _ => return Err(format!("unsupported pseudo weather {id}")),
                }
            }
        }
        if bt.field.weather != Weather::None && bt.field.weather_turns == 0 {
            return Err("weather without duration (permanent weather)".into());
        }

        // Sides.
        let sides_v = v.get("sides").and_then(|x| x.as_array()).ok_or("state: no sides")?;
        for sidx in 0..2 {
            let sv = &sides_v[sidx];
            let side = &mut bt.sides[sidx];
            side.total_fainted = i(sv, "total_fainted") as u8;
            side.fainted_last_turn = b(sv, "fainted_last_turn");
            side.fainted_this_turn = b(sv, "fainted_this_turn");
            side.mega_used = b(sv, "mega_used");
            if let Some(conds) = sv.get("conds").and_then(|x| x.as_object()) {
                for (id, st) in conds {
                    let c = &mut side.conds;
                    let dur = i(st, "duration") as u8;
                    match id.as_str() {
                        "tailwind" => c.tailwind = dur,
                        "reflect" => c.reflect = dur,
                        "lightscreen" => c.light_screen = dur,
                        "auroraveil" => c.aurora_veil = dur,
                        "safeguard" => c.safeguard = dur,
                        "mist" => c.mist = dur,
                        "wideguard" => c.wide_guard = true,
                        "quickguard" => c.quick_guard = true,
                        "stealthrock" => c.stealth_rock = true,
                        "spikes" => c.spikes = i(st, "layers") as u8,
                        "toxicspikes" => c.toxic_spikes = i(st, "layers") as u8,
                        "stickyweb" => c.sticky_web = true,
                        _ => return Err(format!("unsupported side condition {id}")),
                    }
                }
            }
            if let Some(slots) = sv.get("slot_conds").and_then(|x| x.as_array()) {
                for (slot, o) in slots.iter().enumerate().take(2) {
                    for (id, st) in o.as_object().into_iter().flatten() {
                        match id.as_str() {
                            "wish" => side.wish[slot] = (i(st, "duration") as u8, i(st, "hp") as u16),
                            "healingwish" => side.healing_wish[slot] = true,
                            _ => return Err(format!("unsupported slot condition {id}")),
                        }
                    }
                }
            }
            if let Some(order) = sv.get("order").and_then(|x| x.as_array()) {
                let mut o: Vec<u8> = order.iter().filter_map(|x| x.as_u64()).map(|x| x as u8).collect();
                for k in 0..6u8 {
                    if !o.contains(&k) {
                        o.push(k);
                    }
                }
                side.order.copy_from_slice(&o[..6]);
            }
            let active = sv.get("active").and_then(|x| x.as_array()).ok_or("side: no active")?;
            for slot in 0..2 {
                let a = active.get(slot).and_then(|x| x.as_i64()).unwrap_or(-1);
                side.active[slot] = if a < 0 { NO_MON } else { a as u8 };
            }
        }
        for sidx in 0..2 {
            let mons = sides_v[sidx].get("mons").and_then(|x| x.as_array()).ok_or("side: no mons")?;
            for idx in 0..6 {
                let mv = mons.get(idx).unwrap_or(&Value::Null);
                if mv.is_null() {
                    bt.sides[sidx].mons[idx].brought = false;
                    continue;
                }
                bt.load_mon(sidx, idx, mv)?;
            }
            for slot in 0..2 {
                let a = bt.sides[sidx].active[slot];
                if a != NO_MON && bt.sides[sidx].mons[a as usize].slot != slot as i8 {
                    return Err(format!("side {sidx} slot {slot}: active mon {a} has position {}", bt.sides[sidx].mons[a as usize].slot));
                }
            }
        }
        for sidx in 0..2 {
            let sv = &sides_v[sidx];
            if let Some(ss) = v.get("switch_slots").and_then(|x| x.get(sidx)).and_then(|x| x.as_array()) {
                for slot in 0..2 {
                    bt.switch_slots[sidx][slot] = ss.get(slot).and_then(|x| x.as_bool()).unwrap_or(false);
                }
            }
            let _ = sv;
        }

        // Queue (mid-turn states).
        bt.queue = Queue::default();
        if let Some(q) = v.get("queue").and_then(|x| x.as_array()) {
            for a in q {
                let choice = s(a, "choice");
                match choice {
                    "residual" | "beforeTurn" => continue,
                    "move" | "switch" | "megaEvo" => {}
                    _ => return Err(format!("unsupported queued action {choice}")),
                }
                let p = pos_from_code(i(a, "pos"))?;
                let mon = i(a, "mon_team_idx") as u8;
                let prio_total = f(a, "priority");
                let frac_f = f(a, "fractional_priority");
                let frac = (frac_f * 10.0).round() as i8;
                let priority = (prio_total - frac_f).round() as i8;
                let speed = f(a, "speed") as i32;
                let act = match choice {
                    "switch" => Action {
                        kind: ActionKind::Switch { to: i(a, "target") as u8 },
                        pos: p,
                        mon,
                        order: 103,
                        priority: 0,
                        frac: 0,
                        speed,
                        move_id: 0,
                    },
                    "megaEvo" => Action { kind: ActionKind::Mega, pos: p, mon, order: 104, priority: 0, frac: 0, speed, move_id: 0 },
                    _ => {
                        let mid_s = s(a, "move");
                        let m = &bt.sides[p.s()].mons[mon as usize];
                        // A recharge turn is a move action for the locked Pokemon.
                        let mid = if mid_s == "recharge" { m.moves[0] } else { d.move_id(mid_s).ok_or(format!("unknown queued move {mid_s}"))? };
                        let mslot = if mid == d.struggle {
                            0
                        } else {
                            m.move_slot(mid).ok_or(format!("queued move {mid_s} not in moveset"))?
                        };
                        let loc = i(a, "target_loc");
                        let tgt = target_code(p, d.mv(mid).target, loc);
                        Action {
                            kind: ActionKind::Move { mslot: mslot as u8, target: tgt },
                            pos: p,
                            mon,
                            order: 200,
                            priority,
                            frac,
                            speed,
                            move_id: mid,
                        }
                    }
                };
                bt.queue.push(act);
            }
        }
        if bt.phase == Phase::Move && bt.queue.len > 0 {
            return Err("move phase with a non-empty queue".into());
        }
        Ok(bt)
    }

    fn load_mon(&mut self, sidx: usize, idx: usize, mv: &Value) -> R<()> {
        let d = dex();
        let m = &mut self.sides[sidx].mons[idx];
        m.brought = true;
        let sp = s(mv, "species");
        m.species = d.species_id(sp).ok_or(format!("unknown species {sp}"))?;
        m.is_mega = b(mv, "is_mega");
        if m.is_mega && m.species != m.mega_species {
            return Err(format!("mega species {sp} differs from the set's mega"));
        }
        if !m.is_mega && m.species != m.base_species {
            if m.species != m.alt_species || m.alt_species == 0 {
                return Err(format!("forme change to {sp} unsupported"));
            }
            // Zero to Hero's change is permanent (Showdown's baseSpecies changed).
            m.alt_locked = s(mv, "base_species") == d.sp(m.alt_species).id;
            m.weight_hg = m.alt_weight_hg;
        }
        let ab = s(mv, "ability");
        m.ability = d.ability_id(ab).ok_or(format!("unknown ability {ab}"))?;
        let bab = s(mv, "base_ability");
        m.base_ability = d.ability_id(bab).ok_or(format!("unknown ability {bab}"))?;
        for a in [m.ability, m.base_ability] {
            if let Some(r) = d.ability(a).unsupported.first() {
                return Err(format!("unsupported {r}"));
            }
        }
        let it = s(mv, "item");
        m.item = d.item_id(it).ok_or(format!("unknown item {it}"))?;
        if let Some(r) = d.item(m.item).unsupported.first() {
            return Err(format!("unsupported {r}"));
        }
        let li = s(mv, "last_item");
        m.last_item = d.item_id(li).ok_or(format!("unknown item {li}"))?;
        m.ate_berry = b(mv, "ate_berry");
        m.item_knocked = b(mv, "item_knocked");
        m.hp = i(mv, "hp") as u16;
        m.max_hp = i(mv, "maxhp") as u16;
        m.status = status_from(s(mv, "status"))?;
        m.status_turns = i(mv, "status_turns") as u8;
        m.fainted = b(mv, "fainted");
        if let Some(bs) = mv.get("boosts").and_then(|x| x.as_array()) {
            for k in 0..7 {
                m.boosts[k] = bs.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i8;
            }
        }
        let tys: Vec<&str> = mv.get("types").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|t| t.as_str()).collect()).unwrap_or_default();
        let parse_ty = |t: &str| if t == "???" { Some(Type::None) } else { Type::from_name(t) };
        let t0 = tys.first().and_then(|t| parse_ty(t)).ok_or("mon without types")?;
        let t1 = match tys.get(1) {
            Some(t) => parse_ty(t).ok_or(format!("bad type {t}"))?,
            None => Type::None,
        };
        if tys.len() > 2 || !s(mv, "added_type").is_empty() {
            return Err("added types unsupported".into());
        }
        m.types = [t0, t1];
        if m.is_mega {
            m.base_types = m.mega_types;
        }
        if let Some(st) = mv.get("stats").and_then(|x| x.as_array()) {
            for k in 1..6 {
                m.stats[k] = st.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as u16;
            }
        }
        m.weight_hg = i(mv, "weighthg") as u32;
        if b(mv, "transformed") || b(mv, "illusion") {
            return Err("transform / illusion unsupported".into());
        }
        // Moves and PP (must be the set's moves, in order).
        let moves = mv.get("moves").and_then(|x| x.as_array()).ok_or("mon without moves")?;
        if moves.len() != m.n_moves as usize {
            return Err("moveset size differs from the set".into());
        }
        for (k, ms) in moves.iter().enumerate() {
            let id = s(ms, "id");
            if d.move_id(id) != Some(m.moves[k]) {
                return Err(format!("move slot {k} is {id}, set has {}", d.mv(m.moves[k]).id));
            }
            m.pp[k] = i(ms, "pp").clamp(0, 255) as u8;
            m.max_pp[k] = i(ms, "maxpp").clamp(0, 255) as u8;
            if b(ms, "used") {
                m.vol.moves_used |= 1 << k;
            }
        }
        let pos = i(mv, "position");
        m.slot = if b(mv, "active") { pos as i8 } else { -1 };
        m.reveal.seen = m.slot >= 0 || m.fainted;

        // Per-switch-in state.
        let v = &mut m.vol;
        let lm = s(mv, "last_move");
        v.last_move = if lm.is_empty() { 0 } else { d.move_id(lm).ok_or(format!("unknown move {lm}"))? };
        v.last_move_slot = m.moves[..m.n_moves as usize].iter().position(|&x| x == v.last_move && x != 0).unwrap_or(0) as u8;
        v.last_move_failed = get(mv, "move_last_turn_result") == &Value::Bool(false);
        v.this_move_failed = get(mv, "move_this_turn_result") == &Value::Bool(false);
        v.moved_this_turn = !s(mv, "move_this_turn").is_empty();
        v.hurt_this_turn = !get(mv, "hurt_this_turn").is_null();
        v.stats_raised_this_turn = b(mv, "stats_raised_this_turn");
        v.stats_lowered_this_turn = b(mv, "stats_lowered_this_turn");
        v.times_attacked = i(mv, "times_attacked").clamp(0, 255) as u8;
        v.active_turns = i(mv, "active_turns") as u16;
        v.move_actions = i(mv, "active_move_actions") as u16;
        v.newly_switched = b(mv, "newly_switched");
        match get(mv, "switch_flag") {
            Value::Bool(x) => v.switch_flag = *x,
            Value::String(x) => {
                // Showdown stores the move id (or the selfSwitch kind).
                v.switch_flag = true;
                let kind = d.move_id(x).map(|mid| d.mv(mid).self_switch).unwrap_or(0);
                match (x.as_str(), kind) {
                    ("copyvolatile", _) | (_, 2) => v.switch_copyvolatile = true,
                    ("shedtail", _) | (_, 3) => v.switch_shedtail = true,
                    _ => {}
                }
            }
            _ => {}
        }
        v.force_switch = b(mv, "force_switch_flag");
        if b(mv, "being_called_back") {
            return Err("being called back unsupported".into());
        }
        if let Some(f) = get(mv, "ability_state").get("fallen").and_then(|x| x.as_i64()) {
            v.fallen = f as u8;
        }
        v.protean = b(get(mv, "ability_state"), "protean");
        // Counter / Mirror Coat / Metal Burst: the last damage taken this turn.
        if let Some(ab) = mv.get("attacked_by").and_then(|x| x.as_array()) {
            for a in ab.iter().rev() {
                if b(a, "this_turn") && a.get("damage_value").map(|x| x.is_number()).unwrap_or(false) {
                    v.last_damage_taken = i(a, "damage_value").clamp(0, 65535) as u16;
                    let mid = d.move_id(s(a, "move")).unwrap_or(0);
                    v.last_damage_physical = d.mv(mid).category == Category::Physical;
                    v.last_damage_src = i(a, "source").clamp(0, 3) as u8;
                    break;
                }
            }
        }
        let moves_arr = m.moves;
        let n_moves = m.n_moves as usize;
        let slot_of = |id: &str| -> R<u8> {
            let mid = d.move_id(id).ok_or(format!("unknown move {id}"))?;
            moves_arr[..n_moves].iter().position(|&x| x == mid).map(|x| x as u8).ok_or(format!("move {id} not in moveset"))
        };
        let vols = mv.get("volatiles").and_then(|x| x.as_object()).cloned().unwrap_or_default();
        let mut crit_stage = 0u8;
        for (id, st) in &vols {
            let dur = i(st, "duration") as u8;
            match id.as_str() {
                "flinch" => v.flinch = true,
                "protect" => v.protect = ProtectKind::Protect,
                "spikyshield" => v.protect = ProtectKind::SpikyShield,
                "banefulbunker" => v.protect = ProtectKind::BanefulBunker,
                "kingsshield" => v.protect = ProtectKind::KingsShield,
                "silktrap" => v.protect = ProtectKind::SilkTrap,
                "burningbulwark" => v.protect = ProtectKind::BurningBulwark,
                "endure" => v.endure = true,
                "stall" => {
                    v.stall_counter = i(st, "counter") as u16;
                    v.stall_dur = dur;
                }
                "helpinghand" => {
                    let mult = f(st, "multiplier").max(1.0);
                    v.helping_hand = (mult.ln() / 1.5f64.ln()).round() as u8;
                }
                "followme" => v.follow_me = 1,
                "ragepowder" => v.follow_me = 2,
                "confusion" => v.confusion = i(st, "time") as u8,
                "taunt" => v.taunt = dur,
                "encore" => {
                    v.encore = dur;
                    v.encore_slot = slot_of(s(st, "move"))?;
                }
                "disable" => {
                    v.disable = dur;
                    v.disable_slot = slot_of(s(st, "move"))?;
                }
                "throatchop" => v.throat_chop = dur,
                "yawn" => v.yawn = dur,
                "perishsong" => v.perish = dur,
                "substitute" => v.substitute = i(st, "hp") as u16,
                "twoturnmove" => {
                    let mvid = s(st, "move");
                    // Without the move's own volatile the attack already
                    // happened this turn (twoturnmove lingers until residual).
                    if !vols.contains_key(mvid) {
                        continue;
                    }
                    v.charging = slot_of(mvid)? + 1;
                    let tl = vols.get(mvid).map(|x| i(x, "targetLoc")).unwrap_or(0);
                    let p = Pos::new(sidx, pos.max(0) as usize);
                    v.charge_target = target_code(p, d.mv(d.move_id(mvid).unwrap()).target, tl) as i8;
                }
                "fly" | "bounce" => v.semi_inv = SemiInv::Air,
                "dig" => v.semi_inv = SemiInv::Underground,
                "dive" => v.semi_inv = SemiInv::Underwater,
                "phantomforce" | "shadowforce" => v.semi_inv = SemiInv::Shadow,
                "solarbeam" | "solarblade" | "electroshot" | "meteorbeam" => {}
                "mustrecharge" => v.must_recharge = true,
                "focusenergy" => crit_stage = crit_stage.max(2),
                "dragoncheer" => crit_stage = crit_stage.max(if b(st, "hasDragonType") { 2 } else { 1 }),
                "choicelock" => v.choice_lock = slot_of(s(st, "move"))? + 1,
                "glaiverush" => v.glaive_rush = 1,
                "roost" => v.roost = true,
                "imprison" => v.imprison = true,
                "leechseed" => {
                    v.leech_seed = true;
                    v.leech_seed_src = i(st, "sourceSlot").clamp(0, 3) as u8;
                }
                "saltcure" => v.salt_cure = true,
                "curse" => v.curse = true,
                "destinybond" => v.destiny_bond = true,
                "flashfire" => v.flash_fire = true,
                "unburden" => v.unburden = true,
                "torment" => v.torment = true,
                "healblock" => v.heal_block = dur,
                "magnetrise" => v.magnet_rise = dur,
                "aquaring" => v.aqua_ring = true,
                "ingrain" => v.ingrain = true,
                "charge" => v.charge = true,
                "stockpile" => v.stockpile = i(st, "layers") as u8,
                "attract" => v.attract = true,
                "noretreat" => v.no_retreat = true,
                "laserfocus" => v.laser_focus = dur,
                "tarshot" => v.tar_shot = true,
                "smackdown" => v.smacked_down = true,
                "partiallytrapped" => {
                    v.partial_trap = dur;
                    v.partial_trap_src = i(st, "sourceSlot").clamp(0, 3) as u8;
                    v.partial_trap_mon = i(st, "sourceIdx").clamp(0, 5) as u8;
                    if i(st, "boundDivisor") != 0 && i(st, "boundDivisor") != 8 {
                        return Err("binding band unsupported".into());
                    }
                }
                _ => return Err(format!("unsupported volatile {id}")),
            }
        }
        v.crit_stage = crit_stage;
        // A Pokemon whose item was used up gets Unburden's speed boost.
        Ok(())
    }

    // ---- outcome features ---------------------------------------------------------

    /// The features `lib.js` `outcome()` extracts from Showdown, from this battle.
    /// `log_start` is the log length before the step (for the move order).
    pub fn outcome_features(&self, log_start: usize) -> Map<String, Value> {
        let d = dex();
        let mut f = Map::new();
        let req = match self.phase {
            Phase::Ended => format!("end:{}", self.winner.unwrap_or(2)),
            Phase::Move => "move".to_string(),
            Phase::TeamPreview => "teampreview".to_string(),
            Phase::Switch { midturn } => {
                let mut slots = vec![];
                for sd in 0..2 {
                    for sl in 0..2 {
                        if self.switch_slots[sd][sl] {
                            slots.push(format!("p{}{}", sd + 1, if sl == 0 { 'a' } else { 'b' }));
                        }
                    }
                }
                format!("switch{}:{}", if midturn { ":mid" } else { "" }, slots.join(","))
            }
        };
        f.insert("req".into(), json!(req));
        f.insert("turn".into(), json!(self.turn));
        for sd in 0..2 {
            let side = &self.sides[sd];
            let sid = format!("p{}", sd + 1);
            for (idx, m) in side.mons.iter().enumerate() {
                if !m.brought {
                    continue;
                }
                let k = format!("{sid}.{idx}");
                f.insert(format!("{k}.hp"), json!(m.hp));
                let st = if m.fainted {
                    "fnt"
                } else if m.status == Status::None {
                    "-"
                } else {
                    status_id(m.status)
                };
                f.insert(format!("{k}.st"), json!(st));
                if !m.fainted && matches!(m.status, Status::Slp | Status::Frz | Status::Tox) {
                    f.insert(format!("{k}.stt"), json!(m.status_turns));
                }
                let bs: Vec<String> = (0..7)
                    .filter(|&x| m.boosts[x] != 0)
                    .map(|x| format!("{}{}{}", BOOST_NAMES[x], if m.boosts[x] > 0 { "+" } else { "" }, m.boosts[x]))
                    .collect();
                f.insert(format!("{k}.b"), json!(bs.join(" ")));
                let pos = if m.slot >= 0 && !m.fainted { if m.slot == 0 { "a" } else { "b" } } else { "-" };
                f.insert(format!("{k}.pos"), json!(pos));
                let it = if m.item == 0 { "-".to_string() } else { d.item(m.item).id.clone() };
                f.insert(format!("{k}.it"), json!(it));
                f.insert(format!("{k}.ab"), json!(d.ability(m.ability).id));
                f.insert(format!("{k}.sp"), json!(d.sp(m.species).id));
                f.insert(format!("{k}.ty"), json!(types_string(m.types)));
                let vs = if m.slot >= 0 && !m.fainted { self.vol_string(sd, idx) } else { String::new() };
                f.insert(format!("{k}.v"), json!(vs));
                let pp: Vec<String> = m.pp[..m.n_moves as usize].iter().map(|x| x.to_string()).collect();
                f.insert(format!("{k}.pp"), json!(pp.join(",")));
            }
            let c = &side.conds;
            let mut sc: Vec<String> = vec![];
            let mut add = |id: &str, n: u8| {
                if n > 0 {
                    sc.push(format!("{id}:{n}"));
                }
            };
            add("tailwind", c.tailwind);
            add("reflect", c.reflect);
            add("lightscreen", c.light_screen);
            add("auroraveil", c.aurora_veil);
            add("safeguard", c.safeguard);
            add("mist", c.mist);
            add("wideguard", c.wide_guard as u8);
            add("quickguard", c.quick_guard as u8);
            add("stealthrock", c.stealth_rock as u8);
            add("spikes", c.spikes);
            add("toxicspikes", c.toxic_spikes);
            add("stickyweb", c.sticky_web as u8);
            sc.sort();
            f.insert(format!("{sid}.sc"), json!(sc.join(" ")));
            let mut slots: Vec<String> = vec![];
            for sl in 0..2 {
                let l = if sl == 0 { 'a' } else { 'b' };
                if side.wish[sl].0 > 0 {
                    slots.push(format!("{l}.wish:{}", side.wish[sl].0));
                }
                if side.healing_wish[sl] {
                    slots.push(format!("{l}.healingwish:1"));
                }
            }
            slots.sort();
            f.insert(format!("{sid}.slot"), json!(slots.join(" ")));
        }
        let fd = &self.field;
        let w = if fd.weather == Weather::None { "-".to_string() } else { format!("{}:{}", weather_id(fd.weather), fd.weather_turns) };
        f.insert("f.w".into(), json!(w));
        let t = if fd.terrain == Terrain::None { "-".to_string() } else { format!("{}:{}", terrain_id(fd.terrain), fd.terrain_turns) };
        f.insert("f.t".into(), json!(t));
        let mut pw = vec![];
        if fd.gravity > 0 {
            pw.push(format!("gravity:{}", fd.gravity));
        }
        if fd.trick_room > 0 {
            pw.push(format!("trickroom:{}", fd.trick_room));
        }
        f.insert("f.pw".into(), json!(pw.join(" ")));
        let mut order: Vec<String> = vec![];
        for line in self.log_lines.iter().skip(log_start) {
            if let Some(rest) = line.strip_prefix("|move|") {
                if line.contains("[from]") {
                    continue;
                }
                order.push(rest.chars().take(3).collect());
            }
        }
        f.insert("order".into(), json!(order.join(",")));
        f
    }

    /// Tracked volatiles in `lib.js` TRACKED_VOLATILES form.
    fn vol_string(&self, sd: usize, idx: usize) -> String {
        let d = dex();
        let m = &self.sides[sd].mons[idx];
        let v = &m.vol;
        let mut out: Vec<String> = vec![];
        let mut push = |s: String| out.push(s);
        if v.confusion > 0 {
            push("confusion=y".into());
        }
        if v.taunt > 0 {
            push(format!("taunt={}", v.taunt));
        }
        if v.encore > 0 {
            push(format!("encore={}", v.encore));
        }
        if v.disable > 0 {
            push(format!("disable={}", v.disable));
        }
        if v.yawn > 0 {
            push(format!("yawn={}", v.yawn));
        }
        if v.perish > 0 {
            push(format!("perishsong={}", v.perish));
        }
        if v.substitute > 0 {
            push(format!("substitute={}", v.substitute));
        }
        if v.leech_seed {
            push("leechseed=y".into());
        }
        if v.salt_cure {
            push("saltcure=y".into());
        }
        if v.curse {
            push("curse=y".into());
        }
        if v.flash_fire {
            push("flashfire=y".into());
        }
        if v.torment {
            push("torment=y".into());
        }
        if v.heal_block > 0 {
            push(format!("healblock={}", v.heal_block));
        }
        if v.magnet_rise > 0 {
            push(format!("magnetrise={}", v.magnet_rise));
        }
        if v.aqua_ring {
            push("aquaring=y".into());
        }
        if v.ingrain {
            push("ingrain=y".into());
        }
        if v.charge {
            push("charge=y".into());
        }
        if v.stockpile > 0 {
            push(format!("stockpile={}", v.stockpile));
        }
        if v.attract {
            push("attract=y".into());
        }
        if v.no_retreat {
            push("noretreat=y".into());
        }
        if v.laser_focus > 0 {
            push(format!("laserfocus={}", v.laser_focus));
        }
        if v.tar_shot {
            push("tarshot=y".into());
        }
        if v.smacked_down {
            push("smackdown=y".into());
        }
        if v.crit_stage > 0 {
            // focusenergy / dragoncheer are indistinguishable in the engine.
            push("critstage=y".into());
        }
        if v.choice_lock > 0 && is_choice_item(dex().it(m.item)) {
            push(format!("choicelock={}", d.mv(m.moves[(v.choice_lock - 1) as usize]).id));
        }
        if v.stall_dur > 0 {
            push(format!("stall={}/{}", v.stall_counter, v.stall_dur));
        }
        if v.must_recharge {
            push("mustrecharge=y".into());
        }
        if v.charging > 0 {
            push(format!("twoturnmove={}", d.mv(m.moves[(v.charging - 1) as usize]).id));
        }
        if v.glaive_rush > 0 {
            push("glaiverush=y".into());
        }
        if v.throat_chop > 0 {
            push(format!("throatchop={}", v.throat_chop));
        }
        if v.imprison {
            push("imprison=y".into());
        }
        if v.destiny_bond {
            push("destinybond=y".into());
        }
        if v.partial_trap > 0 {
            push(format!("partiallytrapped={}", v.partial_trap));
        }
        if v.unburden {
            push("unburden=y".into());
        }
        out.sort();
        out.join(" ")
    }

    // ---- damage probe ------------------------------------------------------------

    /// Damage of `move_id` from `src` into `tgt` with a fixed random roll
    /// (0..16, Showdown's `100 - random(16)` factor) and crit flag, as
    /// Showdown's `getDamage` computes it (no accuracy, no side effects).
    /// Ok(None) = no damage (immune / 0 base power).
    pub fn damage_probe(&mut self, src: Pos, tgt: Pos, move_id: &str, spread: bool, crit: bool, roll: u32) -> R<Option<u32>> {
        let d = dex();
        let mid = d.move_id(move_id).ok_or(format!("unknown move {move_id}"))?;
        let mslot = self.m(src).move_slot(mid).unwrap_or(0);
        let mut am = self.make_active_move(src, mid, mslot, false);
        self.resolve_move_type(src, &mut am);
        am.spread_hit = spread;
        self.mold_breaker = am.mold_breaker;
        let out = (|| {
            if am.category == Category::Status {
                return None;
            }
            if am.ty != Type::None && self.type_immune(src, am.ty, tgt, &am) {
                return None;
            }
            if am.fx == crate::kinds::MoveFx::Endeavor && self.m(src).hp >= self.m(tgt).hp {
                return None;
            }
            let crit = crit && !matches!(self.tab(tgt), Ab::BattleArmor | Ab::ShellArmor);
            if let Some(fx) = self.fixed_damage(src, tgt, &am) {
                return Some(fx);
            }
            self.calc_damage(src, tgt, &am, crit, roll)
        })();
        self.mold_breaker = false;
        Ok(out)
    }

    // ---- legality summary ------------------------------------------------------------

    /// Per active slot: usable move slots, mega, switch targets (team indices).
    pub fn legal_summary(&self, side: usize) -> Value {
        use crate::actions::{decode, SlotAction};
        let mut out = vec![];
        for slot in 0..2 {
            let mask = self.legal_mask(side, slot);
            let mut moves = std::collections::BTreeSet::new();
            let mut switches = std::collections::BTreeSet::new();
            let mut mega = false;
            let mut pass = false;
            for a in 0..31u8 {
                if mask & (1 << a) == 0 {
                    continue;
                }
                match decode(a) {
                    SlotAction::Pass => pass = true,
                    SlotAction::Switch(t) => {
                        switches.insert(t);
                    }
                    SlotAction::Move { slot: ms, mega: mg, .. } => {
                        moves.insert(ms);
                        mega |= mg;
                    }
                }
            }
            out.push(json!({"moves": moves.into_iter().collect::<Vec<_>>(), "switches": switches.into_iter().collect::<Vec<_>>(), "mega": mega, "pass": pass}));
        }
        Value::Array(out)
    }
}

fn is_choice_item(it: It) -> bool {
    matches!(it, It::ChoiceBand | It::ChoiceScarf | It::ChoiceSpecs)
}

fn types_string(t: [Type; 2]) -> String {
    let names: Vec<&str> = t.iter().filter(|&&x| x != Type::None).map(|x| x.name()).collect();
    if names.is_empty() {
        "???".into()
    } else {
        names.join("/")
    }
}

/// Engine target code (0 foe a, 1 foe b, 2 ally) from a Showdown targetLoc.
pub fn target_code(p: Pos, target: Target, loc: i64) -> u8 {
    match target {
        Target::Normal | Target::Any | Target::AdjacentFoe => {
            if loc > 0 {
                (loc - 1) as u8
            } else if loc < 0 {
                T_ALLY
            } else {
                T_FOE0
            }
        }
        Target::AdjacentAllyOrSelf => {
            if loc < 0 && (-loc - 1) as u8 != p.slot {
                T_ALLY
            } else {
                T_FOE0
            }
        }
        _ => T_FOE0,
    }
}

// ---- parity helpers shared by src/bin/parity.rs and tests/parity.rs ---------

/// Per-feature value counts over runs.
pub type Counts = std::collections::BTreeMap<String, std::collections::BTreeMap<String, u32>>;

/// Engine choice from the JSON `lib.js` `engineChoice` writes.
pub fn choice_from_json(c: &Value) -> R<crate::actions::Choice> {
    use crate::actions::{preview_index, Choice};
    if let Some(po) = c.get("preview_order").and_then(|x| x.as_array()) {
        let o: Vec<u8> = po.iter().map(|x| x.as_u64().unwrap_or(0) as u8).collect();
        if o.len() != 4 {
            return Err("preview_order needs 4".into());
        }
        let p = preview_index([o[0], o[1]], [o[2], o[3]]).ok_or("bad preview order")?;
        return Ok(Choice::preview(p));
    }
    let s = c.get("slots").and_then(|x| x.as_array()).ok_or("choice without slots")?;
    let a = s.first().and_then(|x| x.as_u64()).unwrap_or(0) as u8;
    let b = s.get(1).and_then(|x| x.as_u64()).unwrap_or(0) as u8;
    Ok(Choice::slots(a, b))
}

/// Run the decision `choices` from `state` `runs` times with different seeds
/// and count the outcome features. Returns (counts, step errors).
pub fn outcome_counts(state: &Value, choices: &Value, runs: u64, seed0: u64) -> R<(Counts, Counts)> {
    let base = Battle::from_state_json(state, 1)?;
    let cs = choices.as_array().ok_or("no choices")?;
    let ch = [choice_from_json(&cs[0])?, choice_from_json(&cs[1])?];
    let mut counts = Counts::new();
    let mut errors = Counts::new();
    for k in 0..runs {
        let mut b = base.clone();
        b.rng = crate::rng::Rng::new(seed0.wrapping_mul(1_000_003).wrapping_add(k));
        let log0 = b.log_lines.len();
        if let Err(e) = b.step(ch) {
            *errors.entry("step".into()).or_default().entry(e.0).or_default() += 1;
            continue;
        }
        for (key, v) in b.outcome_features(log0) {
            let vs = match v {
                Value::String(s) => s,
                other => other.to_string(),
            };
            *counts.entry(key).or_default().entry(vs).or_default() += 1;
        }
    }
    Ok((counts, errors))
}

fn lgamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Upper regularized gamma function Q(s, x).
fn gamma_q(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < s + 1.0 {
        let (mut sum, mut term) = (1.0 / s, 1.0 / s);
        for n in 1..500 {
            term *= x / (s + n as f64);
            sum += term;
            if term < sum * 1e-15 {
                break;
            }
        }
        return (1.0 - (-x + s * x.ln() - lgamma(s)).exp() * sum).max(0.0);
    }
    let (mut b, mut c) = (x + 1.0 - s, 1e300);
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..500 {
        let an = -(i as f64) * (i as f64 - s);
        b += 2.0;
        d = an * d + b;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = b + an / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + s * x.ln() - lgamma(s)).exp() * h
}

fn ks_p(d: f64, n: f64, m: f64) -> f64 {
    let ne = n * m / (n + m);
    let lam = (ne.sqrt() + 0.12 + 0.11 / ne.sqrt()) * d;
    if lam < 0.2 {
        return 1.0;
    }
    let mut sum = 0.0;
    for j in 1..=100 {
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        let t = 2.0 * sign * (-2.0 * (j * j) as f64 * lam * lam).exp();
        sum += t;
        if t.abs() < 1e-12 {
            break;
        }
    }
    sum.clamp(0.0, 1.0)
}

/// Homogeneity test of two value-count maps, as turns.js `compareCounts`:
/// chi-square with rare values pooled, and KS for numeric features.
/// Returns (p-value, total variation distance).
pub fn compare_counts(a: &std::collections::BTreeMap<String, u32>, b: &std::collections::BTreeMap<String, u32>) -> (f64, f64) {
    let na: u32 = a.values().sum();
    let nb: u32 = b.values().sum();
    if na == 0 || nb == 0 {
        return (1.0, 0.0);
    }
    let (na, nb) = (na as f64, nb as f64);
    let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
    keys.sort();
    keys.dedup();
    let get = |m: &std::collections::BTreeMap<String, u32>, k: &String| *m.get(k).unwrap_or(&0) as f64;
    let tvd = keys.iter().map(|k| (get(a, k) / na - get(b, k) / nb).abs()).sum::<f64>() / 2.0;
    if keys.len() == 1 {
        return (1.0, 0.0);
    }
    let mut cells: Vec<(f64, f64)> = vec![];
    let (mut ra, mut rb) = (0.0, 0.0);
    for k in &keys {
        let (x, y) = (get(a, k), get(b, k));
        if x + y < 8.0 {
            ra += x;
            rb += y;
        } else {
            cells.push((x, y));
        }
    }
    if ra + rb > 0.0 {
        cells.push((ra, rb));
    }
    let mut chi2 = 0.0;
    for &(x, y) in &cells {
        let p = (x + y) / (na + nb);
        let (ea, eb) = (na * p, nb * p);
        if ea > 0.0 {
            chi2 += (x - ea).powi(2) / ea;
        }
        if eb > 0.0 {
            chi2 += (y - eb).powi(2) / eb;
        }
    }
    let dof = cells.len() as f64 - 1.0;
    let mut p = if dof <= 0.0 { 1.0 } else { gamma_q(dof / 2.0, chi2 / 2.0) };
    let mut nums: Vec<(f64, &String)> = vec![];
    for k in &keys {
        match k.parse::<f64>() {
            Ok(v) => nums.push((v, *k)),
            Err(_) => {
                nums.clear();
                break;
            }
        }
    }
    if nums.len() > 2 {
        nums.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        let (mut ca, mut cb, mut d) = (0.0, 0.0, 0.0f64);
        for (_, k) in &nums {
            ca += get(a, k) / na;
            cb += get(b, k) / nb;
            d = d.max((ca - cb).abs());
        }
        p = p.min(ks_p(d, na, nb));
    }
    (p, tvd)
}
