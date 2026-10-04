//! Observation encoding: the battle as one side sees it.
//!
//! The network gets only what that player could know: its own team in full;
//! for the opponent, the six species from team preview plus whatever has been
//! revealed (moves used, items/abilities that activated, everything under open
//! team sheets), HP as Showdown's Champions percentage, and visible effects.
//! No damage calculator, usage statistics or speed resolver is encoded — only
//! the state and static dex data (exported separately by `static_tables`).
//!
//! Per side:
//!   mon_int   [N_MONS, MON_INT]    ids (embedding lookups)
//!   mon_float [N_MONS, MON_FLOAT]  numeric features
//!   field     [FIELD_FLOAT]
//!   mask      [2, N_SLOT_ACTIONS]  legal actions per own active slot
//! Tokens 0..6 are the viewer's team (team order), 6..12 the opponent's.

use crate::actions::N_SLOT_ACTIONS;
use crate::battle::{Battle, Phase, RequestKind};
use crate::dex::{dex, Category, Status, Target, Terrain, Weather};
use crate::kinds::It;
use crate::state::{Mon, Pos};
use crate::types::{Type, N_TYPES};

pub const N_MONS: usize = 12;

// mon_int columns
pub const I_SPECIES: usize = 0;
pub const I_ABILITY: usize = 1;
pub const I_ITEM: usize = 2;
pub const I_MOVE0: usize = 3; // ..=6
pub const I_LAST_MOVE: usize = 7;
pub const I_STATUS: usize = 8;
pub const I_POSITION: usize = 9;
pub const I_OWNER: usize = 10;
pub const MON_INT: usize = 11;

/// Position codes (I_POSITION).
pub const POS_UNKNOWN: i32 = 0;
pub const POS_ACTIVE0: i32 = 1;
pub const POS_ACTIVE1: i32 = 2;
pub const POS_BENCH: i32 = 3;
pub const POS_FAINTED: i32 = 4;
pub const POS_NOT_BROUGHT: i32 = 5;
pub const N_POSITIONS: usize = 6;

pub const MON_FLOAT: usize = 80;
pub const FIELD_FLOAT: usize = 48;

/// Vocabulary sizes (embedding tables need one extra "unknown" row).
pub struct Vocab {
    pub species: usize,
    pub moves: usize,
    pub items: usize,
    pub abilities: usize,
}

pub fn vocab() -> Vocab {
    let d = dex();
    Vocab { species: d.species.len() + 1, moves: d.moves.len() + 1, items: d.items.len() + 1, abilities: d.abilities.len() + 1 }
}

/// Sentinel ids for "unknown" (one past each table).
pub fn unknown_move() -> i32 {
    dex().moves.len() as i32
}
pub fn unknown_item() -> i32 {
    dex().items.len() as i32
}
pub fn unknown_ability() -> i32 {
    dex().abilities.len() as i32
}

/// Sizes of one side's observation.
pub const fn sizes() -> (usize, usize, usize, usize) {
    (N_MONS * MON_INT, N_MONS * MON_FLOAT, FIELD_FLOAT, 2 * N_SLOT_ACTIONS)
}

fn type_onehot(out: &mut [f32], types: [Type; 2]) {
    for t in types {
        if t != Type::None {
            out[t.idx()] = 1.0;
        }
    }
}

/// Encode the battle from `side`'s point of view into the given buffers.
pub fn encode(b: &Battle, side: usize, ints: &mut [i32], floats: &mut [f32], field: &mut [f32], mask: &mut [u8]) {
    debug_assert_eq!(ints.len(), N_MONS * MON_INT);
    debug_assert_eq!(floats.len(), N_MONS * MON_FLOAT);
    debug_assert_eq!(field.len(), FIELD_FLOAT);
    debug_assert_eq!(mask.len(), 2 * N_SLOT_ACTIONS);
    ints.fill(0);
    floats.fill(0.0);
    field.fill(0.0);
    mask.fill(0);
    let preview = b.phase == Phase::TeamPreview;
    for (k, s) in [side, 1 - side].into_iter().enumerate() {
        let own = k == 0;
        for i in 0..6 {
            let tok = k * 6 + i;
            encode_mon(
                b,
                s,
                i,
                own,
                preview,
                &mut ints[tok * MON_INT..(tok + 1) * MON_INT],
                &mut floats[tok * MON_FLOAT..(tok + 1) * MON_FLOAT],
            );
        }
    }
    encode_field(b, side, field);
    let req = b.request(side);
    if matches!(req.kind, RequestKind::Move | RequestKind::Switch) {
        for slot in 0..2 {
            let m = b.legal_mask(side, slot);
            for a in 0..N_SLOT_ACTIONS {
                mask[slot * N_SLOT_ACTIONS + a] = ((m >> a) & 1) as u8;
            }
        }
    } else {
        mask[0] = 1;
        mask[N_SLOT_ACTIONS] = 1;
    }
}

fn encode_mon(b: &Battle, s: usize, i: usize, own: bool, preview: bool, ints: &mut [i32], f: &mut [f32]) {
    let d = dex();
    let m: &Mon = &b.sides[s].mons[i];
    let sheet = b.sides[s].sheet_open;
    let r = m.reveal;
    ints[I_OWNER] = if own { 0 } else { 1 };
    ints[I_SPECIES] = m.species as i32;
    // Ability / item / moves: own always; opponent only when revealed.
    let know_ability = own || sheet || r.ability;
    let know_item = own || sheet || r.item;
    ints[I_ABILITY] = if know_ability { m.ability as i32 } else { unknown_ability() };
    ints[I_ITEM] = if know_item { m.item as i32 } else { unknown_item() };
    for j in 0..4 {
        let known = own || sheet || (r.moves >> j) & 1 == 1;
        ints[I_MOVE0 + j] = if j >= m.n_moves as usize {
            0
        } else if known {
            m.moves[j] as i32
        } else {
            unknown_move()
        };
    }
    let visible = own || r.seen;
    ints[I_LAST_MOVE] = if visible && m.active() { m.vol.last_move as i32 } else { 0 };
    ints[I_STATUS] = if visible { m.status as i32 } else { 0 };
    ints[I_POSITION] = if preview {
        POS_UNKNOWN
    } else if m.active() {
        if m.fainted {
            POS_FAINTED
        } else if m.slot == 0 {
            POS_ACTIVE0
        } else {
            POS_ACTIVE1
        }
    } else if m.fainted {
        POS_FAINTED
    } else if own {
        if m.brought {
            POS_BENCH
        } else {
            POS_NOT_BROUGHT
        }
    } else if r.seen {
        POS_BENCH
    } else {
        POS_UNKNOWN
    };

    // Floats.
    let hp = if !m.alive() {
        0.0
    } else if own {
        m.hp as f32 / m.max_hp.max(1) as f32
    } else {
        // Champions shares floor(100 * hp / maxhp) (min 1) with the opponent.
        ((100 * m.hp as u32 / m.max_hp.max(1) as u32).max(1)) as f32 / 100.0
    };
    f[0] = if visible || !preview { hp } else { 1.0 };
    f[1] = if m.alive() { 1.0 } else { 0.0 };
    f[2] = if m.active() && m.alive() { 1.0 } else { 0.0 };
    f[3] = if own { 1.0 } else { 0.0 };
    if m.active() {
        for j in 0..7 {
            f[4 + j] = m.boosts[j] as f32 / 6.0;
        }
    }
    type_onehot(&mut f[11..11 + N_TYPES], if visible { m.types } else { d.sp(m.species).types });
    if own {
        for j in 0..6 {
            f[29 + j] = m.stats[j] as f32 / 400.0;
        }
    }
    f[35] = m.weight_hg as f32 / 10000.0;
    f[36] = if m.is_mega { 1.0 } else { 0.0 };
    f[37] = if m.can_mega && !m.is_mega && !b.sides[s].mega_used { 1.0 } else { 0.0 };
    for j in 0..m.n_moves as usize {
        f[38 + j] = if own { m.pp[j] as f32 / m.max_pp[j].max(1) as f32 } else { 1.0 };
    }
    let p = Pos::new(s, m.slot.max(0) as usize);
    let on_field = m.active() && m.alive() && b.phase != Phase::TeamPreview;
    if own && on_field && b.phase == Phase::Move {
        for j in 0..m.n_moves as usize {
            f[42 + j] = if b.move_disabled(p, j) { 0.0 } else { 1.0 };
        }
    }
    if on_field {
        let v = &m.vol;
        f[46] = (v.confusion > 0) as u8 as f32;
        f[47] = v.taunt as f32 / 4.0;
        f[48] = v.encore as f32 / 4.0;
        f[49] = v.disable as f32 / 5.0;
        f[50] = v.throat_chop as f32 / 2.0;
        f[51] = v.yawn as f32 / 2.0;
        f[52] = v.perish as f32 / 4.0;
        f[53] = v.substitute as f32 / m.max_hp.max(1) as f32;
        f[54] = v.leech_seed as u8 as f32;
        f[55] = if v.stall_dur > 0 && v.stall_counter > 0 { (v.stall_counter as f32).log(3.0) / 6.0 } else { 0.0 };
        if own && v.choice_lock > 0 && crate::battle::is_choice_item(d.it(m.item)) {
            f[56] = 1.0;
            f[57 + (v.choice_lock - 1) as usize] = 1.0;
        }
        f[61] = (v.charging > 0) as u8 as f32;
        f[62] = v.must_recharge as u8 as f32;
        f[63] = v.salt_cure as u8 as f32;
        f[64] = v.curse as u8 as f32;
        f[65] = v.flash_fire as u8 as f32;
        f[66] = (v.unburden && m.item == 0) as u8 as f32;
        f[67] = (v.glaive_rush > 0) as u8 as f32;
        f[68] = v.crit_stage as f32 / 3.0;
        f[69] = v.tar_shot as u8 as f32;
        f[70] = (v.active_turns.min(5)) as f32 / 5.0;
        f[71] = (v.move_actions == 0) as u8 as f32;
        f[72] = (v.times_attacked.min(6)) as f32 / 6.0;
        f[79] = b.is_grounded(p) as u8 as f32;
    }
    f[73] = r.seen as u8 as f32;
    f[74] = r.moves.count_ones() as f32 / 4.0;
    f[75] = r.item as u8 as f32;
    f[76] = r.ability as u8 as f32;
    f[77] = (m.item == 0 && (own || r.item)) as u8 as f32;
    f[78] = if own && on_field { b.speed(p) as f32 / 500.0 } else { 0.0 };
}

fn encode_field(b: &Battle, side: usize, f: &mut [f32]) {
    let fl = &b.field;
    let w = match fl.weather {
        Weather::None => 0,
        Weather::Rain => 1,
        Weather::Sun => 2,
        Weather::Sand => 3,
        Weather::Snow => 4,
    };
    f[w] = 1.0;
    f[5] = fl.weather_turns as f32 / 8.0;
    let t = match fl.terrain {
        Terrain::None => 0,
        Terrain::Grassy => 1,
        Terrain::Psychic => 2,
        Terrain::Electric => 3,
        Terrain::Misty => 4,
    };
    f[6 + t] = 1.0;
    f[11] = fl.terrain_turns as f32 / 8.0;
    f[12] = fl.trick_room as f32 / 5.0;
    f[13] = fl.gravity as f32 / 5.0;
    for (k, s) in [side, 1 - side].into_iter().enumerate() {
        let o = 14 + k * 11;
        let sd = &b.sides[s];
        let c = &sd.conds;
        f[o] = c.tailwind as f32 / 4.0;
        f[o + 1] = c.reflect as f32 / 8.0;
        f[o + 2] = c.light_screen as f32 / 8.0;
        f[o + 3] = c.aurora_veil as f32 / 8.0;
        f[o + 4] = c.safeguard as f32 / 5.0;
        f[o + 5] = c.stealth_rock as u8 as f32;
        f[o + 6] = c.spikes as f32 / 3.0;
        f[o + 7] = c.toxic_spikes as f32 / 2.0;
        f[o + 8] = sd.mega_used as u8 as f32;
        f[o + 9] = sd.total_fainted as f32 / 4.0;
        f[o + 10] = (sd.wish[0].0 > 0 || sd.wish[1].0 > 0) as u8 as f32;
    }
    f[36] = (b.turn as f32 / 30.0).min(2.0);
    let req = b.request(side);
    let rk = match req.kind {
        RequestKind::Wait => 0,
        RequestKind::TeamPreview => 1,
        RequestKind::Move => 2,
        RequestKind::Switch => 3,
    };
    f[37 + rk] = 1.0;
    f[41] = req.slots[0] as u8 as f32;
    f[42] = req.slots[1] as u8 as f32;
    f[43] = b.sides[side].sheet_open as u8 as f32;
    f[44] = b.sides[1 - side].sheet_open as u8 as f32;
    f[45] = b.sides[side].fainted_last_turn as u8 as f32;
    f[46] = b.sides[1 - side].fainted_last_turn as u8 as f32;
    f[47] = matches!(b.phase, Phase::Switch { midturn: true }) as u8 as f32;
}

// ---- static dex tables --------------------------------------------------------

pub const SPECIES_FEAT: usize = N_TYPES + 6 + 2;
pub const MOVE_FEAT: usize = 96;
pub const ITEM_FEAT: usize = 8;

/// Static per-id feature tables (rows = vocab size, incl. the unknown row).
pub fn species_table() -> Vec<f32> {
    let d = dex();
    let n = d.species.len() + 1;
    let mut out = vec![0f32; n * SPECIES_FEAT];
    for (i, sp) in d.species.iter().enumerate() {
        let row = &mut out[i * SPECIES_FEAT..(i + 1) * SPECIES_FEAT];
        if sp.id.is_empty() {
            continue;
        }
        type_onehot(&mut row[..N_TYPES], sp.types);
        for j in 0..6 {
            row[N_TYPES + j] = sp.base[j] as f32 / 255.0;
        }
        row[N_TYPES + 6] = sp.weight_hg as f32 / 10000.0;
        row[N_TYPES + 7] = sp.is_mega as u8 as f32;
    }
    out
}

fn target_index(t: Target) -> usize {
    t as usize
}

pub fn move_table() -> Vec<f32> {
    let d = dex();
    let n = d.moves.len() + 1;
    let mut out = vec![0f32; n * MOVE_FEAT];
    for (i, mv) in d.moves.iter().enumerate() {
        if mv.id.is_empty() {
            continue;
        }
        let r = &mut out[i * MOVE_FEAT..(i + 1) * MOVE_FEAT];
        r[0] = mv.base_power as f32 / 250.0;
        r[1] = if mv.accuracy == 0 { 1.0 } else { mv.accuracy as f32 / 100.0 };
        r[2] = (mv.accuracy == 0) as u8 as f32;
        r[3] = mv.priority as f32 / 7.0;
        let c = match mv.category {
            Category::Physical => 0,
            Category::Special => 1,
            Category::Status => 2,
        };
        r[4 + c] = 1.0;
        if mv.ty != Type::None {
            r[7 + mv.ty.idx()] = 1.0;
        }
        r[25 + target_index(mv.target)] = 1.0; // 15 targets: 25..40
        for bit in 0..28 {
            r[40 + bit] = ((mv.flags >> bit) & 1) as f32; // 40..68
        }
        r[68] = mv.crit_ratio as f32 / 3.0;
        r[69] = if mv.multihit.0 > 0 { (mv.multihit.0 as f32 + mv.multihit.1 as f32) / 10.0 } else { 0.0 };
        r[70] = if mv.drain.1 > 0 { mv.drain.0 as f32 / mv.drain.1 as f32 } else { 0.0 };
        r[71] = if mv.recoil.1 > 0 { mv.recoil.0 as f32 / mv.recoil.1 as f32 } else { 0.0 };
        r[72] = if mv.heal.1 > 0 { mv.heal.0 as f32 / mv.heal.1 as f32 } else { 0.0 };
        let sec = mv.secondaries.first();
        r[73] = sec.map(|s| s.chance as f32 / 100.0).unwrap_or(0.0);
        if let Some(s) = sec {
            if s.status != Status::None {
                r[74 + s.status as usize - 1] = 1.0; // 74..80
            }
            r[80] = s.volatile.is_some() as u8 as f32;
            r[81] = s.boosts.map(|b| b.iter().map(|&x| x as f32).sum::<f32>() / 6.0).unwrap_or(0.0);
            r[82] = s.self_boosts.map(|b| b.iter().map(|&x| x as f32).sum::<f32>() / 6.0).unwrap_or(0.0);
        }
        if mv.status != Status::None {
            r[83] = 1.0;
        }
        r[84] = mv.boosts.map(|b| b.iter().map(|&x| x as f32).sum::<f32>() / 6.0).unwrap_or(0.0);
        r[85] = mv.self_boosts.map(|b| b.iter().map(|&x| x as f32).sum::<f32>() / 6.0).unwrap_or(0.0);
        r[86] = (mv.self_switch > 0) as u8 as f32;
        r[87] = mv.force_switch as u8 as f32;
        r[88] = mv.stalling as u8 as f32;
        r[89] = mv.side_condition.is_some() as u8 as f32;
        r[90] = (mv.weather != Weather::None) as u8 as f32;
        r[91] = (mv.terrain != Terrain::None) as u8 as f32;
        r[92] = mv.pseudo_weather.is_some() as u8 as f32;
        r[93] = mv.volatile.is_some() as u8 as f32;
        r[94] = mv.pp as f32 / 20.0;
        r[95] = (mv.selfdestruct > 0) as u8 as f32;
    }
    out
}

pub fn item_table() -> Vec<f32> {
    let d = dex();
    let n = d.items.len() + 1;
    let mut out = vec![0f32; n * ITEM_FEAT];
    for (i, it) in d.items.iter().enumerate() {
        if it.id.is_empty() {
            continue;
        }
        let r = &mut out[i * ITEM_FEAT..(i + 1) * ITEM_FEAT];
        r[0] = it.is_berry as u8 as f32;
        r[1] = it.is_mega_stone as u8 as f32;
        r[2] = matches!(it.kind, It::ChoiceBand | It::ChoiceScarf | It::ChoiceSpecs) as u8 as f32;
        r[3] = crate::battle::type_booster_of(it.kind).is_some() as u8 as f32;
        r[4] = crate::battle::resist_berry_of(it.kind).is_some() as u8 as f32;
        r[5] = it.unsupported.is_empty() as u8 as f32;
        r[6] = it.fling_bp as f32 / 130.0;
        r[7] = 1.0;
    }
    out
}
