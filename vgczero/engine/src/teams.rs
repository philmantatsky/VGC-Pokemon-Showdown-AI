//! Team pool: the JSON written by `showdown/compile_teams.js`, resolved
//! against the dex. Stats come precomputed from Showdown.

use serde_json::Value;

use crate::dex::{dex, read_json, AbilityId, ItemId, MoveId, SpeciesId, NONE};
use crate::types::Type;

#[derive(Copy, Clone, Debug, Default)]
pub struct MegaSpec {
    pub species: SpeciesId,
    pub ability: AbilityId,
    pub types: [Type; 2],
    pub stats: [u16; 6],
    pub weight_hg: u32,
}

/// A battle-only forme the set can change into (Aegislash-Blade by Stance
/// Change, Palafin-Hero by Zero to Hero), with its stats for this set.
#[derive(Copy, Clone, Debug, Default)]
pub struct AltForme {
    pub species: SpeciesId,
    pub stats: [u16; 6],
    pub weight_hg: u32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct SetSpec {
    pub species: SpeciesId,
    pub ability: AbilityId,
    pub item: ItemId,
    pub moves: [MoveId; 4],
    pub n_moves: u8,
    pub stats: [u16; 6],
    pub types: [Type; 2],
    pub weight_hg: u32,
    /// 0 genderless, 1 male, 2 female.
    pub gender: u8,
    pub level: u8,
    pub mega: Option<MegaSpec>,
    pub alt: Option<AltForme>,
}

#[derive(Clone, Debug, Default)]
pub struct TeamSpec {
    pub name: String,
    pub mons: [SetSpec; 6],
    /// Reasons the engine cannot simulate this team faithfully.
    pub unsupported: Vec<String>,
}

impl TeamSpec {
    pub fn supported(&self) -> bool {
        self.unsupported.is_empty()
    }
}

fn types_of(v: Option<&Value>) -> [Type; 2] {
    let a: Vec<Type> = v
        .and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|t| t.as_str().and_then(Type::from_name)).collect())
        .unwrap_or_default();
    [a.first().copied().unwrap_or(Type::Normal), a.get(1).copied().unwrap_or(Type::None)]
}

fn stats_of(v: Option<&Value>) -> [u16; 6] {
    let mut out = [0u16; 6];
    if let Some(a) = v.and_then(|x| x.as_array()) {
        for (i, x) in a.iter().take(6).enumerate() {
            out[i] = x.as_u64().unwrap_or(0) as u16;
        }
    }
    out
}

pub fn parse_set(m: &Value, why: &mut Vec<String>) -> SetSpec {
    let d = dex();
    let sid = m.get("species").and_then(|x| x.as_str()).unwrap_or("");
    let species = d.species_id(sid).unwrap_or_else(|| {
        why.push(format!("unknown species {sid}"));
        NONE
    });
    let aid = m.get("ability").and_then(|x| x.as_str()).unwrap_or("");
    let ability = d.ability_id(aid).unwrap_or_else(|| {
        why.push(format!("unknown ability {aid}"));
        NONE
    });
    let iid = m.get("item").and_then(|x| x.as_str()).unwrap_or("");
    let item = d.item_id(iid).unwrap_or_else(|| {
        why.push(format!("unknown item {iid}"));
        NONE
    });
    let mut moves = [NONE; 4];
    let mut n_moves = 0u8;
    if let Some(a) = m.get("moves").and_then(|x| x.as_array()) {
        for x in a.iter().take(4) {
            let id = x.as_str().unwrap_or("");
            match d.move_id(id) {
                Some(mid) => {
                    moves[n_moves as usize] = mid;
                    n_moves += 1;
                }
                None => why.push(format!("unknown move {id}")),
            }
        }
    }
    let gender = match m.get("gender").and_then(|x| x.as_str()).unwrap_or("") {
        "M" => 1,
        "F" => 2,
        _ => 0,
    };
    let mega = m.get("mega").filter(|x| !x.is_null()).map(|mg| {
        let msid = mg.get("species").and_then(|x| x.as_str()).unwrap_or("");
        let maid = mg.get("ability").and_then(|x| x.as_str()).unwrap_or("");
        MegaSpec {
            species: d.species_id(msid).unwrap_or_else(|| {
                why.push(format!("unknown mega species {msid}"));
                NONE
            }),
            ability: d.ability_id(maid).unwrap_or_else(|| {
                why.push(format!("unknown mega ability {maid}"));
                NONE
            }),
            types: types_of(mg.get("types")),
            stats: stats_of(mg.get("stats")),
            weight_hg: (mg.get("weightkg").and_then(|x| x.as_f64()).unwrap_or(0.0) * 10.0).round() as u32,
        }
    });
    // Champions stats (scripts.ts statModify): HP base + EV + 75, others
    // base + EV + 20 with the nature's 1.1 / 0.9 (truncated).
    let alt_id = match sid {
        "aegislash" => Some("aegislashblade"),
        "palafin" => Some("palafinhero"),
        _ => None,
    };
    let alt = alt_id.and_then(|id| d.species_id(id)).map(|asp| {
        let base = d.sp(asp).base;
        let evs = stats_of(m.get("evs"));
        let nat = d.natures.get(m.get("nature").and_then(|x| x.as_str()).unwrap_or("")).copied().unwrap_or((None, None));
        let mut st = [0u16; 6];
        for k in 0..6 {
            let mut v = base[k] as u32 + evs[k] as u32 + if k == 0 { 75 } else { 20 };
            if k > 0 && nat.0 == Some(k) {
                v = v * 110 / 100;
            } else if k > 0 && nat.1 == Some(k) {
                v = v * 90 / 100;
            }
            st[k] = v as u16;
        }
        AltForme { species: asp, stats: st, weight_hg: d.sp(asp).weight_hg }
    });
    SetSpec {
        alt,
        species,
        ability,
        item,
        moves,
        n_moves,
        stats: stats_of(m.get("stats")),
        types: types_of(m.get("types")),
        weight_hg: (m.get("weightkg").and_then(|x| x.as_f64()).unwrap_or(0.0) * 10.0).round() as u32,
        gender,
        level: m.get("level").and_then(|x| x.as_u64()).unwrap_or(50) as u8,
        mega,
    }
}

/// Everything the engine cannot run faithfully for this set.
pub fn set_support(set: &SetSpec) -> Vec<String> {
    let d = dex();
    let mut why = vec![];
    for &m in &set.moves[..set.n_moves as usize] {
        for r in &d.mv(m).unsupported {
            why.push(format!("{}: {}", d.mv(m).id, r));
        }
    }
    for r in &d.ability(set.ability).unsupported {
        why.push(r.clone());
    }
    if let Some(mg) = &set.mega {
        for r in &d.ability(mg.ability).unsupported {
            why.push(format!("mega {r}"));
        }
    }
    for r in &d.item(set.item).unsupported {
        why.push(r.clone());
    }
    why
}

pub fn parse_team(t: &Value) -> TeamSpec {
    let mut why = vec![];
    let mut mons = [SetSpec::default(); 6];
    if let Some(a) = t.get("mons").and_then(|x| x.as_array()) {
        for (i, m) in a.iter().take(6).enumerate() {
            mons[i] = parse_set(m, &mut why);
        }
        if a.len() != 6 {
            why.push(format!("team has {} mons", a.len()));
        }
    }
    for set in &mons {
        why.extend(set_support(set));
    }
    why.sort();
    why.dedup();
    TeamSpec { name: t.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(), mons, unsupported: why }
}

pub fn load_teams(path: &str) -> Result<Vec<TeamSpec>, String> {
    let v = read_json(path)?;
    let arr = v.get("teams").and_then(|x| x.as_array()).ok_or("teams file: no 'teams'")?;
    Ok(arr.iter().map(parse_team).collect())
}
