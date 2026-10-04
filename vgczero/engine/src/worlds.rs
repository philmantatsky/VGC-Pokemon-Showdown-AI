//! Hidden-information sampling ("worlds") for search.
//!
//! From one player's point of view, the opponent's unrevealed details (moves,
//! item, ability, stat spread, which unseen Pokemon were brought) are unknown.
//! `determinize` replaces them with a set drawn from the tournament team pool
//! that is consistent with everything revealed so far, keeping all visible
//! dynamic state (HP percentage, status, boosts, volatiles, field).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::battle::{Battle, Phase};
use crate::dex::SpeciesId;
use crate::rng::Rng;
use crate::state::Mon;
use crate::teams::{SetSpec, TeamSpec};

pub struct SetPool {
    pub by_species: HashMap<SpeciesId, Vec<SetSpec>>,
}

static POOL: OnceLock<SetPool> = OnceLock::new();

pub fn set_pool() -> Option<&'static SetPool> {
    POOL.get()
}

/// Build the global set pool from teams (first call wins).
pub fn init_set_pool(teams: &[TeamSpec]) -> &'static SetPool {
    POOL.get_or_init(|| SetPool::from_teams(teams))
}

impl SetPool {
    pub fn from_teams(teams: &[TeamSpec]) -> SetPool {
        let mut by_species: HashMap<SpeciesId, Vec<SetSpec>> = HashMap::new();
        for t in teams {
            for s in &t.mons {
                let v = by_species.entry(s.species).or_default();
                // Keep duplicates: frequency in the pool is the prior.
                v.push(*s);
            }
        }
        SetPool { by_species }
    }

    /// Sets for `species` consistent with what `viewer` knows about `m`.
    fn candidates<'a>(&'a self, m: &Mon, sheet_open: bool, strict: u8) -> Vec<&'a SetSpec> {
        let Some(v) = self.by_species.get(&m.base_species) else { return vec![] };
        let r = m.reveal;
        v.iter()
            .filter(|s| {
                // Mega: if it has Mega Evolved, the set must allow it.
                if m.is_mega && s.mega.map(|g| g.species) != Some(m.species) {
                    return false;
                }
                if strict >= 1 {
                    for j in 0..m.n_moves as usize {
                        if sheet_open || (r.moves >> j) & 1 == 1 {
                            if !s.moves[..s.n_moves as usize].contains(&m.moves[j]) {
                                return false;
                            }
                        }
                    }
                }
                if strict >= 2 {
                    if sheet_open || r.item {
                        let known = if m.item != 0 { m.item } else { m.last_item };
                        if known != 0 && s.item != known {
                            return false;
                        }
                    }
                    if sheet_open || r.ability {
                        // Compare like with like: the Mega ability for a Mega,
                        // the set's own ability otherwise.
                        let ok = if m.is_mega {
                            s.mega.map(|g| g.ability == m.base_ability).unwrap_or(false)
                        } else {
                            s.ability == m.base_ability
                        };
                        if !ok {
                            return false;
                        }
                    }
                }
                true
            })
            .collect()
    }
}

/// Replace `src` with a sampled set, keeping the visible dynamic state.
fn resample_mon(src: &Mon, set: &SetSpec) -> Mon {
    let mut m = Mon::from_set(set, src.team_idx);
    // Visible state.
    m.brought = src.brought;
    m.slot = src.slot;
    m.fainted = src.fainted;
    m.status = src.status;
    m.status_turns = src.status_turns;
    m.boosts = src.boosts;
    m.vol = src.vol;
    // Revealed moves by identity (slot order differs between sets).
    m.reveal = src.reveal;
    m.reveal.moves = 0;
    for j in 0..src.n_moves as usize {
        if (src.reveal.moves >> j) & 1 == 1 {
            if let Some(k) = m.move_slot(src.moves[j]) {
                m.reveal.moves |= 1 << k;
            }
        }
    }
    m.ate_berry = src.ate_berry;
    m.item_knocked = src.item_knocked;
    // HP: the same Champions percentage (floor(100 hp / max), min 1) the
    // viewer saw, on the sampled max HP.
    if src.fainted || src.hp == 0 {
        m.hp = 0;
    } else {
        let mx = m.max_hp as u32;
        let pct = (100 * src.hp as u32 / src.max_hp.max(1) as u32).max(1);
        let mut hp = ((pct * mx + 99) / 100).clamp(1, mx);
        while hp > 1 && (100 * hp / mx).max(1) > pct {
            hp -= 1;
        }
        while hp < mx && (100 * hp / mx).max(1) < pct {
            hp += 1;
        }
        if src.hp == src.max_hp {
            hp = mx;
        }
        m.hp = hp as u16;
    }
    // Item: a known-consumed item stays consumed.
    if src.item == 0 && src.reveal.item {
        m.last_item = m.item;
        m.item = 0;
    }
    // Mega forme.
    if src.is_mega {
        m.species = m.mega_species;
        m.types = m.mega_types;
        m.base_types = m.mega_types;
        for i in 1..6 {
            m.stats[i] = m.mega_stats[i];
        }
        m.weight_hg = m.mega_weight_hg;
        m.ability = m.mega_ability;
        m.base_ability = m.mega_ability;
        m.is_mega = true;
    }
    // Visible in-battle changes (Trace, Skill Swap, Soak...) carry over.
    if src.vol.ability_changed {
        m.ability = src.ability;
    }
    if src.vol.type_changed {
        m.types = src.types;
    }
    // PP: copy for moves the opponent has shown, full otherwise.
    for j in 0..m.n_moves as usize {
        if let Some(k) = src.move_slot(m.moves[j]) {
            m.pp[j] = src.pp[k].min(m.max_pp[j]);
        }
    }
    // Choice lock / encore / disable refer to move slots: remap by move id.
    let new_moves = m;
    let remap = |slot1: u8| -> u8 {
        if slot1 == 0 {
            return 0;
        }
        let mid = src.moves[(slot1 - 1) as usize];
        new_moves.move_slot(mid).map(|k| k as u8 + 1).unwrap_or(0)
    };
    m.vol.choice_lock = remap(src.vol.choice_lock);
    m.vol.charging = remap(src.vol.charging);
    if src.vol.encore > 0 {
        match m.move_slot(src.moves[src.vol.encore_slot as usize]) {
            Some(k) => m.vol.encore_slot = k as u8,
            None => m.vol.encore = 0,
        }
    }
    if src.vol.disable > 0 {
        match m.move_slot(src.moves[src.vol.disable_slot as usize]) {
            Some(k) => m.vol.disable_slot = k as u8,
            None => m.vol.disable = 0,
        }
    }
    m
}

/// A copy of `b` in which everything `viewer` cannot see about the opponent
/// is resampled. The battle RNG is reseeded from `rng`.
pub fn determinize(b: &Battle, viewer: usize, pool: &SetPool, rng: &mut Rng) -> Battle {
    let mut out = b.clone();
    out.log_lines.clear();
    out.cfg.log = false;
    out.rng = rng.fork();
    let opp = 1 - viewer;
    let sheet = b.sides[opp].sheet_open;
    let in_preview = b.phase == Phase::TeamPreview;

    // Which unseen opponents were brought?
    if !in_preview {
        let side = &b.sides[opp];
        let seen: Vec<usize> = (0..6).filter(|&i| side.mons[i].reveal.seen).collect();
        let unseen: Vec<usize> = (0..6).filter(|&i| !side.mons[i].reveal.seen).collect();
        let need = 4usize.saturating_sub(seen.len()).min(unseen.len());
        let mut pick = unseen.clone();
        for i in 0..pick.len() {
            let j = i + rng.below((pick.len() - i) as u32) as usize;
            pick.swap(i, j);
        }
        for &i in &unseen {
            out.sides[opp].mons[i].brought = false;
        }
        for &i in pick.iter().take(need) {
            out.sides[opp].mons[i].brought = true;
        }
    }
    for i in 0..6 {
        let src = b.sides[opp].mons[i];
        let mut cands = pool.candidates(&src, sheet, 2);
        if cands.is_empty() {
            cands = pool.candidates(&src, sheet, 1);
        }
        if cands.is_empty() {
            cands = pool.candidates(&src, sheet, 0);
        }
        if cands.is_empty() {
            continue; // unknown species: keep as is
        }
        let set = cands[rng.below(cands.len() as u32) as usize];
        let brought = out.sides[opp].mons[i].brought;
        let mut m = resample_mon(&src, set);
        m.brought = brought;
        out.sides[opp].mons[i] = m;
    }
    out
}
