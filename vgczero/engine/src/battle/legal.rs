//! Legal actions per slot, move disabling, trapping and priority.

use super::{Battle, Phase};
use crate::actions::{move_action, switch_action, PASS, T_ALLY, T_FOE0, T_FOE1};
use crate::dex::{dex, flag, Category, MoveId, Target, Terrain};
use crate::kinds::{Ab, It, MoveFx};
use crate::state::Pos;
use crate::types::Type;

impl Battle {
    /// Bitmask over the 31 slot actions for this side/slot right now.
    pub fn legal_mask(&self, side: usize, slot: usize) -> u32 {
        let p = Pos::new(side, slot);
        match self.phase {
            Phase::Move => {
                if !self.is_live(p) {
                    return 1 << PASS;
                }
                self.move_phase_mask(p)
            }
            Phase::Switch { .. } => {
                if !self.switch_slots[side][slot] {
                    return 1 << PASS;
                }
                let mut mask = 0u32;
                for i in 0..6 {
                    if self.sides[side].can_switch_to(i) {
                        mask |= 1 << switch_action(i);
                    }
                }
                if mask == 0 {
                    mask = 1 << PASS;
                }
                mask
            }
            _ => 1 << PASS,
        }
    }

    fn move_phase_mask(&self, p: Pos) -> u32 {
        let m = self.m(p);
        let mut mask = 0u32;
        // Locked into an action: recharge, second turn of a charge move.
        if m.vol.must_recharge || m.vol.charging > 0 {
            let s = if m.vol.charging > 0 { (m.vol.charging - 1) as usize } else { 0 };
            return 1 << move_action(s, T_FOE0, false);
        }
        let can_mega = self.can_mega(p);
        if self.must_struggle(p) {
            mask |= 1 << move_action(0, T_FOE0, false);
            if can_mega {
                mask |= 1 << move_action(0, T_FOE0, true);
            }
        } else {
            for s in 0..m.n_moves as usize {
                if self.move_disabled(p, s) {
                    continue;
                }
                let mid = m.moves[s];
                for t in self.move_targets(p, mid) {
                    mask |= 1 << move_action(s, t, false);
                    if can_mega {
                        mask |= 1 << move_action(s, t, true);
                    }
                }
            }
        }
        if !self.is_trapped(p) {
            for i in 0..6 {
                if self.sides[p.s()].can_switch_to(i) {
                    mask |= 1 << switch_action(i);
                }
            }
        }
        if mask == 0 {
            mask = 1 << PASS;
        }
        mask
    }

    /// Target choices (encoded 0/1/2) offered for a move.
    pub fn move_targets(&self, p: Pos, mid: MoveId) -> impl Iterator<Item = u8> {
        let mv = dex().mv(mid);
        let mut out = [false; 3];
        match mv.target {
            Target::Normal | Target::Any | Target::AdjacentFoe => {
                let f0 = self.is_live(p.foe(0));
                let f1 = self.is_live(p.foe(1));
                out[T_FOE0 as usize] = f0;
                out[T_FOE1 as usize] = f1;
                if !f0 && !f1 {
                    out[T_FOE0 as usize] = true;
                }
                let ally_ok = mv.target != Target::AdjacentFoe
                    && self.is_live(p.ally())
                    && (mv.category == Category::Status || self.cfg.ally_damage_targeting || ally_target_move(mv.fx));
                out[T_ALLY as usize] = ally_ok;
            }
            Target::AdjacentAllyOrSelf => {
                out[T_FOE0 as usize] = true;
                out[T_ALLY as usize] = self.is_live(p.ally());
            }
            _ => out[T_FOE0 as usize] = true,
        }
        (0..3u8).filter(move |&t| out[t as usize])
    }

    pub fn can_mega(&self, p: Pos) -> bool {
        let m = self.m(p);
        m.can_mega && !m.is_mega && !self.sides[p.s()].mega_used && m.item != 0 && dex().item(m.item).is_mega_stone
    }

    /// True if every move is unusable.
    pub fn must_struggle(&self, p: Pos) -> bool {
        let m = self.m(p);
        (0..m.n_moves as usize).all(|s| self.move_disabled(p, s))
    }

    /// Is move slot `s` unusable this turn?
    pub fn move_disabled(&self, p: Pos, s: usize) -> bool {
        let m = self.m(p);
        if s >= m.n_moves as usize || m.pp[s] == 0 {
            return true;
        }
        let mid = m.moves[s];
        let mv = dex().mv(mid);
        let v = &m.vol;
        if v.encore > 0 && v.encore_slot as usize != s {
            return true;
        }
        if v.choice_lock > 0 && (v.choice_lock - 1) as usize != s && is_choice(self.it(p)) {
            return true;
        }
        if v.disable > 0 && v.disable_slot as usize == s {
            return true;
        }
        if v.taunt > 0 && mv.category == Category::Status {
            return true;
        }
        if v.throat_chop > 0 && mv.flags & flag::SOUND != 0 {
            return true;
        }
        if v.heal_block > 0 && mv.flags & flag::HEAL != 0 {
            return true;
        }
        if self.field.gravity > 0 && mv.flags & flag::GRAVITY != 0 {
            return true;
        }
        if v.torment && v.last_move == mid {
            return true;
        }
        if mv.flags & flag::CANTUSETWICE != 0 && v.last_move == mid {
            return true;
        }
        // Champions: Fake Out / First Impression only on the first action.
        if matches!(mv.fx, MoveFx::FakeOut | MoveFx::FirstImpression) && v.move_actions > 0 {
            return true;
        }
        // Imprison: a foe with Imprison that knows this move.
        for q in self.live_foes(p) {
            let f = self.m(q);
            if f.vol.imprison && f.moves[..f.n_moves as usize].contains(&mid) {
                return true;
            }
        }
        false
    }

    pub fn is_trapped(&self, p: Pos) -> bool {
        let m = self.m(p);
        if m.has_type(Type::Ghost) || self.ab(p) == Ab::RunAway {
            return false;
        }
        if m.vol.ingrain || m.vol.no_retreat || m.vol.partial_trap > 0 {
            return true;
        }
        for q in self.live_foes(p) {
            if self.ab(q) == Ab::ShadowTag && self.ab(p) != Ab::ShadowTag {
                return true;
            }
        }
        false
    }

    /// ModifyPriority: Prankster, Gale Wings, Triage, Grassy Glide.
    pub fn modify_priority(&self, p: Pos, mid: MoveId, base: i8) -> i8 {
        let mv = dex().mv(mid);
        let m = self.m(p);
        let mut pr = base;
        match mv.fx {
            MoveFx::GrassyGlide => {
                if self.field.terrain == Terrain::Grassy && self.is_grounded(p) {
                    pr += 1;
                }
            }
            _ => {}
        }
        match self.ab(p) {
            Ab::Prankster if mv.category == Category::Status => pr += 1,
            Ab::GaleWings if mv.ty == Type::Flying && m.hp == m.max_hp => pr += 1,
            _ => {}
        }
        pr
    }
}

pub fn is_choice(it: It) -> bool {
    matches!(it, It::ChoiceBand | It::ChoiceScarf | It::ChoiceSpecs)
}

/// Damaging moves commonly aimed at an ally on purpose.
fn ally_target_move(fx: MoveFx) -> bool {
    matches!(fx, MoveFx::PollenPuff | MoveFx::BeatUp)
}
