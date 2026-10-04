//! Switching out and in, and switch-in effects.

use super::effects::{BoostSrc, DmgKind};
use super::{blog, Battle};
use crate::dex::{dex, Status, Terrain, Weather, B_SPE};
use crate::kinds::{Ab, It};
use crate::state::{Pos, Vol};
use crate::types::Type;

/// Abilities Trace cannot copy.
const NO_TRACE: &[&str] = &[
    "trace", "illusion", "imposter", "stancechange", "zerotohero", "commander", "receiver",
    "powerofalchemy", "flowergift", "forecast", "hungerswitch", "multitype", "neutralizinggas",
    "rkssystem", "schooling", "shieldsdown", "zenmode", "disguise", "iceface", "gulpmissile",
    "battlebond", "comatose", "protosynthesis", "quarkdrive", "wonderguard",
];

impl Battle {
    /// Replace the Pokemon at `p` with team member `to` (no entry effects).
    pub fn switch_out_in(&mut self, p: Pos, to: usize) {
        let s = p.s();
        let old = self.sides[s].active[p.i()];
        let mut baton: Option<(crate::dex::Boosts, Vol)> = None;
        if old != crate::state::NO_MON {
            let oi = old as usize;
            let alive = self.sides[s].mons[oi].alive();
            if alive {
                // Switch-out abilities.
                let ab = dex().ab(self.sides[s].mons[oi].ability);
                {
                    let m = &mut self.sides[s].mons[oi];
                    match ab {
                        Ab::Regenerator => {
                            let h = (m.max_hp / 3).min(m.max_hp - m.hp);
                            if m.vol.heal_block == 0 {
                                m.hp += h;
                            }
                        }
                        Ab::NaturalCure => {
                            m.status = Status::None;
                            m.status_turns = 0;
                        }
                        _ => {}
                    }
                }
                let m = &self.sides[s].mons[oi];
                if m.vol.switch_copyvolatile {
                    baton = Some((m.boosts, m.vol));
                } else if m.vol.switch_shedtail {
                    let mut v = Vol::default();
                    v.substitute = m.vol.substitute;
                    baton = Some(([0; 7], v));
                }
            }
            let m = &mut self.sides[s].mons[oi];
            m.slot = -1;
            if !m.fainted {
                m.boosts = [0; 7];
                m.vol = Vol::default();
                // Types / ability revert on switch-out (megas stay mega).
                m.types = m.base_types;
                m.ability = m.base_ability;
                if m.status == Status::Tox {
                    m.status_turns = 0;
                }
            }
        }
        self.sides[s].active[p.i()] = to as u8;
        if old != crate::state::NO_MON {
            self.sides[s].swap_order(old, to as u8);
        }
        let m = &mut self.sides[s].mons[to];
        m.slot = p.slot as i8;
        m.vol = Vol::default();
        m.vol.newly_switched = true;
        m.reveal.seen = true;
        if let Some((boosts, v)) = baton {
            if v.substitute > 0 && v.switch_shedtail {
                m.vol.substitute = v.substitute;
            } else {
                m.boosts = boosts;
                m.vol.substitute = v.substitute;
                m.vol.confusion = v.confusion;
                m.vol.leech_seed = v.leech_seed;
                m.vol.leech_seed_src = v.leech_seed_src;
                m.vol.aqua_ring = v.aqua_ring;
                m.vol.ingrain = v.ingrain;
                m.vol.magnet_rise = v.magnet_rise;
                m.vol.crit_stage = v.crit_stage;
                m.vol.perish = v.perish;
                m.vol.curse = v.curse;
                m.vol.heal_block = v.heal_block;
                m.vol.no_retreat = v.no_retreat;
            }
        }
        blog!(self, "|switch|{}|{}/{}", self.name(p), self.m(p).hp, self.m(p).max_hp);
    }

    /// Forced out (Red Card, Roar...): random replacement.
    pub fn drag_out(&mut self, p: Pos) {
        if !self.is_live(p) {
            return;
        }
        let s = p.s();
        let options: Vec<usize> = (0..6).filter(|&i| self.sides[s].can_switch_to(i)).collect();
        if options.is_empty() {
            return;
        }
        let to = options[self.rng.below(options.len() as u32) as usize];
        self.switch_out_in(p, to);
        self.run_switch_in(&[p]);
    }

    /// Run switch-in effects for Pokemon that just entered, in Showdown's
    /// order: handlers sorted by priority, then speed (ties resolved once).
    pub fn run_switch_in(&mut self, switchers: &[Pos]) {
        let mut live: Vec<Pos> = switchers.iter().copied().filter(|&p| self.is_live(p)).collect();
        if live.is_empty() {
            return;
        }
        // Resolve speed order once (random tiebreaks), like speedOrder.
        let mut keyed: Vec<(i32, u32, Pos)> = live.iter().map(|&p| (self.action_speed(p), self.rng.next_u64() as u32, p)).collect();
        keyed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        live = keyed.into_iter().map(|x| x.2).collect();

        // Priority > 0: Healing Wish (slot condition).
        for &p in &live {
            if self.sides[p.s()].healing_wish[p.i()] && self.is_live(p) {
                let m = self.m(p);
                if m.hp < m.max_hp || m.status != Status::None {
                    self.sides[p.s()].healing_wish[p.i()] = false;
                    let mh = self.m(p).max_hp as u32;
                    self.heal(p, mh);
                    self.cure_status(p);
                }
            }
        }
        // Hazards.
        for &p in &live {
            self.entry_hazards(p);
        }
        // Abilities (priority 0), in speed order.
        for &p in &live {
            if self.is_live(p) {
                self.ability_start(p);
            }
        }
        // Items (seeds have onSwitchInPriority -1, after abilities).
        for &p in &live {
            if self.is_live(p) {
                self.item_start(p);
            }
        }
        for &p in &live {
            if self.is_live(p) {
                self.update_items(p);
            }
        }
    }

    fn entry_hazards(&mut self, p: Pos) {
        if !self.is_live(p) {
            return;
        }
        let conds = self.sides[p.s()].conds;
        if conds.stealth_rock {
            let eff = self.effectiveness(Type::Rock, p, None);
            let mh = self.m(p).max_hp as u32;
            let dmg = match eff {
                2 => mh / 2,
                1 => mh / 4,
                0 => mh / 8,
                -1 => mh / 16,
                _ => mh / 32,
            };
            self.damage(p, dmg.max(1), None, DmgKind::Indirect);
        }
        if conds.toxic_spikes > 0 && self.is_live(p) && self.is_grounded(p) {
            if self.m(p).has_type(Type::Poison) {
                self.sides[p.s()].conds.toxic_spikes = 0;
            } else {
                let st = if conds.toxic_spikes >= 2 { Status::Tox } else { Status::Psn };
                self.try_set_status(p, st, None);
            }
        }
        if conds.spikes > 0 && self.is_live(p) && self.is_grounded(p) {
            let mh = self.m(p).max_hp as u32;
            let dmg = match conds.spikes {
                1 => mh / 8,
                2 => mh / 6,
                _ => mh / 4,
            };
            self.damage(p, dmg, None, DmgKind::Indirect);
        }
        if conds.sticky_web && self.is_live(p) && self.is_grounded(p) {
            let mut d = [0i8; 7];
            d[B_SPE] = -1;
            let foe = p.foe(0);
            self.boost(p, &d, BoostSrc::Move(foe));
        }
    }

    /// An ability's onStart (switch-in or gained mid-battle).
    pub fn ability_start(&mut self, p: Pos) {
        if !self.is_live(p) {
            return;
        }
        let ab = self.ab(p);
        let mut fallen = 0;
        match ab {
            Ab::Intimidate => self.intimidate(p),
            Ab::Drizzle => {
                self.reveal_ability(p);
                self.set_weather(Weather::Rain, p);
            }
            Ab::Drought => {
                self.reveal_ability(p);
                self.set_weather(Weather::Sun, p);
            }
            Ab::SandStream => {
                self.reveal_ability(p);
                self.set_weather(Weather::Sand, p);
            }
            Ab::SnowWarning => {
                self.reveal_ability(p);
                self.set_weather(Weather::Snow, p);
            }
            Ab::GrassySurge => {
                self.reveal_ability(p);
                self.set_terrain(Terrain::Grassy, p);
            }
            Ab::PsychicSurge => {
                self.reveal_ability(p);
                self.set_terrain(Terrain::Psychic, p);
            }
            Ab::ElectricSurge => {
                self.reveal_ability(p);
                self.set_terrain(Terrain::Electric, p);
            }
            Ab::MistySurge => {
                self.reveal_ability(p);
                self.set_terrain(Terrain::Misty, p);
            }
            Ab::Hospitality => {
                if let Some(a) = self.live_ally(p) {
                    let amt = self.m(a).max_hp as u32 / 4;
                    if self.m(a).hp < self.m(a).max_hp {
                        self.reveal_ability(p);
                        self.heal(a, amt);
                    }
                }
            }
            Ab::Trace => {
                // Copy a random opposing ability that can be copied.
                let foes: Vec<Pos> = self.live_foes(p).collect();
                let ok: Vec<Pos> = foes
                    .into_iter()
                    .filter(|&q| {
                        let a = dex().ability(self.m(q).ability);
                        !a.cantsuppress && !NO_TRACE.contains(&a.id.as_str())
                    })
                    .collect();
                if !ok.is_empty() {
                    let q = ok[self.rng.below(ok.len() as u32) as usize];
                    let new = self.m(q).ability;
                    self.reveal_ability(p);
                    self.reveal_ability(q);
                    self.mm(p).ability = new;
                    self.mm(p).vol.ability_changed = true;
                    self.ability_start(p);
                }
            }
            Ab::Frisk => {
                for q in self.live_foes(p).collect::<Vec<_>>() {
                    self.reveal_item(q);
                }
                self.reveal_ability(p);
            }
            Ab::SupremeOverlord => {
                fallen = self.sides[p.s()].total_fainted.min(5);
            }
            Ab::FairyAura | Ab::Pressure | Ab::Unnerve | Ab::MoldBreaker | Ab::AirLock | Ab::CloudNine => {
                self.reveal_ability(p);
            }
            Ab::CuriousMedicine => {
                if let Some(a) = self.live_ally(p) {
                    self.mm(a).boosts = [0; 7];
                }
            }
            _ => {}
        }
        if ab == Ab::SupremeOverlord {
            self.mm(p).vol.fallen = fallen;
        }
    }

    /// An item's onStart at switch-in.
    pub fn item_start(&mut self, p: Pos) {
        match self.it(p) {
            It::AirBalloon => self.reveal_item(p),
            It::GrassySeed | It::PsychicSeed | It::ElectricSeed | It::MistySeed => self.check_seed(p),
            _ => {}
        }
    }
}
