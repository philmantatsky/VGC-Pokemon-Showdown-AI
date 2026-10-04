//! End-of-turn effects in Showdown's residual order: every handler from the
//! field, sides and Pokemon is sorted by (order, speed desc, subOrder), and
//! durations tick at their handler's slot.

use super::effects::{BoostSrc, DmgKind};
use super::{blog, Battle};
use crate::dex::{Status, Terrain, Weather, B_SPE};
use crate::kinds::{Ab, It};
use crate::state::{Pos, ProtectKind};
use crate::types::Type;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Res {
    Weather,
    Wish,
    GrassyHeal,
    Healer,
    Leftovers,
    AquaRing,
    Ingrain,
    LeechSeed,
    Poison,
    Burn,
    Curse,
    SaltCure,
    PartialTrap,
    Taunt,
    Encore,
    Disable,
    MagnetRise,
    HealBlock,
    ThroatChop,
    Yawn,
    PerishSong,
    SideConds,
    TrickRoom,
    Gravity,
    Terrain,
    SpeedBoost,
    Moody,
    /// Volatiles with a duration but no residual order (Protect, flinch...).
    EndOfTurnVolatiles,
}

impl Battle {
    pub(crate) fn run_residual(&mut self) {
        // HP before residuals, for Emergency Exit.
        let mut hp_before = [0u16; 4];
        for c in 0..4 {
            let p = crate::state::Pos::from_code(c);
            if self.is_live(p) {
                hp_before[c as usize] = self.m(p).hp;
            }
        }
        // (order, speed, subOrder, kind, position code or 255 for field/side)
        let mut items: Vec<(u16, i32, u8, Res, u8)> = Vec::with_capacity(32);
        items.push((1, 0, 0, Res::Weather, 255));
        items.push((4, 0, 0, Res::Wish, 255));
        for s in 0..2 {
            items.push((26, 0, 0, Res::SideConds, s as u8));
        }
        items.push((27, 0, 1, Res::TrickRoom, 255));
        items.push((27, 0, 2, Res::Gravity, 255));
        items.push((27, 0, 7, Res::Terrain, 255));
        for c in 0..4u8 {
            let p = Pos::from_code(c);
            if !self.is_live(p) {
                continue;
            }
            let spd = self.action_speed(p);
            let m = self.m(p);
            let v = m.vol;
            if self.field.terrain == Terrain::Grassy {
                items.push((5, spd, 2, Res::GrassyHeal, c));
            }
            if self.ab(p) == Ab::Healer {
                items.push((5, spd, 3, Res::Healer, c));
            }
            if self.it(p) == It::Leftovers {
                items.push((5, spd, 4, Res::Leftovers, c));
            }
            if v.aqua_ring {
                items.push((6, spd, 0, Res::AquaRing, c));
            }
            if v.ingrain {
                items.push((7, spd, 0, Res::Ingrain, c));
            }
            if v.leech_seed {
                items.push((8, spd, 0, Res::LeechSeed, c));
            }
            if matches!(m.status, Status::Psn | Status::Tox) {
                items.push((9, spd, 0, Res::Poison, c));
            }
            if m.status == Status::Brn {
                items.push((10, spd, 0, Res::Burn, c));
            }
            if v.curse {
                items.push((12, spd, 0, Res::Curse, c));
            }
            if v.salt_cure {
                items.push((13, spd, 0, Res::SaltCure, c));
            }
            if v.partial_trap > 0 {
                items.push((13, spd, 0, Res::PartialTrap, c));
            }
            if v.taunt > 0 {
                items.push((15, spd, 0, Res::Taunt, c));
            }
            if v.encore > 0 {
                items.push((16, spd, 0, Res::Encore, c));
            }
            if v.disable > 0 {
                items.push((17, spd, 0, Res::Disable, c));
            }
            if v.magnet_rise > 0 {
                items.push((18, spd, 0, Res::MagnetRise, c));
            }
            if v.heal_block > 0 {
                items.push((20, spd, 0, Res::HealBlock, c));
            }
            if v.throat_chop > 0 {
                items.push((22, spd, 0, Res::ThroatChop, c));
            }
            if v.yawn > 0 {
                items.push((23, spd, 0, Res::Yawn, c));
            }
            if v.perish > 0 {
                items.push((24, spd, 0, Res::PerishSong, c));
            }
            if self.ab(p) == Ab::SpeedBoost && v.active_turns > 0 {
                items.push((28, spd, 2, Res::SpeedBoost, c));
            }
            if self.ab(p) == Ab::Moody {
                items.push((28, spd, 2, Res::Moody, c));
            }
            items.push((u16::MAX, spd, 0, Res::EndOfTurnVolatiles, c));
        }
        // Sort: order asc, speed desc, subOrder asc; ties broken randomly.
        let mut keyed: Vec<_> = items.into_iter().map(|x| (x, self.rng.next_u64())).collect();
        keyed.sort_by(|a, b| a.0 .0.cmp(&b.0 .0).then(b.0 .1.cmp(&a.0 .1)).then(a.0 .2.cmp(&b.0 .2)).then(a.1.cmp(&b.1)));
        for ((_, _, _, kind, code), _) in keyed {
            if self.ended() {
                return;
            }
            if code == 255 || kind == Res::SideConds {
                self.residual_field(kind, code);
            } else {
                let p = Pos::from_code(code);
                if !self.is_live(p) {
                    continue;
                }
                self.residual_mon(kind, p);
            }
            // Perish Song faints through its end callback, which Showdown's
            // fieldEvent does not follow with faintMessages: simultaneous
            // perishes faint together (the last one decides a double KO).
            if kind == Res::PerishSong {
                continue;
            }
            self.check_win_quiet();
        }
        // Per-turn counters.
        for c in 0..4 {
            let p = Pos::from_code(c);
            if self.is_live(p) {
                let m = self.mm(p);
                m.vol.active_turns = m.vol.active_turns.saturating_add(1);
            }
        }
        // Emergency Exit from residual damage.
        for c in 0..4u8 {
            let p = Pos::from_code(c);
            if self.is_live(p) {
                let m = self.m(p);
                let half = m.max_hp as u32;
                if (m.hp as u32) * 2 <= half && (hp_before[c as usize] as u32) * 2 > half {
                    self.emergency_exit(p);
                }
            }
        }
    }

    fn check_win_quiet(&mut self) {
        // Faint Pokemon at 0 HP as residuals go (Showdown faints after each handler).
        self.process_faints();
    }

    fn residual_field(&mut self, kind: Res, code: u8) {
        match kind {
            Res::Weather => {
                if self.field.weather == Weather::None {
                    return;
                }
                self.field.weather_turns = self.field.weather_turns.saturating_sub(1);
                if self.field.weather_turns == 0 {
                    blog!(self, "|-weather|none");
                    self.field.weather = Weather::None;
                    return;
                }
                let w = self.weather();
                // Weather effects on each active Pokemon, by speed.
                let mut ps: Vec<(i32, u64, Pos)> =
                    self.live_positions().collect::<Vec<_>>().into_iter().map(|p| (self.action_speed(p), self.rng.next_u64(), p)).collect();
                ps.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                for (_, _, p) in ps {
                    if !self.is_live(p) {
                        continue;
                    }
                    self.weather_effect(p, w);
                    self.process_faints();
                    if self.ended() {
                        return;
                    }
                }
            }
            Res::Wish => {
                for s in 0..2 {
                    for slot in 0..2 {
                        let (t, amt) = self.sides[s].wish[slot];
                        if t == 0 {
                            continue;
                        }
                        if t == 1 {
                            self.sides[s].wish[slot] = (0, 0);
                            let p = Pos::new(s, slot);
                            if self.is_live(p) {
                                self.heal(p, amt as u32);
                            }
                        } else {
                            self.sides[s].wish[slot] = (t - 1, amt);
                        }
                    }
                }
            }
            Res::SideConds => {
                let c = &mut self.sides[code as usize].conds;
                // subOrders: reflect 1, light screen 2, safeguard 3, tailwind 5, aurora veil 10
                c.reflect = c.reflect.saturating_sub(1);
                c.light_screen = c.light_screen.saturating_sub(1);
                c.safeguard = c.safeguard.saturating_sub(1);
                c.mist = c.mist.saturating_sub(1);
                c.tailwind = c.tailwind.saturating_sub(1);
                c.aurora_veil = c.aurora_veil.saturating_sub(1);
                c.wide_guard = false;
                c.quick_guard = false;
            }
            Res::TrickRoom => {
                if self.field.trick_room > 0 {
                    self.field.trick_room -= 1;
                    if self.field.trick_room == 0 {
                        blog!(self, "|-fieldend|move: Trick Room");
                    }
                }
            }
            Res::Gravity => {
                self.field.gravity = self.field.gravity.saturating_sub(1);
            }
            Res::Terrain => {
                if self.field.terrain != Terrain::None {
                    self.field.terrain_turns = self.field.terrain_turns.saturating_sub(1);
                    if self.field.terrain_turns == 0 {
                        blog!(self, "|-fieldend|{:?}", self.field.terrain);
                        self.field.terrain = Terrain::None;
                    }
                }
            }
            _ => {}
        }
    }

    fn weather_effect(&mut self, p: Pos, w: Weather) {
        let mh = self.m(p).max_hp as u32;
        let ab = self.ab(p);
        match w {
            Weather::Sand => {
                let m = self.m(p);
                let immune = m.has_type(Type::Rock)
                    || m.has_type(Type::Ground)
                    || m.has_type(Type::Steel)
                    || matches!(ab, Ab::SandVeil | Ab::SandRush | Ab::SandForce | Ab::Overcoat | Ab::MagicGuard);
                if !immune {
                    self.damage(p, (mh / 16).max(1), None, DmgKind::Indirect);
                }
            }
            Weather::Rain => match ab {
                Ab::RainDish => {
                    self.heal(p, (mh / 16).max(1));
                }
                Ab::DrySkin => {
                    self.heal(p, (mh / 8).max(1));
                }
                _ => {}
            },
            Weather::Sun => match ab {
                Ab::DrySkin | Ab::SolarPower => {
                    self.damage(p, (mh / 8).max(1), None, DmgKind::Indirect);
                }
                _ => {}
            },
            Weather::Snow => {
                if ab == Ab::IceBody {
                    self.heal(p, (mh / 16).max(1));
                }
            }
            Weather::None => {}
        }
        self.update_items(p);
    }

    fn residual_mon(&mut self, kind: Res, p: Pos) {
        let mh = self.m(p).max_hp as u32;
        match kind {
            Res::GrassyHeal => {
                if self.field.terrain == Terrain::Grassy && self.is_grounded(p) && self.m(p).vol.semi_inv == crate::state::SemiInv::None {
                    self.heal(p, (mh / 16).max(1));
                }
            }
            Res::Healer => {
                if let Some(a) = self.live_ally(p) {
                    if self.m(a).status != Status::None && self.rng.chance(1, 2) {
                        self.cure_status(a);
                    }
                }
            }
            Res::Leftovers => {
                if self.m(p).hp < self.m(p).max_hp {
                    self.reveal_item(p);
                }
                self.heal(p, (mh / 16).max(1));
            }
            Res::AquaRing | Res::Ingrain => {
                let mut amt = (mh / 16).max(1);
                if self.it(p) == It::BigRoot {
                    amt = amt * 13 / 10;
                }
                self.heal(p, amt);
            }
            Res::LeechSeed => {
                let src = Pos::from_code(self.m(p).vol.leech_seed_src);
                let dealt = self.damage(p, (mh / 8).max(1), Some(src), DmgKind::Indirect);
                if dealt > 0 && self.is_live(src) {
                    self.heal(src, dealt);
                }
            }
            Res::Poison => {
                let st = self.m(p).status;
                let dmg = if st == Status::Tox {
                    let m = self.mm(p);
                    if m.status_turns < 15 {
                        m.status_turns += 1;
                    }
                    (mh / 16).max(1) * m.status_turns as u32
                } else {
                    (mh / 8).max(1)
                };
                self.damage(p, dmg, None, DmgKind::Indirect);
            }
            Res::Burn => {
                self.damage(p, (mh / 16).max(1), None, DmgKind::Indirect);
            }
            Res::Curse => {
                self.damage(p, (mh / 4).max(1), None, DmgKind::Indirect);
            }
            Res::SaltCure => {
                // Champions: 1/16 (1/8 for Water and Steel types).
                let m = self.m(p);
                let frac = if m.has_type(Type::Water) || m.has_type(Type::Steel) { 8 } else { 16 };
                self.damage(p, (mh / frac).max(1), None, DmgKind::Indirect);
            }
            Res::PartialTrap => {
                // Duration ticks first; then it ends if the trapper left.
                self.mm(p).vol.partial_trap -= 1;
                if self.m(p).vol.partial_trap == 0 {
                    return;
                }
                if !self.partial_trapper_active(p) {
                    self.mm(p).vol.partial_trap = 0;
                    return;
                }
                self.damage(p, (mh / 8).max(1), None, DmgKind::Indirect);
            }
            Res::Taunt => self.mm(p).vol.taunt -= 1,
            Res::Encore => {
                let m = self.mm(p);
                m.vol.encore -= 1;
                // Encore also ends if the move runs out of PP.
                let s = m.vol.encore_slot as usize;
                if m.pp[s] == 0 {
                    m.vol.encore = 0;
                }
            }
            Res::Disable => self.mm(p).vol.disable -= 1,
            Res::MagnetRise => self.mm(p).vol.magnet_rise -= 1,
            Res::HealBlock => self.mm(p).vol.heal_block -= 1,
            Res::ThroatChop => self.mm(p).vol.throat_chop -= 1,
            Res::Yawn => {
                let m = self.mm(p);
                m.vol.yawn -= 1;
                if m.vol.yawn == 0 {
                    self.try_set_status(p, Status::Slp, None);
                }
            }
            Res::PerishSong => {
                let m = self.mm(p);
                m.vol.perish -= 1;
                let left = m.vol.perish;
                blog!(self, "|-start|{}|perish{}", self.name(p), left);
                if left == 0 {
                    let hp = self.m(p).hp as u32;
                    self.damage(p, hp, None, DmgKind::SelfCost);
                }
            }
            Res::SpeedBoost => {
                let mut up = [0i8; 7];
                up[B_SPE] = 1;
                self.reveal_ability(p);
                self.boost(p, &up, BoostSrc::SelfInflicted);
            }
            Res::Moody => {
                let boosts = self.m(p).boosts;
                let ups: Vec<usize> = (0..5).filter(|&i| boosts[i] < 6).collect();
                let mut b = [0i8; 7];
                if !ups.is_empty() {
                    let u = ups[self.rng.below(ups.len() as u32) as usize];
                    b[u] = 2;
                }
                let downs: Vec<usize> = (0..5).filter(|&i| boosts[i] > -6 && b[i] == 0).collect();
                if !downs.is_empty() {
                    let d = downs[self.rng.below(downs.len() as u32) as usize];
                    b[d] = -1;
                }
                self.reveal_ability(p);
                self.boost(p, &b, BoostSrc::SelfInflicted);
            }
            Res::EndOfTurnVolatiles => {
                let m = self.mm(p);
                let v = &mut m.vol;
                v.flinch = false;
                v.protect = ProtectKind::None;
                v.endure = false;
                v.helping_hand = 0;
                v.follow_me = 0;
                v.roost = false;
                v.destiny_bond_pending = false;
                if v.stall_dur > 0 {
                    v.stall_dur -= 1;
                    if v.stall_dur == 0 {
                        v.stall_counter = 0;
                    }
                }
                if v.laser_focus > 0 {
                    v.laser_focus -= 1;
                }
            }
            _ => {}
        }
        if self.is_live(p) {
            self.update_items(p);
        }
    }
}
