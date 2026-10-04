//! Stats, speed, types and the damage formula, with Showdown's integer
//! arithmetic (4096-based modifiers, truncation at each step).

use super::Battle;
use crate::dex::{
    dex, flag, Category, MoveId, Target, Terrain, Weather, ATK, B_ACC, B_EVA, DEF, SPA, SPD, SPE,
};
use crate::dex::Status;
use crate::kinds::{Ab, It, MoveFx};
use crate::rng::Rng;
use crate::state::{Pos, SemiInv};
use crate::types::Type;

/// Showdown's `modify(value, numerator/denominator)`.
#[inline]
pub fn modify(value: u32, num: u32, den: u32) -> u32 {
    let m = (num as u64 * 4096 / den as u64) as u64;
    ((value as u64 * m + 2047) / 4096) as u32
}

/// A chain of 4096-based modifiers (Showdown's `chainModify`).
#[derive(Copy, Clone, Debug)]
pub struct Chain(pub u32);

impl Chain {
    pub fn new() -> Chain {
        Chain(4096)
    }
    #[inline]
    pub fn mul(&mut self, num: u32, den: u32) {
        let next = (num as u64 * 4096 / den as u64) as u64;
        self.0 = ((self.0 as u64 * next + 2048) >> 12) as u32;
    }
    #[inline]
    pub fn mul4096(&mut self, m: u32) {
        self.0 = ((self.0 as u64 * m as u64 + 2048) >> 12) as u32;
    }
    #[inline]
    pub fn apply(&self, value: u32) -> u32 {
        if self.0 == 4096 {
            return value;
        }
        ((value as u64 * self.0 as u64 + 2047) / 4096) as u32
    }
}

// Showdown effect-type subOrders (`Battle.resolvePriority`).
const SUB_MOVE: u8 = 0;
const SUB_COND: u8 = 2;
const SUB_SIDE: u8 = 4;
const SUB_FIELD: u8 = 5;
const SUB_ABIL: u8 = 7;
const SUB_ITEM: u8 = 8;

/// One event handler's modifier with Showdown's handler sort keys. `holder`
/// is the position code of the Pokemon holding the effect (its speed sorts
/// the handler), or NO_HOLDER for field / side effects (speed 0).
#[derive(Copy, Clone, Debug)]
struct Hm {
    prio: i16,
    holder: u8,
    speed: i32,
    sub: u8,
    m: u32,
}

const NO_HOLDER: u8 = 4;

/// The modifiers an event's handlers apply, chained in Showdown's handler
/// order (`speedSort`: priority desc, holder speed desc, subOrder asc, exact
/// ties in random order). The order matters because every chainModify step
/// rounds.
struct Handlers {
    v: [Hm; 12],
    n: usize,
}

impl Handlers {
    #[inline]
    fn new() -> Handlers {
        Handlers { v: [Hm { prio: 0, holder: NO_HOLDER, speed: 0, sub: 0, m: 4096 }; 12], n: 0 }
    }
    /// chainModify([m, 4096]).
    #[inline]
    fn add(&mut self, prio: i16, holder: u8, sub: u8, m: u32) {
        if self.n < self.v.len() {
            self.v[self.n] = Hm { prio, holder, speed: 0, sub, m };
            self.n += 1;
        }
    }
    /// chainModify(num / den).
    #[inline]
    fn frac(&mut self, prio: i16, holder: u8, sub: u8, num: u32, den: u32) {
        self.add(prio, holder, sub, (num as u64 * 4096 / den as u64) as u32);
    }
    fn chain(&mut self, rng: &mut Rng) -> Chain {
        let n = self.n;
        let mut c = Chain::new();
        if n == 0 {
            return c;
        }
        if n > 1 {
            let key = |h: &Hm| (-(h.prio as i32), -(h.speed as i64), h.sub);
            let v = &mut self.v[..n];
            v.sort_by_key(key);
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n && key(&v[j]) == key(&v[i]) {
                    j += 1;
                }
                // Shuffle exact ties (Fisher-Yates), like PRNG.shuffle.
                for k in (i + 1..j).rev() {
                    let r = i + rng.below((k - i + 1) as u32) as usize;
                    v.swap(k, r);
                }
                i = j;
            }
        }
        for h in &self.v[..n] {
            c.mul4096(h.m);
        }
        c
    }
}

const BOOST_NUM: [u32; 7] = [2, 3, 4, 5, 6, 7, 8];

/// Apply a stat stage to a stat (floor(stat * table)).
#[inline]
pub fn boosted(stat: u32, boost: i8) -> u32 {
    let b = boost.clamp(-6, 6);
    if b >= 0 {
        stat * BOOST_NUM[b as usize] / 2
    } else {
        stat * 2 / BOOST_NUM[(-b) as usize]
    }
}

/// One use of a move, with everything resolved at use time.
#[derive(Copy, Clone, Debug)]
pub struct ActiveMove {
    pub id: MoveId,
    pub fx: MoveFx,
    pub ty: Type,
    pub category: Category,
    pub base_power: u16,
    pub accuracy: u8,
    pub priority: i8,
    pub target: Target,
    pub flags: u32,
    pub crit_ratio: u8,
    pub will_crit: bool,
    pub spread_hit: bool,
    /// Pixilate & co. converted the type (1.2x).
    pub type_changer: bool,
    /// Current hit number (1-based) for multi-hit moves.
    pub hit: u8,
    pub sheer_force: bool,
    pub prankster: bool,
    pub infiltrates: bool,
    pub mold_breaker: bool,
    pub helping_hand: u8,
    pub mslot: u8,
    pub struggle: bool,
    pub confusion_hit: bool,
    /// Parental Bond second hit.
    pub parental_bond_hit: bool,
    /// Bypassed protection (Unseen Fist / Piercing Drill): damage x0.25.
    pub bypass_protect_quarter: bool,
}

impl Battle {
    /// Chain an event's handlers in Showdown's order (holder speeds are only
    /// computed when there is more than one handler).
    fn chain_handlers(&mut self, h: &mut Handlers) -> Chain {
        if h.n > 1 {
            let mut sp = [0i32; 4];
            let mut have = [false; 4];
            for k in 0..h.n {
                let c = h.v[k].holder as usize;
                if c < 4 {
                    if !have[c] {
                        sp[c] = self.action_speed(Pos::from_code(c as u8));
                        have[c] = true;
                    }
                    h.v[k].speed = sp[c];
                }
            }
        }
        h.chain(&mut self.rng)
    }

    // ---- abilities / items -----------------------------------------------

    /// Ability of the Pokemon at `p` (suppression aside).
    #[inline]
    pub fn ab(&self, p: Pos) -> Ab {
        dex().ab(self.m(p).ability)
    }

    /// Ability of a *target* of the current move: breakable abilities are
    /// ignored while a Mold Breaker user's move executes.
    #[inline]
    pub fn tab(&self, p: Pos) -> Ab {
        let id = self.m(p).ability;
        if self.mold_breaker && dex().ability(id).breakable {
            return Ab::Other;
        }
        dex().ab(id)
    }

    #[inline]
    pub fn it(&self, p: Pos) -> It {
        dex().it(self.m(p).item)
    }

    pub fn ability_on_field(&self, a: Ab) -> bool {
        let d = dex();
        for s in 0..2 {
            for slot in 0..2 {
                let i = self.sides[s].active[slot];
                if i != crate::state::NO_MON {
                    let m = &self.sides[s].mons[i as usize];
                    if m.alive() && d.ab(m.ability) == a {
                        return true;
                    }
                }
            }
        }
        false
    }

    // ---- field -------------------------------------------------------------

    /// Field weather, unless Cloud Nine / Air Lock is out.
    pub fn weather(&self) -> Weather {
        if self.field.weather == Weather::None {
            return Weather::None;
        }
        if self.ability_on_field(Ab::CloudNine) || self.ability_on_field(Ab::AirLock) {
            return Weather::None;
        }
        self.field.weather
    }

    /// Weather as seen by a move's user (Mega Sol counts as sun).
    pub fn user_weather(&self, p: Pos) -> Weather {
        if self.is_live(p) && self.ab(p) == Ab::MegaSol {
            return Weather::Sun;
        }
        self.weather()
    }

    pub fn is_grounded(&self, p: Pos) -> bool {
        let m = self.m(p);
        if self.field.gravity > 0 || m.vol.ingrain || m.vol.smacked_down {
            return true;
        }
        let item = self.it(p);
        if item == It::IronBall {
            return true;
        }
        if m.has_type(Type::Flying) && !m.vol.roost {
            return false;
        }
        if matches!(self.tab(p), Ab::Levitate | Ab::Eelevate) {
            return false;
        }
        if m.vol.magnet_rise > 0 {
            return false;
        }
        item != It::AirBalloon
    }

    pub fn terrain_for(&self, p: Pos) -> Terrain {
        if self.is_grounded(p) && self.m(p).vol.semi_inv == SemiInv::None {
            self.field.terrain
        } else {
            Terrain::None
        }
    }

    // ---- stats ----------------------------------------------------------------

    /// Boost stage after Unaware / Simple-type adjustments (ModifyBoost).
    #[inline]
    pub fn boost_of(&self, p: Pos, b: usize) -> i8 {
        self.m(p).boosts[b]
    }

    /// Speed with boosts and modifiers (getStat('spe')).
    pub fn speed(&self, p: Pos) -> u32 {
        let m = self.m(p);
        let mut spe = boosted(m.stats[SPE] as u32, m.boosts[SPE - 1]);
        let mut c = Chain::new();
        let w = self.weather();
        match self.ab(p) {
            Ab::SwiftSwim if w == Weather::Rain => c.mul(2, 1),
            Ab::Chlorophyll if w == Weather::Sun => c.mul(2, 1),
            Ab::SandRush if w == Weather::Sand => c.mul(2, 1),
            Ab::SlushRush if w == Weather::Snow => c.mul(2, 1),
            Ab::SurgeSurfer if self.field.terrain == Terrain::Electric => c.mul(2, 1),
            Ab::Unburden if m.vol.unburden && m.item == 0 => c.mul(2, 1),
            Ab::QuickFeet if m.status != Status::None => c.mul(3, 2),
            _ => {}
        }
        match self.it(p) {
            It::ChoiceScarf => c.mul(3, 2),
            It::IronBall => c.mul(1, 2),
            _ => {}
        }
        if self.sides[p.s()].conds.tailwind > 0 {
            c.mul(2, 1);
        }
        spe = c.apply(spe);
        // Paralysis (priority -101): finalModify, then halve.
        if m.status == Status::Par && self.ab(p) != Ab::QuickFeet {
            spe = spe * 50 / 100;
        }
        spe.min(10000)
    }

    /// Speed used for ordering (negated under Trick Room).
    pub fn action_speed(&self, p: Pos) -> i32 {
        let s = self.speed(p) as i32;
        if self.field.trick_room > 0 {
            -s
        } else {
            s
        }
    }

    // ---- types -------------------------------------------------------------

    pub fn types_of(&self, p: Pos) -> [Type; 2] {
        let m = self.m(p);
        let mut t = m.types;
        if m.vol.roost {
            // Roost removes Flying for the turn.
            if t[0] == Type::Flying {
                t[0] = if t[1] == Type::None { Type::Normal } else { t[1] };
                t[1] = Type::None;
            } else if t[1] == Type::Flying {
                t[1] = Type::None;
            }
        }
        t
    }

    /// Showdown's runEffectiveness (log2 sum over the target's types).
    pub fn effectiveness(&self, ty: Type, tgt: Pos, am: Option<&ActiveMove>) -> i32 {
        let chart = &dex().chart;
        let mut total = 0;
        for t in self.types_of(tgt) {
            if t == Type::None {
                continue;
            }
            let mut e = chart.log2(ty, t);
            if let Some(am) = am {
                if am.fx == MoveFx::FreezeDry && t == Type::Water {
                    e = 1;
                }
            }
            total += e;
        }
        if ty == Type::Fire && self.m(tgt).vol.tar_shot {
            total += 1;
        }
        total.clamp(-6, 6)
    }

    /// Type-chart (and grounding) immunity of `tgt` to a move of type `ty`.
    pub fn type_immune(&self, src: Pos, ty: Type, tgt: Pos, am: &ActiveMove) -> bool {
        if ty == Type::Ground {
            if !self.is_grounded(tgt) && !am.confusion_hit {
                return true;
            }
        }
        let chart = &dex().chart;
        for t in self.types_of(tgt) {
            if t == Type::None {
                continue;
            }
            if chart.immune(ty, t) {
                if t == Type::Ghost
                    && (ty == Type::Normal || ty == Type::Fighting)
                    && self.is_live(src)
                    && self.ab(src) == Ab::Scrappy
                {
                    continue;
                }
                if ty == Type::Ground && (self.field.gravity > 0 || self.m(tgt).vol.smacked_down || self.it(tgt) == It::IronBall || self.m(tgt).vol.ingrain) {
                    continue;
                }
                return true;
            }
        }
        false
    }

    // ---- moves: type and base power ---------------------------------------------

    /// Resolve the move's type (Weather Ball, Terrain Pulse, -ate abilities,
    /// Liquid Voice, Raging Bull...).
    pub fn resolve_move_type(&self, src: Pos, am: &mut ActiveMove) {
        let mv = dex().mv(am.id);
        match am.fx {
            MoveFx::WeatherBall => {
                am.ty = match self.user_weather(src) {
                    Weather::Rain => Type::Water,
                    Weather::Sun => Type::Fire,
                    Weather::Sand => Type::Rock,
                    Weather::Snow => Type::Ice,
                    Weather::None => Type::Normal,
                };
                if am.ty != Type::Normal {
                    am.base_power = mv.base_power * 2;
                }
            }
            MoveFx::TerrainPulse => {
                if self.is_grounded(src) {
                    am.ty = match self.field.terrain {
                        Terrain::Grassy => Type::Grass,
                        Terrain::Psychic => Type::Psychic,
                        Terrain::Electric => Type::Electric,
                        Terrain::Misty => Type::Fairy,
                        Terrain::None => Type::Normal,
                    };
                    if self.field.terrain != Terrain::None {
                        am.base_power = mv.base_power * 2;
                    }
                }
            }
            MoveFx::RagingBull => {
                // Tauros-Paldea formes use their breed's type; others stay Normal.
                am.ty = match dex().sp(self.m(src).species).id.as_str() {
                    "taurospaldeacombat" => Type::Fighting,
                    "taurospaldeablaze" => Type::Fire,
                    "taurospaldeaaqua" => Type::Water,
                    _ => am.ty,
                };
            }
            MoveFx::BurnUp | MoveFx::DoubleShock => {}
            _ => {}
        }
        if am.struggle || am.confusion_hit {
            return;
        }
        // -ate abilities (onModifyTypePriority -1, after the move's own change).
        let no_ate = matches!(am.fx, MoveFx::WeatherBall | MoveFx::TerrainPulse);
        let ab = self.ab(src);
        if am.ty == Type::Normal && !no_ate {
            let to = match ab {
                Ab::Pixilate => Some(Type::Fairy),
                Ab::Aerilate => Some(Type::Flying),
                Ab::Refrigerate => Some(Type::Ice),
                Ab::Galvanize => Some(Type::Electric),
                Ab::Dragonize => Some(Type::Dragon),
                _ => None,
            };
            if let Some(t) = to {
                am.ty = t;
                am.type_changer = true;
            }
        }
        if ab == Ab::LiquidVoice && am.flags & flag::SOUND != 0 {
            am.ty = Type::Water;
        }
    }

    /// basePowerCallback equivalents.
    pub fn base_power_callback(&self, src: Pos, tgt: Pos, am: &ActiveMove) -> u32 {
        let bp = am.base_power as u32;
        let s = self.m(src);
        let t = self.m(tgt);
        match am.fx {
            MoveFx::Eruption | MoveFx::WaterSpout | MoveFx::DragonEnergy => {
                (bp * s.hp as u32 / s.max_hp.max(1) as u32).max(1)
            }
            MoveFx::LastRespects => 50 + 50 * self.sides[src.s()].total_fainted.min(100) as u32,
            MoveFx::LowKick | MoveFx::GrassKnot => {
                let w = t.weight_hg;
                if w >= 2000 {
                    120
                } else if w >= 1000 {
                    100
                } else if w >= 500 {
                    80
                } else if w >= 250 {
                    60
                } else if w >= 100 {
                    40
                } else {
                    20
                }
            }
            MoveFx::HeavySlam | MoveFx::HeatCrash => {
                let ratio = s.weight_hg as f64 / t.weight_hg.max(1) as f64;
                if ratio >= 5.0 {
                    120
                } else if ratio >= 4.0 {
                    100
                } else if ratio >= 3.0 {
                    80
                } else if ratio >= 2.0 {
                    60
                } else {
                    40
                }
            }
            MoveFx::Acrobatics => {
                if s.item == 0 {
                    bp * 2
                } else {
                    bp
                }
            }
            MoveFx::StompingTantrum | MoveFx::TemperFlare => {
                if s.vol.last_move_failed {
                    bp * 2
                } else {
                    bp
                }
            }
            MoveFx::RageFist => (50 + 50 * s.vol.times_attacked as u32).min(350),
            MoveFx::Hex => {
                if t.status != Status::None {
                    bp * 2
                } else {
                    bp
                }
            }
            MoveFx::StoredPower | MoveFx::PowerTrip => {
                let pos: u32 = s.boosts.iter().filter(|&&b| b > 0).map(|&b| b as u32).sum();
                bp + 20 * pos
            }
            MoveFx::ElectroBall => {
                let ratio = self.speed(src) / self.speed(tgt).max(1);
                [40, 60, 80, 120, 150][(ratio as usize).min(4)]
            }
            MoveFx::HardPress => (100 * t.hp as u32 / t.max_hp.max(1) as u32).max(1),
            MoveFx::RisingVoltage => {
                if self.field.terrain == Terrain::Electric && self.is_grounded(tgt) {
                    bp * 2
                } else {
                    bp
                }
            }
            MoveFx::TripleAxel => 20 * am.hit as u32,
            MoveFx::BeatUp => {
                // 5 + base Attack / 10 of the hit-th party member.
                let side = &self.sides[src.s()];
                let (allies, n) = side.beat_up_allies(side.active[src.i()]);
                let k = (am.hit.max(1) as usize - 1).min(n.saturating_sub(1));
                // The set's species: a Mega counts as its base forme.
                let sp = side.mons[allies[k] as usize].base_species;
                5 + dex().sp(sp).base[1] as u32 / 10
            }
            MoveFx::Payback => {
                // Doubled unless the target just switched in or has yet to move.
                if t.vol.newly_switched || self.queue.will_move(tgt).is_some() {
                    bp
                } else {
                    bp * 2
                }
            }
            MoveFx::Assurance => {
                if t.vol.hurt_this_turn {
                    bp * 2
                } else {
                    bp
                }
            }
            MoveFx::Round => {
                if self.round_used {
                    bp * 2
                } else {
                    bp
                }
            }
            _ => bp,
        }
    }

    /// The BasePower event, after the crit roll: every handler's modifier,
    /// chained in Showdown's handler order (priority, holder speed, effect type).
    pub fn base_power_mods(&mut self, src: Pos, tgt: Pos, am: &ActiveMove, bp: u32) -> u32 {
        let s = *self.m(src);
        let t = *self.m(tgt);
        let sa = src.code();
        let sd = tgt.code();
        let mut h = Handlers::new();
        // The move's own onBasePower (priority 0, held by the user).
        let own = match am.fx {
            MoveFx::ExpandingForce if self.field.terrain == Terrain::Psychic && self.is_grounded(src) => Some((3, 2)),
            MoveFx::KnockOff if t.item != 0 && !dex().item(t.item).is_mega_stone => Some((3, 2)),
            MoveFx::Facade if s.status != Status::None && s.status != Status::Slp => Some((2, 1)),
            MoveFx::Venoshock | MoveFx::BarbBarrage if matches!(t.status, Status::Psn | Status::Tox) => Some((2, 1)),
            MoveFx::LashOut if s.vol.stats_lowered_this_turn => Some((2, 1)),
            MoveFx::SolarBeam | MoveFx::SolarBlade
                if matches!(self.user_weather(src), Weather::Rain | Weather::Sand | Weather::Snow) =>
            {
                Some((1, 2))
            }
            _ => None,
        };
        if let Some((n, d)) = own {
            h.frac(0, sa, SUB_MOVE, n, d);
        }
        // The user's ability.
        match self.ab(src) {
            Ab::Technician if bp <= 60 => h.frac(30, sa, SUB_ABIL, 3, 2),
            Ab::Rivalry if s.gender != 0 && t.gender != 0 => {
                if s.gender == t.gender {
                    h.frac(24, sa, SUB_ABIL, 5, 4)
                } else {
                    h.frac(24, sa, SUB_ABIL, 3, 4)
                }
            }
            Ab::IronFist if am.flags & flag::PUNCH != 0 => h.add(23, sa, SUB_ABIL, 4915),
            Ab::ToughClaws if am.flags & flag::CONTACT != 0 => h.add(21, sa, SUB_ABIL, 5325),
            Ab::SheerForce if am.sheer_force || dex().mv(am.id).has_sheer_force_boost => h.add(21, sa, SUB_ABIL, 5325),
            Ab::Analytic if !self.any_other_will_move(src) => h.add(21, sa, SUB_ABIL, 5325),
            Ab::SandForce if self.weather() == Weather::Sand && matches!(am.ty, Type::Rock | Type::Ground | Type::Steel) => {
                h.add(21, sa, SUB_ABIL, 5325)
            }
            Ab::SupremeOverlord if s.vol.fallen > 0 => {
                const POW: [u32; 6] = [4096, 4506, 4915, 5325, 5734, 6144];
                h.add(21, sa, SUB_ABIL, POW[(s.vol.fallen as usize).min(5)])
            }
            Ab::Sharpness if am.flags & flag::SLICING != 0 => h.frac(19, sa, SUB_ABIL, 3, 2),
            Ab::StrongJaw if am.flags & flag::BITE != 0 => h.frac(19, sa, SUB_ABIL, 3, 2),
            Ab::MegaLauncher if am.flags & flag::PULSE != 0 => h.frac(19, sa, SUB_ABIL, 3, 2),
            Ab::PunkRock if am.flags & flag::SOUND != 0 => h.add(7, sa, SUB_ABIL, 5325),
            _ => {}
        }
        // -ate abilities (priority 23, only if they changed the type).
        if am.type_changer {
            h.add(23, sa, SUB_ABIL, 4915);
        }
        // Fairy Aura (onAnyBasePower 20): one boost however many auras.
        if am.ty == Type::Fairy && src != tgt && self.ability_on_field(Ab::FairyAura) {
            h.add(20, NO_HOLDER, SUB_ABIL, 5448);
        }
        // Dry Skin on the target (onSourceBasePower 17).
        if am.ty == Type::Fire && self.tab(tgt) == Ab::DrySkin {
            h.frac(17, sd, SUB_ABIL, 5, 4);
        }
        // The user's item.
        match self.it(src) {
            It::MuscleBand if am.category == Category::Physical => h.add(16, sa, SUB_ITEM, 4505),
            It::WiseGlasses if am.category == Category::Special => h.add(16, sa, SUB_ITEM, 4505),
            it => {
                if type_booster(it) == Some(am.ty) {
                    h.add(15, sa, SUB_ITEM, 4915);
                }
            }
        }
        // Volatiles on the user: Helping Hand (10, one chainModify of 1.5^n), Charge (9).
        if am.helping_hand > 0 {
            let mult = 1.5f64.powi(am.helping_hand as i32);
            h.add(10, sa, SUB_COND, (mult * 4096.0) as u32);
        }
        if s.vol.charge && am.ty == Type::Electric {
            h.frac(9, sa, SUB_COND, 2, 1);
        }
        // Terrain (field condition, priority 6).
        let src_grounded = self.is_grounded(src) && s.vol.semi_inv == SemiInv::None;
        let tgt_grounded = self.is_grounded(tgt) && t.vol.semi_inv == SemiInv::None;
        match self.field.terrain {
            Terrain::Grassy => {
                if matches!(dex().mv(am.id).id.as_str(), "earthquake" | "bulldoze" | "magnitude") && tgt_grounded {
                    h.frac(6, NO_HOLDER, SUB_FIELD, 1, 2);
                } else if am.ty == Type::Grass && self.is_grounded(src) {
                    h.add(6, NO_HOLDER, SUB_FIELD, 5325);
                }
            }
            Terrain::Psychic if am.ty == Type::Psychic && src_grounded => h.add(6, NO_HOLDER, SUB_FIELD, 5325),
            Terrain::Electric if am.ty == Type::Electric && src_grounded => h.add(6, NO_HOLDER, SUB_FIELD, 5325),
            Terrain::Misty if am.ty == Type::Dragon && tgt_grounded => h.frac(6, NO_HOLDER, SUB_FIELD, 1, 2),
            _ => {}
        }
        self.chain_handlers(&mut h).apply(bp).max(1)
    }

    /// Analytic: no other active Pokemon still has a move action this turn.
    fn any_other_will_move(&self, p: Pos) -> bool {
        (0..4).map(Pos::from_code).any(|q| q != p && self.is_live(q) && self.queue.will_move(q).is_some())
    }

    // ---- damage ----------------------------------------------------------------

    /// Critical hit roll for one target.
    pub fn roll_crit(&mut self, src: Pos, tgt: Pos, am: &ActiveMove) -> bool {
        if am.confusion_hit {
            return false;
        }
        let s = self.m(src);
        let will = am.will_crit || s.vol.laser_focus > 0;
        let mut ratio = am.crit_ratio as i32 + s.vol.crit_stage as i32;
        if self.ab(src) == Ab::SuperLuck {
            ratio += 1;
        }
        if matches!(self.it(src), It::ScopeLens | It::RazorClaw) {
            ratio += 1;
        }
        let crit = if will {
            true
        } else if ratio <= 0 {
            false
        } else {
            let r = ratio.clamp(0, 4) as usize;
            let den = [0u32, 24, 8, 2, 1][r];
            self.rng.chance(1, den)
        };
        if crit && matches!(self.tab(tgt), Ab::BattleArmor | Ab::ShellArmor) {
            return false;
        }
        crit
    }

    /// Attack stat used by a move (getDamage's attacker side), after boosts
    /// and the ModifyAtk / ModifySpA event.
    fn attack_stat(&mut self, src: Pos, tgt: Pos, am: &ActiveMove, crit: bool) -> u32 {
        let mv = dex().mv(am.id);
        let physical = am.category == Category::Physical;
        let atk_holder = if mv.foul_play { tgt } else { src };
        let stat_idx = mv.override_offensive_stat.unwrap_or(if physical { ATK } else { SPA });
        let hm = self.m(atk_holder);
        let mut boost = hm.boosts[stat_idx - 1];
        // Unaware on the target ignores the attacker's boosts.
        if self.tab(tgt) == Ab::Unaware {
            boost = 0;
        }
        if crit && boost < 0 {
            boost = 0;
        }
        let mut atk = boosted(hm.stats[stat_idx] as u32, boost);
        // ModifyAtk (physical) / ModifySpA (special) on the user.
        let s = *self.m(src);
        let ab = self.ab(src);
        let sa = src.code();
        let sd = tgt.code();
        // Hustle modifies the stat directly instead of chaining.
        if ab == Ab::Hustle && physical {
            atk = modify(atk, 3, 2);
        }
        let mut h = Handlers::new();
        let low_hp = (s.hp as u32) * 3 <= s.max_hp as u32;
        match ab {
            Ab::HugePower | Ab::PurePower if physical => h.frac(5, sa, SUB_ABIL, 2, 1),
            Ab::Guts if physical && s.status != Status::None => h.frac(5, sa, SUB_ABIL, 3, 2),
            Ab::Blaze if am.ty == Type::Fire && low_hp => h.frac(5, sa, SUB_ABIL, 3, 2),
            Ab::Torrent if am.ty == Type::Water && low_hp => h.frac(5, sa, SUB_ABIL, 3, 2),
            Ab::Overgrow if am.ty == Type::Grass && low_hp => h.frac(5, sa, SUB_ABIL, 3, 2),
            Ab::Swarm if am.ty == Type::Bug && low_hp => h.frac(5, sa, SUB_ABIL, 3, 2),
            Ab::SolarPower if !physical && self.user_weather(src) == Weather::Sun => h.frac(5, sa, SUB_ABIL, 3, 2),
            Ab::FireMane if am.ty == Type::Fire => h.frac(5, sa, SUB_ABIL, 3, 2),
            Ab::WaterBubble if am.ty == Type::Water => h.frac(0, sa, SUB_ABIL, 2, 1),
            _ => {}
        }
        // Flash Fire's volatile (priority 5).
        if s.vol.flash_fire && am.ty == Type::Fire && ab == Ab::FlashFire {
            h.frac(5, sa, SUB_COND, 3, 2);
        }
        match self.it(src) {
            It::ChoiceBand if physical => h.frac(1, sa, SUB_ITEM, 3, 2),
            It::ChoiceSpecs if !physical => h.frac(1, sa, SUB_ITEM, 3, 2),
            _ => {}
        }
        // The target's onSourceModifyAtk / onSourceModifySpA.
        match self.tab(tgt) {
            Ab::ThickFat if matches!(am.ty, Type::Fire | Type::Ice) => h.frac(if physical { 6 } else { 5 }, sd, SUB_ABIL, 1, 2),
            Ab::WaterBubble if am.ty == Type::Fire => h.frac(5, sd, SUB_ABIL, 1, 2),
            _ => {}
        }
        atk = self.chain_handlers(&mut h).apply(atk);
        atk.max(1)
    }

    fn defense_stat(&mut self, src: Pos, tgt: Pos, am: &ActiveMove, crit: bool) -> u32 {
        let mv = dex().mv(am.id);
        let physical = am.category == Category::Physical;
        let stat_idx = mv.override_defensive_stat.unwrap_or(if physical { DEF } else { SPD });
        let t = *self.m(tgt);
        let mut boost = t.boosts[stat_idx - 1];
        if mv.ignore_defensive || (self.is_live(src) && self.ab(src) == Ab::Unaware) {
            boost = 0;
        }
        if crit && boost > 0 {
            boost = 0;
        }
        let mut def = boosted(t.stats[stat_idx] as u32, boost);
        // Weather (priority 10) modifies the stat directly. During a Mega Sol
        // user's move every Pokemon's effective weather is sun.
        let w = if self.is_live(src) && self.ab(src) == Ab::MegaSol { Weather::Sun } else { self.weather() };
        if stat_idx == SPD && w == Weather::Sand && t.has_type(Type::Rock) {
            def = modify(def, 3, 2);
        }
        if stat_idx == DEF && w == Weather::Snow && t.has_type(Type::Ice) {
            def = modify(def, 3, 2);
        }
        if stat_idx == DEF && self.tab(tgt) == Ab::MarvelScale && t.status != Status::None {
            def = Chain(6144).apply(def);
        }
        def.max(1)
    }

    /// Full damage for one hit (getDamage + modifyDamage). Returns None if the
    /// move does no damage (0 base power), Some(dmg) otherwise.
    pub fn calc_damage(&mut self, src: Pos, tgt: Pos, am: &ActiveMove, crit: bool, roll: u32) -> Option<u32> {
        let bp0 = self.base_power_callback(src, tgt, am);
        if bp0 == 0 {
            return None;
        }
        let bp = self.base_power_mods(src, tgt, am, bp0);
        let level = self.m(src).level as u32;
        let atk = self.attack_stat(src, tgt, am, crit);
        let def = self.defense_stat(src, tgt, am, crit);
        let base = ((2 * level / 5 + 2) * bp * atk / def) / 50;
        Some(self.modify_damage(base, src, tgt, am, crit, roll))
    }

    /// modifyDamage: spread, weather, crit, random, STAB, type, burn, then the
    /// ModifyDamage event (chained in Showdown's handler order).
    pub fn modify_damage(&mut self, base: u32, src: Pos, tgt: Pos, am: &ActiveMove, crit: bool, roll: u32) -> u32 {
        let mut dmg = base + 2;
        if am.spread_hit {
            dmg = modify(dmg, 3, 4);
        } else if am.parental_bond_hit {
            dmg = modify(dmg, 1, 4);
        }
        // Weather.
        let w = self.user_weather(src);
        match (w, am.ty) {
            (Weather::Rain, Type::Water) | (Weather::Sun, Type::Fire) => dmg = modify(dmg, 3, 2),
            (Weather::Rain, Type::Fire) | (Weather::Sun, Type::Water) => dmg = modify(dmg, 1, 2),
            _ => {}
        }
        if crit {
            dmg = dmg * 3 / 2;
        }
        // Random factor: 85..100.
        dmg = dmg * (100 - roll.min(15)) / 100;
        // STAB.
        if am.ty != Type::None && !am.confusion_hit {
            let s = self.m(src);
            if s.has_type(am.ty) {
                let stab = if self.ab(src) == Ab::Adaptability { (2, 1) } else { (3, 2) };
                dmg = modify(dmg, stab.0, stab.1);
            }
        }
        // Type effectiveness.
        let type_mod = if am.ty == Type::None { 0 } else { self.effectiveness(am.ty, tgt, Some(am)) };
        if type_mod > 0 {
            for _ in 0..type_mod {
                dmg *= 2;
            }
        } else if type_mod < 0 {
            for _ in 0..(-type_mod) {
                dmg /= 2;
            }
        }
        // Burn.
        let s = *self.m(src);
        if s.status == Status::Brn && am.category == Category::Physical && self.ab(src) != Ab::Guts && am.fx != MoveFx::Facade {
            dmg = modify(dmg, 1, 2);
        }
        // ModifyDamage handlers.
        let t = *self.m(tgt);
        let sa = src.code();
        let sd = tgt.code();
        let mut h = Handlers::new();
        // Screens: side conditions (holder speed 0, subOrder 4).
        if !crit && !am.infiltrates && src != tgt {
            let tside = &self.sides[tgt.s()].conds;
            let screen = match am.category {
                Category::Physical => tside.reflect > 0 || tside.aurora_veil > 0,
                Category::Special => tside.light_screen > 0 || tside.aurora_veil > 0,
                Category::Status => false,
            };
            if screen {
                h.add(0, NO_HOLDER, SUB_SIDE, 2732);
            }
        }
        // The user's ability and item.
        match self.ab(src) {
            Ab::Sniper if crit => h.frac(0, sa, SUB_ABIL, 3, 2),
            Ab::TintedLens if type_mod < 0 => h.frac(0, sa, SUB_ABIL, 2, 1),
            _ => {}
        }
        match self.it(src) {
            It::LifeOrb => h.add(0, sa, SUB_ITEM, 5324),
            It::ExpertBelt if type_mod > 0 => h.add(0, sa, SUB_ITEM, 4915),
            _ => {}
        }
        // The target's ability, item and volatiles (onSourceModifyDamage).
        match self.tab(tgt) {
            Ab::Multiscale if t.hp >= t.max_hp => h.frac(0, sd, SUB_ABIL, 1, 2),
            Ab::Filter | Ab::SolidRock if type_mod > 0 => h.frac(0, sd, SUB_ABIL, 3, 4),
            Ab::Fluffy => {
                let contact = am.flags & flag::CONTACT != 0;
                let fire = am.ty == Type::Fire;
                if contact && !fire {
                    h.frac(0, sd, SUB_ABIL, 1, 2);
                } else if fire && !contact {
                    h.frac(0, sd, SUB_ABIL, 2, 1);
                }
            }
            Ab::PunkRock if am.flags & flag::SOUND != 0 => h.frac(0, sd, SUB_ABIL, 1, 2),
            Ab::AuraGuard if am.flags & flag::CONTACT != 0 => h.frac(0, sd, SUB_ABIL, 1, 2),
            _ => {}
        }
        if let Some(bt) = resist_berry(self.it(tgt)) {
            let hit_sub = t.vol.substitute > 0 && am.flags & flag::BYPASSSUB == 0 && !am.infiltrates;
            if bt == am.ty && (type_mod > 0 || bt == Type::Normal) && !hit_sub && !self.unnerved(tgt) {
                h.frac(0, sd, SUB_ITEM, 1, 2);
            }
        }
        if t.vol.glaive_rush > 0 {
            h.frac(0, sd, SUB_COND, 2, 1);
        }
        let mid = dex().mv(am.id).id.as_str();
        match t.vol.semi_inv {
            SemiInv::Underground if matches!(mid, "earthquake" | "magnitude") => h.frac(0, sd, SUB_COND, 2, 1),
            SemiInv::Underwater if matches!(mid, "surf" | "whirlpool") => h.frac(0, sd, SUB_COND, 2, 1),
            _ => {}
        }
        // Friend Guard on the target's ally (onAnyModifyDamage).
        if let Some(a) = self.live_ally(tgt) {
            if self.tab(a) == Ab::FriendGuard {
                h.frac(0, a.code(), SUB_ABIL, 3, 4);
            }
        }
        dmg = self.chain_handlers(&mut h).apply(dmg);
        if am.bypass_protect_quarter {
            dmg = modify(dmg, 1, 4);
        }
        if dmg == 0 {
            return 1;
        }
        dmg & 0xFFFF
    }

    /// Is `p` prevented from eating berries (Unnerve on the other side)?
    pub fn unnerved(&self, p: Pos) -> bool {
        self.live_foes(p).any(|q| self.ab(q) == Ab::Unnerve)
    }

    /// Accuracy check for one target. Returns true if the move hits.
    pub fn accuracy_hits(&mut self, src: Pos, tgt: Pos, am: &ActiveMove) -> bool {
        let mv = dex().mv(am.id);
        if am.accuracy == 0 {
            return true;
        }
        if mv.target == Target::User && am.category == Category::Status && self.m(tgt).vol.semi_inv == SemiInv::None {
            return true;
        }
        if self.ab(src) == Ab::NoGuard || self.tab(tgt) == Ab::NoGuard {
            return true;
        }
        if self.m(tgt).vol.glaive_rush > 0 {
            return true;
        }
        let mut acc = am.accuracy as u32;
        // Weather accuracy (onModifyMove).
        let w = self.user_weather(src);
        match am.fx {
            MoveFx::Hurricane | MoveFx::Thunder => match w {
                Weather::Rain => return true,
                Weather::Sun => acc = 50,
                _ => {}
            },
            MoveFx::Blizzard if w == Weather::Snow => return true,
            _ => {}
        }
        if mv.ohko {
            let (sl, tl) = (self.m(src).level as u32, self.m(tgt).level as u32);
            if tl > sl {
                return false;
            }
            acc += sl - tl;
            return self.rng.chance(acc.min(100), 100);
        }
        // ModifyAccuracy chain.
        let mut c = Chain::new();
        if self.field.gravity > 0 {
            c.mul(5, 3);
        }
        match self.ab(src) {
            Ab::CompoundEyes => c.mul4096(5325),
            Ab::Hustle if am.category == Category::Physical => c.mul4096(3277),
            _ => {}
        }
        match self.it(src) {
            It::WideLens => c.mul4096(4505),
            It::ZoomLens if self.m(tgt).vol.moved_this_turn => c.mul4096(4915),
            _ => {}
        }
        match self.tab(tgt) {
            Ab::SandVeil if self.weather() == Weather::Sand => c.mul4096(3277),
            Ab::SnowCloak if self.weather() == Weather::Snow => c.mul4096(3277),
            _ => {}
        }
        if self.it(tgt) == It::BrightPowder {
            c.mul4096(3686);
        }
        acc = c.apply(acc);
        // Accuracy / evasion stages.
        let mut boost: i32 = 0;
        if self.tab(tgt) != Ab::Unaware {
            boost = self.m(src).boosts[B_ACC] as i32;
        }
        if !mv.ignore_evasion && self.ab(src) != Ab::Unaware && !(self.ab(src) == Ab::Illuminate) {
            boost = (boost - self.m(tgt).boosts[B_EVA] as i32).clamp(-6, 6);
        }
        let boost = boost.clamp(-6, 6);
        if boost > 0 {
            acc = acc * (3 + boost as u32) / 3;
        } else if boost < 0 {
            acc = acc * 3 / (3 + (-boost) as u32);
        }
        self.rng.chance(acc.min(100), 100)
    }
}

/// Type-boosting held items.
pub fn type_booster(it: It) -> Option<Type> {
    Some(match it {
        It::MiracleSeed => Type::Grass,
        It::Charcoal => Type::Fire,
        It::MysticWater => Type::Water,
        It::BlackGlasses => Type::Dark,
        It::FairyFeather => Type::Fairy,
        It::SharpBeak => Type::Flying,
        It::NeverMeltIce => Type::Ice,
        It::TwistedSpoon => Type::Psychic,
        It::DragonFang => Type::Dragon,
        It::SpellTag => Type::Ghost,
        It::BlackBelt => Type::Fighting,
        It::MetalCoat => Type::Steel,
        It::HardStone => Type::Rock,
        It::SilkScarf => Type::Normal,
        It::Magnet => Type::Electric,
        It::PoisonBarb => Type::Poison,
        It::SoftSand => Type::Ground,
        It::SilverPowder => Type::Bug,
        _ => return None,
    })
}

/// Damage-halving berries.
pub fn resist_berry(it: It) -> Option<Type> {
    Some(match it {
        It::BabiriBerry => Type::Steel,
        It::ChartiBerry => Type::Rock,
        It::ChilanBerry => Type::Normal,
        It::ChopleBerry => Type::Fighting,
        It::CobaBerry => Type::Flying,
        It::ColburBerry => Type::Dark,
        It::HabanBerry => Type::Dragon,
        It::KasibBerry => Type::Ghost,
        It::KebiaBerry => Type::Poison,
        It::OccaBerry => Type::Fire,
        It::PasshoBerry => Type::Water,
        It::PayapaBerry => Type::Psychic,
        It::RindoBerry => Type::Grass,
        It::RoseliBerry => Type::Fairy,
        It::ShucaBerry => Type::Ground,
        It::TangaBerry => Type::Bug,
        It::WacanBerry => Type::Electric,
        It::YacheBerry => Type::Ice,
        _ => return None,
    })
}
