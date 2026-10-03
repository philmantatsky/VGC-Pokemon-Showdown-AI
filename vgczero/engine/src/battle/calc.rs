//! Stats, speed, types and the damage formula, with Showdown's integer
//! arithmetic (4096-based modifiers, truncation at each step).

use super::Battle;
use crate::dex::{
    dex, flag, Category, MoveId, Target, Terrain, Weather, ATK, B_ACC, B_EVA, DEF, SPA, SPD, SPE,
};
use crate::kinds::{Ab, It, MoveFx};
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
        self.live_positions().any(|q| self.ab(q) == a)
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
            Ab::QuickFeet if m.status != crate::dex::Status::None => c.mul(3, 2),
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
        if m.status == crate::dex::Status::Par && self.ab(p) != Ab::QuickFeet {
            c.mul(1, 2);
        }
        spe = c.apply(spe);
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
                // Tauros-Paldea formes take their own type; others stay Normal.
                let m = self.m(src);
                if m.types[1] != Type::None && m.types[0] == Type::Fighting {
                    am.ty = m.types[1];
                }
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
                if t.status != crate::dex::Status::None {
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
            MoveFx::Payback => {
                if t.vol.moved_this_turn || t.vol.newly_switched {
                    bp * 2
                } else {
                    bp
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

    /// The BasePower event: chained modifiers after the crit roll.
    pub fn base_power_mods(&self, src: Pos, tgt: Pos, am: &ActiveMove, bp: u32) -> u32 {
        let s = self.m(src);
        let t = self.m(tgt);
        let mut c = Chain::new();
        // The move's own onBasePower runs first.
        match am.fx {
            MoveFx::ExpandingForce => {
                if self.field.terrain == Terrain::Psychic && self.is_grounded(src) {
                    c.mul(3, 2);
                }
            }
            MoveFx::KnockOff => {
                if t.item != 0 && !dex().item(t.item).is_mega_stone {
                    c.mul(3, 2);
                }
            }
            MoveFx::Facade => {
                if s.status != crate::dex::Status::None && s.status != crate::dex::Status::Slp {
                    c.mul(2, 1);
                }
            }
            MoveFx::Venoshock | MoveFx::BarbBarrage => {
                if matches!(t.status, crate::dex::Status::Psn | crate::dex::Status::Tox) {
                    c.mul(2, 1);
                }
            }
            MoveFx::LashOut => {
                if s.vol.stats_lowered_this_turn {
                    c.mul(2, 1);
                }
            }
            MoveFx::SolarBeam | MoveFx::SolarBlade => {
                let w = self.user_weather(src);
                if !matches!(w, Weather::None | Weather::Sun) {
                    c.mul(1, 2);
                }
            }
            _ => {}
        }
        let ab = self.ab(src);
        let bp_raw = bp;
        // Technician (priority 30).
        if ab == Ab::Technician && bp_raw <= 60 {
            c.mul(3, 2);
        }
        // Priority 24: Rivalry.
        if ab == Ab::Rivalry && s.gender != 0 && t.gender != 0 {
            if s.gender == t.gender {
                c.mul(5, 4);
            } else {
                c.mul(3, 4);
            }
        }
        // Priority 23: Iron Fist, -ate abilities.
        if ab == Ab::IronFist && am.flags & flag::PUNCH != 0 {
            c.mul4096(4915);
        }
        if am.type_changer {
            c.mul4096(4915);
        }
        // Priority 21: Tough Claws, Sheer Force, Analytic, Sand Force, Supreme Overlord.
        if ab == Ab::ToughClaws && am.flags & flag::CONTACT != 0 {
            c.mul4096(5325);
        }
        if ab == Ab::SheerForce && am.sheer_force {
            c.mul4096(5325);
        }
        if ab == Ab::Analytic && self.queue.len == 0 {
            c.mul4096(5325);
        }
        if ab == Ab::SandForce
            && self.weather() == Weather::Sand
            && matches!(am.ty, Type::Rock | Type::Ground | Type::Steel)
        {
            c.mul4096(5325);
        }
        if ab == Ab::SupremeOverlord {
            let fainted = s.vol.fallen as u32;
            if fainted > 0 {
                c.mul4096(4096 + 410 * fainted);
            }
        }
        // Priority 20: Fairy Aura (anyone on the field).
        if am.ty == Type::Fairy && self.ability_on_field(Ab::FairyAura) {
            c.mul4096(5448);
        }
        // Priority 19: Sharpness, Strong Jaw, Mega Launcher.
        if ab == Ab::Sharpness && am.flags & flag::SLICING != 0 {
            c.mul(3, 2);
        }
        if ab == Ab::StrongJaw && am.flags & flag::BITE != 0 {
            c.mul(3, 2);
        }
        if ab == Ab::MegaLauncher && am.flags & flag::PULSE != 0 {
            c.mul(3, 2);
        }
        // Items: Muscle Band / Wise Glasses (16), type boosters (15).
        match self.it(src) {
            It::MuscleBand if am.category == Category::Physical => c.mul4096(4505),
            It::WiseGlasses if am.category == Category::Special => c.mul4096(4505),
            it => {
                if let Some(t) = type_booster(it) {
                    if t == am.ty {
                        c.mul4096(4915);
                    }
                }
            }
        }
        // Helping Hand (10).
        for _ in 0..am.helping_hand {
            c.mul(3, 2);
        }
        // Charge.
        if s.vol.charge && am.ty == Type::Electric {
            c.mul(2, 1);
        }
        // Punk Rock (7).
        if ab == Ab::PunkRock && am.flags & flag::SOUND != 0 {
            c.mul4096(5325);
        }
        // Terrain (6).
        match self.field.terrain {
            Terrain::Grassy => {
                if am.ty == Type::Grass && self.is_grounded(src) && s.vol.semi_inv == SemiInv::None {
                    c.mul4096(5325);
                }
                if matches!(dex().mv(am.id).id.as_str(), "earthquake" | "bulldoze" | "magnitude")
                    && self.is_grounded(tgt)
                    && t.vol.semi_inv == SemiInv::None
                {
                    c.mul(1, 2);
                }
            }
            Terrain::Psychic => {
                if am.ty == Type::Psychic && self.is_grounded(src) && s.vol.semi_inv == SemiInv::None {
                    c.mul4096(5325);
                }
            }
            Terrain::Electric => {
                if am.ty == Type::Electric && self.is_grounded(src) && s.vol.semi_inv == SemiInv::None {
                    c.mul4096(5325);
                }
            }
            Terrain::Misty => {
                if am.ty == Type::Dragon && self.is_grounded(tgt) && t.vol.semi_inv == SemiInv::None {
                    c.mul(1, 2);
                }
            }
            Terrain::None => {}
        }
        c.apply(bp).max(1)
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
    /// and the ModifyAtk / ModifySpA events.
    fn attack_stat(&self, src: Pos, tgt: Pos, am: &ActiveMove, crit: bool) -> u32 {
        let mv = dex().mv(am.id);
        let physical = am.category == Category::Physical;
        let atk_holder = if mv.foul_play { tgt } else { src };
        let stat_idx = mv.override_offensive_stat.unwrap_or(if physical { ATK } else { SPA });
        let h = self.m(atk_holder);
        let mut boost = h.boosts[stat_idx - 1];
        // Unaware on the target ignores the attacker's boosts.
        if self.tab(tgt) == Ab::Unaware {
            boost = 0;
        }
        if crit && boost < 0 {
            boost = 0;
        }
        let mut atk = boosted(h.stats[stat_idx] as u32, boost);
        // ModifyAtk / ModifySpA event (by category, holder = user).
        let s = self.m(src);
        let ab = self.ab(src);
        let mut c = Chain::new();
        let low_hp = (s.hp as u32) * 3 <= s.max_hp as u32;
        match ab {
            Ab::HugePower | Ab::PurePower if physical => c.mul(2, 1),
            Ab::Guts if physical && s.status != crate::dex::Status::None => c.mul(3, 2),
            Ab::Hustle if physical => c.mul(3, 2),
            Ab::Blaze if am.ty == Type::Fire && low_hp => c.mul(3, 2),
            Ab::Torrent if am.ty == Type::Water && low_hp => c.mul(3, 2),
            Ab::Overgrow if am.ty == Type::Grass && low_hp => c.mul(3, 2),
            Ab::Swarm if am.ty == Type::Bug && low_hp => c.mul(3, 2),
            Ab::FlashFire if s.vol.flash_fire && am.ty == Type::Fire => c.mul(3, 2),
            Ab::SolarPower if !physical && self.weather() == Weather::Sun => c.mul(3, 2),
            Ab::WaterBubble if am.ty == Type::Water => c.mul(2, 1),
            Ab::FireMane if am.ty == Type::Fire => c.mul(3, 2),
            _ => {}
        }
        match self.it(src) {
            It::ChoiceBand if physical => c.mul(3, 2),
            It::ChoiceSpecs if !physical => c.mul(3, 2),
            _ => {}
        }
        // Target-side attack modifiers (onSourceModifyAtk).
        match self.tab(tgt) {
            Ab::ThickFat if matches!(am.ty, Type::Fire | Type::Ice) => c.mul(1, 2),
            Ab::WaterBubble if am.ty == Type::Fire => c.mul(1, 2),
            _ => {}
        }
        atk = c.apply(atk);
        atk.max(1)
    }

    fn defense_stat(&self, src: Pos, tgt: Pos, am: &ActiveMove, crit: bool) -> u32 {
        let mv = dex().mv(am.id);
        let physical = am.category == Category::Physical;
        let stat_idx = mv.override_defensive_stat.unwrap_or(if physical { DEF } else { SPD });
        let t = self.m(tgt);
        let mut boost = t.boosts[stat_idx - 1];
        if mv.ignore_defensive || (self.is_live(src) && self.ab(src) == Ab::Unaware) {
            boost = 0;
        }
        if crit && boost > 0 {
            boost = 0;
        }
        let mut def = boosted(t.stats[stat_idx] as u32, boost);
        let mut c = Chain::new();
        let w = self.weather();
        if stat_idx == SPD && w == Weather::Sand && t.has_type(Type::Rock) {
            c.mul(3, 2);
        }
        if stat_idx == DEF && w == Weather::Snow && t.has_type(Type::Ice) {
            c.mul(3, 2);
        }
        if stat_idx == DEF && self.tab(tgt) == Ab::MarvelScale && t.status != crate::dex::Status::None {
            c.mul(3, 2);
        }
        def = c.apply(def);
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

    /// modifyDamage: spread, weather, crit, random, STAB, type, burn, final.
    pub fn modify_damage(&self, base: u32, src: Pos, tgt: Pos, am: &ActiveMove, crit: bool, roll: u32) -> u32 {
        let mut dmg = base + 2;
        if am.spread_hit {
            dmg = modify(dmg, 3, 4);
        }
        if am.parental_bond_hit {
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
        let s = self.m(src);
        if s.status == crate::dex::Status::Brn
            && am.category == Category::Physical
            && self.ab(src) != Ab::Guts
            && am.fx != MoveFx::Facade
        {
            dmg = modify(dmg, 1, 2);
        }
        // Final modifiers (ModifyDamage).
        let mut c = Chain::new();
        let t = self.m(tgt);
        let tside = &self.sides[tgt.s()].conds;
        let infiltrates = am.infiltrates;
        if !crit && !infiltrates {
            let screen = match am.category {
                Category::Physical => tside.reflect > 0 || tside.aurora_veil > 0,
                Category::Special => tside.light_screen > 0 || tside.aurora_veil > 0,
                Category::Status => false,
            };
            if screen {
                c.mul4096(2732);
            }
        }
        // Attacker: Sniper, Tinted Lens, Life Orb, Expert Belt.
        let ab = self.ab(src);
        if ab == Ab::Sniper && crit {
            c.mul(3, 2);
        }
        if ab == Ab::TintedLens && type_mod < 0 {
            c.mul(2, 1);
        }
        match self.it(src) {
            It::LifeOrb => c.mul4096(5324),
            It::ExpertBelt if type_mod > 0 => c.mul4096(4915),
            _ => {}
        }
        // Target: Multiscale, Filter/Solid Rock, Fluffy, Punk Rock, Aura Guard,
        // Friend Guard on the ally, resist berries, Glaive Rush.
        let tab = self.tab(tgt);
        if tab == Ab::Multiscale && t.hp == t.max_hp {
            c.mul(1, 2);
        }
        if matches!(tab, Ab::Filter | Ab::SolidRock) && type_mod > 0 {
            c.mul(3, 4);
        }
        if tab == Ab::Fluffy {
            if am.flags & flag::CONTACT != 0 && am.ty != Type::Fire {
                c.mul(1, 2);
            }
            if am.ty == Type::Fire && am.flags & flag::CONTACT == 0 {
                c.mul(2, 1);
            }
        }
        if tab == Ab::PunkRock && am.flags & flag::SOUND != 0 {
            c.mul(1, 2);
        }
        if tab == Ab::AuraGuard && am.flags & flag::CONTACT != 0 {
            c.mul(1, 2);
        }
        if let Some(a) = self.live_ally(tgt) {
            if self.tab(a) == Ab::FriendGuard {
                c.mul4096(3072);
            }
        }
        if let Some(bt) = resist_berry(self.it(tgt)) {
            if bt == am.ty && (type_mod > 0 || bt == Type::Normal) && !self.unnerved(tgt) {
                c.mul(1, 2);
            }
        }
        if t.vol.glaive_rush > 0 {
            c.mul(2, 1);
        }
        dmg = c.apply(dmg);
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
