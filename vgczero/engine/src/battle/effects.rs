//! Stat changes, status, volatiles, damage, healing, items, fainting,
//! weather/terrain and Mega Evolution.

use super::{blog, Battle};
use crate::dex::{dex, Boosts, Category, Status, Terrain, VolKind, Weather, B_ACC, B_ATK, B_DEF, B_SPA, B_SPE, B_SPD};
use crate::kinds::{Ab, It};
use crate::state::{Pos, ProtectKind};
use crate::types::Type;

/// Where a stat change comes from (for ability interactions).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoostSrc {
    /// The Pokemon's own move / ability / item.
    SelfInflicted,
    /// An opponent's (or ally's) move effect.
    Move(Pos),
    /// Intimidate from a Pokemon.
    Intimidate(Pos),
    /// Other ability (Cotton Down, etc.) from a Pokemon.
    Ability(Pos),
}

impl BoostSrc {
    fn source(self) -> Option<Pos> {
        match self {
            BoostSrc::SelfInflicted => None,
            BoostSrc::Move(p) | BoostSrc::Intimidate(p) | BoostSrc::Ability(p) => Some(p),
        }
    }
}

/// Kinds of damage for Magic Guard / Sturdy / Focus Sash rules.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DmgKind {
    /// A move's direct hit.
    Attack,
    /// Residual, recoil, Life Orb, Rough Skin... (Magic Guard blocks).
    Indirect,
    /// Confusion self-hit (a move-like hit; Magic Guard does not block).
    Confusion,
    /// HP cost the Pokemon pays itself (Substitute, Belly Drum, Steel Beam).
    SelfCost,
}

const STAT_NAMES: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

impl Battle {
    // ---- boosts ------------------------------------------------------------------

    /// Apply stat stage changes. Returns true if any stat changed.
    pub fn boost(&mut self, tgt: Pos, boosts: &Boosts, src: BoostSrc) -> bool {
        if !self.is_live(tgt) {
            return false;
        }
        let mut b = *boosts;
        let from_other = match src.source() {
            Some(s) => s != tgt,
            None => false,
        };
        let tab = self.tab(tgt);
        // ChangeBoost: Contrary inverts.
        if tab == Ab::Contrary {
            for x in b.iter_mut() {
                *x = -*x;
            }
        }
        // TryBoost: drop prevention from others.
        if from_other {
            let mut blocked_all = false;
            match tab {
                Ab::ClearBody | Ab::WhiteSmoke | Ab::FullMetalBody => blocked_all = true,
                Ab::HyperCutter => {
                    if b[B_ATK] < 0 {
                        b[B_ATK] = 0;
                    }
                }
                Ab::BigPecks => {
                    if b[B_DEF] < 0 {
                        b[B_DEF] = 0;
                    }
                }
                Ab::Illuminate => {
                    if b[B_ACC] < 0 {
                        b[B_ACC] = 0;
                    }
                }
                Ab::MirrorArmor => {
                    // Reflect drops back to the source.
                    let mut reflected = [0i8; 7];
                    let mut any = false;
                    for i in 0..7 {
                        if b[i] < 0 {
                            reflected[i] = b[i];
                            b[i] = 0;
                            any = true;
                        }
                    }
                    if any {
                        if let Some(s) = src.source() {
                            if self.is_live(s) && self.ab(s) != Ab::MirrorArmor {
                                blog!(self, "|-ability|{}|Mirror Armor", self.name(tgt));
                                self.boost(s, &reflected, BoostSrc::Ability(tgt));
                            }
                        }
                    }
                }
                _ => {}
            }
            if blocked_all {
                for x in b.iter_mut() {
                    if *x < 0 {
                        *x = 0;
                    }
                }
            }
            // Flower Veil protects Grass-type allies (and itself).
            if self.m(tgt).has_type(Type::Grass) {
                let veiled = tab == Ab::FlowerVeil || self.live_ally(tgt).map(|a| self.tab(a) == Ab::FlowerVeil).unwrap_or(false);
                if veiled {
                    for x in b.iter_mut() {
                        if *x < 0 {
                            *x = 0;
                        }
                    }
                }
            }
            if self.sides[tgt.s()].conds.mist > 0 && !self.attacker_infiltrates(src.source()) {
                for x in b.iter_mut() {
                    if *x < 0 {
                        *x = 0;
                    }
                }
            }
        }
        // Each stat is applied in turn; Defiant / Competitive react to each
        // stat an opponent lowered (Showdown's AfterEachBoost).
        let foe_src = src.source().map(|s| s.side != tgt.side).unwrap_or(false);
        let mut changed = false;
        let mut lowered = false;
        let mut raised = false;
        for i in 0..7 {
            if b[i] == 0 || !self.is_live(tgt) {
                continue;
            }
            let m = self.mm(tgt);
            let old = m.boosts[i];
            let new = (old as i32 + b[i] as i32).clamp(-6, 6) as i8;
            m.boosts[i] = new;
            let delta = new - old;
            if delta == 0 {
                continue;
            }
            changed = true;
            if b[i] < 0 {
                lowered = true;
            } else {
                raised = true;
            }
            blog!(self, "|{}|{}|{}|{}", if delta > 0 { "-boost" } else { "-unboost" }, self.name(tgt), STAT_NAMES[i], delta.abs());
            if b[i] < 0 && foe_src {
                let up_stat = match self.ab(tgt) {
                    Ab::Defiant => Some(B_ATK),
                    Ab::Competitive => Some(B_SPA),
                    _ => None,
                };
                if let Some(st) = up_stat {
                    self.reveal_ability(tgt);
                    let mut up = [0i8; 7];
                    up[st] = 2;
                    self.boost(tgt, &up, BoostSrc::SelfInflicted);
                }
            }
        }
        if raised {
            self.mm(tgt).vol.stats_raised_this_turn = true;
        }
        if lowered {
            self.mm(tgt).vol.stats_lowered_this_turn = true;
            self.check_white_herb(tgt);
        }
        changed
    }

    fn attacker_infiltrates(&self, src: Option<Pos>) -> bool {
        match src {
            Some(s) if self.is_live(s) => self.ab(s) == Ab::Infiltrator,
            _ => false,
        }
    }

    /// White Herb: restore lowered stats (Update event).
    pub fn check_white_herb(&mut self, p: Pos) {
        if !self.is_live(p) || self.it(p) != It::WhiteHerb {
            return;
        }
        if self.m(p).boosts.iter().any(|&x| x < 0) {
            for x in self.mm(p).boosts.iter_mut() {
                if *x < 0 {
                    *x = 0;
                }
            }
            blog!(self, "|-enditem|{}|White Herb", self.name(p));
            self.consume_item(p);
        }
    }

    /// Intimidate on all adjacent foes.
    pub fn intimidate(&mut self, src: Pos) {
        self.reveal_ability(src);
        blog!(self, "|-ability|{}|Intimidate|boost", self.name(src));
        let foes: Vec<Pos> = self.live_foes(src).collect();
        for t in foes {
            if self.m(t).vol.substitute > 0 {
                continue;
            }
            match self.tab(t) {
                Ab::InnerFocus | Ab::Oblivious | Ab::OwnTempo | Ab::Scrappy => {
                    self.reveal_ability(t);
                    blog!(self, "|-fail|{}|unboost|[from] ability", self.name(t));
                    continue;
                }
                _ => {}
            }
            let mut d = [0i8; 7];
            d[B_ATK] = -1;
            self.boost(t, &d, BoostSrc::Intimidate(src));
            if self.is_live(t) && self.ab(t) == Ab::Rattled {
                let mut up = [0i8; 7];
                up[B_SPE] = 1;
                self.boost(t, &up, BoostSrc::SelfInflicted);
            }
        }
    }

    // ---- status ------------------------------------------------------------------

    /// Can `st` be inflicted on `tgt`? (type, ability, terrain, Safeguard).
    pub fn status_immune(&self, tgt: Pos, st: Status, src: Option<Pos>) -> bool {
        let m = self.m(tgt);
        let corrosion = src.map(|s| self.is_live(s) && self.ab(s) == Ab::Corrosion).unwrap_or(false);
        match st {
            Status::Par if m.has_type(Type::Electric) => return true,
            Status::Brn if m.has_type(Type::Fire) => return true,
            Status::Frz if m.has_type(Type::Ice) => return true,
            Status::Psn | Status::Tox if (m.has_type(Type::Poison) || m.has_type(Type::Steel)) && !corrosion => {
                return true
            }
            _ => {}
        }
        let from_other = src.map(|s| s != tgt).unwrap_or(false);
        match (self.tab(tgt), st) {
            (Ab::Limber, Status::Par) => return true,
            (Ab::Insomnia | Ab::VitalSpirit, Status::Slp) => return true,
            (Ab::WaterBubble | Ab::ThermalExchange, Status::Brn) => return true,
            (Ab::MagmaArmor, Status::Frz) => return true,
            (Ab::LeafGuard, _) if self.weather() == Weather::Sun => return true,
            _ => {}
        }
        if st == Status::Slp {
            if self.tab(tgt) == Ab::SweetVeil || self.live_ally(tgt).map(|a| self.tab(a) == Ab::SweetVeil).unwrap_or(false) {
                return true;
            }
        }
        if from_other && m.has_type(Type::Grass) {
            let veiled = self.tab(tgt) == Ab::FlowerVeil
                || self.live_ally(tgt).map(|a| self.tab(a) == Ab::FlowerVeil).unwrap_or(false);
            if veiled {
                return true;
            }
        }
        if self.is_grounded(tgt) && m.vol.semi_inv == crate::state::SemiInv::None {
            match self.field.terrain {
                Terrain::Misty => return true,
                Terrain::Electric if st == Status::Slp => return true,
                _ => {}
            }
        }
        if from_other && self.sides[tgt.s()].conds.safeguard > 0 && !self.attacker_infiltrates(src) {
            return true;
        }
        false
    }

    /// Try to inflict a major status. Returns true on success.
    pub fn try_set_status(&mut self, tgt: Pos, st: Status, src: Option<Pos>) -> bool {
        if !self.is_live(tgt) || st == Status::None {
            return false;
        }
        if self.m(tgt).status != Status::None {
            return false;
        }
        if self.status_immune(tgt, st, src) {
            return false;
        }
        let turns = match st {
            Status::Slp => {
                // Champions: sleep lasts 2 or 3 (startTime sampled from [2, 3, 3]).
                let t = [2u8, 3, 3][self.rng.below(3) as usize];
                t
            }
            Status::Frz => 3,
            _ => 0,
        };
        {
            let m = self.mm(tgt);
            m.status = st;
            m.status_turns = turns;
        }
        blog!(self, "|-status|{}|{:?}", self.name(tgt), st);
        // Synchronize.
        if let Some(s) = src {
            if s.side != tgt.side
                && self.ab(tgt) == Ab::Synchronize
                && matches!(st, Status::Brn | Status::Par | Status::Psn | Status::Tox)
                && self.is_live(s)
            {
                self.reveal_ability(tgt);
                let st2 = st;
                self.try_set_status(s, st2, Some(tgt));
            }
        }
        self.check_status_berry(tgt);
        true
    }

    pub fn cure_status(&mut self, p: Pos) {
        let m = self.mm(p);
        m.status = Status::None;
        m.status_turns = 0;
    }

    /// Lum / Chesto / etc. (Update event).
    pub fn check_status_berry(&mut self, p: Pos) {
        if !self.is_live(p) || self.unnerved(p) {
            return;
        }
        let st = self.m(p).status;
        let conf = self.m(p).vol.confusion > 0;
        let cure = match self.it(p) {
            It::LumBerry => st != Status::None || conf,
            It::ChestoBerry => st == Status::Slp,
            It::CheriBerry => st == Status::Par,
            It::PechaBerry => matches!(st, Status::Psn | Status::Tox),
            It::RawstBerry => st == Status::Brn,
            It::AspearBerry => st == Status::Frz,
            It::PersimBerry => conf,
            _ => false,
        };
        if cure {
            match self.it(p) {
                It::LumBerry => {
                    self.cure_status(p);
                    self.mm(p).vol.confusion = 0;
                }
                It::PersimBerry => self.mm(p).vol.confusion = 0,
                _ => self.cure_status(p),
            }
            blog!(self, "|-enditem|{}|berry|[eat]", self.name(p));
            self.eat_item(p);
        }
    }

    // ---- volatiles ---------------------------------------------------------------------

    /// Add a volatile status. Returns true on success.
    pub fn add_volatile(&mut self, tgt: Pos, v: VolKind, src: Option<Pos>) -> bool {
        if !self.is_live(tgt) {
            return false;
        }
        let tab = self.tab(tgt);
        let grounded_misty =
            self.field.terrain == Terrain::Misty && self.is_grounded(tgt) && self.m(tgt).vol.semi_inv == crate::state::SemiInv::None;
        match v {
            VolKind::Flinch => {
                if tab == Ab::InnerFocus {
                    return false;
                }
                self.mm(tgt).vol.flinch = true;
            }
            VolKind::Confusion => {
                if self.m(tgt).vol.confusion > 0 || tab == Ab::OwnTempo || grounded_misty {
                    return false;
                }
                if src.map(|s| s != tgt).unwrap_or(false)
                    && self.sides[tgt.s()].conds.safeguard > 0
                    && !self.attacker_infiltrates(src)
                {
                    return false;
                }
                let t = self.rng.range(2, 6) as u8;
                self.mm(tgt).vol.confusion = t;
                blog!(self, "|-start|{}|confusion", self.name(tgt));
                self.check_status_berry(tgt);
            }
            VolKind::MustRecharge => self.mm(tgt).vol.must_recharge = true,
            VolKind::Taunt => {
                if self.m(tgt).vol.taunt > 0 || tab == Ab::Oblivious {
                    return false;
                }
                let mut d = 3;
                if self.m(tgt).vol.active_turns > 0 && self.queue.will_move(tgt).is_none() {
                    d += 1;
                }
                self.mm(tgt).vol.taunt = d;
                blog!(self, "|-start|{}|move: Taunt", self.name(tgt));
                if self.it(tgt) == It::MentalHerb {
                    self.mm(tgt).vol.taunt = 0;
                    self.consume_item(tgt);
                }
            }
            VolKind::Yawn => {
                if self.m(tgt).vol.yawn > 0 || self.m(tgt).status != Status::None {
                    return false;
                }
                if self.status_immune(tgt, Status::Slp, src) {
                    return false;
                }
                self.mm(tgt).vol.yawn = 2;
            }
            VolKind::FocusEnergy => {
                if self.m(tgt).vol.crit_stage >= 2 {
                    return false;
                }
                self.mm(tgt).vol.crit_stage = 2;
            }
            VolKind::DragonCheer => {
                if self.m(tgt).vol.crit_stage > 0 {
                    return false;
                }
                self.mm(tgt).vol.crit_stage = if self.m(tgt).has_type(Type::Dragon) { 2 } else { 1 };
            }
            VolKind::LeechSeed => {
                if self.m(tgt).has_type(Type::Grass) || self.m(tgt).vol.leech_seed {
                    return false;
                }
                let code = src.map(|s| s.code()).unwrap_or(0);
                let m = self.mm(tgt);
                m.vol.leech_seed = true;
                m.vol.leech_seed_src = code;
            }
            VolKind::Roost => self.mm(tgt).vol.roost = true,
            VolKind::Imprison => {
                if self.m(tgt).vol.imprison {
                    return false;
                }
                self.mm(tgt).vol.imprison = true;
            }
            VolKind::GlaiveRush => self.mm(tgt).vol.glaive_rush = 1,
            VolKind::SaltCure => {
                if self.m(tgt).vol.salt_cure {
                    return false;
                }
                self.mm(tgt).vol.salt_cure = true;
            }
            VolKind::Curse => self.mm(tgt).vol.curse = true,
            VolKind::Torment => {
                if self.m(tgt).vol.torment {
                    return false;
                }
                self.mm(tgt).vol.torment = true;
            }
            VolKind::HealBlock => {
                if self.m(tgt).vol.heal_block > 0 {
                    return false;
                }
                self.mm(tgt).vol.heal_block = 5;
            }
            VolKind::MagnetRise => {
                if self.m(tgt).vol.magnet_rise > 0 || self.m(tgt).vol.smacked_down || self.m(tgt).vol.ingrain {
                    return false;
                }
                self.mm(tgt).vol.magnet_rise = 5;
            }
            VolKind::AquaRing => {
                if self.m(tgt).vol.aqua_ring {
                    return false;
                }
                self.mm(tgt).vol.aqua_ring = true;
            }
            VolKind::Ingrain => {
                if self.m(tgt).vol.ingrain {
                    return false;
                }
                self.mm(tgt).vol.ingrain = true;
            }
            VolKind::Charge => self.mm(tgt).vol.charge = true,
            VolKind::LaserFocus => self.mm(tgt).vol.laser_focus = 2,
            VolKind::TarShot => {
                if self.m(tgt).vol.tar_shot {
                    return false;
                }
                self.mm(tgt).vol.tar_shot = true;
            }
            VolKind::SmackDown => {
                let m = self.mm(tgt);
                m.vol.smacked_down = true;
                m.vol.magnet_rise = 0;
            }
            VolKind::NoRetreat => {
                if self.m(tgt).vol.no_retreat {
                    return false;
                }
                self.mm(tgt).vol.no_retreat = true;
            }
            VolKind::Attract => {
                let (sg, tg) = (src.map(|s| self.m(s).gender).unwrap_or(0), self.m(tgt).gender);
                if sg == 0 || tg == 0 || sg == tg || self.m(tgt).vol.attract || tab == Ab::Oblivious {
                    return false;
                }
                self.mm(tgt).vol.attract = true;
            }
            VolKind::Stockpile => {
                if self.m(tgt).vol.stockpile >= 3 {
                    return false;
                }
                self.mm(tgt).vol.stockpile += 1;
                let mut up = [0i8; 7];
                up[B_DEF] = 1;
                up[B_SPD] = 1;
                self.boost(tgt, &up, BoostSrc::SelfInflicted);
            }
            VolKind::PartiallyTrapped => {
                // Fire Spin, Whirlpool, Infestation...: 5 or 6 turns.
                if self.m(tgt).vol.partial_trap > 0 {
                    return false;
                }
                let s = src.unwrap_or(tgt);
                let mon = self.sides[s.s()].active[s.i()];
                let dur = self.rng.range(5, 7) as u8;
                let m = self.mm(tgt);
                m.vol.partial_trap = dur;
                m.vol.partial_trap_src = s.code();
                m.vol.partial_trap_mon = mon;
            }
            VolKind::Endure => self.mm(tgt).vol.endure = true,
            VolKind::DestinyBond => self.mm(tgt).vol.destiny_bond = true,
            VolKind::Protect => self.mm(tgt).vol.protect = ProtectKind::Protect,
            // Volatiles applied by scripted moves themselves.
            VolKind::Encore
            | VolKind::Disable
            | VolKind::Substitute
            | VolKind::HelpingHand
            | VolKind::FollowMe
            | VolKind::RagePowder
            | VolKind::ThroatChop
            | VolKind::Unsupported => return false,
        }
        true
    }

    // ---- HP ------------------------------------------------------------------------------

    /// Deal damage. Returns damage actually dealt.
    pub fn damage(&mut self, tgt: Pos, amount: u32, src: Option<Pos>, kind: DmgKind) -> u32 {
        if !self.is_live(tgt) || amount == 0 {
            return 0;
        }
        if kind == DmgKind::Indirect && self.ab(tgt) == Ab::MagicGuard {
            return 0;
        }
        let mut dmg = amount.min(self.m(tgt).hp as u32);
        if kind == DmgKind::Attack {
            let m = self.m(tgt);
            let full = m.hp == m.max_hp;
            if dmg >= m.hp as u32 {
                if m.vol.endure {
                    dmg = m.hp as u32 - 1;
                    blog!(self, "|-activate|{}|move: Endure", self.name(tgt));
                } else if full && m.hp > 1 && self.tab(tgt) == Ab::Sturdy {
                    dmg = m.hp as u32 - 1;
                    self.reveal_ability(tgt);
                    blog!(self, "|-activate|{}|ability: Sturdy", self.name(tgt));
                } else if full && m.hp > 1 && self.it(tgt) == It::FocusSash {
                    dmg = m.hp as u32 - 1;
                    blog!(self, "|-enditem|{}|Focus Sash", self.name(tgt));
                    self.consume_item(tgt);
                }
            }
        }
        let m = self.mm(tgt);
        m.hp -= dmg as u16;
        if kind == DmgKind::Attack {
            m.vol.hurt_this_turn = true;
        }
        let hp = m.hp;
        blog!(self, "|-damage|{}|{}/{}", self.name(tgt), hp, self.m(tgt).max_hp);
        if hp == 0 {
            self.queue_faint(tgt);
        }
        let _ = src;
        dmg
    }

    pub fn heal(&mut self, tgt: Pos, amount: u32) -> u32 {
        if !self.is_live(tgt) || amount == 0 {
            return 0;
        }
        if self.m(tgt).vol.heal_block > 0 {
            return 0;
        }
        let m = self.mm(tgt);
        let room = (m.max_hp - m.hp) as u32;
        let h = amount.min(room);
        m.hp += h as u16;
        if h > 0 {
            blog!(self, "|-heal|{}|{}/{}", self.name(tgt), self.m(tgt).hp, self.m(tgt).max_hp);
        }
        h
    }

    /// Sitrus Berry & co. after HP changes (Update event).
    pub fn check_hp_berry(&mut self, p: Pos) {
        if !self.is_live(p) || self.unnerved(p) {
            return;
        }
        let m = self.m(p);
        match self.it(p) {
            It::SitrusBerry if (m.hp as u32) * 2 <= m.max_hp as u32 => {
                let amt = m.max_hp as u32 / 4;
                blog!(self, "|-enditem|{}|Sitrus Berry|[eat]", self.name(p));
                self.eat_item(p);
                self.heal(p, amt);
            }
            It::OranBerry if (m.hp as u32) * 2 <= m.max_hp as u32 => {
                self.eat_item(p);
                self.heal(p, 10);
            }
            _ => {}
        }
    }

    /// All Update-event item checks for one Pokemon.
    pub fn update_items(&mut self, p: Pos) {
        self.check_hp_berry(p);
        self.check_status_berry(p);
        self.check_white_herb(p);
    }

    // ---- items --------------------------------------------------------------------------

    /// Remove the held item (used up or knocked off).
    pub fn remove_item(&mut self, p: Pos) {
        let had = self.m(p).item;
        if had == 0 {
            return;
        }
        {
            let m = self.mm(p);
            m.last_item = had;
            m.item = 0;
            m.vol.choice_lock = 0;
            m.reveal.item = true;
        }
        if self.ab(p) == Ab::Unburden {
            self.mm(p).vol.unburden = true;
        }
    }

    pub fn consume_item(&mut self, p: Pos) {
        self.remove_item(p);
        self.symbiosis(p);
    }

    pub fn eat_item(&mut self, p: Pos) {
        self.mm(p).ate_berry = true;
        self.consume_item(p);
    }

    fn symbiosis(&mut self, p: Pos) {
        if let Some(a) = self.live_ally(p) {
            if self.ab(a) == Ab::Symbiosis && self.m(a).item != 0 && self.m(p).item == 0 {
                let it = self.m(a).item;
                if dex().item(it).is_mega_stone {
                    return;
                }
                self.mm(a).item = 0;
                self.mm(p).item = it;
                self.mm(p).vol.unburden = false;
                self.reveal_ability(a);
            }
        }
    }

    // ---- faint ---------------------------------------------------------------------------

    pub(crate) fn queue_faint(&mut self, p: Pos) {
        // Faints are processed after the action; record the order they hit 0.
        self.last_faint_side = Some(p.side);
        let _ = p;
    }

    pub fn faint(&mut self, p: Pos) {
        let name = self.name(p);
        {
            let side = &mut self.sides[p.s()];
            side.total_fainted = side.total_fainted.saturating_add(1);
            side.fainted_this_turn = true;
            let m = side.mons.get_mut(side.active[p.i()] as usize).unwrap();
            m.fainted = true;
            m.hp = 0;
            m.status = Status::None;
            m.boosts = [0; 7];
            m.vol = Default::default();
            // clearVolatile: ability and types back to the (Mega) forme's own.
            m.ability = m.base_ability;
            m.types = m.base_types;
        }
        blog!(self, "|faint|{}", name);
        // The slot stays occupied by the fainted Pokemon until replaced.
    }

    // ---- field ----------------------------------------------------------------------------

    pub fn set_weather(&mut self, w: Weather, src: Pos) -> bool {
        if self.field.weather == w {
            return false;
        }
        let rock = match (w, self.it(src)) {
            (Weather::Rain, It::DampRock) | (Weather::Sun, It::HeatRock) | (Weather::Sand, It::SmoothRock) | (Weather::Snow, It::IcyRock) => true,
            _ => false,
        };
        self.field.weather = w;
        self.field.weather_turns = if rock { 8 } else { 5 };
        blog!(self, "|-weather|{:?}", w);
        true
    }

    pub fn set_terrain(&mut self, t: Terrain, src: Pos) -> bool {
        if self.field.terrain == t {
            return false;
        }
        self.field.terrain = t;
        self.field.terrain_turns = if self.it(src) == It::TerrainExtender { 8 } else { 5 };
        blog!(self, "|-fieldstart|{:?}", t);
        // Seeds activate when their terrain starts.
        let all: Vec<Pos> = self.live_positions().collect();
        for p in all {
            self.check_seed(p);
        }
        true
    }

    pub fn check_seed(&mut self, p: Pos) {
        if !self.is_live(p) {
            return;
        }
        let t = self.field.terrain;
        let stat = match (self.it(p), t) {
            (It::GrassySeed, Terrain::Grassy) | (It::ElectricSeed, Terrain::Electric) => B_DEF,
            (It::PsychicSeed, Terrain::Psychic) | (It::MistySeed, Terrain::Misty) => B_SPD,
            _ => return,
        };
        let mut up = [0i8; 7];
        up[stat] = 1;
        blog!(self, "|-enditem|{}|seed", self.name(p));
        self.consume_item(p);
        self.boost(p, &up, BoostSrc::SelfInflicted);
    }

    // ---- Mega Evolution -----------------------------------------------------------------

    pub fn mega_evolve(&mut self, p: Pos) {
        if !self.is_live(p) || !self.can_mega(p) {
            return;
        }
        {
            let m = self.mm(p);
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
            m.reveal.item = true;
            m.reveal.ability = true;
        }
        self.sides[p.s()].mega_used = true;
        blog!(self, "|-mega|{}", self.name(p));
        // The new ability starts immediately.
        self.ability_start(p);
    }

    pub fn reveal_ability(&mut self, p: Pos) {
        if self.is_live(p) {
            self.mm(p).reveal.ability = true;
        }
    }

    pub fn reveal_item(&mut self, p: Pos) {
        if self.is_live(p) {
            self.mm(p).reveal.item = true;
        }
    }

    /// Is this move a "status" move for Good as Gold / Magic Bounce purposes?
    pub fn is_status_move(&self, cat: Category) -> bool {
        cat == Category::Status
    }
}
