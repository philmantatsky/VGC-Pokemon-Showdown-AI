//! Running a move: before-move checks, targeting and redirection, Showdown's
//! hit steps (protection, immunity, accuracy, the hit loop), move effects,
//! secondaries, contact/on-hit reactions, and scripted moves.

use super::calc::ActiveMove;
use super::effects::{BoostSrc, DmgKind};
use super::legal::is_choice;
use super::{blog, ActionKind, Battle};
use crate::actions::{T_ALLY, T_FOE0, T_FOE1};
use crate::dex::{
    dex, flag, Boosts, Category, MoveId, PseudoWeather, SideCond, Status, Target, Terrain, VolKind, Weather, B_ATK,
    B_DEF, B_SPA, B_SPD, B_SPE,
};
use crate::kinds::{Ab, It, MoveFx};
use crate::state::{Pos, ProtectKind, SemiInv};
use crate::types::Type;

/// Outcome of the hit steps for one target.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Hit {
    /// Still being hit.
    Pending,
    /// Blocked / immune / missed: counts as a failure for this target.
    Failed,
    /// Stopped without counting as a failure (Protect returns NOT_FAIL).
    NotFail,
}

const MAX_TARGETS: usize = 3;

fn ups(stat: usize, n: i8) -> Boosts {
    let mut b = [0i8; 7];
    b[stat] = n;
    b
}

impl Battle {
    // ---- the move action -----------------------------------------------------------

    pub(crate) fn run_move_action(&mut self, p: Pos, mslot: usize, target: u8, move_id: MoveId) {
        if !self.is_live(p) {
            return;
        }
        let d = dex();
        {
            let m = self.mm(p);
            m.vol.move_actions = m.vol.move_actions.saturating_add(1);
            m.vol.moved_this_turn = true;
        }
        // Recharge turn.
        if self.m(p).vol.must_recharge {
            self.mm(p).vol.must_recharge = false;
            blog!(self, "|cant|{}|recharge", self.name(p));
            return;
        }
        // Encore (Champions changes the queued move when Encore starts, so a
        // differing choice here means Encore landed after the choice).
        let mut mslot = mslot;
        let mut move_id = move_id;
        let mut target = target;
        let charging = self.m(p).vol.charging > 0;
        if charging {
            mslot = (self.m(p).vol.charging - 1) as usize;
            move_id = self.m(p).moves[mslot];
            target = self.m(p).vol.charge_target.max(0) as u8;
        } else if move_id != d.struggle {
            let enc = self.m(p).vol.encore;
            if enc > 0 && self.m(p).vol.encore_slot as usize != mslot {
                mslot = self.m(p).vol.encore_slot as usize;
                move_id = self.m(p).moves[mslot];
                target = self.random_target_code(p);
            }
        }
        // An action changed by Encore has no chosen target.
        if target == crate::battle::TARGET_RANDOM {
            target = self.random_target_code(p);
        }
        // Glaive Rush ends when the user next tries to move.
        self.mm(p).vol.glaive_rush = 0;

        if let Err(counts_as_fail) = self.before_move(p, move_id) {
            self.mm(p).vol.this_move_failed = counts_as_fail;
            self.mm(p).vol.charging = 0;
            self.mm(p).vol.semi_inv = SemiInv::None;
            return;
        }

        // PP and last-move bookkeeping (moveUsed).
        let struggle = move_id == d.struggle;
        if !struggle && !charging {
            let m = self.mm(p);
            if m.pp[mslot] == 0 {
                blog!(self, "|cant|{}|nopp", self.name(p));
                self.mm(p).vol.this_move_failed = true;
                return;
            }
            m.pp[mslot] -= 1;
        }
        {
            let m = self.mm(p);
            m.vol.last_move = move_id;
            m.vol.last_move_slot = mslot as u8;
            if !struggle {
                m.reveal.moves |= 1 << mslot;
            }
            if is_choice(dex().it(m.item)) && m.vol.choice_lock == 0 && !struggle {
                m.vol.choice_lock = mslot as u8 + 1;
            }
        }
        let ok = self.use_move(p, move_id, mslot, target, struggle);
        if self.is_live(p) {
            self.mm(p).vol.this_move_failed = !ok;
        }
        if dex().mv(move_id).fx == MoveFx::Round {
            self.round_used = true;
            // The next queued Round (either side) moves next.
            let round = dex().move_id("round");
            if let Some(i) = self.queue.list[..self.queue.len]
                .iter()
                .position(|a| a.map(|a| matches!(a.kind, ActionKind::Move { .. }) && Some(a.move_id) == round).unwrap_or(false))
            {
                self.prioritize_action(i);
            }
        }
    }

    /// Before-move checks in Showdown's BeforeMove priority order.
    /// Err(true) = the move fails (counts for Stomping Tantrum).
    fn before_move(&mut self, p: Pos, move_id: MoveId) -> Result<(), bool> {
        let mv = dex().mv(move_id);
        // Sleep (10).
        if self.m(p).status == Status::Slp {
            let m = self.mm(p);
            m.status_turns = m.status_turns.saturating_sub(1);
            if m.status_turns == 0 {
                self.cure_status(p);
                blog!(self, "|-curestatus|{}|slp", self.name(p));
            } else {
                blog!(self, "|cant|{}|slp", self.name(p));
                if !mv.sleep_usable {
                    return Err(true);
                }
            }
        }
        // Freeze (10).
        if self.m(p).status == Status::Frz {
            if mv.flags & flag::DEFROST != 0 {
                self.cure_status(p);
            } else {
                let m = self.mm(p);
                m.status_turns = m.status_turns.saturating_sub(1);
                let thaw = m.status_turns == 0 || self.rng.chance(1, 4);
                if thaw {
                    self.cure_status(p);
                } else {
                    blog!(self, "|cant|{}|frz", self.name(p));
                    return Err(true);
                }
            }
        }
        // Flinch (8).
        if self.m(p).vol.flinch {
            blog!(self, "|cant|{}|flinch", self.name(p));
            if self.ab(p) == Ab::Steadfast {
                self.boost(p, &ups(B_SPE, 1), BoostSrc::SelfInflicted);
            }
            return Err(true);
        }
        // Disable (7).
        let v = self.m(p).vol;
        if v.disable > 0 && self.m(p).moves[v.disable_slot as usize] == move_id {
            return Err(true);
        }
        // Throat Chop (6), Heal Block.
        if v.throat_chop > 0 && mv.flags & flag::SOUND != 0 {
            return Err(true);
        }
        if v.heal_block > 0 && mv.flags & flag::HEAL != 0 {
            return Err(true);
        }
        // Taunt (5).
        if v.taunt > 0 && mv.category == Category::Status {
            blog!(self, "|cant|{}|move: Taunt", self.name(p));
            return Err(true);
        }
        // Gravity.
        if self.field.gravity > 0 && mv.flags & flag::GRAVITY != 0 {
            return Err(true);
        }
        // Imprison (a foe's, priority 4): the move fails if the imprisoner knows it.
        if move_id != dex().struggle {
            for q in (0..2).map(|i| p.foe(i)) {
                if self.is_live(q) {
                    let f = self.m(q);
                    if f.vol.imprison && f.moves[..f.n_moves as usize].contains(&move_id) {
                        blog!(self, "|cant|{}|move: Imprison|{}", self.name(p), mv.name);
                        return Err(true);
                    }
                }
            }
        }
        // Confusion (3).
        if self.m(p).vol.confusion > 0 {
            let m = self.mm(p);
            m.vol.confusion -= 1;
            if m.vol.confusion == 0 {
                blog!(self, "|-end|{}|confusion", self.name(p));
            } else if self.rng.chance(33, 100) {
                let dmg = self.confusion_damage(p);
                blog!(self, "|-activate|{}|confusion", self.name(p));
                self.damage(p, dmg, Some(p), DmgKind::Confusion);
                return Err(true);
            }
        }
        // Attract (2).
        if self.m(p).vol.attract && self.rng.chance(1, 2) {
            blog!(self, "|cant|{}|Attract", self.name(p));
            return Err(true);
        }
        // Paralysis (1): Champions full-paralysis chance is 1/8.
        if self.m(p).status == Status::Par && self.rng.chance(1, 8) {
            blog!(self, "|cant|{}|par", self.name(p));
            return Err(true);
        }
        Ok(())
    }

    fn confusion_damage(&mut self, p: Pos) -> u32 {
        let m = self.m(p);
        let atk = super::calc::boosted(m.stats[1] as u32, m.boosts[B_ATK]);
        let def = super::calc::boosted(m.stats[2] as u32, m.boosts[B_DEF]);
        let level = m.level as u32;
        let base = ((2 * level / 5 + 2) * 40 * atk / def.max(1)) / 50 + 2;
        let base = base & 0xFFFF;
        let roll = self.rng.below(16);
        (base * (100 - roll) / 100).max(1)
    }

    fn random_target_code(&mut self, p: Pos) -> u8 {
        let foes: Vec<u8> = (0..2u8).filter(|&i| self.is_live(p.foe(i as usize))).collect();
        if foes.is_empty() {
            T_FOE0
        } else {
            foes[self.rng.below(foes.len() as u32) as usize]
        }
    }

    // ---- useMove ---------------------------------------------------------------------

    pub(crate) fn make_active_move(&self, p: Pos, move_id: MoveId, mslot: usize, struggle: bool) -> ActiveMove {
        let mv = dex().mv(move_id);
        let mut am = ActiveMove {
            id: move_id,
            fx: mv.fx,
            ty: mv.ty,
            category: mv.category,
            base_power: mv.base_power,
            accuracy: mv.accuracy,
            priority: self.modify_priority(p, move_id, mv.priority),
            target: mv.target,
            flags: mv.flags,
            crit_ratio: mv.crit_ratio,
            will_crit: mv.will_crit,
            spread_hit: false,
            type_changer: false,
            hit: 1,
            sheer_force: false,
            prankster: false,
            infiltrates: self.ab(p) == Ab::Infiltrator,
            mold_breaker: self.ab(p) == Ab::MoldBreaker || mv.ignore_ability,
            helping_hand: self.m(p).vol.helping_hand,
            mslot: mslot as u8,
            struggle,
            confusion_hit: false,
            parental_bond_hit: false,
            bypass_protect_quarter: false,
        };
        am.prankster = self.ab(p) == Ab::Prankster && mv.category == Category::Status;
        // Sheer Force (onModifyMove): moves with secondaries lose them (and
        // their `self` effects) and get the 1.3x boost.
        if self.ab(p) == Ab::SheerForce && !mv.secondaries.is_empty() && !mv.has_sheer_force_boost {
            am.sheer_force = true;
        }
        if mv.fx == MoveFx::ExpandingForce && self.field.terrain == Terrain::Psychic && self.is_grounded(p) {
            am.target = Target::AllAdjacentFoes;
        }
        if mv.fx == MoveFx::Curse && !self.m(p).has_type(Type::Ghost) {
            am.target = Target::User;
        }
        am
    }

    /// Returns true if the move "did something" (not a failure).
    fn use_move(&mut self, p: Pos, move_id: MoveId, mslot: usize, target: u8, struggle: bool) -> bool {
        let mut am = self.make_active_move(p, move_id, mslot, struggle);
        self.resolve_move_type(p, &mut am);
        self.mold_breaker = am.mold_breaker;
        let ok = self.use_move_inner(p, &mut am, target);
        self.mold_breaker = false;
        // Throat Spray, Life Orb etc. run in after-move hooks inside.
        ok
    }

    fn use_move_inner(&mut self, p: Pos, am: &mut ActiveMove, target: u8) -> bool {
        let mv = dex().mv(am.id);
        blog!(self, "|move|{}|{}", self.name(p), mv.name);

        // getMoveTargets (retargeting, redirection), before TryMove.
        let field = matches!(am.target, Target::All | Target::FoeSide | Target::AllySide | Target::AllyTeam);
        let (targets, n) = if field { ([p; MAX_TARGETS], 0) } else { self.move_targets_resolved(p, am, target) };
        if !field && n == 0 {
            blog!(self, "|-fail|{}|[notarget]", self.name(p));
            return false;
        }
        // Pressure: one extra PP per opposing Pressure Pokemon targeted.
        self.pressure_pp(p, am, &targets[..n]);
        // TryMove: Dazzling / Queenly Majesty / Armor Tail (onFoeTryMove).
        if self.blocked_by_dazzling(p, am, if n > 0 { Some(targets[n - 1]) } else { None }) {
            return false;
        }

        // Two-turn moves: charge turn.
        if self.charge_turn(p, am, target) {
            return true;
        }

        // Self-destruct style moves faint the user up front.
        if mv.selfdestruct == 1 {
            if self.ability_on_field(Ab::Damp) {
                return false;
            }
        }

        // Field / side moves.
        if field {
            if !self.move_try(p, am, &[p]) {
                blog!(self, "|-fail|{}", self.name(p));
                self.after_move_fail(p, am);
                return false;
            }
            self.protean(p, am);
            let ok = self.field_move(p, am);
            self.after_move(p, am, &[], ok);
            return ok;
        }

        if mv.selfdestruct == 1 {
            let hp = self.m(p).hp as u32;
            self.damage(p, hp, Some(p), DmgKind::SelfCost);
        }
        if n > 1 {
            am.spread_hit = true;
        }
        // onTry / PrepareHit (move-specific failure conditions).
        if !self.move_try(p, am, &targets[..n]) {
            blog!(self, "|-fail|{}", self.name(p));
            self.after_move_fail(p, am);
            return false;
        }
        self.protean(p, am);
        let ok = self.try_spread_move_hit(p, am, &targets[..n]);
        if !ok {
            self.after_move_fail(p, am);
        }
        ok
    }

    /// Protean / Libero (onPrepareHit): once per switch-in, become the move's type.
    fn protean(&mut self, p: Pos, am: &ActiveMove) {
        if !self.is_live(p) || !matches!(self.ab(p), Ab::Protean | Ab::Libero) || self.m(p).vol.protean {
            return;
        }
        if am.ty == Type::None || am.struggle {
            return;
        }
        if self.types_of(p) == [am.ty, Type::None] {
            return;
        }
        self.reveal_ability(p);
        let m = self.mm(p);
        m.types = [am.ty, Type::None];
        m.vol.type_changed = true;
        m.vol.protean = true;
    }

    /// Pressure (onDeductPP): an extra PP for each opposing Pressure Pokemon
    /// among the move's targets (all foes for field moves and `mustpressure`
    /// moves, none for moves aimed at the foe's side).
    fn pressure_pp(&mut self, p: Pos, am: &ActiveMove, targets: &[Pos]) {
        if am.struggle {
            return;
        }
        let all_foes = am.target == Target::All || am.flags & flag::MUSTPRESSURE != 0;
        let mut extra = 0u8;
        for q in (0..2).map(|i| p.foe(i)) {
            if !self.is_live(q) || self.ab(q) != Ab::Pressure {
                continue;
            }
            let targeted = if all_foes {
                true
            } else if matches!(am.target, Target::FoeSide | Target::AllySide | Target::AllyTeam) {
                false
            } else {
                targets.contains(&q)
            };
            if targeted {
                extra += 1;
            }
        }
        if extra > 0 {
            let s = am.mslot as usize;
            let m = self.mm(p);
            m.pp[s] = m.pp[s].saturating_sub(extra);
        }
    }

    /// Dazzling, Queenly Majesty and Armor Tail (onFoeTryMove): a priority
    /// move whose (last) target is on the holder's side fails. Moves aimed at
    /// the foe's side are exempt, field-wide moves only if not Perish Song,
    /// Flower Shield or Rototiller.
    fn blocked_by_dazzling(&mut self, p: Pos, am: &ActiveMove, last_target: Option<Pos>) -> bool {
        if am.priority <= 0 {
            return false;
        }
        let exception = matches!(dex().mv(am.id).id.as_str(), "perishsong" | "flowershield" | "rototiller");
        match am.target {
            Target::FoeSide => return false,
            Target::All if !exception => return false,
            _ => {}
        }
        for q in (0..2).map(|i| p.foe(i)) {
            if !self.is_live(q) || !matches!(self.tab(q), Ab::Dazzling | Ab::QueenlyMajesty | Ab::ArmorTail) {
                continue;
            }
            let aimed = am.target == Target::All || last_target.map(|t| t.side == q.side).unwrap_or(false);
            if aimed {
                self.reveal_ability(q);
                blog!(self, "|cant|{}|ability: Dazzling|{}", self.name(q), dex().mv(am.id).name);
                return true;
            }
        }
        false
    }

    fn target_pos(&self, p: Pos, code: u8) -> Option<Pos> {
        match code {
            T_FOE0 => Some(p.foe(0)),
            T_FOE1 => Some(p.foe(1)),
            T_ALLY => Some(p.ally()),
            _ => None,
        }
    }

    /// getMoveTargets with retargeting and redirection.
    fn move_targets_resolved(&mut self, p: Pos, am: &ActiveMove, code: u8) -> ([Pos; MAX_TARGETS], usize) {
        let mut out = [p; MAX_TARGETS];
        let mut n = 0;
        match am.target {
            Target::User => {
                out[0] = p;
                n = 1;
            }
            Target::Allies => {
                out[0] = p;
                n = 1;
                if let Some(a) = self.live_ally(p) {
                    out[1] = a;
                    n = 2;
                }
            }
            Target::AdjacentAlly => {
                if let Some(a) = self.live_ally(p) {
                    out[0] = a;
                    n = 1;
                }
            }
            Target::AdjacentAllyOrSelf => {
                if code == T_ALLY && self.is_live(p.ally()) {
                    out[0] = p.ally();
                } else {
                    out[0] = p;
                }
                n = 1;
            }
            Target::AllAdjacent => {
                if let Some(a) = self.live_ally(p) {
                    out[n] = a;
                    n += 1;
                }
                for q in self.live_foes(p).collect::<Vec<_>>() {
                    out[n] = q;
                    n += 1;
                }
            }
            Target::AllAdjacentFoes => {
                for q in self.live_foes(p).collect::<Vec<_>>() {
                    out[n] = q;
                    n += 1;
                }
            }
            Target::RandomNormal | Target::Scripted => {
                // Counter / Mirror Coat / Metal Burst: the last attacker.
                let mut q = p.foe(self.random_target_code(p) as usize);
                if matches!(am.fx, MoveFx::Counter | MoveFx::MirrorCoat | MoveFx::MetalBurst) {
                    let src = Pos::from_code(self.m(p).vol.last_damage_src);
                    if src.side != p.side && self.is_live(src) {
                        q = src;
                    }
                } else if am.target == Target::RandomNormal {
                    q = self.redirect(p, am, q);
                }
                if self.is_live(q) {
                    out[0] = q;
                    n = 1;
                }
            }
            _ => {
                // Single target: a fainted foe is replaced by a random foe; a
                // fainted ally stays the target (the move fails) unless redirected.
                let mut t = self.target_pos(p, code).unwrap_or(p.foe(0));
                if !self.is_live(t) && t.side != p.side {
                    let c = self.random_target_code(p);
                    t = p.foe(c as usize);
                    if !self.is_live(t) {
                        return (out, 0);
                    }
                }
                t = self.redirect(p, am, t);
                if !self.is_live(t) {
                    return (out, 0);
                }
                out[0] = t;
                n = 1;
            }
        }
        (out, n)
    }

    /// RedirectTarget (Showdown's priorityEvent, handlers sorted by priority
    /// then speed): Follow Me / Rage Powder of the user's foes (priority 1,
    /// also for moves aimed at the user's own ally), then Lightning Rod /
    /// Storm Drain of any other Pokemon (priority 0).
    fn redirect(&mut self, p: Pos, am: &ActiveMove, t: Pos) -> Pos {
        if matches!(self.ab(p), Ab::Stalwart) {
            return t;
        }
        let powder_immune = self.m(p).has_type(Type::Grass) || self.ab(p) == Ab::Overcoat;
        let mut best: Option<(i32, Pos)> = None;
        for q in [p.foe(0), p.foe(1)] {
            if !self.is_live(q) {
                continue;
            }
            let fm = self.m(q).vol.follow_me;
            if fm == 1 || (fm == 2 && !powder_immune) {
                let sp = self.action_speed(q);
                let better = match best {
                    None => true,
                    Some((bs, _)) => sp > bs || (sp == bs && self.rng.chance(1, 2)),
                };
                if better {
                    best = Some((sp, q));
                }
            }
        }
        if let Some((_, q)) = best {
            return q;
        }
        let absorber = match am.ty {
            Type::Electric => Ab::LightningRod,
            Type::Water => Ab::StormDrain,
            _ => return t,
        };
        let mut best: Option<(i32, Pos)> = None;
        for c in 0..4 {
            let q = Pos::from_code(c);
            if q == p || !self.is_live(q) || self.tab(q) != absorber {
                continue;
            }
            let sp = self.action_speed(q);
            let better = match best {
                None => true,
                Some((bs, _)) => sp > bs || (sp == bs && self.rng.chance(1, 2)),
            };
            if better {
                best = Some((sp, q));
            }
        }
        if let Some((_, q)) = best {
            if q != t {
                self.reveal_ability(q);
            }
            return q;
        }
        t
    }

    /// Two-turn moves. Returns true if this was the charge turn.
    fn charge_turn(&mut self, p: Pos, am: &ActiveMove, target: u8) -> bool {
        let fx = am.fx;
        let semi = match fx {
            MoveFx::Fly | MoveFx::Bounce => SemiInv::Air,
            MoveFx::Dig => SemiInv::Underground,
            MoveFx::Dive => SemiInv::Underwater,
            MoveFx::PhantomForce | MoveFx::ShadowForce => SemiInv::Shadow,
            _ => SemiInv::None,
        };
        let is_charge = semi != SemiInv::None
            || matches!(fx, MoveFx::SolarBeam | MoveFx::SolarBlade | MoveFx::ElectroShot | MoveFx::MeteorBeam);
        if !is_charge {
            return false;
        }
        if self.m(p).vol.charging > 0 {
            // Second turn: attack.
            let m = self.mm(p);
            m.vol.charging = 0;
            m.vol.semi_inv = SemiInv::None;
            return false;
        }
        let w = self.user_weather(p);
        let skip = match fx {
            MoveFx::SolarBeam | MoveFx::SolarBlade => w == Weather::Sun,
            MoveFx::ElectroShot => w == Weather::Rain,
            _ => false,
        };
        if matches!(fx, MoveFx::ElectroShot | MoveFx::MeteorBeam) {
            self.boost(p, &ups(B_SPA, 1), BoostSrc::SelfInflicted);
        }
        if skip {
            return false;
        }
        // twoturnmove's onStart runs PrepareHit (Protean) on the charge turn.
        self.protean(p, am);
        let m = self.mm(p);
        m.vol.charging = am.mslot + 1;
        m.vol.charge_target = target as i8;
        m.vol.semi_inv = semi;
        blog!(self, "|-prepare|{}|charge", self.name(p));
        true
    }

    /// Move-specific failure conditions (onTry / onPrepareHit).
    fn move_try(&mut self, p: Pos, am: &ActiveMove, targets: &[Pos]) -> bool {
        let t0 = targets[0];
        match am.fx {
            MoveFx::FakeOut | MoveFx::FirstImpression => self.m(p).vol.move_actions <= 1,
            MoveFx::SuckerPunch => {
                let tgt = t0;
                match self.queue.will_move(tgt) {
                    Some(a) => {
                        let mvt = dex().mv(a.move_id);
                        mvt.category != Category::Status && !self.m(tgt).vol.moved_this_turn
                    }
                    None => false,
                }
            }
            MoveFx::UpperHand => match self.queue.will_move(t0) {
                Some(a) => a.priority > 0 && dex().mv(a.move_id).category != Category::Status,
                None => false,
            },
            MoveFx::Poltergeist => self.is_live(t0) && self.m(t0).item != 0,
            MoveFx::AuroraVeil => self.weather() == Weather::Snow,
            MoveFx::SteelRoller => self.field.terrain != Terrain::None,
            MoveFx::DoubleShock => self.m(p).has_type(Type::Electric),
            MoveFx::BurnUp => self.m(p).has_type(Type::Fire),
            MoveFx::Rest => {
                let m = self.m(p);
                m.hp < m.max_hp && m.status != Status::Slp && !matches!(self.ab(p), Ab::Insomnia | Ab::VitalSpirit)
            }
            MoveFx::ClangorousSoul => (self.m(p).hp as u32) * 3 > self.m(p).max_hp as u32,
            MoveFx::BellyDrum => (self.m(p).hp as u32) * 2 > self.m(p).max_hp as u32 && self.m(p).boosts[B_ATK] < 6,
            MoveFx::HelpingHand => {
                // Fails if the ally already moved (and did not just switch in).
                targets.iter().all(|&t| self.is_live(t) && (self.queue.will_move(t).is_some() || self.m(t).vol.newly_switched))
            }
            MoveFx::Protect
            | MoveFx::Detect
            | MoveFx::SpikyShield
            | MoveFx::BanefulBunker
            | MoveFx::KingsShield
            | MoveFx::SilkTrap
            | MoveFx::BurningBulwark
            | MoveFx::Endure => self.stall_check(p),
            MoveFx::Substitute => {
                let m = self.m(p);
                m.vol.substitute == 0 && (m.hp as u32) > m.max_hp as u32 / 4
            }
            MoveFx::ShedTail => {
                let m = self.m(p);
                m.vol.substitute == 0 && (m.hp as u32) * 2 > m.max_hp as u32 && self.sides[p.s()].bench_available() > 0
            }
            MoveFx::HealingWish => self.sides[p.s()].bench_available() > 0,
            MoveFx::Recycle => self.m(p).item == 0 && self.m(p).last_item != 0,
            MoveFx::LastResort => {
                let m = self.m(p);
                m.n_moves > 1
            }
            MoveFx::Counter | MoveFx::MirrorCoat | MoveFx::MetalBurst => {
                // Damaged by a foe this turn (Mirror Coat: special, Counter: physical).
                let v = self.m(p).vol;
                let foe = Pos::from_code(v.last_damage_src).side != p.side;
                v.last_damage_taken > 0
                    && foe
                    && match am.fx {
                        MoveFx::Counter => v.last_damage_physical,
                        MoveFx::MirrorCoat => !v.last_damage_physical,
                        _ => true,
                    }
            }
            MoveFx::NoRetreat => !self.m(p).vol.no_retreat,
            MoveFx::DestinyBond => !(self.m(p).vol.last_move_slot == am.mslot && self.m(p).vol.destiny_bond),
            MoveFx::AfterYou | MoveFx::Quash => self.queue.will_move(t0).is_some(),
            _ => true,
        }
    }

    /// Protect-family stall check: fails if last to act or the stall roll fails.
    fn stall_check(&mut self, p: Pos) -> bool {
        // queue.willAct(): anyone still to act this turn.
        if self.queue.len == 0 {
            self.mm(p).vol.stall_counter = 0;
            self.mm(p).vol.stall_dur = 0;
            return false;
        }
        let m = self.m(p);
        if m.vol.stall_dur > 0 && m.vol.stall_counter > 1 {
            let c = m.vol.stall_counter as u32;
            if !self.rng.chance(1, c) {
                let m = self.mm(p);
                m.vol.stall_counter = 0;
                m.vol.stall_dur = 0;
                return false;
            }
        }
        true
    }

    fn add_stall(&mut self, p: Pos) {
        let m = self.mm(p);
        if m.vol.stall_dur > 0 && m.vol.stall_counter > 0 {
            m.vol.stall_counter = (m.vol.stall_counter * 3).min(729);
        } else {
            m.vol.stall_counter = 3;
        }
        m.vol.stall_dur = 2;
        m.vol.stall_used_this_turn = true;
    }

    // ---- field / side moves -------------------------------------------------------------

    fn field_move(&mut self, p: Pos, am: &ActiveMove) -> bool {
        let mv = dex().mv(am.id);
        let s = p.s();
        let mut ok = false;
        if let Some(sc) = mv.side_condition {
            let side = match am.target {
                Target::FoeSide => 1 - s,
                _ => s,
            };
            ok = self.add_side_condition(side, sc, p, am);
        }
        if let Some(pw) = mv.pseudo_weather {
            match pw {
                PseudoWeather::TrickRoom => {
                    if self.field.trick_room > 0 {
                        self.field.trick_room = 0;
                        blog!(self, "|-fieldend|move: Trick Room");
                    } else {
                        self.field.trick_room = 5;
                        blog!(self, "|-fieldstart|move: Trick Room");
                    }
                    ok = true;
                }
                PseudoWeather::Gravity => {
                    if self.field.gravity == 0 {
                        self.field.gravity = 5;
                        ok = true;
                    }
                }
                _ => {}
            }
        }
        if mv.weather != Weather::None {
            ok = self.set_weather(mv.weather, p);
        }
        if mv.terrain != Terrain::None {
            ok = self.set_terrain(mv.terrain, p);
        }
        match am.fx {
            MoveFx::Haze => {
                for q in self.live_positions().collect::<Vec<_>>() {
                    self.mm(q).boosts = [0; 7];
                }
                ok = true;
            }
            MoveFx::PerishSong => {
                for q in self.live_positions().collect::<Vec<_>>() {
                    if self.m(q).vol.perish == 0 && (self.tab(q) != Ab::Soundproof || q == p) {
                        self.mm(q).vol.perish = 4;
                        ok = true;
                    }
                }
            }
            MoveFx::StealthRock => {
                let c = &mut self.sides[1 - s].conds;
                if !c.stealth_rock {
                    c.stealth_rock = true;
                    ok = true;
                }
            }
            MoveFx::ToxicSpikes => {
                let c = &mut self.sides[1 - s].conds;
                if c.toxic_spikes < 2 {
                    c.toxic_spikes += 1;
                    ok = true;
                }
            }
            _ => {}
        }
        // Wide Guard / Quick Guard are stall moves too.
        if matches!(am.fx, MoveFx::WideGuard | MoveFx::QuickGuard) && ok {
            self.add_stall(p);
        }
        ok
    }

    fn add_side_condition(&mut self, side: usize, sc: SideCond, p: Pos, am: &ActiveMove) -> bool {
        let light_clay = self.it(p) == It::LightClay;
        let screen_turns = if light_clay { 8 } else { 5 };
        let c = &mut self.sides[side].conds;
        match sc {
            SideCond::Tailwind => {
                if c.tailwind > 0 {
                    return false;
                }
                c.tailwind = 4;
                // Wind Rider allies get +1 Atk.
                for slot in 0..2 {
                    let q = Pos::new(side, slot);
                    if self.is_live(q) && self.ab(q) == Ab::WindRider {
                        self.boost(q, &ups(B_ATK, 1), BoostSrc::SelfInflicted);
                    }
                }
                true
            }
            SideCond::Reflect => {
                if c.reflect > 0 {
                    return false;
                }
                c.reflect = screen_turns;
                true
            }
            SideCond::LightScreen => {
                if c.light_screen > 0 {
                    return false;
                }
                c.light_screen = screen_turns;
                true
            }
            SideCond::AuroraVeil => {
                if c.aurora_veil > 0 {
                    return false;
                }
                c.aurora_veil = screen_turns;
                true
            }
            SideCond::Safeguard => {
                if c.safeguard > 0 {
                    return false;
                }
                c.safeguard = 5;
                true
            }
            SideCond::Mist => {
                if c.mist > 0 {
                    return false;
                }
                c.mist = 5;
                true
            }
            SideCond::WideGuard | SideCond::QuickGuard => {
                // onTry: anyone still to act (no stall roll; the stall counter
                // still goes up for the next Protect).
                if self.queue.len == 0 {
                    return false;
                }
                let c = &mut self.sides[side].conds;
                if sc == SideCond::WideGuard {
                    c.wide_guard = true;
                } else {
                    c.quick_guard = true;
                }
                true
            }
            SideCond::StealthRock | SideCond::ToxicSpikes | SideCond::Spikes | SideCond::StickyWeb => {
                let c = &mut self.sides[side].conds;
                match sc {
                    SideCond::StealthRock if !c.stealth_rock => c.stealth_rock = true,
                    SideCond::ToxicSpikes if c.toxic_spikes < 2 => c.toxic_spikes += 1,
                    SideCond::Spikes if c.spikes < 3 => c.spikes += 1,
                    SideCond::StickyWeb if !c.sticky_web => c.sticky_web = true,
                    _ => return false,
                }
                let _ = am;
                true
            }
            _ => false,
        }
    }

    // ---- hit steps -------------------------------------------------------------------------

    fn try_spread_move_hit(&mut self, p: Pos, am: &mut ActiveMove, targets: &[Pos]) -> bool {
        let n = targets.len();
        let mut res = [Hit::Pending; MAX_TARGETS];
        let mut tg = [p; MAX_TARGETS];
        tg[..n].copy_from_slice(targets);

        // 0. Semi-invulnerability.
        for i in 0..n {
            let t = tg[i];
            if t == p {
                continue;
            }
            let si = self.m(t).vol.semi_inv;
            if si != SemiInv::None && self.ab(p) != Ab::NoGuard && self.tab(t) != Ab::NoGuard {
                let hits = match (si, dex().mv(am.id).id.as_str()) {
                    (SemiInv::Air, "hurricane" | "thunder" | "skyuppercut" | "smackdown" | "gust" | "twister" | "thousandarrows") => true,
                    (SemiInv::Underground, "earthquake" | "magnitude" | "fissure") => true,
                    (SemiInv::Underwater, "surf" | "whirlpool") => true,
                    _ => false,
                };
                if !hits {
                    blog!(self, "|-miss|{}|{}", self.name(p), self.name(t));
                    res[i] = Hit::Failed;
                }
            }
        }
        // 1. TryHit: protection, abilities, terrain.
        for i in 0..n {
            if res[i] != Hit::Pending {
                continue;
            }
            res[i] = self.try_hit(p, tg[i], am);
        }
        // 2-3. Type immunity and move-specific immunity.
        for i in 0..n {
            if res[i] != Hit::Pending {
                continue;
            }
            let t = tg[i];
            if !self.immunity_check(p, t, am) {
                blog!(self, "|-immune|{}", self.name(t));
                res[i] = Hit::Failed;
            }
        }
        // 4. Accuracy.
        for i in 0..n {
            if res[i] != Hit::Pending {
                continue;
            }
            let t = tg[i];
            if !self.accuracy_hits(p, t, am) {
                blog!(self, "|-miss|{}|{}", self.name(p), self.name(t));
                res[i] = Hit::Failed;
                if dex().mv(am.id).crash_damage {
                    let mh = self.m(p).max_hp as u32;
                    self.damage(p, mh / 2, Some(p), DmgKind::Indirect);
                }
            }
        }
        // 5. Break protection (Feint) is folded into try_hit.
        let live: Vec<Pos> = (0..n).filter(|&i| res[i] == Hit::Pending).map(|i| tg[i]).collect();
        let any_fail = (0..n).any(|i| res[i] == Hit::Failed);
        if live.is_empty() {
            // Protect-style blocks (NotFail) still count as "did nothing".
            let _ = any_fail;
            return false;
        }
        // Screen-breaking moves act before damage.
        if matches!(am.fx, MoveFx::PsychicFangs | MoveFx::BrickBreak | MoveFx::RagingBull) {
            let side = live[0].s();
            let c = &mut self.sides[side].conds;
            c.reflect = 0;
            c.light_screen = 0;
            c.aurora_veil = 0;
        }
        self.move_hit_loop(p, am, &live);
        true
    }

    /// TryHit for one target (Protect & co., absorbing abilities, Psychic
    /// Terrain, Good as Gold, Magic Bounce, Substitute vs status).
    fn try_hit(&mut self, p: Pos, t: Pos, am: &mut ActiveMove) -> Hit {
        let mv = dex().mv(am.id);
        if t == p {
            return Hit::Pending;
        }
        let foe = t.side != p.side;
        // Wide Guard / Quick Guard (side conditions on the target's side).
        let conds = self.sides[t.s()].conds;
        if conds.wide_guard && am.target.is_spread() {
            if mv.fx != MoveFx::Feint {
                blog!(self, "|-activate|{}|move: Wide Guard", self.name(t));
                return Hit::NotFail;
            }
        }
        if conds.quick_guard && am.priority > 0 && am.flags & flag::PROTECT != 0 && foe {
            if mv.fx != MoveFx::Feint {
                blog!(self, "|-activate|{}|move: Quick Guard", self.name(t));
                return Hit::NotFail;
            }
        }
        // Protect family.
        let prot = self.m(t).vol.protect;
        if prot != ProtectKind::None && am.flags & flag::PROTECT != 0 {
            let blocks = match prot {
                ProtectKind::KingsShield | ProtectKind::SilkTrap | ProtectKind::BurningBulwark => {
                    am.category != Category::Status
                }
                _ => true,
            };
            if blocks {
                if mv.fx == MoveFx::Feint {
                    self.mm(t).vol.protect = ProtectKind::None;
                } else if am.flags & flag::CONTACT != 0 && matches!(self.ab(p), Ab::UnseenFist | Ab::PiercingDrill) {
                    am.bypass_protect_quarter = true;
                } else {
                    blog!(self, "|-activate|{}|move: Protect", self.name(t));
                    if am.flags & flag::CONTACT != 0 && self.makes_contact(p, am) {
                        match prot {
                            ProtectKind::SpikyShield => {
                                let mh = self.m(p).max_hp as u32;
                                self.damage(p, (mh / 8).max(1), Some(t), DmgKind::Indirect);
                            }
                            ProtectKind::BanefulBunker => {
                                self.try_set_status(p, Status::Psn, Some(t));
                            }
                            ProtectKind::BurningBulwark => {
                                self.try_set_status(p, Status::Brn, Some(t));
                            }
                            ProtectKind::KingsShield => {
                                self.boost(p, &ups(B_ATK, -1), BoostSrc::Move(t));
                            }
                            ProtectKind::SilkTrap => {
                                self.boost(p, &ups(B_SPE, -1), BoostSrc::Move(t));
                            }
                            _ => {}
                        }
                    }
                    return Hit::NotFail;
                }
            }
        }
        // Psychic Terrain blocks opposing priority moves on grounded targets.
        if foe && am.priority > 0 && self.field.terrain == Terrain::Psychic && self.is_grounded(t) && self.m(t).vol.semi_inv == SemiInv::None {
            blog!(self, "|-activate|{}|move: Psychic Terrain", self.name(t));
            return Hit::Failed;
        }
        // Magic Bounce: reflect reflectable status moves.
        if am.flags & flag::REFLECTABLE != 0 && self.tab(t) == Ab::MagicBounce && foe {
            self.reveal_ability(t);
            blog!(self, "|-activate|{}|ability: Magic Bounce", self.name(t));
            let mut bounced = *am;
            bounced.flags &= !flag::REFLECTABLE;
            if self.is_live(p) {
                self.status_move_effects(t, p, &bounced);
            }
            return Hit::NotFail;
        }
        // Good as Gold.
        if am.category == Category::Status && self.tab(t) == Ab::GoodAsGold {
            self.reveal_ability(t);
            blog!(self, "|-immune|{}|[from] ability: Good as Gold", self.name(t));
            return Hit::Failed;
        }
        // Telepathy: ally's attacks.
        if !foe && am.category != Category::Status && self.tab(t) == Ab::Telepathy {
            return Hit::Failed;
        }
        // Absorbing abilities.
        let tab = self.tab(t);
        let absorb = match (tab, am.ty) {
            (Ab::VoltAbsorb, Type::Electric) | (Ab::WaterAbsorb, Type::Water) | (Ab::DrySkin, Type::Water) | (Ab::EarthEater, Type::Ground) => {
                let mh = self.m(t).max_hp as u32;
                self.heal(t, mh / 4);
                true
            }
            (Ab::LightningRod, Type::Electric) | (Ab::StormDrain, Type::Water) => {
                self.boost(t, &ups(B_SPA, 1), BoostSrc::SelfInflicted);
                true
            }
            (Ab::Sapsipper, Type::Grass) => {
                self.boost(t, &ups(B_ATK, 1), BoostSrc::SelfInflicted);
                true
            }
            (Ab::MotorDrive, Type::Electric) => {
                self.boost(t, &ups(B_SPE, 1), BoostSrc::SelfInflicted);
                true
            }
            (Ab::WellBakedBody, Type::Fire) => {
                self.boost(t, &ups(B_DEF, 2), BoostSrc::SelfInflicted);
                true
            }
            (Ab::FlashFire, Type::Fire) => {
                self.mm(t).vol.flash_fire = true;
                true
            }
            (Ab::WindRider, _) if am.flags & flag::WIND != 0 => {
                self.boost(t, &ups(B_ATK, 1), BoostSrc::SelfInflicted);
                true
            }
            _ => false,
        };
        if absorb && !(am.category == Category::Status && am.target == Target::User) {
            self.reveal_ability(t);
            blog!(self, "|-immune|{}|[from] ability", self.name(t));
            return Hit::Failed;
        }
        if am.flags & flag::SOUND != 0 && tab == Ab::Soundproof && t != p {
            self.reveal_ability(t);
            return Hit::Failed;
        }
        // Substitute blocks most status moves.
        if self.m(t).vol.substitute > 0
            && am.category == Category::Status
            && am.flags & flag::BYPASSSUB == 0
            && !am.infiltrates
            && am.flags & flag::SOUND == 0
        {
            return Hit::Failed;
        }
        Hit::Pending
    }

    /// Type immunity and TryImmunity (powder, Prankster vs Dark, Ground vs airborne...).
    pub(crate) fn immunity_check(&mut self, p: Pos, t: Pos, am: &ActiveMove) -> bool {
        let mv = dex().mv(am.id);
        if t == p {
            return true;
        }
        let check_type = am.category != Category::Status || !mv.ignore_immunity;
        if check_type && am.ty != Type::None && self.type_immune(p, am.ty, t, am) {
            return false;
        }
        if am.flags & flag::POWDER != 0 {
            if self.m(t).has_type(Type::Grass) || self.tab(t) == Ab::Overcoat {
                return false;
            }
        }
        if am.prankster && t.side != p.side && self.m(t).has_type(Type::Dark) {
            return false;
        }
        match am.fx {
            MoveFx::LowKick | MoveFx::GrassKnot | MoveFx::HeavySlam | MoveFx::HeatCrash => true,
            MoveFx::LeechSeed => !self.m(t).has_type(Type::Grass),
            // Endeavor's onTryImmunity: the user must have less HP.
            MoveFx::Endeavor => self.m(p).hp < self.m(t).hp,
            _ => true,
        }
    }

    pub(crate) fn makes_contact(&self, p: Pos, am: &ActiveMove) -> bool {
        am.flags & flag::CONTACT != 0 && self.is_live(p)
    }

    // ---- the hit loop ---------------------------------------------------------------------

    fn move_hit_loop(&mut self, p: Pos, am: &mut ActiveMove, targets: &[Pos]) {
        let mv = dex().mv(am.id);
        let n = targets.len();
        // Number of hits.
        let mut hits: u8 = 1;
        if mv.multihit.0 > 0 {
            let (lo, hi) = mv.multihit;
            if lo == 2 && hi == 5 {
                const DIST: [u8; 20] = [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5];
                hits = DIST[self.rng.below(20) as usize];
            } else {
                hits = self.rng.range(lo as u32, hi as u32 + 1) as u8;
            }
        }
        if am.fx == MoveFx::BeatUp {
            let user = self.sides[p.s()].active[p.i()];
            hits = self.sides[p.s()].beat_up_allies(user).1.max(1) as u8;
        }
        // Parental Bond (onPrepareHit): not for spread, multi-hit, charge or
        // noparentalbond moves.
        let parental = self.ab(p) == Ab::ParentalBond
            && hits == 1
            && mv.multihit.0 == 0
            && am.fx != MoveFx::BeatUp
            && !am.spread_hit
            && am.category != Category::Status
            && mv.flags & (flag::NOPARENTALBOND | flag::CHARGE | flag::FUTUREMOVE) == 0;
        if parental {
            hits = 2;
        }
        let mut total_damage = [0u32; MAX_TARGETS];
        let mut hp_before = [0u16; MAX_TARGETS];
        for i in 0..n {
            hp_before[i] = self.m(targets[i]).hp;
        }
        let mut damaged = [false; MAX_TARGETS];
        let mut any_hit = false;
        let mut hit_no = 0u8;
        let mut self_dropped = false;
        for h in 1..=hits {
            if !self.is_live(p) {
                break;
            }
            if h > 1 && self.m(p).status == Status::Slp {
                break;
            }
            if targets.iter().all(|&t| !self.is_live(t)) {
                break;
            }
            am.hit = h;
            am.parental_bond_hit = parental && h == 2;
            if mv.multiaccuracy && h > 1 {
                let t = targets[0];
                if !self.accuracy_hits(p, t, am) {
                    break;
                }
            }
            hit_no = h;
            // spreadMoveHit for this hit.
            let dmg = self.spread_move_hit(p, am, targets, &mut self_dropped);
            for i in 0..n {
                if let Some(d) = dmg[i] {
                    total_damage[i] += d;
                    damaged[i] = true;
                    any_hit = true;
                }
            }
            if !any_hit && dmg.iter().take(n).all(|x| x.is_none()) && am.category != Category::Status {
                // No target took damage on this hit (e.g. Substitute broke).
            }
            for &t in targets {
                if self.is_live(t) {
                    self.update_items(t);
                }
            }
            if self.is_live(p) {
                self.update_items(p);
            }
            if !self.is_live(p) && n == 1 {
                break;
            }
        }
        if hits > 1 && hit_no > 0 {
            blog!(self, "|-hitcount|{}|{}", self.name(targets[0]), hit_no);
        }
        // Recoil from total damage.
        let total: u32 = total_damage.iter().sum();
        if total > 0 && self.is_live(p) {
            if mv.recoil.0 > 0 && !matches!(self.ab(p), Ab::RockHead | Ab::MagicGuard) {
                let r = ((total as f64) * mv.recoil.0 as f64 / mv.recoil.1 as f64).round() as u32;
                self.damage(p, r.max(1), Some(p), DmgKind::Indirect);
            }
            if mv.struggle_recoil || am.struggle {
                let mh = self.m(p).max_hp as u32;
                let r = ((mh as f64) / 4.0).round() as u32;
                self.damage(p, r.max(1), Some(p), DmgKind::SelfCost);
            }
        }
        if mv.mind_blown_recoil && self.is_live(p) {
            let mh = self.m(p).max_hp as u32;
            self.damage(p, (mh + 1) / 2, Some(p), DmgKind::SelfCost);
        }
        if am.fx == MoveFx::SteelBeam && self.is_live(p) {
            let mh = self.m(p).max_hp as u32;
            let r = ((mh as f64) / 2.0).round() as u32;
            self.damage(p, r, Some(p), DmgKind::SelfCost);
        }
        // eachEvent('Update') after recoil (Sitrus Berry...).
        for c in 0..4 {
            let q = Pos::from_code(c);
            if self.is_live(q) {
                self.update_items(q);
            }
        }
        // Times attacked (Rage Fist).
        for i in 0..n {
            let t = targets[i];
            if damaged[i] && t != p && self.is_live(t) {
                let m = self.mm(t);
                m.vol.times_attacked = m.vol.times_attacked.saturating_add(hit_no);
            }
        }
        // After-move-secondary: Eject Button, Red Card, Emergency Exit.
        self.after_move_secondary(p, am, targets, &damaged, &hp_before, total);
        self.after_move(p, am, targets, any_hit || am.category == Category::Status);
    }

    /// One hit against all remaining targets. Returns damage dealt per target.
    fn spread_move_hit(&mut self, p: Pos, am: &mut ActiveMove, targets: &[Pos], self_dropped: &mut bool) -> [Option<u32>; MAX_TARGETS] {
        let mv = dex().mv(am.id);
        let n = targets.len();
        let mut out = [None; MAX_TARGETS];
        // Damage for each target first, then apply (spread damage is simultaneous).
        let mut dmgs = [None::<(u32, bool)>; MAX_TARGETS];
        for i in 0..n {
            let t = targets[i];
            if !self.is_live(t) {
                continue;
            }
            if am.category == Category::Status {
                continue;
            }
            // Pollen Puff on an ally heals instead.
            if am.fx == MoveFx::PollenPuff && t.side == p.side {
                continue;
            }
            let fixed = self.fixed_damage(p, t, am);
            let (dmg, crit) = if let Some(f) = fixed {
                (f, false)
            } else {
                let crit = self.roll_crit(p, t, am);
                let roll = self.rng.below(16);
                match self.calc_damage(p, t, am, crit, roll) {
                    Some(d) => (d, crit),
                    None => continue,
                }
            };
            dmgs[i] = Some((dmg, crit));
        }
        // Apply damage (substitute absorbs).
        let mut sub_hit = [false; MAX_TARGETS];
        for i in 0..n {
            let t = targets[i];
            let Some((dmg, crit)) = dmgs[i] else { continue };
            if !self.is_live(t) {
                continue;
            }
            if crit {
                blog!(self, "|-crit|{}", self.name(t));
            }
            let sub = self.m(t).vol.substitute;
            let hits_sub = sub > 0 && t != p && am.flags & flag::SOUND == 0 && !am.infiltrates && am.flags & flag::BYPASSSUB == 0;
            // Resist berry is eaten when it weakens the hit (not through a Substitute).
            if let Some(bt) = super::calc::resist_berry(self.it(t)) {
                let eff = self.effectiveness(am.ty, t, Some(am));
                if bt == am.ty && (eff > 0 || bt == Type::Normal) && !self.unnerved(t) && !hits_sub {
                    blog!(self, "|-enditem|{}|berry|[eat]", self.name(t));
                    self.eat_item(t);
                }
            }
            if hits_sub {
                let absorbed = dmg.min(sub as u32);
                self.mm(t).vol.substitute -= absorbed as u16;
                if self.m(t).vol.substitute == 0 {
                    blog!(self, "|-end|{}|Substitute", self.name(t));
                }
                // HIT_SUBSTITUTE: the target takes no further effects.
                sub_hit[i] = true;
                out[i] = Some(0);
                continue;
            }
            let dealt = self.damage(t, dmg, Some(p), DmgKind::Attack);
            {
                let m = self.mm(t);
                m.vol.last_damage_taken = dealt as u16;
                m.vol.last_damage_physical = am.category == Category::Physical;
                m.vol.last_damage_src = p.code();
            }
            out[i] = Some(dealt);
            if self.m(t).hp == 0 && self.m(t).vol.destiny_bond && t.side != p.side && self.is_live(p) {
                blog!(self, "|-activate|{}|move: Destiny Bond", self.name(t));
                let hp = self.m(p).hp as u32;
                self.damage(p, hp, Some(t), DmgKind::SelfCost);
            }
            // Drain.
            if mv.drain.0 > 0 && dealt > 0 && self.is_live(p) {
                let mut h = ((dealt as f64) * mv.drain.0 as f64 / mv.drain.1 as f64).round() as u32;
                if self.it(p) == It::BigRoot {
                    h = h * 13 / 10;
                }
                self.heal(p, h.max(1));
            }
        }
        // Move effects per target (status moves and damaging moves' main effects).
        for i in 0..n {
            let t = targets[i];
            if am.category == Category::Status || (am.fx == MoveFx::PollenPuff && t.side == p.side) {
                if self.is_live(t) || t == p {
                    self.status_move_effects(p, t, am);
                }
                continue;
            }
            if out[i].is_none() {
                continue;
            }
            self.damaging_move_effects(p, t, am, out[i].unwrap_or(0), sub_hit[i]);
        }
        // Self drops (Close Combat) and other self effects, once.
        if !*self_dropped && self.is_live(p) && out.iter().take(n).any(|x| x.is_some()) && am.category != Category::Status {
            *self_dropped = true;
            if let Some(b) = mv.self_boosts {
                if !am.sheer_force {
                    self.boost(p, &b, BoostSrc::SelfInflicted);
                }
            }
            if let Some(v) = mv.self_volatile {
                self.add_volatile(p, v, Some(p));
            }
            self.self_hit_effects(p, am);
        } else if !*self_dropped && self.is_live(p) && am.category == Category::Status && out.iter().take(n).any(|x| x.is_some()) {
            *self_dropped = true;
            if let Some(v) = mv.self_volatile {
                self.add_volatile(p, v, Some(p));
            }
        }
        // Secondaries (user effects apply even if the target fainted or a
        // Substitute took the hit).
        if !am.sheer_force && am.category != Category::Status {
            for i in 0..n {
                if out[i].is_some() {
                    self.secondaries(p, targets[i], am, sub_hit[i]);
                }
            }
        }
        // Force switch (Dragon Tail, Circle Throw).
        if mv.force_switch && am.category != Category::Status {
            for i in 0..n {
                let t = targets[i];
                if out[i].is_some() && !sub_hit[i] && self.is_live(t) && self.is_live(p) {
                    self.mm(t).vol.force_switch = true;
                }
            }
        }
        // DamagingHit reactions (contact abilities, items); not on a Substitute.
        let user_hp0 = self.m(p).hp;
        for i in 0..n {
            let t = targets[i];
            if let Some(d) = out[i] {
                if t != p && am.category != Category::Status && !sub_hit[i] {
                    self.on_damaging_hit(p, t, am, d);
                }
            }
        }
        // AfterHit (Knock Off...).
        if am.category != Category::Status {
            for i in 0..n {
                if out[i].is_some() {
                    self.after_hit(p, targets[i], am, sub_hit[i]);
                }
            }
        }
        // The user's Emergency Exit after contact damage.
        if self.is_live(p) {
            let max = self.m(p).max_hp as u32;
            if (self.m(p).hp as u32) * 2 <= max && (user_hp0 as u32) * 2 > max {
                self.emergency_exit(p);
            }
        }
        out
    }

    /// damageCallback moves.
    pub(crate) fn fixed_damage(&self, p: Pos, t: Pos, am: &ActiveMove) -> Option<u32> {
        let s = self.m(p);
        let tm = self.m(t);
        match am.fx {
            MoveFx::SuperFang => Some((tm.hp as u32 / 2).max(1)),
            MoveFx::FinalGambit => Some(s.hp as u32),
            MoveFx::Endeavor => Some((tm.hp as u32).saturating_sub(s.hp as u32)),
            MoveFx::Counter | MoveFx::MirrorCoat | MoveFx::MetalBurst => {
                let v = s.vol;
                let ok = match am.fx {
                    MoveFx::Counter => v.last_damage_physical,
                    MoveFx::MirrorCoat => !v.last_damage_physical,
                    _ => true,
                };
                if !ok || v.last_damage_taken == 0 {
                    return Some(0);
                }
                let mult = if am.fx == MoveFx::MetalBurst { 3 } else { 4 };
                Some((v.last_damage_taken as u32 * mult / 2).max(1))
            }
            _ => {
                if dex().mv(am.id).ohko {
                    Some(tm.max_hp as u32)
                } else {
                    None
                }
            }
        }
    }

    /// Main effects of a damaging move on one target (after damage).
    /// runMoveEffects of a damaging move on one target (its onHit, primary
    /// volatile, self switch). A Substitute hit only switches the user out.
    fn damaging_move_effects(&mut self, p: Pos, t: Pos, am: &ActiveMove, dealt: u32, sub_hit: bool) {
        let mv = dex().mv(am.id);
        if !sub_hit {
            match am.fx {
                MoveFx::BugBite => {
                    if self.is_live(t) && dex().item(self.m(t).item).is_berry {
                        self.remove_item(t);
                    }
                }
                MoveFx::ClearSmog => {
                    if self.is_live(t) {
                        self.mm(t).boosts = [0; 7];
                    }
                }
                MoveFx::FinalGambit => {
                    let hp = self.m(p).hp as u32;
                    self.damage(p, hp, Some(p), DmgKind::SelfCost);
                }
                _ => {}
            }
            // Move-level volatile on damaging moves (Infestation, Whirlpool...).
            if let Some(v) = mv.volatile {
                if mv.category != Category::Status && self.is_live(t) {
                    self.add_volatile(t, v, Some(p));
                }
            }
        }
        // Self-switch (U-turn, Flip Turn, Volt Switch).
        if mv.self_switch == 1 && self.is_live(p) && self.sides[p.s()].bench_available() > 0 {
            self.mm(p).vol.switch_flag = true;
        }
        let _ = dealt;
    }

    /// A damaging move's `self` effects that are not stat drops (once).
    fn self_hit_effects(&mut self, p: Pos, am: &ActiveMove) {
        if let MoveFx::DoubleShock | MoveFx::BurnUp = am.fx {
            let ty = if am.fx == MoveFx::DoubleShock { Type::Electric } else { Type::Fire };
            let m = self.mm(p);
            if m.types[0] == ty {
                m.types[0] = if m.types[1] == Type::None { Type::None } else { m.types[1] };
                m.types[1] = Type::None;
            } else if m.types[1] == ty {
                m.types[1] = Type::None;
            }
            m.vol.type_changed = true;
        }
    }

    /// onAfterHit (after the DamagingHit reactions), and onAfterSubDamage for
    /// the moves that have it. Runs even if the user or target fainted.
    fn after_hit(&mut self, p: Pos, t: Pos, am: &ActiveMove, sub_hit: bool) {
        match am.fx {
            MoveFx::KnockOff if !sub_hit => {
                if self.m(t).item != 0 && !dex().item(self.m(t).item).is_mega_stone {
                    blog!(self, "|-enditem|{}|knocked off", self.name(t));
                    self.mm(t).item_knocked = true;
                    self.remove_item(t);
                }
            }
            MoveFx::StoneAxe if self.is_live(p) && !am.sheer_force => {
                self.sides[t.s()].conds.stealth_rock = true;
            }
            MoveFx::CeaselessEdge if self.is_live(p) && !am.sheer_force => {
                let c = &mut self.sides[t.s()].conds;
                if c.spikes < 3 {
                    c.spikes += 1;
                }
            }
            MoveFx::MortalSpin if !am.sheer_force => {
                // (The poison is a secondary.)
                let c = &mut self.sides[p.s()].conds;
                c.stealth_rock = false;
                c.spikes = 0;
                c.toxic_spikes = 0;
                c.sticky_web = false;
                if self.is_live(p) {
                    let m = self.mm(p);
                    m.vol.leech_seed = false;
                    m.vol.partial_trap = 0;
                }
            }
            MoveFx::IceSpinner | MoveFx::SteelRoller if self.is_live(p) => {
                if self.field.terrain != Terrain::None {
                    self.field.terrain = Terrain::None;
                    self.field.terrain_turns = 0;
                }
            }
            _ => {}
        }
    }

    /// Effects of a status move (or Pollen Puff on an ally) on one target.
    pub(crate) fn status_move_effects(&mut self, p: Pos, t: Pos, am: &ActiveMove) -> bool {
        let mv = dex().mv(am.id);
        let mut ok = false;
        let from = if t == p { BoostSrc::SelfInflicted } else { BoostSrc::Move(p) };
        match am.fx {
            MoveFx::Protect | MoveFx::Detect | MoveFx::SpikyShield | MoveFx::BanefulBunker | MoveFx::KingsShield | MoveFx::SilkTrap | MoveFx::BurningBulwark => {
                let k = match am.fx {
                    MoveFx::SpikyShield => ProtectKind::SpikyShield,
                    MoveFx::BanefulBunker => ProtectKind::BanefulBunker,
                    MoveFx::KingsShield => ProtectKind::KingsShield,
                    MoveFx::SilkTrap => ProtectKind::SilkTrap,
                    MoveFx::BurningBulwark => ProtectKind::BurningBulwark,
                    _ => ProtectKind::Protect,
                };
                self.mm(p).vol.protect = k;
                self.add_stall(p);
                return true;
            }
            MoveFx::Endure => {
                self.mm(p).vol.endure = true;
                self.add_stall(p);
                return true;
            }
            MoveFx::HelpingHand => {
                if self.is_live(t) {
                    self.mm(t).vol.helping_hand += 1;
                    // Already-queued actions pick it up at use time.
                    return true;
                }
                return false;
            }
            MoveFx::FollowMe => {
                self.mm(p).vol.follow_me = 1;
                return true;
            }
            MoveFx::RagePowder => {
                self.mm(p).vol.follow_me = 2;
                return true;
            }
            MoveFx::PartingShot => {
                let b = mv.boosts.unwrap_or([-1, 0, -1, 0, 0, 0, 0]);
                let mirror = self.tab(t) == Ab::MirrorArmor;
                let success = self.boost(t, &b, from);
                if (success || mirror) && self.is_live(p) && self.sides[p.s()].bench_available() > 0 {
                    self.mm(p).vol.switch_flag = true;
                }
                return success;
            }
            MoveFx::Encore => {
                if !self.is_live(t) || self.m(t).vol.encore > 0 {
                    return false;
                }
                let last = self.m(t).vol.last_move;
                let Some(slot) = self.m(t).move_slot(last) else { return false };
                if last == 0 || self.m(t).pp[slot] == 0 || dex().mv(last).flags & flag::FAILENCORE != 0 {
                    return false;
                }
                let mut dur = 3;
                {
                    let tv = self.m(t).vol;
                    let _ = tv;
                }
                // Champions: change the queued action to the encored move.
                let willmove = self.queue.will_move(t).copied();
                match willmove {
                    None => dur += 1,
                    Some(a) => {
                        if a.move_id != last && self.it(t) != It::MentalHerb {
                            for i in 0..self.queue.len {
                                if let Some(mut q) = self.queue.list[i] {
                                    if q.pos == t {
                                        if let ActionKind::Move { .. } = q.kind {
                                            q.kind = ActionKind::Move { mslot: slot as u8, target: crate::battle::TARGET_RANDOM };
                                            q.move_id = last;
                                            q.order = 200;
                                            self.queue.list[i] = Some(q);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let m = self.mm(t);
                m.vol.encore = dur;
                m.vol.encore_slot = slot as u8;
                if self.it(t) == It::MentalHerb {
                    self.mm(t).vol.encore = 0;
                    self.consume_item(t);
                }
                return true;
            }
            MoveFx::Disable => {
                if !self.is_live(t) || self.m(t).vol.disable > 0 {
                    return false;
                }
                let last = self.m(t).vol.last_move;
                let Some(slot) = self.m(t).move_slot(last) else { return false };
                let mut d = 4;
                if self.queue.will_move(t).is_none() {
                    d += 1;
                }
                let m = self.mm(t);
                m.vol.disable = d;
                m.vol.disable_slot = slot as u8;
                if self.it(t) == It::MentalHerb {
                    self.mm(t).vol.disable = 0;
                    self.consume_item(t);
                }
                return true;
            }
            MoveFx::Taunt => return self.add_volatile(t, VolKind::Taunt, Some(p)),
            MoveFx::Imprison => return self.add_volatile(p, VolKind::Imprison, Some(p)),
            MoveFx::Yawn => return self.add_volatile(t, VolKind::Yawn, Some(p)),
            MoveFx::LeechSeed => return self.add_volatile(t, VolKind::LeechSeed, Some(p)),
            MoveFx::Substitute => {
                let mh = self.m(p).max_hp as u32;
                let cost = mh / 4;
                self.damage(p, cost, Some(p), DmgKind::SelfCost);
                self.mm(p).vol.substitute = cost as u16;
                return true;
            }
            MoveFx::ShedTail => {
                let mh = self.m(p).max_hp as u32;
                let cost = ((mh as f64) / 2.0).ceil() as u32;
                self.damage(p, cost, Some(p), DmgKind::SelfCost);
                let m = self.mm(p);
                m.vol.substitute = (mh / 4) as u16;
                m.vol.switch_shedtail = true;
                m.vol.switch_flag = true;
                return true;
            }
            MoveFx::BatonPass => {
                if self.sides[p.s()].bench_available() == 0 {
                    return false;
                }
                let m = self.mm(p);
                m.vol.switch_copyvolatile = true;
                m.vol.switch_flag = true;
                return true;
            }
            MoveFx::Trick | MoveFx::Switcheroo => {
                if !self.is_live(t) {
                    return false;
                }
                let (a, b) = (self.m(p).item, self.m(t).item);
                if a == 0 && b == 0 {
                    return false;
                }
                if dex().item(a).is_mega_stone || dex().item(b).is_mega_stone {
                    return false;
                }
                self.mm(p).item = b;
                self.mm(t).item = a;
                self.mm(p).vol.choice_lock = 0;
                self.mm(t).vol.choice_lock = 0;
                self.reveal_item(p);
                self.reveal_item(t);
                return true;
            }
            MoveFx::Roost => {
                let mh = self.m(p).max_hp as u32;
                let h = self.heal(p, (mh + 1) / 2);
                self.mm(p).vol.roost = true;
                return h > 0;
            }
            MoveFx::Soak => {
                if !self.is_live(t) || self.m(t).types == [Type::Water, Type::None] {
                    return false;
                }
                let m = self.mm(t);
                m.types = [Type::Water, Type::None];
                m.vol.type_changed = true;
                return true;
            }
            MoveFx::SkillSwap => {
                if !self.is_live(t) {
                    return false;
                }
                let (a, b) = (self.m(p).ability, self.m(t).ability);
                if dex().ability(a).cantsuppress || dex().ability(b).cantsuppress {
                    return false;
                }
                self.mm(p).ability = b;
                self.mm(t).ability = a;
                self.reveal_ability(p);
                self.reveal_ability(t);
                if t.side != p.side {
                    self.ability_start(p);
                    self.ability_start(t);
                }
                return true;
            }
            MoveFx::Entrainment | MoveFx::WorrySeed => {
                if !self.is_live(t) {
                    return false;
                }
                let new = if am.fx == MoveFx::WorrySeed {
                    dex().ability_id("insomnia").unwrap_or(0)
                } else {
                    self.m(p).ability
                };
                if self.m(t).ability == new || dex().ability(self.m(t).ability).cantsuppress {
                    return false;
                }
                self.mm(t).ability = new;
                self.mm(t).vol.ability_changed = true;
                if am.fx == MoveFx::WorrySeed && self.m(t).status == Status::Slp {
                    self.cure_status(t);
                }
                return true;
            }
            MoveFx::StrengthSap => {
                if !self.is_live(t) || self.m(t).boosts[B_ATK] == -6 {
                    return false;
                }
                let atk = super::calc::boosted(self.m(t).stats[1] as u32, self.m(t).boosts[B_ATK]);
                self.boost(t, &ups(B_ATK, -1), from);
                self.heal(p, atk);
                return true;
            }
            MoveFx::HealPulse | MoveFx::FloralHealing => {
                if !self.is_live(t) {
                    return false;
                }
                let mh = self.m(t).max_hp as u32;
                let amt = if am.fx == MoveFx::FloralHealing && self.field.terrain == Terrain::Grassy {
                    ((mh as f64) * 0.667).round() as u32
                } else {
                    (mh + 1) / 2
                };
                return self.heal(t, amt) > 0;
            }
            MoveFx::PollenPuff => {
                if t.side == p.side && self.is_live(t) {
                    let mh = self.m(t).max_hp as u32;
                    return self.heal(t, (mh + 1) / 2) > 0;
                }
                return false;
            }
            MoveFx::PainSplit => {
                if !self.is_live(t) {
                    return false;
                }
                let avg = (self.m(p).hp as u32 + self.m(t).hp as u32) / 2;
                for q in [p, t] {
                    let m = self.mm(q);
                    m.hp = (avg as u16).min(m.max_hp);
                }
                return true;
            }
            MoveFx::BellyDrum => {
                let mh = self.m(p).max_hp as u32;
                self.damage(p, mh / 2, Some(p), DmgKind::SelfCost);
                self.mm(p).boosts[B_ATK] = 6;
                return true;
            }
            MoveFx::ClangorousSoul => {
                let mh = self.m(p).max_hp as u32;
                self.damage(p, mh / 3, Some(p), DmgKind::SelfCost);
                self.boost(p, &[1, 1, 1, 1, 1, 0, 0], BoostSrc::SelfInflicted);
                return true;
            }
            MoveFx::NoRetreat => {
                self.boost(p, &[1, 1, 1, 1, 1, 0, 0], BoostSrc::SelfInflicted);
                self.mm(p).vol.no_retreat = true;
                return true;
            }
            MoveFx::TidyUp => {
                for s in 0..2 {
                    let c = &mut self.sides[s].conds;
                    c.stealth_rock = false;
                    c.spikes = 0;
                    c.toxic_spikes = 0;
                    c.sticky_web = false;
                }
                self.boost(p, &[1, 0, 0, 0, 1, 0, 0], BoostSrc::SelfInflicted);
                return true;
            }
            MoveFx::Growth => {
                let n = if self.user_weather(p) == Weather::Sun { 2 } else { 1 };
                return self.boost(p, &[n, 0, n, 0, 0, 0, 0], BoostSrc::SelfInflicted);
            }
            MoveFx::Acupressure => {
                let m = self.m(t);
                let opts: Vec<usize> = (0..7).filter(|&i| m.boosts[i] < 6).collect();
                if opts.is_empty() {
                    return false;
                }
                let i = opts[self.rng.below(opts.len() as u32) as usize];
                return self.boost(t, &ups(i, 2), BoostSrc::SelfInflicted);
            }
            MoveFx::PsychUp => {
                if !self.is_live(t) {
                    return false;
                }
                let b = self.m(t).boosts;
                let cs = self.m(t).vol.crit_stage;
                let m = self.mm(p);
                m.boosts = b;
                m.vol.crit_stage = cs;
                return true;
            }
            MoveFx::TopsyTurvy => {
                if !self.is_live(t) || self.m(t).boosts.iter().all(|&x| x == 0) {
                    return false;
                }
                for x in self.mm(t).boosts.iter_mut() {
                    *x = -*x;
                }
                return true;
            }
            MoveFx::Rest => {
                let mh = self.m(p).max_hp as u32;
                {
                    let m = self.mm(p);
                    m.status = Status::Slp;
                    m.status_turns = 3;
                }
                self.heal(p, mh);
                self.check_status_berry(p);
                return true;
            }
            MoveFx::Wish => {
                let slot = p.i();
                if self.sides[p.s()].wish[slot].0 > 0 {
                    return false;
                }
                let amt = (self.m(p).max_hp / 2) as u16;
                self.sides[p.s()].wish[slot] = (2, amt);
                return true;
            }
            MoveFx::HealingWish => {
                self.sides[p.s()].healing_wish[p.i()] = true;
                let hp = self.m(p).hp as u32;
                self.damage(p, hp, Some(p), DmgKind::SelfCost);
                return true;
            }
            MoveFx::Moonlight | MoveFx::MorningSun | MoveFx::Synthesis | MoveFx::ShoreUp => {
                let mh = self.m(p).max_hp as u32;
                let w = self.user_weather(p);
                let amt = if am.fx == MoveFx::ShoreUp {
                    if w == Weather::Sand {
                        ((mh as f64) * 0.667).round() as u32
                    } else {
                        (mh + 1) / 2
                    }
                } else {
                    match w {
                        Weather::Sun => ((mh as f64) * 0.667).round() as u32,
                        Weather::None => (mh + 1) / 2,
                        _ => ((mh as f64) * 0.25).round() as u32,
                    }
                };
                return self.heal(p, amt) > 0;
            }
            MoveFx::DestinyBond => {
                self.mm(p).vol.destiny_bond = true;
                return true;
            }
            MoveFx::Recycle => {
                let it = self.m(p).last_item;
                let m = self.mm(p);
                m.item = it;
                m.last_item = 0;
                m.vol.unburden = false;
                return true;
            }
            MoveFx::Curse => {
                if !self.m(p).has_type(Type::Ghost) {
                    return self.boost(p, &[1, 1, 0, 0, -1, 0, 0], BoostSrc::SelfInflicted);
                }
                let mh = self.m(p).max_hp as u32;
                self.damage(p, mh / 2, Some(p), DmgKind::SelfCost);
                let mut tt = t;
                if t.side == p.side {
                    let c = self.random_target_code(p);
                    tt = p.foe(c as usize);
                }
                if self.is_live(tt) {
                    self.mm(tt).vol.curse = true;
                    return true;
                }
                return false;
            }
            MoveFx::AfterYou | MoveFx::Quash => {
                if let Some(i) = self.queue.list[..self.queue.len]
                    .iter()
                    .position(|a| a.map(|a| a.pos == t && matches!(a.kind, ActionKind::Move { .. })).unwrap_or(false))
                {
                    if am.fx == MoveFx::AfterYou {
                        self.prioritize_action(i);
                    } else if let Some(mut a) = self.queue.list[i] {
                        // Quash: order 201, after every other move.
                        a.order = 201;
                        self.queue.list[i] = Some(a);
                    }
                    return true;
                }
                return false;
            }
            MoveFx::FocusEnergy => return self.add_volatile(p, VolKind::FocusEnergy, Some(p)),
            MoveFx::DragonCheer => {
                if let Some(a) = self.live_ally(p) {
                    return self.add_volatile(a, VolKind::DragonCheer, Some(p));
                }
                return false;
            }
            MoveFx::Defog => {
                let ts = t.s();
                let c = &mut self.sides[ts].conds;
                c.reflect = 0;
                c.light_screen = 0;
                c.aurora_veil = 0;
                c.safeguard = 0;
                c.mist = 0;
                for s in 0..2 {
                    let c = &mut self.sides[s].conds;
                    c.stealth_rock = false;
                    c.spikes = 0;
                    c.toxic_spikes = 0;
                    c.sticky_web = false;
                }
                self.field.terrain = Terrain::None;
                if self.is_live(t) {
                    self.boost(t, &ups(crate::dex::B_EVA, -1), from);
                }
                return true;
            }
            _ => {}
        }
        // Generic status-move data.
        if let Some(b) = mv.boosts {
            if self.is_live(t) {
                ok |= self.boost(t, &b, from);
            }
        }
        if mv.status != Status::None && self.is_live(t) {
            ok |= self.try_set_status(t, mv.status, Some(p));
        }
        if let Some(v) = mv.volatile {
            if self.is_live(t) {
                ok |= self.add_volatile(t, v, Some(p));
            }
        }
        if mv.heal.0 > 0 && self.is_live(t) {
            let mh = self.m(t).max_hp as u32;
            let amt = ((mh as f64) * mv.heal.0 as f64 / mv.heal.1 as f64).round() as u32;
            ok |= self.heal(t, amt) > 0;
        }
        if let Some(b) = mv.self_boosts {
            ok |= self.boost(p, &b, BoostSrc::SelfInflicted);
        }
        if mv.self_switch == 1 && self.is_live(p) && self.sides[p.s()].bench_available() > 0 {
            self.mm(p).vol.switch_flag = true;
            ok = true;
        }
        if mv.force_switch && self.is_live(t) && t != p {
            self.mm(t).vol.force_switch = true;
            ok = true;
        }
        if !ok {
            blog!(self, "|-fail|{}", self.name(p));
        }
        ok
    }

    /// Secondary effects of a damaging move on one target.
    fn secondaries(&mut self, p: Pos, t: Pos, am: &ActiveMove, sub_hit: bool) {
        let mv = dex().mv(am.id);
        if mv.secondaries.is_empty() {
            return;
        }
        let shielded = self.tab(t) == Ab::ShieldDust;
        for sec in &mv.secondaries {
            let roll = self.rng.below(100);
            if roll >= sec.chance as u32 {
                continue;
            }
            // Effects on the user (Flame Charge etc.) ignore Shield Dust.
            if let Some(b) = sec.self_boosts {
                if self.is_live(p) {
                    self.boost(p, &b, BoostSrc::SelfInflicted);
                }
            }
            if shielded || !self.is_live(t) || sub_hit {
                continue;
            }
            if sec.scripted {
                match am.fx {
                    MoveFx::DireClaw => {
                        let st = [Status::Psn, Status::Par, Status::Slp][self.rng.below(3) as usize];
                        self.try_set_status(t, st, Some(p));
                    }
                    MoveFx::TriAttack => {
                        let st = [Status::Par, Status::Brn, Status::Frz][self.rng.below(3) as usize];
                        self.try_set_status(t, st, Some(p));
                    }
                    MoveFx::ThroatChop => {
                        if self.m(t).vol.throat_chop == 0 {
                            self.mm(t).vol.throat_chop = 2;
                        }
                    }
                    MoveFx::BurningJealousy => {
                        if self.m(t).vol.stats_raised_this_turn {
                            self.try_set_status(t, Status::Brn, Some(p));
                        }
                    }
                    MoveFx::AlluringVoice => {
                        if self.m(t).vol.stats_raised_this_turn {
                            self.add_volatile(t, VolKind::Confusion, Some(p));
                        }
                    }
                    _ => {}
                }
                continue;
            }
            if sec.status != Status::None {
                self.try_set_status(t, sec.status, Some(p));
            }
            if let Some(v) = sec.volatile {
                if v == VolKind::HealBlock && mv.id == "psychicnoise" {
                    // Psychic Noise's Heal Block lasts 2 turns.
                    if self.add_volatile(t, v, Some(p)) {
                        self.mm(t).vol.heal_block = 2;
                    }
                } else if v == VolKind::Flinch {
                    // Flinch only matters if the target has not moved yet.
                    if self.queue.will_move(t).is_some() {
                        self.add_volatile(t, v, Some(p));
                    }
                } else {
                    self.add_volatile(t, v, Some(p));
                }
            }
            if let Some(b) = sec.boosts {
                self.boost(t, &b, BoostSrc::Move(p));
            }
        }
    }

    /// onDamagingHit: contact abilities, Rocky Helmet, Stamina, Weakness
    /// Policy, Justified, Cursed Body...
    fn on_damaging_hit(&mut self, p: Pos, t: Pos, am: &ActiveMove, dealt: u32) {
        let _ = dealt;
        let contact = self.makes_contact(p, am);
        // A Fire-type hit thaws a frozen target.
        if self.is_live(t) && self.m(t).status == Status::Frz && am.ty == Type::Fire && am.category != Category::Status {
            self.cure_status(t);
        }
        // The target's onDamagingHit runs even when the hit knocked it out
        // (it faints only after the move).
        let tab = dex().ab(self.m(t).ability);
        if self.is_live(t) {
            match tab {
                Ab::Stamina => {
                    self.reveal_ability(t);
                    self.boost(t, &ups(B_DEF, 1), BoostSrc::SelfInflicted);
                }
                Ab::Justified if am.ty == Type::Dark => {
                    self.boost(t, &ups(B_ATK, 1), BoostSrc::SelfInflicted);
                }
                Ab::ThermalExchange if am.ty == Type::Fire => {
                    self.boost(t, &ups(B_ATK, 1), BoostSrc::SelfInflicted);
                }
                Ab::WeakArmor if am.category == Category::Physical => {
                    self.boost(t, &[0, -1, 0, 0, 2, 0, 0], BoostSrc::SelfInflicted);
                }
                Ab::Electromorphosis => self.mm(t).vol.charge = true,
                Ab::Berserk => {
                    let m = self.m(t);
                    if (m.hp as u32) * 2 <= m.max_hp as u32 && (m.hp as u32 + dealt) * 2 > m.max_hp as u32 {
                        self.boost(t, &ups(B_SPA, 1), BoostSrc::SelfInflicted);
                    }
                }
                _ => {}
            }
        }
        match tab {
            Ab::CursedBody if self.is_live(p) && !am.struggle && self.m(p).vol.disable == 0 && self.rng.chance(3, 10) => {
                if self.m(p).vol.last_move != 0 {
                    // Disable lasts 4 turns once the attacker has moved this turn.
                    if let Some(slot) = self.m(p).move_slot(self.m(p).vol.last_move).filter(|&s| self.m(p).pp[s] > 0) {
                        let m = self.mm(p);
                        m.vol.disable = 4;
                        m.vol.disable_slot = slot as u8;
                    }
                }
            }
            Ab::ToxicDebris if am.category == Category::Physical => {
                let c = &mut self.sides[p.s()].conds;
                if c.toxic_spikes < 2 {
                    c.toxic_spikes += 1;
                }
            }
            Ab::CuteCharm if contact && self.is_live(p) && self.rng.chance(3, 10) => {
                self.add_volatile(p, VolKind::Attract, Some(t));
            }
            _ => {}
        }
        if !self.is_live(p) {
            return;
        }
        // Contact reactions on the attacker.
        if contact {
            // (A target the hit knocked out still reacts: it faints after the move.)
            let tab = dex().ab(self.m(t).ability);
            match tab {
                Ab::RoughSkin | Ab::IronBarbs => {
                    let mh = self.m(p).max_hp as u32;
                    self.reveal_ability(t);
                    self.damage(p, (mh / 8).max(1), Some(t), DmgKind::Indirect);
                }
                Ab::FlameBody if self.rng.chance(3, 10) => {
                    self.try_set_status(p, Status::Brn, Some(t));
                }
                Ab::Static if self.rng.chance(3, 10) => {
                    self.try_set_status(p, Status::Par, Some(t));
                }
                Ab::PoisonPoint if self.rng.chance(3, 10) => {
                    self.try_set_status(p, Status::Psn, Some(t));
                }
                _ => {}
            }
            if self.is_live(p) && self.m(t).item != 0 && dex().it(self.m(t).item) == It::RockyHelmet {
                let mh = self.m(p).max_hp as u32;
                self.reveal_item(t);
                self.damage(p, (mh / 6).max(1), Some(t), DmgKind::Indirect);
            }
            // Poison Touch (attacker's ability).
            if self.is_live(p) && self.ab(p) == Ab::PoisonTouch && self.is_live(t) && self.rng.chance(3, 10) {
                if self.tab(t) != Ab::ShieldDust {
                    self.try_set_status(t, Status::Psn, Some(p));
                }
            }
        }
        // Spicy Spray burns whoever hit it.
        if self.m(t).ability != 0 && dex().ab(self.m(t).ability) == Ab::SpicySpray && self.is_live(p) {
            self.try_set_status(p, Status::Brn, Some(t));
        }
        // Innards Out.
        if !self.is_live(t) && dex().ab(self.m(t).ability) == Ab::InnardsOut && self.is_live(p) {
            self.damage(p, dealt, Some(t), DmgKind::Indirect);
        }
    }

    /// Eject Button, Red Card, Emergency Exit after the move's hits.
    fn after_move_secondary(&mut self, p: Pos, am: &ActiveMove, targets: &[Pos], damaged: &[bool; MAX_TARGETS], hp_before: &[u16; MAX_TARGETS], _total: u32) {
        // Scald & co. thaw a frozen target.
        if dex().mv(am.id).thaws_target {
            for (i, &t) in targets.iter().enumerate() {
                if damaged[i] && self.is_live(t) && self.m(t).status == Status::Frz {
                    self.cure_status(t);
                }
            }
        }
        for (i, &t) in targets.iter().enumerate() {
            if !damaged[i] || t == p || !self.is_live(t) {
                continue;
            }
            match self.it(t) {
                It::EjectButton if self.sides[t.s()].bench_available() > 0 => {
                    blog!(self, "|-enditem|{}|Eject Button", self.name(t));
                    self.consume_item(t);
                    self.mm(t).vol.switch_flag = true;
                    if self.is_live(p) {
                        self.mm(p).vol.switch_flag = false;
                    }
                }
                It::RedCard if self.is_live(p) && self.sides[p.s()].bench_available() > 0 => {
                    self.consume_item(t);
                    self.mm(p).vol.force_switch = true;
                }
                _ => {}
            }
        }
        // Emergency Exit / Wimp Out.
        if !(am.sheer_force) {
            for (i, &t) in targets.iter().enumerate() {
                if !damaged[i] || !self.is_live(t) || t == p {
                    continue;
                }
                let m = self.m(t);
                let max = m.max_hp as u32;
                if (m.hp as u32) * 2 <= max && (hp_before[i] as u32) * 2 > max {
                    self.emergency_exit(t);
                }
            }
        }
    }

    pub(crate) fn emergency_exit(&mut self, t: Pos) {
        if !self.is_live(t) || !matches!(self.ab(t), Ab::EmergencyExit | Ab::WimpOut) {
            return;
        }
        if self.sides[t.s()].bench_available() == 0 {
            return;
        }
        let m = self.m(t);
        if m.vol.switch_flag || m.vol.force_switch {
            return;
        }
        if self.ab(t) == Ab::EmergencyExit {
            // Showdown (base) clears everyone's switch flags; Champions only
            // returns if the holder is already switching.
        }
        self.reveal_ability(t);
        blog!(self, "|-activate|{}|ability: Emergency Exit", self.name(t));
        self.mm(t).vol.switch_flag = true;
    }

    /// After a move that did something: Life Orb, Throat Spray, Moxie...
    fn after_move(&mut self, p: Pos, am: &ActiveMove, targets: &[Pos], ok: bool) {
        if !self.is_live(p) {
            // Destiny Bond etc. not needed when the user fainted.
            return;
        }
        // Life Orb recoil whenever the move connected (AfterMoveSecondarySelf).
        let hit_something = targets.iter().any(|&t| t != p);
        if ok && am.category != Category::Status && self.it(p) == It::LifeOrb && hit_something && !am.sheer_force {
            if self.ab(p) != Ab::MagicGuard {
                let mh = self.m(p).max_hp as u32;
                let hp0 = self.m(p).hp as u32;
                self.reveal_item(p);
                self.damage(p, (mh / 10).max(1), Some(p), DmgKind::Indirect);
                if self.is_live(p) && (self.m(p).hp as u32) * 2 <= mh && hp0 * 2 > mh {
                    self.emergency_exit(p);
                }
            }
        }
        // Shell Bell.
        // Moxie / Eelevate: KO'd a target.
        let kos = targets.iter().filter(|&&t| t != p && !self.is_live(t)).count() as i8;
        if kos > 0 && self.is_live(p) {
            match self.ab(p) {
                Ab::Moxie => {
                    self.boost(p, &ups(B_ATK, kos), BoostSrc::SelfInflicted);
                }
                Ab::Eelevate => {
                    let m = self.m(p);
                    let best = (1..6).max_by_key(|&i| m.stats[i]).unwrap_or(1);
                    self.boost(p, &ups(best - 1, kos), BoostSrc::SelfInflicted);
                }
                _ => {}
            }
        }
        // Charge is used up by an Electric move.
        if am.ty == Type::Electric && am.category != Category::Status && self.is_live(p) {
            self.mm(p).vol.charge = false;
        }
        // Destiny Bond lasts until the user's next move.
        if am.fx != MoveFx::DestinyBond && self.is_live(p) {
            self.mm(p).vol.destiny_bond = false;
        }
        self.mm(p).vol.laser_focus = self.m(p).vol.laser_focus.min(1);
        let _ = B_SPD;
    }

    fn after_move_fail(&mut self, p: Pos, am: &ActiveMove) {
        if self.is_live(p) && am.fx != MoveFx::DestinyBond {
            self.mm(p).vol.destiny_bond = false;
        }
    }
}
