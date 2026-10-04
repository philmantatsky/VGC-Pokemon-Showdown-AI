//! The battle engine: requests, choices, the action queue and turn flow.
//!
//! Mechanics are split across submodules, all `impl Battle`:
//!   calc      stats, speed, types, effectiveness, damage formula
//!   moves     running a move (targets, hit steps, effects, scripted moves)
//!   effects   boosts, status, volatiles, healing, items, forme changes
//!   switching switching in/out and entry effects
//!   residual  end-of-turn effects
//!   legal     legal actions per slot

mod calc;
mod effects;
mod legal;
mod moves;
mod residual;
mod switching;

pub use calc::ActiveMove;

use crate::actions::{decode, preview_table, Choice, SlotAction, PASS};
use crate::dex::dex;
use crate::rng::Rng;
use crate::state::{Field, Mon, Pos, Side, SideConds, NO_MON};
use crate::teams::TeamSpec;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Phase {
    TeamPreview,
    /// Both sides choose moves for the turn.
    Move,
    /// Some slots must choose a replacement (after faints at end of turn, or
    /// mid-turn after U-turn / Parting Shot / Emergency Exit / Eject Button).
    Switch { midturn: bool },
    Ended,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RequestKind {
    /// Nothing to choose this step (the choice is ignored).
    Wait,
    TeamPreview,
    Move,
    Switch,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub kind: RequestKind,
    /// Slots that act in this request.
    pub slots: [bool; 2],
}

/// Final result: Some(0) / Some(1) side wins, Some(2) draw.
pub type Winner = Option<u8>;

#[derive(Copy, Clone, Debug)]
pub struct BattleConfig {
    /// Draw when this turn number is reached.
    pub turn_limit: u16,
    /// Allow choosing the ally as the target of damaging single-target moves.
    pub ally_damage_targeting: bool,
    /// Record a human-readable log.
    pub log: bool,
    /// Open team sheets for side 0 / side 1 (each seen by the opponent).
    pub sheets: [bool; 2],
}

impl Default for BattleConfig {
    fn default() -> Self {
        BattleConfig { turn_limit: 100, ally_damage_targeting: false, log: false, sheets: [false, false] }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ActionKind {
    Switch { to: u8 },
    Mega,
    Move { mslot: u8, target: u8 },
}

/// A queued action for one active slot.
#[derive(Copy, Clone, Debug)]
pub struct Action {
    pub kind: ActionKind,
    pub pos: Pos,
    /// Mon index that chose the action (actions are dropped if it left).
    pub mon: u8,
    /// Showdown action order: 103 switch, 104 mega, 200 move.
    pub order: u16,
    pub priority: i8,
    /// Fractional priority (Quick Claw +0.1, Lagging Tail / Quash -0.1) x10.
    pub frac: i8,
    pub speed: i32,
    /// The move this action will use (for Sucker Punch, Encore changes).
    pub move_id: u16,
}

pub const MAX_QUEUE: usize = 16;

/// Move action target for an action Encore rewrote: chosen at random on use.
pub const TARGET_RANDOM: u8 = 3;

#[derive(Copy, Clone, Debug)]
pub struct Queue {
    pub list: [Option<Action>; MAX_QUEUE],
    pub len: usize,
}

impl Default for Queue {
    fn default() -> Self {
        Queue { list: [None; MAX_QUEUE], len: 0 }
    }
}

impl Queue {
    pub fn push(&mut self, a: Action) {
        if self.len < MAX_QUEUE {
            self.list[self.len] = Some(a);
            self.len += 1;
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &Action> {
        self.list[..self.len].iter().filter_map(|x| x.as_ref())
    }
    pub fn pop_front(&mut self) -> Option<Action> {
        if self.len == 0 {
            return None;
        }
        let a = self.list[0];
        for i in 1..self.len {
            self.list[i - 1] = self.list[i];
        }
        self.len -= 1;
        self.list[self.len] = None;
        a
    }
    pub fn remove_where(&mut self, f: impl Fn(&Action) -> bool) {
        let mut j = 0;
        for i in 0..self.len {
            if let Some(a) = self.list[i] {
                if !f(&a) {
                    self.list[j] = Some(a);
                    j += 1;
                }
            }
        }
        for k in j..self.len {
            self.list[k] = None;
        }
        self.len = j;
    }
    /// Move action still to come for this position, if any.
    pub fn will_move(&self, p: Pos) -> Option<&Action> {
        self.iter().find(|a| a.pos == p && matches!(a.kind, ActionKind::Move { .. }))
    }
}

#[derive(Clone, Debug)]
pub struct Battle {
    pub sides: [Side; 2],
    pub field: Field,
    pub turn: u16,
    pub rng: Rng,
    pub phase: Phase,
    pub queue: Queue,
    /// Slots that must choose a switch in the current Switch phase.
    pub switch_slots: [[bool; 2]; 2],
    pub winner: Winner,
    pub cfg: BattleConfig,
    /// Side whose Pokemon fainted last (Gen 5+ double-KO rule).
    pub last_faint_side: Option<u8>,
    /// Speed-ordered field positions for this switch-in event.
    pub log_lines: Vec<String>,
    /// The move currently executing ignores breakable abilities (Mold Breaker).
    pub(crate) mold_breaker: bool,
    /// Gen 9: a move already used this turn by this side (Round, Fusion...).
    pub(crate) round_used: bool,
    /// Number of turns where nothing happened (endless battle guard).
    pub stall_turns: u16,
}

macro_rules! blog {
    ($b:expr, $($arg:tt)*) => {
        if $b.cfg.log { $b.log_lines.push(format!($($arg)*)); }
    };
}
pub(crate) use blog;

#[derive(Debug)]
pub struct ChoiceError(pub String);

impl std::fmt::Display for ChoiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Battle {
    /// A new battle at team preview.
    pub fn new(teams: [&TeamSpec; 2], seed: u64, cfg: BattleConfig) -> Battle {
        let mk_side = |t: &TeamSpec, sheet: bool| {
            let mut mons = [Mon::from_set(&t.mons[0], 0); 6];
            for (i, set) in t.mons.iter().enumerate() {
                mons[i] = Mon::from_set(set, i as u8);
            }
            Side {
                mons,
                active: [NO_MON; 2],
                conds: SideConds::default(),
                mega_used: false,
                total_fainted: 0,
                fainted_this_turn: false,
                fainted_last_turn: false,
                healing_wish: [false; 2],
                wish: [(0, 0); 2],
                sheet_open: sheet,
                order: [0, 1, 2, 3, 4, 5],
            }
        };
        let mut b = Battle {
            sides: [mk_side(teams[0], cfg.sheets[0]), mk_side(teams[1], cfg.sheets[1])],
            field: Field::default(),
            turn: 0,
            rng: Rng::new(seed),
            phase: Phase::TeamPreview,
            queue: Queue::default(),
            switch_slots: [[false; 2]; 2],
            winner: None,
            cfg,
            last_faint_side: None,
            log_lines: vec![],
            mold_breaker: false,
            round_used: false,
            stall_turns: 0,
        };
        for s in 0..2 {
            if b.sides[s].sheet_open {
                for m in b.sides[s].mons.iter_mut() {
                    m.reveal.moves = (1u8 << m.n_moves) - 1;
                    m.reveal.ability = true;
                    m.reveal.item = true;
                }
            }
        }
        b
    }

    #[inline]
    pub fn ended(&self) -> bool {
        self.phase == Phase::Ended
    }

    // ---- accessors -------------------------------------------------------

    /// The Pokemon at a field position (must exist).
    #[inline]
    pub fn m(&self, p: Pos) -> &Mon {
        let i = self.sides[p.s()].active[p.i()];
        &self.sides[p.s()].mons[i as usize]
    }
    #[inline]
    pub fn mm(&mut self, p: Pos) -> &mut Mon {
        let i = self.sides[p.s()].active[p.i()];
        &mut self.sides[p.s()].mons[i as usize]
    }
    /// A living Pokemon is at this position.
    #[inline]
    pub fn is_live(&self, p: Pos) -> bool {
        let i = self.sides[p.s()].active[p.i()];
        i != NO_MON && self.sides[p.s()].mons[i as usize].alive()
    }
    /// Positions of all living active Pokemon (side 0 slot 0, 0/1, 1/0, 1/1).
    pub fn live_positions(&self) -> impl Iterator<Item = Pos> + '_ {
        (0..4).map(|c| Pos::from_code(c)).filter(move |&p| self.is_live(p))
    }
    pub fn live_foes(&self, p: Pos) -> impl Iterator<Item = Pos> + '_ {
        (0..2).map(move |i| p.foe(i)).filter(move |&q| self.is_live(q))
    }
    pub fn live_ally(&self, p: Pos) -> Option<Pos> {
        let a = p.ally();
        if self.is_live(a) {
            Some(a)
        } else {
            None
        }
    }

    // ---- requests ---------------------------------------------------------

    pub fn request(&self, side: usize) -> Request {
        match self.phase {
            Phase::TeamPreview => Request { kind: RequestKind::TeamPreview, slots: [false; 2] },
            Phase::Move => {
                let slots = [self.is_live(Pos::new(side, 0)), self.is_live(Pos::new(side, 1))];
                if slots[0] || slots[1] {
                    Request { kind: RequestKind::Move, slots }
                } else {
                    Request { kind: RequestKind::Wait, slots }
                }
            }
            Phase::Switch { .. } => {
                let slots = self.switch_slots[side];
                if slots[0] || slots[1] {
                    Request { kind: RequestKind::Switch, slots }
                } else {
                    Request { kind: RequestKind::Wait, slots }
                }
            }
            Phase::Ended => Request { kind: RequestKind::Wait, slots: [false; 2] },
        }
    }

    // ---- stepping --------------------------------------------------------

    /// Apply both sides' choices and run until the next decision point.
    pub fn step(&mut self, choices: [Choice; 2]) -> Result<(), ChoiceError> {
        match self.phase {
            Phase::Ended => Err(ChoiceError("battle is over".into())),
            Phase::TeamPreview => {
                for (s, c) in choices.iter().enumerate() {
                    if c.preview as usize >= preview_table().len() {
                        return Err(ChoiceError(format!("side {s}: bad preview {}", c.preview)));
                    }
                }
                for (s, c) in choices.iter().enumerate() {
                    let row = preview_table()[c.preview as usize];
                    for &i in &row {
                        self.sides[s].mons[i as usize].brought = true;
                    }
                    self.sides[s].active = [row[0], row[1]];
                    let mut order = [row[0], row[1], row[2], row[3], 0, 0];
                    let mut k = 4;
                    for i in 0..6u8 {
                        if !row.contains(&i) {
                            order[k] = i;
                            k += 1;
                        }
                    }
                    self.sides[s].order = order;
                    for slot in 0..2 {
                        let mi = row[slot] as usize;
                        self.sides[s].mons[mi].slot = slot as i8;
                        self.sides[s].mons[mi].reveal.seen = true;
                    }
                }
                blog!(self, "|start");
                self.turn = 1;
                let all: Vec<Pos> = (0..4).map(Pos::from_code).collect();
                self.run_switch_in(&all);
                self.after_actions_check();
                if self.phase != Phase::Ended && !matches!(self.phase, Phase::Switch { .. }) {
                    self.begin_turn();
                }
                Ok(())
            }
            Phase::Move => {
                for s in 0..2 {
                    self.validate_move_choice(s, &choices[s])?;
                }
                self.build_queue(&choices);
                self.run_queue();
                Ok(())
            }
            Phase::Switch { midturn } => {
                for s in 0..2 {
                    self.validate_switch_choice(s, &choices[s])?;
                }
                let mut switched = vec![];
                for s in 0..2 {
                    for slot in 0..2 {
                        if !self.switch_slots[s][slot] {
                            continue;
                        }
                        if let SlotAction::Switch(to) = decode(choices[s].slots[slot]) {
                            let p = Pos::new(s, slot);
                            self.switch_out_in(p, to as usize);
                            switched.push(p);
                        }
                    }
                }
                self.switch_slots = [[false; 2]; 2];
                self.phase = Phase::Move;
                self.run_switch_in(&switched);
                self.after_actions_check();
                if matches!(self.phase, Phase::Switch { .. }) || self.ended() {
                    // A switch-in caused another pending switch (e.g. hazards
                    // triggering Emergency Exit); keep the original timing.
                    if let Phase::Switch { .. } = self.phase {
                        self.phase = Phase::Switch { midturn };
                    }
                    return Ok(());
                }
                if midturn {
                    // The queue is re-sorted after the switch (speeds may have
                    // changed: weather, Intimidate...).
                    if self.queue.len > 0 {
                        self.update_queue_speeds();
                        self.sort_queue();
                    }
                    self.run_queue();
                } else {
                    self.check_end_of_turn_replacements();
                }
                Ok(())
            }
        }
    }

    fn validate_switch_choice(&self, s: usize, c: &Choice) -> Result<(), ChoiceError> {
        let mut used = [false; 6];
        let mut switches = 0;
        for slot in 0..2 {
            if !self.switch_slots[s][slot] {
                continue;
            }
            match decode(c.slots[slot]) {
                SlotAction::Switch(to) => {
                    let to = to as usize;
                    if !self.sides[s].can_switch_to(to) || used[to] {
                        return Err(ChoiceError(format!("side {s} slot {slot}: cannot switch to {to}")));
                    }
                    used[to] = true;
                    switches += 1;
                }
                SlotAction::Pass if self.switch_passes_allowed(s) => {}
                _ => return Err(ChoiceError(format!("side {s} slot {slot}: expected a switch"))),
            }
        }
        // As many replacements as possible (Showdown's forcedSwitchesLeft).
        if switches < self.forced_switches(s) {
            return Err(ChoiceError(format!("side {s}: {switches} switches, {} required", self.forced_switches(s))));
        }
        Ok(())
    }

    /// In a switch phase: how many of the side's flagged slots must switch
    /// (the rest pass, when there are fewer replacements than slots).
    pub fn forced_switches(&self, s: usize) -> usize {
        let flagged = self.switch_slots[s].iter().filter(|&&x| x).count();
        flagged.min(self.sides[s].bench_available() as usize)
    }

    /// A flagged slot may pass (Showdown's forcedPassesLeft > 0).
    pub fn switch_passes_allowed(&self, s: usize) -> bool {
        let flagged = self.switch_slots[s].iter().filter(|&&x| x).count();
        flagged > self.sides[s].bench_available() as usize
    }

    fn validate_move_choice(&self, s: usize, c: &Choice) -> Result<(), ChoiceError> {
        let mut used = [false; 6];
        let mut megas = 0;
        for slot in 0..2 {
            let p = Pos::new(s, slot);
            let a = c.slots[slot];
            let mask = self.legal_mask(s, slot);
            if mask & (1u32 << a) == 0 {
                return Err(ChoiceError(format!(
                    "side {s} slot {slot}: illegal action {a} ({:?}); legal mask {mask:#b}",
                    decode(a)
                )));
            }
            match decode(a) {
                SlotAction::Switch(to) => {
                    if used[to as usize] {
                        return Err(ChoiceError(format!("side {s}: both slots switch to {to}")));
                    }
                    used[to as usize] = true;
                }
                SlotAction::Move { mega: true, .. } => {
                    megas += 1;
                    let _ = p;
                }
                _ => {}
            }
        }
        if megas > 1 {
            return Err(ChoiceError(format!("side {s}: two Mega Evolutions")));
        }
        Ok(())
    }

    // ---- turn flow ---------------------------------------------------------

    /// Start-of-turn bookkeeping, then wait for move choices.
    pub(crate) fn begin_turn(&mut self) {
        if self.turn >= self.cfg.turn_limit {
            blog!(self, "|tie|turn limit");
            self.winner = Some(2);
            self.phase = Phase::Ended;
            return;
        }
        blog!(self, "|turn|{}", self.turn);
        for s in 0..2 {
            self.sides[s].fainted_last_turn = self.sides[s].fainted_this_turn;
            self.sides[s].fainted_this_turn = false;
            for m in self.sides[s].mons.iter_mut() {
                let v = &mut m.vol;
                v.last_move_failed = v.this_move_failed;
                v.this_move_failed = false;
                v.moved_this_turn = false;
                v.hurt_this_turn = false;
                v.stats_raised_this_turn = false;
                v.stats_lowered_this_turn = false;
                v.newly_switched = false;
                v.last_damage_taken = 0;
                v.stall_used_this_turn = false;
            }
        }
        self.round_used = false;
        self.phase = Phase::Move;
    }

    fn build_queue(&mut self, choices: &[Choice; 2]) {
        self.queue = Queue::default();
        let d = dex();
        for s in 0..2 {
            for slot in 0..2 {
                let p = Pos::new(s, slot);
                if !self.is_live(p) {
                    continue;
                }
                let mon_idx = self.sides[s].active[slot];
                match decode(choices[s].slots[slot]) {
                    SlotAction::Pass => {}
                    SlotAction::Switch(to) => {
                        self.queue.push(Action {
                            kind: ActionKind::Switch { to },
                            pos: p,
                            mon: mon_idx,
                            order: 103,
                            priority: 0,
                            frac: 0,
                            speed: 0,
                            move_id: 0,
                        });
                    }
                    SlotAction::Move { slot: ms, target, mega } => {
                        if mega {
                            self.queue.push(Action {
                                kind: ActionKind::Mega,
                                pos: p,
                                mon: mon_idx,
                                order: 104,
                                priority: 0,
                                frac: 0,
                                speed: 0,
                                move_id: 0,
                            });
                        }
                        let (mid, ms) = self.resolve_chosen_move(p, ms as usize);
                        let mv = d.mv(mid);
                        let mut a = Action {
                            kind: ActionKind::Move { mslot: ms as u8, target },
                            pos: p,
                            mon: mon_idx,
                            order: 200,
                            priority: mv.priority,
                            frac: 0,
                            speed: 0,
                            move_id: mid,
                        };
                        a.priority = self.modify_priority(p, mid, a.priority);
                        if self.m(p).item != 0 && d.it(self.m(p).item) == crate::kinds::It::QuickClaw && self.rng.chance(1, 5) {
                            a.frac = 1;
                            blog!(self, "|-activate|{}|item: Quick Claw", self.name(p));
                        }
                        self.queue.push(a);
                    }
                }
            }
        }
        self.update_queue_speeds();
        self.sort_queue();
    }

    /// The move a slot will actually use (charging / locked / Struggle).
    pub(crate) fn resolve_chosen_move(&self, p: Pos, ms: usize) -> (u16, usize) {
        let m = self.m(p);
        if m.vol.charging > 0 {
            let s = (m.vol.charging - 1) as usize;
            return (m.moves[s], s);
        }
        if self.must_struggle(p) {
            return (dex().struggle, 0);
        }
        if m.vol.encore > 0 {
            let s = m.vol.encore_slot as usize;
            return (m.moves[s], s);
        }
        (m.moves[ms.min(m.n_moves.saturating_sub(1) as usize)], ms)
    }

    /// Showdown's getActionSpeed for every queued action: the speed, and for
    /// moves the priority (ModifyPriority: Grassy Glide, Prankster...).
    pub(crate) fn update_queue_speeds(&mut self) {
        for i in 0..self.queue.len {
            if let Some(mut a) = self.queue.list[i] {
                if self.sides[a.pos.s()].active[a.pos.i()] == a.mon && self.is_live(a.pos) {
                    a.speed = self.action_speed(a.pos);
                    if let ActionKind::Move { .. } = a.kind {
                        let base = dex().mv(a.move_id).priority;
                        a.priority = self.modify_priority(a.pos, a.move_id, base);
                    }
                }
                self.queue.list[i] = Some(a);
            }
        }
    }

    /// Showdown's queue.prioritizeAction: the action goes next (order 3).
    pub(crate) fn prioritize_action(&mut self, i: usize) {
        if let Some(mut a) = self.queue.list[i] {
            for j in (1..=i).rev() {
                self.queue.list[j] = self.queue.list[j - 1];
            }
            a.order = 3;
            self.queue.list[0] = Some(a);
        }
    }

    /// Showdown's speedSort: order asc, priority desc, speed desc; ties shuffled.
    pub(crate) fn sort_queue(&mut self) {
        // Drop empty entries, then an in-place selection sort with random
        // tie resolution (as Showdown does).
        let mut n = 0;
        for i in 0..self.queue.len {
            if self.queue.list[i].is_some() {
                self.queue.list[n] = self.queue.list[i];
                n += 1;
            }
        }
        for i in n..self.queue.len {
            self.queue.list[i] = None;
        }
        self.queue.len = n;
        #[inline]
        fn key(a: &Option<Action>) -> (i64, i64, i64, i64) {
            let a = a.as_ref().unwrap();
            (a.order as i64, -(a.priority as i64), -(a.frac as i64), -(a.speed as i64))
        }
        let list = &mut self.queue.list;
        for sorted in 0..n {
            let mut best = key(&list[sorted]);
            let mut ties = 1u32;
            for i in sorted + 1..n {
                let k = key(&list[i]);
                if k < best {
                    best = k;
                    ties = 1;
                } else if k == best {
                    ties += 1;
                }
            }
            let rank = if ties > 1 { self.rng.below(ties) } else { 0 };
            let mut r = 0;
            let mut idx = sorted;
            for i in sorted..n {
                if key(&list[i]) == best {
                    if r == rank {
                        idx = i;
                        break;
                    }
                    r += 1;
                }
            }
            let a = list[idx];
            for j in (sorted..idx).rev() {
                list[j + 1] = list[j];
            }
            list[sorted] = a;
        }
    }

    /// Process queued actions until the queue empties (then end the turn), a
    /// mid-turn switch is needed, or the battle ends.
    pub(crate) fn run_queue(&mut self) {
        while let Some(a) = self.queue.pop_front() {
            // Skip actions whose Pokemon left or fainted.
            if self.sides[a.pos.s()].active[a.pos.i()] != a.mon || !self.is_live(a.pos) {
                continue;
            }
            match a.kind {
                ActionKind::Switch { to } => {
                    if self.sides[a.pos.s()].can_switch_to(to as usize) {
                        self.switch_out_in(a.pos, to as usize);
                        self.run_switch_in(&[a.pos]);
                    } else if let Some(alt) = (0..6).find(|&i| self.sides[a.pos.s()].can_switch_to(i)) {
                        // The chosen Pokemon was already sent in by the ally slot.
                        self.switch_out_in(a.pos, alt);
                        self.run_switch_in(&[a.pos]);
                    }
                }
                ActionKind::Mega => self.mega_evolve(a.pos),
                ActionKind::Move { mslot, target } => {
                    self.run_move_action(a.pos, mslot as usize, target, a.move_id);
                }
            }
            if self.ended() {
                return;
            }
            self.after_actions_check();
            if self.ended() || matches!(self.phase, Phase::Switch { .. }) {
                return;
            }
            // Gen 8+: speeds update and the queue re-sorts after every action.
            if self.queue.len > 0 {
                self.update_queue_speeds();
                self.sort_queue();
            }
        }
        self.end_turn();
    }

    /// After any action: faints, win check, pending self-switches.
    pub(crate) fn after_actions_check(&mut self) {
        self.process_faints();
        if self.ended() {
            return;
        }
        self.update_all();
        self.process_faints();
        if self.ended() {
            return;
        }
        // Mid-turn switches (U-turn, Parting Shot, Emergency Exit, Eject Button).
        let mut any = false;
        let mut slots = [[false; 2]; 2];
        for s in 0..2 {
            for slot in 0..2 {
                let p = Pos::new(s, slot);
                if !self.is_live(p) {
                    continue;
                }
                let m = self.m(p);
                if m.vol.switch_flag {
                    if self.sides[s].bench_available() > 0 {
                        slots[s][slot] = true;
                        any = true;
                    } else {
                        self.mm(p).vol.switch_flag = false;
                    }
                }
                if self.m(p).vol.force_switch {
                    self.mm(p).vol.force_switch = false;
                    self.drag_out(p);
                }
            }
        }
        if any {
            // A side with two pending self-switches but one bench Pokemon
            // chooses which slot switches (the other passes).
            for s in 0..2 {
                for slot in 0..2 {
                    if slots[s][slot] {
                        let p = Pos::new(s, slot);
                        self.mm(p).vol.switch_flag = false;
                    }
                }
            }
            self.switch_slots = slots;
            self.phase = Phase::Switch { midturn: true };
        }
    }

    /// Faint any Pokemon at 0 HP and check for a winner.
    pub(crate) fn process_faints(&mut self) {
        for c in 0..4 {
            let p = Pos::from_code(c);
            let i = self.sides[p.s()].active[p.i()];
            if i == NO_MON {
                continue;
            }
            let m = &self.sides[p.s()].mons[i as usize];
            if m.hp == 0 && !m.fainted {
                self.faint(p);
            }
        }
        self.check_win();
    }

    /// Showdown's faintMessages inside an action: faint every active Pokemon
    /// at 0 HP, checking for a winner only when asked.
    pub(crate) fn faint_pending(&mut self, check_win: bool) {
        for c in 0..4 {
            let p = Pos::from_code(c);
            let i = self.sides[p.s()].active[p.i()];
            if i == NO_MON {
                continue;
            }
            let m = &self.sides[p.s()].mons[i as usize];
            if m.hp == 0 && !m.fainted {
                self.faint(p);
            }
        }
        if check_win {
            self.check_win();
        }
    }

    /// `side.foePokemonLeft()`: the foe side still has an unfainted Pokemon.
    #[inline]
    pub(crate) fn foe_pokemon_left(&self, side: usize) -> bool {
        self.sides[1 - side].mons.iter().any(|m| m.brought && !m.fainted)
    }

    pub(crate) fn check_win(&mut self) {
        if self.ended() {
            return;
        }
        let left = [self.sides[0].alive_brought(), self.sides[1].alive_brought()];
        let w = match (left[0], left[1]) {
            (0, 0) => Some(match self.last_faint_side {
                // Gen 5+: the side whose Pokemon fainted last wins.
                Some(s) => s,
                None => 2,
            }),
            (0, _) => Some(1),
            (_, 0) => Some(0),
            _ => None,
        };
        if let Some(w) = w {
            blog!(self, "|win|{}", w);
            self.winner = Some(w);
            self.phase = Phase::Ended;
        }
    }

    fn end_turn(&mut self) {
        self.run_residual();
        self.process_faints();
        if self.ended() {
            return;
        }
        // Emergency Exit can trigger from residual damage.
        self.after_actions_check();
        if self.ended() {
            return;
        }
        if let Phase::Switch { .. } = self.phase {
            // A residual-triggered self-switch is handled like an end-of-turn
            // replacement.
            self.phase = Phase::Switch { midturn: false };
            return;
        }
        self.check_end_of_turn_replacements();
    }

    /// End of turn: request replacements for fainted actives, else next turn.
    pub(crate) fn check_end_of_turn_replacements(&mut self) {
        let mut slots = [[false; 2]; 2];
        let mut any = false;
        for s in 0..2 {
            // Every fainted slot is flagged if any replacement exists; with
            // fewer replacements than slots the player picks which pass.
            if self.sides[s].bench_available() == 0 {
                continue;
            }
            for slot in 0..2 {
                let i = self.sides[s].active[slot];
                let empty = i == NO_MON || self.sides[s].mons[i as usize].fainted;
                if empty {
                    slots[s][slot] = true;
                    any = true;
                }
            }
        }
        if any {
            self.switch_slots = slots;
            self.phase = Phase::Switch { midturn: false };
        } else {
            self.turn += 1;
            self.begin_turn();
        }
    }

    // ---- helpers ------------------------------------------------------------

    pub fn name(&self, p: Pos) -> String {
        let i = self.sides[p.s()].active[p.i()];
        if i == NO_MON {
            return format!("p{}{}: (empty)", p.side + 1, if p.slot == 0 { 'a' } else { 'b' });
        }
        format!(
            "p{}{}: {}",
            p.side + 1,
            if p.slot == 0 { 'a' } else { 'b' },
            dex().sp(self.sides[p.s()].mons[i as usize].species).name
        )
    }

    pub fn mon_name(&self, side: usize, i: usize) -> &'static str {
        &dex().sp(self.sides[side].mons[i].species).name
    }

    /// Log lines recorded so far (cfg.log).
    pub fn log(&self) -> &[String] {
        &self.log_lines
    }

    /// Rough per-turn snapshot used by tests and the CLI.
    pub fn summary(&self) -> String {
        let mut out = format!("turn {} phase {:?}\n", self.turn, self.phase);
        for s in 0..2 {
            for i in 0..6 {
                let m = &self.sides[s].mons[i];
                if !m.brought && self.phase != Phase::TeamPreview {
                    continue;
                }
                out += &format!(
                    "  p{} {}{} {}/{} {:?} {:?}{}\n",
                    s + 1,
                    if m.active() { "*" } else { " " },
                    dex().sp(m.species).name,
                    m.hp,
                    m.max_hp,
                    m.status,
                    m.boosts,
                    if m.fainted { " fnt" } else { "" }
                );
            }
        }
        out
    }

    /// Default choice helper: the first legal action for every acting slot.
    pub fn first_legal_choice(&self, side: usize) -> Choice {
        let r = self.request(side);
        match r.kind {
            RequestKind::TeamPreview => Choice::preview(0),
            RequestKind::Wait => Choice::default(),
            _ => {
                let mut c = Choice::slots(PASS, PASS);
                let mut used = [false; 6];
                for slot in 0..2 {
                    let mask = self.legal_mask(side, slot);
                    for a in 0..31u8 {
                        if mask & (1 << a) == 0 {
                            continue;
                        }
                        if let SlotAction::Switch(to) = decode(a) {
                            if used[to as usize] {
                                continue;
                            }
                            used[to as usize] = true;
                        }
                        if let SlotAction::Move { mega: true, .. } = decode(a) {
                            continue;
                        }
                        c.slots[slot] = a;
                        break;
                    }
                }
                c
            }
        }
    }
}
