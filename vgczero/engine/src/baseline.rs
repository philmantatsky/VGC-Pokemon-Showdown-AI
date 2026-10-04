//! Fixed baseline players for evaluation: uniform random, and a greedy
//! "max expected damage" player (mega-evolves when it can, attacks the target
//! where its best move does the most expected damage, never switches by
//! choice).

use crate::actions::{decode, move_action, Choice, SlotAction, PASS, T_FOE0, T_FOE1};
use crate::battle::{Battle, RequestKind};
use crate::dex::{dex, Category, Target};
use crate::rng::Rng;
use crate::state::Pos;

/// Uniform over legal actions (respecting the joint constraints).
pub fn random_choice(b: &Battle, side: usize, rng: &mut Rng) -> Choice {
    let r = b.request(side);
    match r.kind {
        RequestKind::TeamPreview => Choice::preview(rng.below(90) as u8),
        RequestKind::Wait => Choice::default(),
        _ => {
            let mut c = Choice::slots(PASS, PASS);
            let mut used = [false; 6];
            let mut mega = false;
            for slot in 0..2 {
                let mask = b.legal_mask(side, slot);
                let opts: Vec<u8> = (0..31u8)
                    .filter(|&a| mask & (1 << a) != 0)
                    .filter(|&a| match decode(a) {
                        SlotAction::Switch(t) => !used[t as usize],
                        SlotAction::Move { mega: m, .. } => !(m && mega),
                        _ => true,
                    })
                    .collect();
                let a = if opts.is_empty() { PASS } else { opts[rng.below(opts.len() as u32) as usize] };
                match decode(a) {
                    SlotAction::Switch(t) => used[t as usize] = true,
                    SlotAction::Move { mega: true, .. } => mega = true,
                    _ => {}
                }
                c.slots[slot] = a;
            }
            c
        }
    }
}

/// Expected damage of move slot `ms` from `p` into `t` (fraction of target HP, capped at 1).
fn expected_damage(b: &Battle, p: Pos, ms: usize, t: Pos) -> f32 {
    if !b.is_live(t) {
        return 0.0;
    }
    let mid = b.m(p).moves[ms];
    let mv = dex().mv(mid);
    if mv.category == Category::Status {
        return 0.0;
    }
    let mut sim = b.clone();
    sim.cfg.log = false;
    let mut am = sim.make_active_move(p, mid, ms, false);
    sim.resolve_move_type(p, &mut am);
    if mv.target.is_spread() {
        let n = sim.live_foes(p).count() + if mv.target == Target::AllAdjacent { sim.live_ally(p).is_some() as usize } else { 0 };
        am.spread_hit = n > 1;
    }
    if sim.type_immune(p, am.ty, t, &am) {
        return 0.0;
    }
    let dmg = sim.calc_damage(p, t, &am, false, 7).unwrap_or(0) as f32;
    let hp = sim.m(t).hp.max(1) as f32;
    let acc = if mv.accuracy == 0 { 1.0 } else { mv.accuracy as f32 / 100.0 };
    (dmg / hp).min(1.0) * acc
}

pub fn greedy_choice(b: &Battle, side: usize, rng: &mut Rng) -> Choice {
    let r = b.request(side);
    match r.kind {
        RequestKind::Move => {}
        _ => return random_choice(b, side, rng),
    }
    let mut c = Choice::slots(PASS, PASS);
    let mut mega_taken = false;
    for slot in 0..2 {
        let p = Pos::new(side, slot);
        let mask = b.legal_mask(side, slot);
        if !b.is_live(p) {
            continue;
        }
        let m = b.m(p);
        let mut best = (-1.0f32, PASS);
        for ms in 0..m.n_moves as usize {
            let mv = dex().mv(m.moves[ms]);
            let spread = mv.target.is_spread();
            for t in [T_FOE0, T_FOE1] {
                let a = move_action(ms, t, false);
                if mask & (1 << a) == 0 {
                    continue;
                }
                let score = if spread {
                    (0..2).map(|i| expected_damage(b, p, ms, p.foe(i))).sum::<f32>()
                } else {
                    expected_damage(b, p, ms, p.foe(t as usize))
                };
                let score = score + rng.f32() * 1e-3;
                if score > best.0 {
                    best = (score, a);
                }
            }
        }
        let mut a = best.1;
        if best.0 <= 0.0 {
            // Nothing damaging: any legal non-switch action, else anything.
            let opts: Vec<u8> = (7..31u8).filter(|&x| mask & (1 << x) != 0 && (x - 7) % 2 == 0).collect();
            a = if opts.is_empty() { (0..31u8).find(|&x| mask & (1 << x) != 0).unwrap_or(PASS) } else { opts[rng.below(opts.len() as u32) as usize] };
        }
        if let SlotAction::Move { slot: ms, target, mega: false } = decode(a) {
            let am = move_action(ms as usize, target, true);
            if !mega_taken && mask & (1 << am) != 0 {
                a = am;
                mega_taken = true;
            }
        }
        c.slots[slot] = a;
    }
    c
}
