//! Random self-play benchmark and team coverage report.
//!
//!   cargo run --release --bin bench -- [games] [--all-teams]

use std::time::Instant;

use vgczero_engine::actions::{decode, Choice, SlotAction, PASS};
use vgczero_engine::battle::{Battle, BattleConfig, RequestKind};
use vgczero_engine::rng::Rng;
use vgczero_engine::{load_dex, load_teams};

fn random_choice(b: &Battle, side: usize, rng: &mut Rng) -> Choice {
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
                let mut opts: Vec<u8> = (0..31u8)
                    .filter(|&a| mask & (1 << a) != 0)
                    .filter(|&a| match decode(a) {
                        SlotAction::Switch(t) => !used[t as usize],
                        SlotAction::Move { mega: m, .. } => !(m && mega),
                        _ => true,
                    })
                    .collect();
                if opts.is_empty() {
                    opts.push(PASS);
                }
                let a = opts[rng.below(opts.len() as u32) as usize];
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let all = args.iter().any(|a| a == "--all-teams");
    let root = env!("CARGO_MANIFEST_DIR");
    load_dex(&format!("{root}/../data/dex_regmc.json.gz")).unwrap();
    let teams = load_teams(&format!("{root}/../data/teams_regmc.json.gz")).unwrap();
    let supported: Vec<_> = teams.iter().filter(|t| t.supported()).collect();
    println!("teams: {} total, {} fully supported ({:.1}%)", teams.len(), supported.len(), 100.0 * supported.len() as f64 / teams.len() as f64);
    let mut reasons = std::collections::HashMap::<String, usize>::new();
    for t in &teams {
        for r in &t.unsupported {
            *reasons.entry(r.clone()).or_default() += 1;
        }
    }
    let mut rs: Vec<_> = reasons.into_iter().collect();
    rs.sort_by(|a, b| b.1.cmp(&a.1));
    println!("top blockers (teams affected):");
    for (r, c) in rs.iter().take(25) {
        println!("  {c:5}  {r}");
    }
    let pool: Vec<_> = if all { teams.iter().collect() } else { supported };
    if pool.is_empty() {
        return;
    }
    let mut rng = Rng::new(12345);
    let t0 = Instant::now();
    let (mut turns, mut decisions, mut wins) = (0u64, 0u64, [0u64; 3]);
    let mut errors = 0;
    for g in 0..games {
        let a = pool[rng.below(pool.len() as u32) as usize];
        let bteam = pool[rng.below(pool.len() as u32) as usize];
        let mut b = Battle::new([a, bteam], g as u64, BattleConfig::default());
        let mut steps = 0;
        while !b.ended() && steps < 1000 {
            let c = [random_choice(&b, 0, &mut rng), random_choice(&b, 1, &mut rng)];
            if let Err(e) = b.step(c) {
                errors += 1;
                if errors < 5 {
                    eprintln!("game {g}: {e}\n{}", b.summary());
                }
                break;
            }
            steps += 1;
            decisions += 1;
        }
        turns += b.turn as u64;
        if let Some(w) = b.winner {
            wins[w as usize] += 1;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{games} games in {dt:.2}s: {:.0} games/s, {:.0} decision steps/s, {:.1} turns/game, wins {:?}, errors {errors}",
        games as f64 / dt,
        decisions as f64 / dt,
        turns as f64 / games as f64,
        wins
    );
}
