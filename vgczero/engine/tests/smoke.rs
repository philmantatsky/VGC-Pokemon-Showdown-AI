use vgczero_engine::actions::Choice;
use vgczero_engine::battle::{Battle, BattleConfig};
use vgczero_engine::rng::Rng;
use vgczero_engine::{load_dex, load_teams, TeamSpec};

pub fn setup() -> Vec<TeamSpec> {
    let root = env!("CARGO_MANIFEST_DIR");
    load_dex(&format!("{root}/../data/dex_regmc.json.gz")).unwrap();
    load_teams(&format!("{root}/../data/teams_regmc.json.gz")).unwrap()
}

fn random_choice(b: &Battle, side: usize, rng: &mut Rng) -> Choice {
    b.random_choice(side, rng)
}

#[test]
fn one_logged_game() {
    let teams = setup();
    let sup: Vec<_> = teams.iter().filter(|t| t.supported()).collect();
    let cfg = BattleConfig { log: true, ..Default::default() };
    let mut b = Battle::new([sup[0], sup[1]], 7, cfg);
    let mut rng = Rng::new(1);
    for step in 0..40 {
        if b.ended() { break; }
        let c = [random_choice(&b, 0, &mut rng), random_choice(&b, 1, &mut rng)];
        println!("step {step} phase {:?} req {:?} {:?} choice {:?}", b.phase, b.request(0), b.request(1), c);
        b.step(c).unwrap();
    }
    for l in b.log() { println!("{l}"); }
    println!("{}", b.summary());
}

/// Per-slot sampling from `legal_mask` (only avoiding the same switch target
/// twice and two Mega Evolutions), as an RL policy does, never makes `step`
/// fail: joint constraints the masks cannot express are completed by the
/// engine (e.g. which slot gets the only replacement).
#[test]
fn per_slot_mask_sampling_is_always_accepted() {
    use vgczero_engine::actions::{decode, SlotAction, PASS};
    use vgczero_engine::battle::RequestKind;
    let teams = setup();
    let sup: Vec<_> = teams.iter().filter(|t| t.supported()).collect();
    let mut rng = Rng::new(3);
    for g in 0..300u64 {
        let ta = sup[rng.below(sup.len() as u32) as usize];
        let tb = sup[rng.below(sup.len() as u32) as usize];
        let mut b = Battle::new([ta, tb], g, BattleConfig::default());
        while !b.ended() {
            let mut c = [Choice::default(), Choice::default()];
            for s in 0..2 {
                if b.request(s).kind == RequestKind::TeamPreview {
                    c[s] = Choice::preview(rng.below(90) as u8);
                    continue;
                }
                let (mut used, mut mega) = ([false; 6], false);
                let mut slots = [PASS; 2];
                for slot in 0..2 {
                    let mask = b.legal_mask(s, slot);
                    let opts: Vec<u8> = (0..31u8)
                        .filter(|&a| mask & (1 << a) != 0)
                        .filter(|&a| match decode(a) {
                            SlotAction::Switch(t) => !used[t as usize],
                            SlotAction::Move { mega: m, .. } => !(m && mega),
                            SlotAction::Pass => true,
                        })
                        .collect();
                    let a = if opts.is_empty() { PASS } else { opts[rng.below(opts.len() as u32) as usize] };
                    match decode(a) {
                        SlotAction::Switch(t) => used[t as usize] = true,
                        SlotAction::Move { mega: true, .. } => mega = true,
                        _ => {}
                    }
                    slots[slot] = a;
                }
                c[s] = Choice::slots(slots[0], slots[1]);
            }
            b.step(c).unwrap_or_else(|e| panic!("game {g}: {}\n{}", e.0, b.summary()));
        }
    }
}
