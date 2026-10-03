use vgczero_engine::actions::{decode, Choice, SlotAction, PASS};
use vgczero_engine::battle::{Battle, BattleConfig, RequestKind};
use vgczero_engine::rng::Rng;
use vgczero_engine::{load_dex, load_teams, TeamSpec};

pub fn setup() -> Vec<TeamSpec> {
    let root = env!("CARGO_MANIFEST_DIR");
    load_dex(&format!("{root}/../data/dex_regmc.json.gz")).unwrap();
    load_teams(&format!("{root}/../data/teams_regmc.json.gz")).unwrap()
}

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
