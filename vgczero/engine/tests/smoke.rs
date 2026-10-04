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
