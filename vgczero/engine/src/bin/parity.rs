//! Engine side of the Showdown parity tests (driven by showdown/parity/*.js).
//!
//!   parity supported           -> {"teams": [pool indices the engine supports], abilities, items, moves}
//!   parity damage   < jsonl    -> per line {id, normal: [16], crit: [16]} (null = no damage)
//!   parity turn N   < jsonl    -> per line {id, counts: {feature: {value: n}}, errors} over N seeds
//!   parity turnlog S < jsonl   -> per line one run (seed S) with the engine log, for debugging
//!   parity legal    < jsonl    -> per line {id, sides: [summary, summary]}
//!
//! Input states are Showdown battles exported by lib.js `exportState`.

use std::io::{BufRead, Write};

use serde_json::{json, Value};
use vgczero_engine::battle::Battle;
use vgczero_engine::snapshot::{choice_from_json, outcome_counts};
use vgczero_engine::state::Pos;
use vgczero_engine::{dex, load_dex, load_teams};

fn read_lines() -> Vec<Value> {
    let stdin = std::io::stdin();
    let mut out = vec![];
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).expect("bad json line"));
    }
    out
}

/// Run `f` for every input line, writing its result or `{id, err}`.
fn each_line(out: &mut impl Write, f: impl Fn(&Value) -> Result<Value, String>) {
    for line in read_lines() {
        let id = line.get("id").cloned().unwrap_or(Value::Null);
        match f(&line) {
            Ok(mut v) => {
                v["id"] = id;
                writeln!(out, "{v}").unwrap();
            }
            Err(e) => writeln!(out, "{}", json!({"id": id, "err": e})).unwrap(),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = env!("CARGO_MANIFEST_DIR");
    load_dex(&format!("{root}/../data/dex_regmc.json.gz")).unwrap();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let num_arg = |dflt: u64| args.get(2).and_then(|s| s.parse().ok()).unwrap_or(dflt);
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    match cmd {
        "supported" => {
            let teams = load_teams(&format!("{root}/../data/teams_regmc.json.gz")).unwrap();
            let idx: Vec<usize> = (0..teams.len()).filter(|&i| teams[i].supported()).collect();
            let d = dex();
            let mut abilities = std::collections::BTreeSet::new();
            let mut items = std::collections::BTreeSet::new();
            let mut moves = std::collections::BTreeSet::new();
            for &i in &idx {
                for m in &teams[i].mons {
                    abilities.insert(d.ability(m.ability).id.clone());
                    if let Some(mg) = &m.mega {
                        abilities.insert(d.ability(mg.ability).id.clone());
                    }
                    items.insert(d.item(m.item).id.clone());
                    for &mv in &m.moves[..m.n_moves as usize] {
                        moves.insert(d.mv(mv).id.clone());
                    }
                }
            }
            writeln!(out, "{}", json!({"teams": idx, "abilities": abilities, "items": items, "moves": moves})).unwrap();
        }
        "damage" => each_line(&mut out, |line| {
            let b = Battle::from_state_json(line.get("state").ok_or("no state")?, 1)?;
            let src = Pos::from_code(line["src"].as_u64().ok_or("no src")? as u8);
            let tgt = Pos::from_code(line["tgt"].as_u64().ok_or("no tgt")? as u8);
            let mv = line["move"].as_str().ok_or("no move")?;
            let spread = line["spread"].as_bool().unwrap_or(false);
            let (mut normal, mut crit) = (vec![], vec![]);
            for roll in 0..16 {
                normal.push(b.clone().damage_probe(src, tgt, mv, spread, false, roll)?);
                crit.push(b.clone().damage_probe(src, tgt, mv, spread, true, roll)?);
            }
            Ok(json!({"normal": normal, "crit": crit}))
        }),
        "turn" => {
            let runs = num_arg(100);
            each_line(&mut out, |line| {
                let seed = line.get("seed").and_then(|x| x.as_u64()).unwrap_or(1);
                let (counts, errors) = outcome_counts(line.get("state").ok_or("no state")?, &line["choices"], runs, seed)?;
                let errors: Vec<&String> = errors.values().flat_map(|m| m.keys()).collect();
                Ok(json!({"counts": counts, "errors": errors.iter().map(|e| (e.to_string(), 1)).collect::<std::collections::BTreeMap<_, _>>()}))
            })
        }
        "turnlog" => {
            let seed = num_arg(1);
            each_line(&mut out, |line| {
                let mut b = Battle::from_state_json(line.get("state").ok_or("no state")?, seed)?;
                let cs = line.get("choices").and_then(|x| x.as_array()).ok_or("no choices")?;
                let choices = [choice_from_json(&cs[0])?, choice_from_json(&cs[1])?];
                let log0 = b.log_lines.len();
                let before = b.summary();
                b.step(choices).map_err(|e| e.0)?;
                let log: Vec<String> = b.log_lines[log0..].to_vec();
                Ok(json!({"before": before, "log": log, "after": b.summary(), "features": b.outcome_features(log0)}))
            })
        }
        "legal" => each_line(&mut out, |line| {
            let b = Battle::from_state_json(line.get("state").ok_or("no state")?, 1)?;
            Ok(json!({"sides": [b.legal_summary(0), b.legal_summary(1)]}))
        }),
        _ => {
            eprintln!("usage: parity supported | damage | turn N | turnlog SEED | legal  (JSON lines on stdin)");
            std::process::exit(2);
        }
    }
}
