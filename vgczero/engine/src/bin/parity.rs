//! Engine side of the Showdown parity tests (driven by showdown/parity/*.js).
//!
//!   parity supported            -> {"teams": [pool indices the engine supports], ...}
//!   parity damage   < jsonl     -> per line {id, normal: [16], crit: [16]} (null = no damage)
//!   parity turn N   < jsonl     -> per line {id, counts: {feature: {value: n}}} over N seeds
//!   parity legal    < jsonl     -> per line {id, sides: [summary, summary]}
//!
//! Input states are Showdown battles exported by lib.js `exportState`.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use serde_json::{json, Value};
use vgczero_engine::actions::{preview_index, Choice};
use vgczero_engine::battle::Battle;
use vgczero_engine::rng::Rng;
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

fn choice_of(b: &Battle, c: &Value) -> Result<Choice, String> {
    if let Some(po) = c.get("preview_order").and_then(|x| x.as_array()) {
        let o: Vec<u8> = po.iter().map(|x| x.as_u64().unwrap_or(0) as u8).collect();
        if o.len() != 4 {
            return Err("preview_order needs 4".into());
        }
        let p = preview_index([o[0], o[1]], [o[2], o[3]]).ok_or("bad preview order")?;
        return Ok(Choice::preview(p));
    }
    let _ = b;
    let s = c.get("slots").and_then(|x| x.as_array()).ok_or("choice without slots")?;
    let a = s.first().and_then(|x| x.as_u64()).unwrap_or(0) as u8;
    let bb = s.get(1).and_then(|x| x.as_u64()).unwrap_or(0) as u8;
    Ok(Choice::slots(a, bb))
}

/// For a team preview state, the two leads must go to the slots Showdown put
/// them in: Showdown keeps the chosen order (first chosen = slot a).
fn apply_preview(b: &mut Battle, choices: &[Choice; 2], orders: &[Option<Vec<u8>>; 2]) -> Result<(), String> {
    b.step(*choices).map_err(|e| e.0)?;
    let _ = orders;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = env!("CARGO_MANIFEST_DIR");
    load_dex(&format!("{root}/../data/dex_regmc.json.gz")).unwrap();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
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
        "damage" => {
            for line in read_lines() {
                let id = line.get("id").cloned().unwrap_or(Value::Null);
                let res = (|| -> Result<Value, String> {
                    let state = line.get("state").ok_or("no state")?;
                    let mut b = Battle::from_state_json(state, 1)?;
                    let src = Pos::from_code(line["src"].as_u64().ok_or("no src")? as u8);
                    let tgt = Pos::from_code(line["tgt"].as_u64().ok_or("no tgt")? as u8);
                    let mv = line["move"].as_str().ok_or("no move")?;
                    let spread = line["spread"].as_bool().unwrap_or(false);
                    let mut normal = vec![];
                    let mut crit = vec![];
                    for roll in 0..16 {
                        let mut c = b.clone();
                        normal.push(c.damage_probe(src, tgt, mv, spread, false, roll)?);
                        let mut c = b.clone();
                        crit.push(c.damage_probe(src, tgt, mv, spread, true, roll)?);
                    }
                    let _ = &mut b;
                    Ok(json!({"id": id, "normal": normal, "crit": crit}))
                })();
                match res {
                    Ok(v) => writeln!(out, "{v}").unwrap(),
                    Err(e) => writeln!(out, "{}", json!({"id": id, "err": e})).unwrap(),
                }
            }
        }
        "turn" => {
            let runs: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
            for line in read_lines() {
                let id = line.get("id").cloned().unwrap_or(Value::Null);
                let res = (|| -> Result<Value, String> {
                    let state = line.get("state").ok_or("no state")?;
                    let base = Battle::from_state_json(state, 1)?;
                    let cs = line.get("choices").and_then(|x| x.as_array()).ok_or("no choices")?;
                    let choices = [choice_of(&base, &cs[0])?, choice_of(&base, &cs[1])?];
                    let seed0 = line.get("seed").and_then(|x| x.as_u64()).unwrap_or(1);
                    let mut counts: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
                    let mut errors: BTreeMap<String, u32> = BTreeMap::new();
                    for k in 0..runs {
                        let mut b = base.clone();
                        b.rng = Rng::new(seed0.wrapping_mul(1_000_003).wrapping_add(k));
                        let log0 = b.log_lines.len();
                        let r = if matches!(b.phase, vgczero_engine::battle::Phase::TeamPreview) {
                            apply_preview(&mut b, &choices, &[None, None])
                        } else {
                            b.step(choices).map_err(|e| e.0)
                        };
                        if let Err(e) = r {
                            *errors.entry(e).or_default() += 1;
                            continue;
                        }
                        for (key, v) in b.outcome_features(log0) {
                            let vs = match v {
                                Value::String(s) => s,
                                other => other.to_string(),
                            };
                            *counts.entry(key).or_default().entry(vs).or_default() += 1;
                        }
                    }
                    Ok(json!({"id": id, "counts": counts, "errors": errors}))
                })();
                match res {
                    Ok(v) => writeln!(out, "{v}").unwrap(),
                    Err(e) => writeln!(out, "{}", json!({"id": id, "err": e})).unwrap(),
                }
            }
        }
        "turnlog" => {
            // One run of each input line, printing the engine's log for the step.
            let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
            for line in read_lines() {
                let id = line.get("id").cloned().unwrap_or(Value::Null);
                let res = (|| -> Result<Value, String> {
                    let state = line.get("state").ok_or("no state")?;
                    let mut b = Battle::from_state_json(state, seed)?;
                    let cs = line.get("choices").and_then(|x| x.as_array()).ok_or("no choices")?;
                    let choices = [choice_of(&b, &cs[0])?, choice_of(&b, &cs[1])?];
                    let log0 = b.log_lines.len();
                    let before = b.summary();
                    b.step(choices).map_err(|e| e.0)?;
                    let log: Vec<String> = b.log_lines[log0..].to_vec();
                    Ok(json!({"id": id, "before": before, "log": log, "after": b.summary(), "features": b.outcome_features(log0)}))
                })();
                match res {
                    Ok(v) => writeln!(out, "{v}").unwrap(),
                    Err(e) => writeln!(out, "{}", json!({"id": id, "err": e})).unwrap(),
                }
            }
        }
        "legal" => {
            for line in read_lines() {
                let id = line.get("id").cloned().unwrap_or(Value::Null);
                let res = (|| -> Result<Value, String> {
                    let state = line.get("state").ok_or("no state")?;
                    let b = Battle::from_state_json(state, 1)?;
                    Ok(json!({"id": id, "sides": [b.legal_summary(0), b.legal_summary(1)]}))
                })();
                match res {
                    Ok(v) => writeln!(out, "{v}").unwrap(),
                    Err(e) => writeln!(out, "{}", json!({"id": id, "err": e})).unwrap(),
                }
            }
        }
        _ => {
            eprintln!("usage: parity supported | damage | turn N | legal  (JSON lines on stdin)");
            std::process::exit(2);
        }
    }
}
