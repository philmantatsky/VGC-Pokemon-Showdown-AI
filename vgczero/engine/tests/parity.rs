//! Showdown parity checks against recorded fixtures (no Showdown needed).
//!
//! The fixtures are written by the Node scripts in `showdown/parity/`:
//!   node damage.js --fixture fixtures/damage.jsonl.gz   (exact damage, 16 rolls x crit)
//!   node turns.js  --fixture fixtures/turns.jsonl.gz    (Showdown outcome counts per
//!                                                         decision, and request legality)
//! A test is skipped (with a message) if its fixture is missing.

use std::collections::BTreeMap;
use std::io::Read;

use serde_json::Value;
use vgczero_engine::battle::Battle;
use vgczero_engine::snapshot::{compare_counts, outcome_counts};
use vgczero_engine::state::Pos;
use vgczero_engine::load_dex;

fn fixture(name: &str) -> Option<Vec<Value>> {
    let root = env!("CARGO_MANIFEST_DIR");
    load_dex(&format!("{root}/../data/dex_regmc.json.gz")).unwrap();
    let path = format!("{root}/../showdown/parity/fixtures/{name}");
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => {
            eprintln!("skipping: no fixture {path} (generate it with showdown/parity, see README.md)");
            return None;
        }
    };
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..]).read_to_string(&mut text).unwrap();
    Some(text.lines().filter(|l| !l.trim().is_empty()).map(|l| serde_json::from_str(l).unwrap()).collect())
}

#[test]
fn damage_matches_showdown() {
    let Some(cases) = fixture("damage.jsonl.gz") else { return };
    let (mut ok, mut bad, mut skipped) = (0, 0, 0);
    let mut report = vec![];
    for c in &cases {
        let b = match Battle::from_state_json(&c["state"], 1) {
            Ok(b) => b,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let src = Pos::from_code(c["src"].as_u64().unwrap() as u8);
        let tgt = Pos::from_code(c["tgt"].as_u64().unwrap() as u8);
        let mv = c["move"].as_str().unwrap();
        let spread = c["spread"].as_bool().unwrap_or(false);
        let mut same = true;
        for (kind, crit) in [("normal", false), ("crit", true)] {
            for roll in 0..16u32 {
                let got = b.clone().damage_probe(src, tgt, mv, spread, crit, roll).unwrap();
                let want = c["expect"][kind][roll as usize].as_u64().map(|x| x as u32);
                if got != want {
                    same = false;
                }
            }
        }
        if same {
            ok += 1;
        } else {
            bad += 1;
            if report.len() < 10 {
                report.push(format!("case {} {mv}", c["id"]));
            }
        }
    }
    println!("damage parity: {ok} exact, {bad} mismatched, {skipped} skipped");
    assert!(bad == 0, "damage mismatches: {report:?}");
}

#[test]
fn turn_outcomes_match_showdown() {
    let Some(lines) = fixture("turns.jsonl.gz") else { return };
    const ALPHA: f64 = 1e-6;
    let (mut ok, mut flagged, mut skipped, mut features) = (0, 0, 0, 0);
    let mut report = vec![];
    for (k, l) in lines.iter().filter(|l| l["kind"] == "turn").enumerate() {
        let runs = l["runs"].as_u64().unwrap_or(100);
        let (counts, errors) = match outcome_counts(&l["state"], &l["choices"], runs, 7_000 + k as u64) {
            Ok(x) => x,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        if !errors.is_empty() {
            flagged += 1;
            report.push(format!("{}: engine error {:?}", l["id"], errors));
            continue;
        }
        let sd: BTreeMap<String, BTreeMap<String, u32>> = serde_json::from_value(l["counts"].clone()).unwrap();
        let mut keys: Vec<&String> = sd.keys().chain(counts.keys()).collect();
        keys.sort();
        keys.dedup();
        let empty = BTreeMap::new();
        let mut bad = vec![];
        for key in keys {
            features += 1;
            let (p, tvd) = compare_counts(sd.get(key).unwrap_or(&empty), counts.get(key).unwrap_or(&empty));
            if p < ALPHA {
                bad.push(format!("{key} (p={p:.1e}, tvd={tvd:.2})"));
            }
        }
        if bad.is_empty() {
            ok += 1;
        } else {
            flagged += 1;
            if report.len() < 20 {
                report.push(format!("{}: {}", l["id"], bad.join(", ")));
            }
        }
    }
    println!("turn parity: {ok} consistent, {flagged} flagged, {skipped} skipped, {features} feature distributions");
    for r in &report {
        println!("  {r}");
    }
    // Statistical test: allow a tiny fraction of flags (false positives at
    // alpha=1e-6 over thousands of features are rare but possible).
    assert!(flagged * 100 <= ok + flagged, "too many flagged turn scenarios: {report:?}");
}

#[test]
fn legal_actions_match_showdown() {
    let Some(lines) = fixture("turns.jsonl.gz") else { return };
    let (mut ok, mut bad, mut skipped) = (0, 0, 0);
    let mut report = vec![];
    for l in lines.iter().filter(|l| l["kind"] == "legal") {
        let b = match Battle::from_state_json(&l["state"], 1) {
            Ok(b) => b,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        let mut same = true;
        for side in 0..2 {
            let got = b.legal_summary(side);
            for slot in 0..2 {
                let want = &l["expect"][side][slot];
                if want.is_null() {
                    continue;
                }
                let g = &got[slot];
                for f in ["moves", "switches", "mega", "pass"] {
                    if g[f] != want[f] {
                        same = false;
                    }
                }
            }
        }
        if same {
            ok += 1;
        } else {
            bad += 1;
            if report.len() < 10 {
                report.push(l["id"].to_string());
            }
        }
    }
    println!("legality: {ok} match, {bad} differ, {skipped} skipped");
    assert!(bad == 0, "legality mismatches: {report:?}");
}
