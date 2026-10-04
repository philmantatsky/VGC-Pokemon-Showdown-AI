#!/usr/bin/env node
'use strict';
// Turn-outcome parity (statistical) and request legality.
//
// Plays random Showdown battles in the format with engine-supported teams and
// random legal choices. At sampled decision points (team preview, move
// requests, switch requests incl. mid-turn ones) it records the battle state
// and both sides' choices. Each recorded decision is then replayed RUNS times
// with different seeds in Showdown (State deserialize + new PRNG) and in the
// engine (Battle::from_state_json + new seed), and the distributions of the
// outcome features at the next decision point are compared (HP, faint,
// status and its counters, boosts, position, item, ability, species, types,
// tracked volatiles, PP, side/slot conditions, weather/terrain/rooms, move
// order, the next request). A feature is flagged when a chi-square
// homogeneity test (rare values pooled) or, for numbers, a KS test gives
// p < --alpha (default 1e-6).
//
// It also checks, at every decision point of the random battles, that the
// engine's legal actions match Showdown's request (usable moves, Mega
// Evolution, switch targets, forced switches).
//
// Usage: node turns.js [--battles 60] [--sample 0.35] [--runs 200] [--seed 1]
//                      [--alpha 1e-6] [--show 25] [--jobs 4] [--dump file]
//                      [--require species1,species2]   (side 1 brings one of these)
//                      [--fixture out.jsonl.gz] [--fixture-max 250]
// --dump writes the flagged scenarios (file), request mismatches
// (file.legal.json) and engine step errors (file.err.json) for debug.js /
// debug_hits.js. --fixture records Showdown's outcome counts for
// engine/tests/parity.rs. Exit code 1 when anything is flagged.

const fs = require('fs');
const cp = require('child_process');
const os = require('os');
const L = require('./lib');

function arg(name, dflt) {
	const i = process.argv.indexOf(name);
	return i >= 0 ? process.argv[i + 1] : dflt;
}
const BATTLES = Number(arg('--battles', 60));
const SAMPLE = Number(arg('--sample', 0.35));
const RUNS = Number(arg('--runs', 200));
const SEED = Number(arg('--seed', 1));
const ALPHA = Number(arg('--alpha', 1e-6));
const SHOW = Number(arg('--show', 25));
const JOBS = Number(arg('--jobs', Math.max(1, Math.min(4, os.cpus().length))));
const DUMP = arg('--dump', null);
// Write the compared scenarios with Showdown's outcome counts (and the
// legality checks) as a gzipped JSON-lines fixture for the Rust tests.
const FIXTURE = arg('--fixture', null);
const FIXTURE_MAX = Number(arg('--fixture-max', 250));
const WORKER = process.argv.includes('--worker');
// Only teams with one of these species on side 1 (focus a mechanic).
const REQUIRE = arg('--require', null);

// ---- statistics ----------------------------------------------------------------

function lgamma(x) {
	// Lanczos approximation.
	const g = 7;
	const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
		-176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
	if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
	x -= 1;
	let a = c[0];
	const t = x + g + 0.5;
	for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
	return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
/** Upper regularized gamma Q(s, x). */
function gammaQ(s, x) {
	if (x <= 0) return 1;
	if (x < s + 1) {
		let sum = 1 / s, term = 1 / s;
		for (let n = 1; n < 500; n++) {
			term *= x / (s + n);
			sum += term;
			if (term < sum * 1e-15) break;
		}
		return Math.max(0, 1 - Math.exp(-x + s * Math.log(x) - lgamma(s)) * sum);
	}
	let b = x + 1 - s, c = 1e300, d = 1 / b, h = d;
	for (let i = 1; i < 500; i++) {
		const an = -i * (i - s);
		b += 2;
		d = an * d + b;
		if (Math.abs(d) < 1e-300) d = 1e-300;
		c = b + an / c;
		if (Math.abs(c) < 1e-300) c = 1e-300;
		d = 1 / d;
		const del = d * c;
		h *= del;
		if (Math.abs(del - 1) < 1e-15) break;
	}
	return Math.exp(-x + s * Math.log(x) - lgamma(s)) * h;
}
function chi2p(chi2, dof) {
	return dof <= 0 ? 1 : gammaQ(dof / 2, chi2 / 2);
}
function ksp(d, n, m) {
	const ne = n * m / (n + m);
	const lam = (Math.sqrt(ne) + 0.12 + 0.11 / Math.sqrt(ne)) * d;
	if (lam < 0.2) return 1;
	let sum = 0;
	for (let j = 1; j <= 100; j++) {
		const t = 2 * Math.pow(-1, j - 1) * Math.exp(-2 * j * j * lam * lam);
		sum += t;
		if (Math.abs(t) < 1e-12) break;
	}
	return Math.max(0, Math.min(1, sum));
}

/** Compare two count maps {value: n}. Returns {p, tvd, test}. */
function compareCounts(a, b) {
	const na = Object.values(a).reduce((x, y) => x + y, 0);
	const nb = Object.values(b).reduce((x, y) => x + y, 0);
	if (!na || !nb) return { p: 1, tvd: 0, test: 'empty' };
	const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
	let tvd = 0;
	for (const k of keys) tvd += Math.abs((a[k] || 0) / na - (b[k] || 0) / nb);
	tvd /= 2;
	if (keys.size === 1) return { p: 1, tvd: 0, test: 'same' };
	// Chi-square with rare values pooled.
	const cells = [];
	let rareA = 0, rareB = 0;
	for (const k of keys) {
		const x = a[k] || 0, y = b[k] || 0;
		if (x + y < 8) {
			rareA += x;
			rareB += y;
		} else {
			cells.push([x, y]);
		}
	}
	if (rareA + rareB > 0) cells.push([rareA, rareB]);
	let chi2 = 0;
	for (const [x, y] of cells) {
		const p = (x + y) / (na + nb);
		const ea = na * p, eb = nb * p;
		if (ea > 0) chi2 += (x - ea) ** 2 / ea;
		if (eb > 0) chi2 += (y - eb) ** 2 / eb;
	}
	let p = chi2p(chi2, cells.length - 1);
	let test = 'chi2';
	// KS for numeric features.
	const nums = [...keys].every(k => k !== '' && !isNaN(Number(k)));
	if (nums && keys.size > 2) {
		const vals = [...keys].map(Number).sort((x, y) => x - y);
		let ca = 0, cb = 0, d = 0;
		for (const v of vals) {
			ca += (a[String(v)] || 0) / na;
			cb += (b[String(v)] || 0) / nb;
			d = Math.max(d, Math.abs(ca - cb));
		}
		const pk = ksp(d, na, nb);
		if (pk < p) {
			p = pk;
			test = 'ks';
		}
	}
	return { p, tvd, test };
}

// ---- Showdown side ------------------------------------------------------------

function legalFromShowdown(battle) {
	const out = [];
	for (const side of battle.sides) {
		const req = side.activeRequest;
		const slots = [];
		for (let i = 0; i < 2; i++) {
			const p = side.active[i];
			const bench = side.pokemon.filter(q => !q.isActive && !q.fainted).map(L.teamIdx).sort((x, y) => x - y);
			if (!req || req.wait || req.teamPreview) {
				slots.push(null);
				continue;
			}
			if (req.forceSwitch) {
				const flagged = req.forceSwitch.filter(Boolean).length;
				if (req.forceSwitch[i] && side.slotConditions[i] && side.slotConditions[i].revivalblessing) {
					// Revival Blessing: any fainted party member; pass only
					// when there is no healthy benched Pokemon.
					const fainted = side.pokemon.filter(q => q.fainted).map(L.teamIdx).sort((x, y) => x - y);
					slots.push({ moves: [], switches: fainted, mega: false, pass: flagged > bench.length });
					continue;
				}
				slots.push(req.forceSwitch[i] && bench.length ?
					{ moves: [], switches: bench, mega: false, pass: flagged > bench.length } :
					{ moves: [], switches: [], mega: false, pass: true });
				continue;
			}
			if (!p || p.fainted) {
				slots.push({ moves: [], switches: [], mega: false, pass: true });
				continue;
			}
			const a = req.active[i];
			const moves = new Set();
			for (const m of a.moves) {
				const ms = p.moveSlots.find(s => s.id === m.id);
				if (m.disabled || (ms && ms.disabled)) continue;
				if (m.id === 'struggle' || m.id === 'recharge') {
					moves.add(0);
					continue;
				}
				const k = p.moveSlots.findIndex(s => s.id === m.id);
				moves.add(k);
			}
			slots.push({
				moves: [...moves].sort((x, y) => x - y),
				switches: p.trapped ? [] : bench,
				mega: !!a.canMegaEvo,
				pass: false,
			});
		}
		out.push(slots);
	}
	return out;
}

function sortedPreview(order) {
	const lead = order.slice(0, 2).sort((x, y) => x - y);
	const back = order.slice(2, 4).sort((x, y) => x - y);
	return [...lead, ...back];
}

function choose(battle, choices) {
	for (let i = 0; i < 2; i++) {
		const c = choices[i];
		if (!c) continue;
		const side = battle.sides[i];
		const str = L.choiceString(side, c);
		if (!battle.choose(side.id, str)) {
			if (process.env.DEBUG_CHOICE) {
				const req = side.activeRequest;
				console.error(`REJECTED ${side.id} '${str}' request:`, JSON.stringify(req && (req.active || req.forceSwitch)).slice(0, 1500));
				console.error(side.active.map(p => p && `${p.name} hp=${p.hp} fainted=${p.fainted} vols=${Object.keys(p.volatiles)}`));
				console.error(battle.log.slice(-25).join('\n'));
			}
			throw new Error(`choice rejected: ${side.id} '${str}'`);
		}
	}
}

/** Play random battles; return recorded scenarios and legality checks. */
function generate(seed, nBattles) {
	const pool = L.loadPool();
	const sup = L.supportedTeams();
	const req = REQUIRE ? REQUIRE.split(',') : null;
	const supA = req ? sup.filter(i => pool[i].mons.some(m => req.includes(m.species))) : sup;
	const rand = new L.Rand(seed);
	const scenarios = [];
	const legal = [];
	let errors = 0;
	for (let g = 0; g < nBattles; g++) {
		const ta = pool[rand.pick(supA)], tb = pool[rand.pick(sup)];
		const teams = [ta, tb];
		const bseed = rand.seed();
		let battle = L.newBattle(ta, tb, bseed);
		let steps = 0;
		while (!battle.ended && steps < 120) {
			steps++;
			const choices = battle.sides.map(s => L.randomChoice(battle, s, rand, { allyDamage: rand.chance(0.1) }));
			if (battle.requestState === 'teampreview') {
				for (const c of choices) c.preview = sortedPreview(c.preview);
			}
			const tag = `b${seed}.${g}.${steps}`;
			if (battle.requestState !== 'teampreview') {
				legal.push({ id: tag, state: L.exportState(battle, teams), expect: legalFromShowdown(battle), turn: battle.turn });
			}
			if (rand.chance(battle.requestState === 'teampreview' ? 0.3 : SAMPLE)) {
				scenarios.push({
					id: tag,
					kind: battle.requestState + (battle.requestState === 'switch' && battle.queue.list.some(a => a.choice === 'residual') ? ':mid' : ''),
					serialized: battle.requestState === 'teampreview' ? null : L.serialize(battle),
					teams: battle.requestState === 'teampreview' ? [ta, tb] : null,
					state: L.exportState(battle, teams),
					choices,
					engineChoices: choices.map((c, i) => L.engineChoice(battle.sides[i], c)),
					desc: describe(battle, choices),
				});
			}
			try {
				choose(battle, choices);
			} catch (e) {
				errors++;
				if (errors < 5) console.error(`${tag}: ${e.message}`);
				break;
			}
		}
	}
	return { scenarios, legal, errors };
}

function describe(battle, choices) {
	const parts = [];
	battle.sides.forEach((side, si) => {
		side.active.forEach((p, i) => {
			if (!p) return;
			const c = choices[si] && choices[si].slots ? choices[si].slots[i] : null;
			let act = '';
			if (c) {
				if (c.kind === 'move') act = `${c.moveid}${c.targetLoc ? '@' + c.targetLoc : ''}${c.mega ? '+mega' : ''}`;
				else if (c.kind === 'switch') act = `->${side.pokemon.find(q => L.teamIdx(q) === c.to).species.name}`;
				else act = 'pass';
			}
			parts.push(`${side.id}${'ab'[i]} ${p.species.name}[${p.ability}]@${p.item || '-'} ${p.hp}/${p.maxhp}` +
				`${p.status ? ' ' + p.status : ''}${Object.keys(p.volatiles).length ? ' {' + Object.keys(p.volatiles).join(',') + '}' : ''}: ${act}`);
		});
		const sc = Object.keys(side.sideConditions);
		if (sc.length) parts.push(`${side.id} sc: ${sc.join(',')}`);
	});
	const f = battle.field;
	parts.push(`field: ${f.weather || '-'} ${f.terrain || '-'} ${Object.keys(f.pseudoWeather).join(',')} turn ${battle.turn}`);
	return parts.join('\n      ');
}

/** Run one scenario RUNS times in Showdown; returns {counts, errors}. */
function showdownRuns(sc, runs, seed) {
	const counts = {};
	const errors = {};
	const rand = new L.Rand(seed);
	for (let k = 0; k < runs; k++) {
		let b;
		try {
			if (sc.serialized) {
				b = L.deserialize(sc.serialized);
				L.reseed(b, rand.seed());
			} else {
				b = L.newBattle(sc.teams[0], sc.teams[1], rand.seed());
			}
			const logStart = b.log.length;
			choose(b, sc.choices);
			const f = L.outcome(b, logStart);
			for (const [key, v] of Object.entries(f)) {
				const vs = String(v);
				counts[key] = counts[key] || {};
				counts[key][vs] = (counts[key][vs] || 0) + 1;
			}
		} catch (e) {
			const m = e.message.slice(0, 120);
			errors[m] = (errors[m] || 0) + 1;
		}
	}
	return { counts, errors };
}

// ---- worker mode (Showdown runs in parallel) -----------------------------------------

if (WORKER) {
	const input = JSON.parse(fs.readFileSync(arg('--in'), 'utf8'));
	const out = input.map(sc => ({ id: sc.id, ...showdownRuns(sc, RUNS, sc.runSeed) }));
	fs.writeFileSync(arg('--out'), JSON.stringify(out));
	process.exit(0);
}

function showdownParallel(scenarios) {
	const tmp = fs.mkdtempSync(require('path').join(os.tmpdir(), 'vgcz-turns-'));
	const chunks = Array.from({ length: JOBS }, () => []);
	scenarios.forEach((s, i) => chunks[i % JOBS].push(s));
	const procs = chunks.map((chunk, j) => {
		const inF = `${tmp}/in${j}.json`, outF = `${tmp}/out${j}.json`;
		fs.writeFileSync(inF, JSON.stringify(chunk));
		return { outF, p: cp.spawn(process.execPath, [__filename, '--worker', '--in', inF, '--out', outF, '--runs', String(RUNS)], { stdio: 'inherit' }) };
	});
	return Promise.all(procs.map(({ outF, p }) => new Promise((res, rej) => {
		p.on('exit', code => (code === 0 ? res(JSON.parse(fs.readFileSync(outF, 'utf8'))) : rej(new Error('worker failed'))));
	}))).then(rs => {
		fs.rmSync(tmp, { recursive: true, force: true });
		return rs.flat();
	});
}

// ---- main ------------------------------------------------------------------------

async function main() {
	const t0 = Date.now();
	const { scenarios, legal, errors } = generate(SEED, BATTLES);
	scenarios.forEach((s, i) => { s.runSeed = SEED * 1000003 + i; });
	console.log(`generated ${scenarios.length} scenarios and ${legal.length} legality checks from ${BATTLES} battles` +
		` (${errors} battles aborted) in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

	// Legality.
	const legalRes = L.runEngine(['legal'], legal.map(l => ({ id: l.id, state: l.state })));
	const legalById = new Map(legalRes.map(r => [r.id, r]));
	let lOk = 0, lBad = 0, lSkip = 0;
	const lSkipWhy = {};
	const lBads = [];
	for (const l of legal) {
		const r = legalById.get(l.id);
		if (!r || r.err) {
			lSkip++;
			const w = r ? r.err.replace(/[0-9]+/g, 'N') : 'missing';
			lSkipWhy[w] = (lSkipWhy[w] || 0) + 1;
			continue;
		}
		let bad = [];
		for (let s = 0; s < 2; s++) {
			for (let i = 0; i < 2; i++) {
				const e = l.expect[s][i];
				if (!e) continue;
				const g = r.sides[s][i];
				const ge = { moves: g.moves, switches: g.switches, mega: g.mega, pass: g.pass };
				const ee = { moves: e.moves, switches: e.switches, mega: e.mega, pass: e.pass };
				if (JSON.stringify(ge) !== JSON.stringify(ee)) bad.push(`p${s + 1}${'ab'[i]} showdown ${JSON.stringify(ee)} engine ${JSON.stringify(ge)}`);
			}
		}
		if (bad.length) {
			lBad++;
			lBads.push({ l, bad });
		} else {
			lOk++;
		}
	}
	console.log(`legality: ${legal.length} decision points, ${lOk} match, ${lBad} differ, ${lSkip} skipped`);
	for (const [k, v] of Object.entries(lSkipWhy).sort((a, b) => b[1] - a[1]).slice(0, 8)) console.log(`    skipped ${v}: ${k}`);
	for (const { l, bad } of lBads.slice(0, Math.min(SHOW, 12))) {
		console.log(`  ${l.id} (turn ${l.turn}): ${bad.join('; ')}`);
	}

	// Showdown distributions.
	const t1 = Date.now();
	const sd = await showdownParallel(scenarios);
	const sdById = new Map(sd.map(r => [r.id, r]));
	console.log(`showdown: ${scenarios.length} x ${RUNS} runs in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
	const t2 = Date.now();
	const en = L.runEngine(['turn', String(RUNS)], scenarios.map(s => ({ id: s.id, state: s.state, choices: s.engineChoices, seed: s.runSeed })));
	const enById = new Map(en.map(r => [r.id, r]));
	console.log(`engine: ${scenarios.length} x ${RUNS} runs in ${((Date.now() - t2) / 1000).toFixed(1)}s`);

	// Compare.
	let compared = 0, skipped = 0, flaggedScen = 0, features = 0, flaggedFeat = 0;
	const skipWhy = {};
	const flags = [];
	const errs = [];
	const byKind = {};
	for (const s of scenarios) {
		const a = sdById.get(s.id), b = enById.get(s.id);
		if (!b || b.err || !a) {
			skipped++;
			const w = !b ? 'missing' : b.err ? b.err.replace(/[0-9]+/g, 'N') : 'showdown missing';
			skipWhy[w] = (skipWhy[w] || 0) + 1;
			continue;
		}
		if (Object.keys(b.errors || {}).length) {
			skipped++;
			const w = 'engine step error: ' + Object.keys(b.errors)[0].replace(/[0-9]+/g, 'N');
			skipWhy[w] = (skipWhy[w] || 0) + 1;
			errs.push({ id: s.id, err: Object.keys(b.errors)[0], state: s.state, choices: s.choices, engineChoices: s.engineChoices, serialized: s.serialized });
			continue;
		}
		if (Object.keys(a.errors).length) {
			skipped++;
			const w = 'showdown error: ' + Object.keys(a.errors)[0];
			skipWhy[w] = (skipWhy[w] || 0) + 1;
			continue;
		}
		compared++;
		const k = s.kind;
		byKind[k] = byKind[k] || { n: 0, bad: 0 };
		byKind[k].n++;
		let any = false;
		const keys = new Set([...Object.keys(a.counts), ...Object.keys(b.counts)]);
		for (const key of keys) {
			features++;
			const r = compareCounts(a.counts[key] || {}, b.counts[key] || {});
			if (r.p < ALPHA) {
				flaggedFeat++;
				any = true;
				flags.push({ s, key, r, sd: a.counts[key] || {}, en: b.counts[key] || {} });
			}
		}
		if (any) {
			flaggedScen++;
			byKind[k].bad++;
		}
	}
	console.log(`turn parity: ${scenarios.length} scenarios, ${compared} compared, ${compared - flaggedScen} consistent, ` +
		`${flaggedScen} with flagged features, ${skipped} skipped; ${features} feature distributions, ${flaggedFeat} flagged (p < ${ALPHA})`);
	console.log('  by decision kind: ' + Object.entries(byKind).map(([k, v]) => `${k} ${v.n - v.bad}/${v.n}`).join(', '));
	for (const [k, v] of Object.entries(skipWhy).sort((a, b) => b[1] - a[1]).slice(0, 12)) console.log(`    skipped ${v}: ${k}`);
	// Group flags by feature kind for a summary.
	const kindOf = key => key.replace(/^p[12]\.\d\./, 'mon.').replace(/^p[12]\./, 'side.');
	const byFeat = {};
	for (const f of flags) byFeat[kindOf(f.key)] = (byFeat[kindOf(f.key)] || 0) + 1;
	console.log('  flagged by feature: ' + Object.entries(byFeat).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}:${v}`).join(' '));
	flags.sort((x, y) => x.r.p - y.r.p);
	const shown = new Set();
	let n = 0;
	for (const f of flags) {
		if (n >= SHOW) break;
		const key = f.s.id + kindOf(f.key);
		if (shown.has(f.s.id)) continue;
		shown.add(f.s.id);
		n++;
		const top = c => Object.entries(c).sort((x, y) => y[1] - x[1]).slice(0, 6).map(([v, c2]) => `${JSON.stringify(v)}:${c2}`).join(' ');
		console.log(`\n  [${f.s.id} ${f.s.kind}] ${f.key} p=${f.r.p.toExponential(1)} tvd=${f.r.tvd.toFixed(2)} (${f.r.test})`);
		console.log(`      showdown: ${top(f.sd)}`);
		console.log(`      engine:   ${top(f.en)}`);
		const others = flags.filter(g => g.s === f.s && g !== f).map(g => g.key);
		if (others.length) console.log(`      also: ${others.slice(0, 10).join(' ')}`);
		console.log(`      ${f.s.desc}`);
	}
	if (FIXTURE) {
		const zlib = require('zlib');
		const lines = [];
		const ok = scenarios.filter(s => {
			const a = sdById.get(s.id), b = enById.get(s.id);
			return a && b && !b.err && !Object.keys(b.errors || {}).length && !Object.keys(a.errors).length;
		});
		for (const s of ok.slice(0, FIXTURE_MAX)) {
			lines.push(JSON.stringify({ kind: 'turn', id: s.id, state: s.state, choices: s.engineChoices, runs: RUNS, counts: sdById.get(s.id).counts }));
		}
		for (const l of legal.filter((x, i) => i % 4 === 0).slice(0, FIXTURE_MAX * 2)) {
			lines.push(JSON.stringify({ kind: 'legal', id: l.id, state: l.state, expect: l.expect }));
		}
		fs.writeFileSync(FIXTURE, zlib.gzipSync(lines.join('\n') + '\n', { level: 9 }));
		console.log(`wrote fixture ${FIXTURE} (${Math.min(ok.length, FIXTURE_MAX)} turn scenarios)`);
	}
	if (DUMP) {
		fs.writeFileSync(DUMP + '.err.json', JSON.stringify(errs));
		fs.writeFileSync(DUMP + '.legal.json', JSON.stringify(lBads.map(({ l, bad }) => ({ id: l.id, bad, state: l.state }))));
		fs.writeFileSync(DUMP, JSON.stringify(flags.map(f => ({ id: f.s.id, key: f.key, p: f.r.p, sd: f.sd, en: f.en, state: f.s.state, choices: f.s.choices, engineChoices: f.s.engineChoices, serialized: f.s.serialized }))));
	}
	process.exitCode = flaggedScen || lBad ? 1 : 0;
}

main().catch(e => {
	console.error(e);
	process.exit(2);
});
