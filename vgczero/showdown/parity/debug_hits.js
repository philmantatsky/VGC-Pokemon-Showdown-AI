#!/usr/bin/env node
'use strict';
// For a scenario flagged by turns.js (--dump file), run it many times in both
// simulators and tabulate the HP change of every `-damage` / `-heal` event by
// (move user, move, affected Pokemon), to find which step of a turn differs.
//
// Usage: node debug_hits.js <dump.json> <scenario id> [--runs 200]

const fs = require('fs');
const L = require('./lib');

const [dumpFile, id] = process.argv.slice(2);
const RUNS = Number((process.argv.indexOf('--runs') >= 0 && process.argv[process.argv.indexOf('--runs') + 1]) || 200);
const f = JSON.parse(fs.readFileSync(dumpFile, 'utf8')).find(x => x.id === id);
if (!f) throw new Error('scenario not found');

function events(lines, nameOf) {
	// Track HP per Pokemon ("p1a") and attribute changes to the last |move|.
	const out = [];
	let cur = 'start';
	const hp = {};
	for (const l of lines) {
		const parts = l.split('|');
		if (parts[1] === 'move') cur = `${parts[2].slice(0, 3)}:${parts[3]}`;
		if (parts[1] === 'upkeep' || /^\|-weather\|.*\[upkeep\]/.test(l)) cur = 'residual';
		if (parts[1] === '-damage' || parts[1] === '-heal') {
			const who = parts[2].slice(0, 3);
			const m = /^(\d+)\/(\d+)/.exec(parts[3]);
			const v = m ? Number(m[1]) : 0;
			if (m && Number(m[2]) === 100 && nameOf.hp100) continue; // Showdown's percentage lines
			const before = hp[who];
			hp[who] = v;
			if (before !== undefined) out.push(`${cur} -> ${who} ${v - before > 0 ? '+' : ''}${v - before}`);
			else out.push(`${cur} -> ${who} =${v}`);
		}
	}
	return out;
}

const sd = {}, en = {};
const add = (t, k) => { t[k] = (t[k] || 0) + 1; };
for (let k = 0; k < RUNS; k++) {
	const b = L.deserialize(f.serialized);
	L.reseed(b, [k, k + 7, k + 13, k + 29]);
	// Initial HP of the actives, so deltas can be computed.
	const init = [];
	for (const s of b.sides) s.active.forEach((p, i) => { if (p) init.push(`|-damage|${s.id}${'ab'[i]}: x|${p.hp}/${p.maxhp}`); });
	const start = b.log.length;
	for (let i = 0; i < 2; i++) if (f.choices[i]) b.choose(b.sides[i].id, L.choiceString(b.sides[i], f.choices[i]));
	const lines = b.log.slice(start).filter(l => !l.startsWith('|split'));
	// drop Showdown's duplicated percentage lines: keep the line whose denominator is the max HP
	const filtered = [];
	for (let i = 0; i < lines.length; i++) {
		const l = lines[i];
		// Showdown writes each HP change twice (exact, then percentage).
		if ((l.startsWith('|-damage|') || l.startsWith('|-heal|')) && i > 0 && lines[i - 1].split('|').slice(0, 3).join('|') === l.split('|').slice(0, 3).join('|')) continue;
		filtered.push(l);
	}
	for (const e of events([...init.map(x => x.replace('|-damage|', '|-heal|')), ...filtered], {})) add(sd, e);
}
const r = L.runEngine(['turn', '1'], []); // warm-up check of the binary
void r;
for (let k = 0; k < RUNS; k++) {
	const out = L.runEngine(['turnlog', String(1000 + k)], [{ id: f.id, state: f.state, choices: f.engineChoices }])[0];
	if (out.err) throw new Error(out.err);
	const init = [];
	f.state.sides.forEach((s, si) => s.active.forEach((ti, i) => { if (ti >= 0) { const m = s.mons[ti]; init.push(`|-heal|p${si + 1}${'ab'[i]}: x|${m.hp}/${m.maxhp}`); } }));
	for (const e of events([...init, ...out.log], {})) add(en, e);
}
const keys = [...new Set([...Object.keys(sd), ...Object.keys(en)])].filter(k => !k.startsWith('start'));
const byStep = {};
for (const k of keys) {
	const step = k.replace(/ [+-]?=?\d+$/, '');
	byStep[step] = byStep[step] || [];
	byStep[step].push(k);
}
for (const [step, ks] of Object.entries(byStep)) {
	const vals = ks.map(k => [Number(k.match(/([+-]?\d+)$/)[1]), sd[k] || 0, en[k] || 0]).sort((a, b) => a[0] - b[0]);
	const mean = (i) => { let s = 0, n = 0; for (const v of vals) { s += v[0] * v[i]; n += v[i]; } return n ? (s / n).toFixed(1) : '-'; };
	const nS = vals.reduce((a, v) => a + v[1], 0), nE = vals.reduce((a, v) => a + v[2], 0);
	console.log(`${step}: showdown n=${nS} mean ${mean(1)} | engine n=${nE} mean ${mean(2)}`);
}
