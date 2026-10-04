#!/usr/bin/env node
'use strict';
// Show one Showdown run and one engine run of a scenario flagged by
// turns.js (--dump file), to debug a mismatch.
//
// Usage: node debug.js <dump.json> <scenario id> [--seed N] [--key feature]

const fs = require('fs');
const L = require('./lib');

const [dumpFile, id] = process.argv.slice(2);
function arg(name, dflt) {
	const i = process.argv.indexOf(name);
	return i >= 0 ? process.argv[i + 1] : dflt;
}
const SEED = Number(arg('--seed', 1));
const KEY = arg('--key', null);

const flags = JSON.parse(fs.readFileSync(dumpFile, 'utf8'));
const f = flags.find(x => x.id === id && (!KEY || x.key === KEY));
if (!f) {
	console.error('not found; ids:', [...new Set(flags.map(x => x.id))].join(' '));
	process.exit(1);
}
console.log(`== ${f.id} flagged ${flags.filter(x => x.id === id).map(x => x.key).join(' ')}`);
console.log('showdown counts:', JSON.stringify(f.sd));
console.log('engine counts:  ', JSON.stringify(f.en));
console.log('choices:', JSON.stringify(f.choices), 'engine:', JSON.stringify(f.engineChoices));

// Showdown run.
const b = L.deserialize(f.serialized);
L.reseed(b, [SEED, SEED + 1, SEED + 2, SEED + 3]);
const start = b.log.length;
for (let i = 0; i < 2; i++) {
	if (!f.choices[i]) continue;
	const str = L.choiceString(b.sides[i], f.choices[i]);
	console.log(`showdown choose ${b.sides[i].id}: ${str} -> ${b.choose(b.sides[i].id, str)}`);
}
console.log('--- showdown log');
console.log(b.log.slice(start).filter(l => !l.startsWith('|split') && !/^\|(t:|)$/.test(l)).join('\n'));
const o = L.outcome(b, start);
if (KEY) console.log(`showdown ${KEY} = ${o[KEY]}`);

// Engine run.
const r = L.runEngine(['turnlog', String(SEED)], [{ id: f.id, state: f.state, choices: f.engineChoices }])[0];
if (r.err) {
	console.log('engine error:', r.err);
} else {
	console.log('--- engine before\n' + r.before);
	console.log('--- engine log');
	console.log(r.log.join('\n'));
	console.log('--- engine after\n' + r.after);
	if (KEY) console.log(`engine ${KEY} = ${r.features[KEY]}`);
}
