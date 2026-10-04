#!/usr/bin/env node
// Compile Showdown-export team files into the JSON team pool the engine reads.
//
// Every team is parsed and validated by Showdown's own TeamValidator for the
// format; invalid teams are reported and left out. Final stats (and the Mega
// forme's stats, ability and types where the item allows a Mega Evolution)
// are computed by Showdown's stat code with the format's mod, so the engine
// never re-derives the Champions stat formula. Duplicate teams (same six sets
// in any order) are kept once.
//
// Usage: node vgczero/showdown/compile_teams.js teams/reg_mc [more dirs...]
//          [--out vgczero/data/teams_regmc.json.gz]
'use strict';

const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const PS_DIR = process.env.PS_DIR || path.join(HERE, 'ps');
const FORMAT = process.env.FORMAT || 'gen9championsvgc2026regmc';

const argv = process.argv.slice(2);
let OUT = path.join(HERE, '..', 'data', 'teams_regmc.json.gz');
const dirs = [];
for (let i = 0; i < argv.length; i++) {
	if (argv[i] === '--out') OUT = argv[++i];
	else dirs.push(argv[i]);
}
if (!dirs.length) {
	console.error('usage: compile_teams.js <team dir> [...] [--out file]');
	process.exit(2);
}

const { Teams, TeamValidator, Battle, toID } = require(path.join(PS_DIR, 'dist', 'sim'));
const validator = TeamValidator.get(FORMAT);
const battle = new Battle({ formatid: FORMAT });
const dex = battle.dex;

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const statArray = s => STATS.map(k => s[k]);

function compileSet(set) {
	const species = dex.species.get(set.species);
	const item = dex.items.get(set.item);
	const out = {
		species: species.id,
		item: item.exists ? item.id : '',
		ability: toID(set.ability),
		moves: set.moves.map(m => dex.moves.get(m).id),
		nature: toID(set.nature),
		gender: set.gender || '',
		level: set.level,
		evs: statArray(set.evs),
		stats: statArray(battle.spreadModify(species.baseStats, set)),
		types: species.types,
		weightkg: species.weightkg,
		mega: null,
	};
	const megaName = item.exists && item.megaStone ? item.megaStone[species.name] : null;
	if (megaName) {
		const mega = dex.species.get(megaName);
		out.mega = {
			species: mega.id,
			ability: toID(mega.abilities['0']),
			types: mega.types,
			stats: statArray(battle.spreadModify(mega.baseStats, set)),
			weightkg: mega.weightkg,
		};
	}
	return out;
}

const seen = new Map();
const teams = [];
const rejected = [];
let files = 0;
for (const dir of dirs) {
	for (const f of fs.readdirSync(dir).filter(x => x.endsWith('.txt')).sort()) {
		files++;
		const file = path.join(dir, f);
		const text = fs.readFileSync(file, 'utf8');
		let team;
		try {
			team = Teams.import(text);
		} catch (e) {
			rejected.push({ file, errors: [`parse: ${e.message}`] });
			continue;
		}
		if (!team || team.length !== 6) {
			rejected.push({ file, errors: [`team size ${team ? team.length : 0}`] });
			continue;
		}
		const errors = validator.validateTeam(team);
		if (errors) {
			rejected.push({ file, errors });
			continue;
		}
		const mons = team.map(compileSet);
		const key = JSON.stringify(mons.map(m => JSON.stringify(m)).sort());
		if (seen.has(key)) {
			seen.get(key).duplicates.push(path.basename(f, '.txt'));
			continue;
		}
		const entry = {
			name: path.basename(f, '.txt'), source: file, duplicates: [], mons,
			// Showdown export text, so the live client can submit the team.
			export: Teams.export(team),
			packed: Teams.pack(team),
		};
		seen.set(key, entry);
		teams.push(entry);
	}
}

const out = {
	meta: { format: FORMAT, sources: dirs, files, compiled: teams.length, rejected: rejected.length },
	teams,
	rejected,
};
fs.mkdirSync(path.dirname(OUT), { recursive: true });
const body = JSON.stringify(out);
fs.writeFileSync(OUT, OUT.endsWith('.gz') ? require('zlib').gzipSync(body, { level: 9 }) : body);
console.log(`wrote ${OUT}: ${teams.length} unique valid teams from ${files} files ` +
	`(${rejected.length} rejected, ${files - teams.length - rejected.length} duplicates)`);
const reasons = {};
for (const r of rejected) for (const e of r.errors) {
	const k = e.replace(/^[^:]*'s /, '').slice(0, 70);
	reasons[k] = (reasons[k] || 0) + 1;
}
for (const [k, v] of Object.entries(reasons).sort((a, b) => b[1] - a[1]).slice(0, 10)) {
	console.log(`  rejected x${v}: ${k}`);
}
