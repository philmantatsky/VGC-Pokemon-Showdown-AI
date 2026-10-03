#!/usr/bin/env node
// Export the Champions Reg M-C dex from a built Pokemon Showdown checkout into
// one JSON file the Rust engine loads at startup.
//
// Showdown is the source of truth: every number here (base stats, move data,
// type chart) comes from its Dex with the format's mod applied. Each move,
// item and ability also lists the names of its scripted callbacks
// (onHit, basePowerCallback, ...). The engine compares those lists with the
// handlers it implements, so "supported" is checked, not assumed.
//
// Usage: node vgczero/showdown/export_dex.js [--out vgczero/data/dex_regmc.json.gz]
//        PS_DIR=/path/to/pokemon-showdown (default vgczero/showdown/ps)
'use strict';

const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const PS_DIR = process.env.PS_DIR || path.join(HERE, 'ps');
const FORMAT = process.env.FORMAT || 'gen9championsvgc2026regmc';

function arg(name, dflt) {
	const i = process.argv.indexOf(name);
	return i >= 0 ? process.argv[i + 1] : dflt;
}
const OUT = arg('--out', path.join(HERE, '..', 'data', 'dex_regmc.json.gz'));

const { Dex } = require(path.join(PS_DIR, 'dist', 'sim'));
const dex = Dex.forFormat(FORMAT);

function callbacks(obj) {
	if (!obj) return [];
	return Object.keys(obj).filter(k => typeof obj[k] === 'function').sort();
}

// Keep plain data only; drop prose and functions.
const DROP = new Set([
	'desc', 'shortDesc', 'contestType', 'effectType', 'exists', 'gen', 'kind',
	'fullname', 'sourceEffect', 'duration', 'realMove', 'isNonstandard', 'tier',
	'doublesTier', 'natDexTier', 'color', 'heightm', 'eggGroups', 'evos', 'prevo',
	'evoLevel', 'evoType', 'evoCondition', 'evoItem', 'evoMove', 'canHatch',
	'tags', 'spriteid', 'unreleasedHidden', 'maleOnlyHidden', 'cosmeticFormes',
	'formeOrder', 'changesFrom', 'mother', 'zMove', 'maxMove', 'zMoveEffect',
	'zMoveBoost', 'zMovePower', 'gmaxPower', 'baseMoveType', 'noMetronome',
	'rating', 'flingBasePower', 'itemUser', 'spritenum', 'isGigantamax',
	'battleOnly', 'learnset', 'nfe', 'gmaxUnreleased', 'canGigantamax',
]);

function plain(obj) {
	const out = {};
	for (const k of Object.keys(obj)) {
		if (DROP.has(k)) continue;
		const v = obj[k];
		if (typeof v === 'function' || v === undefined) continue;
		if (Array.isArray(v)) {
			out[k] = v.map(x => {
				if (!x || typeof x !== 'object') return x;
				const inner = plain(x);
				const cbs = callbacks(x);
				if (cbs.length) inner.callbacks = cbs;
				return inner;
			});
		} else if (v && typeof v === 'object') {
			const inner = plain(v);
			const cbs = callbacks(v);
			if (cbs.length) inner.callbacks = cbs;
			out[k] = inner;
		} else {
			out[k] = v;
		}
	}
	return out;
}

function legal(table) {
	return table.all().filter(e => e.exists && !e.isNonstandard)
		.sort((a, b) => a.id.localeCompare(b.id));
}

const species = legal(dex.species).map(s => ({
	id: s.id,
	name: s.name,
	num: s.num,
	baseSpecies: s.baseSpecies,
	forme: s.forme || '',
	types: s.types,
	baseStats: s.baseStats,
	abilities: s.abilities,
	weightkg: s.weightkg,
	isMega: !!s.isMega,
	requiredItem: s.requiredItem || null,
	requiredItems: s.requiredItems || null,
	otherFormes: s.otherFormes || [],
	gender: s.gender || '',
	genderRatio: s.genderRatio || null,
}));

const moves = legal(dex.moves).map(m => {
	const out = plain(m);
	out.callbacks = callbacks(m);
	if (m.condition) out.conditionCallbacks = callbacks(m.condition);
	return out;
});

const items = legal(dex.items).map(it => {
	const out = plain(it);
	out.callbacks = callbacks(it);
	if (it.condition) out.conditionCallbacks = callbacks(it.condition);
	return out;
});

// Abilities: the legal ones plus any a legal species can have (some new Mega
// abilities are still marked "Future" in the base dex).
const abilityIds = new Set(legal(dex.abilities).map(a => a.id));
for (const s of species) {
	for (const name of Object.values(s.abilities)) {
		const a = dex.abilities.get(name);
		if (a.exists) abilityIds.add(a.id);
	}
}
const abilities = [...abilityIds].sort().map(id => dex.abilities.get(id)).map(a => {
	const out = plain(a);
	out.callbacks = callbacks(a);
	if (a.condition) out.conditionCallbacks = callbacks(a.condition);
	return out;
});

// Type chart: typechart[defType][atkType] = 0 normal, 1 super, 2 resist, 3 immune
// (Showdown's damageTaken encoding).
const types = dex.types.all().filter(t => t.exists && !t.isNonstandard).map(t => t.name)
	.filter(n => n !== 'Stellar');
const typechart = {};
for (const def of types) {
	typechart[def] = {};
	for (const atk of types) typechart[def][atk] = dex.types.get(def).damageTaken[atk] || 0;
}

const natures = dex.natures.all().map(n => ({ id: n.id, name: n.name, plus: n.plus || null, minus: n.minus || null }));

const format = dex.formats.get(FORMAT);
const out = {
	meta: {
		format: FORMAT,
		formatName: format.name,
		mod: dex.currentMod,
		gen: dex.gen,
		showdownCommit: (() => {
			try {
				return require('child_process').execSync('git rev-parse HEAD', { cwd: PS_DIR }).toString().trim();
			} catch { return null; }
		})(),
		exportedAt: new Date().toISOString(),
	},
	types,
	typechart,
	natures,
	species,
	moves,
	items,
	abilities,
};

fs.mkdirSync(path.dirname(OUT), { recursive: true });
const body = JSON.stringify(out);
fs.writeFileSync(OUT, OUT.endsWith('.gz') ? require('zlib').gzipSync(body, { level: 9 }) : body);
console.log(`wrote ${OUT}: ${species.length} species, ${moves.length} moves, ` +
	`${items.length} items, ${abilities.length} abilities, ${types.length} types`);
