#!/usr/bin/env node
'use strict';
// Damage parity: for random (attacker, defender, move) combinations drawn from
// the engine-supported team pool, with random variations (boosts, weather,
// terrain, screens, Helping Hand, status, Mega Evolution, items, abilities,
// HP...), compare the engine's damage for all 16 random rolls, with and
// without a critical hit, against Showdown's `battle.actions.getDamage`.
//
// Usage: node damage.js [--n 3000] [--seed 1] [--show 20] [--out file.jsonl]
//                       [--fixture out.jsonl.gz] [--focus ability|item:<id>]

const fs = require('fs');
const zlib = require('zlib');
const L = require('./lib');

function arg(name, dflt) {
	const i = process.argv.indexOf(name);
	return i >= 0 ? process.argv[i + 1] : dflt;
}
const N = Number(arg('--n', 3000));
const SEED = Number(arg('--seed', 1));
const SHOW = Number(arg('--show', 15));
const FIXTURE = arg('--fixture', null);
let FOCUS = arg('--focus', null);
const FOCUS_ALL = process.argv.includes('--focus-all');
const PER = Number(arg('--per', 60));
const tr = Math.trunc;

const WEATHERS = ['raindance', 'sunnyday', 'sandstorm', 'snowscape'];
const TERRAINS = ['electricterrain', 'grassyterrain', 'psychicterrain', 'mistyterrain'];

function isDamaging(id) {
	const m = L.dex.moves.get(id);
	return m.exists && m.category !== 'Status';
}

/** Snapshot and restore everything getDamage can touch (berries are eaten). */
function snapshotMons(battle) {
	return battle.getAllActive().map(p => ({
		p, item: p.item, itemState: { ...p.itemState }, lastItem: p.lastItem,
		usedItemThisTurn: p.usedItemThisTurn, ateBerry: p.ateBerry, hp: p.hp,
		volatiles: { ...p.volatiles }, boosts: { ...p.boosts }, abilityState: { ...p.abilityState },
	}));
}
function restoreMons(snap) {
	for (const s of snap) {
		Object.assign(s.p, {
			item: s.item, itemState: s.itemState, lastItem: s.lastItem,
			usedItemThisTurn: s.usedItemThisTurn, ateBerry: s.ateBerry, hp: s.hp,
			volatiles: s.volatiles, boosts: s.boosts, abilityState: s.abilityState,
		});
	}
}

/** The move as Showdown's useMove prepares it (ModifyType / ModifyMove events). */
function activeMove(battle, src, tgt, moveid) {
	let move = battle.dex.getActiveMove(moveid);
	battle.setActiveMove(move, src, tgt);
	battle.singleEvent('ModifyType', move, null, src, tgt, move, move);
	battle.singleEvent('ModifyMove', move, null, src, tgt, move, move);
	move = battle.runEvent('ModifyType', src, tgt, move, move);
	move = battle.runEvent('ModifyMove', src, tgt, move, move);
	move.hit = 1;
	return move;
}

function showdownDamage(battle, src, tgt, moveid, spread, crit, roll) {
	const snap = snapshotMons(battle);
	const move = activeMove(battle, src, tgt, moveid);
	move.spreadHit = spread;
	move.willCrit = crit;
	const origRand = battle.randomizer;
	battle.randomizer = bd => tr(tr(bd * (100 - roll)) / 100);
	let d;
	try {
		// The move's own TryImmunity (Endeavor) runs before damage in a battle.
		if (move.onTryImmunity && !move.onTryImmunity.call(battle, tgt, src, move)) return null;
		d = battle.actions.getDamage(src, tgt, move, true);
	} finally {
		battle.randomizer = origRand;
		battle.clearActiveMove();
		restoreMons(snap);
	}
	// Negative "damage" (Endeavor when the user has more HP) never happens in
	// a battle: the move fails its TryImmunity first.
	return typeof d === 'number' && !Number.isNaN(d) && d >= 0 ? d : null;
}

function makeScenario(rand, pool, sup, info, k) {
	const ta = pool[rand.pick(sup)];
	const tb = pool[rand.pick(sup)];
	// Attacker: a team member with a damaging move (or any, with a random move).
	const order = i => rand.shuffle([0, 1, 2, 3, 4, 5].filter(x => x !== i));
	const a = rand.int(6);
	const d = rand.int(6);
	const ra = order(a).slice(0, 3);
	const rd = order(d).slice(0, 3);
	const aSlot = rand.int(2);
	const dSlot = rand.int(2);
	const leadA = aSlot === 0 ? [a, ra[0]] : [ra[0], a];
	const leadD = dSlot === 0 ? [d, rd[0]] : [rd[0], d];
	const battle = L.newBattle(ta, tb, rand.seed());
	battle.choose('p1', L.teamChoice([...leadA, ra[1], ra[2]]));
	battle.choose('p2', L.teamChoice([...leadD, rd[1], rd[2]]));
	if (battle.ended || battle.requestState !== 'move') return null;
	const src = battle.p1.active[aSlot];
	// 10%: hit the ally instead of a foe.
	const allyHit = rand.chance(0.1);
	const tgt = allyHit ? battle.p1.active[1 - aSlot] : battle.p2.active[dSlot];
	if (!src || !tgt || src.fainted || tgt.fainted) return null;
	const srcAlly = battle.p1.active[1 - aSlot];
	const tgtAlly = allyHit ? null : battle.p2.active[1 - dSlot];
	const notes = [];
	const pickAb = () => rand.pick(info.abilities);
	const pickIt = () => rand.pick(info.items.filter(x => x && !L.dex.items.get(x).megaStone));
	const safe = (what, fn) => {
		try {
			fn();
			notes.push(what);
		} catch (e) {
			notes.push(`${what}(failed: ${e.message})`);
		}
	};
	// Focus mode: force a given ability or item on the attacker or defender.
	if (FOCUS) {
		const [kind, id] = FOCUS.split(':');
		const who = rand.chance(0.5) ? src : tgt;
		if (kind === 'ability') safe(`${who === src ? 'atk' : 'def'} ability ${id}`, () => who.setAbility(id, null, null, true));
		if (kind === 'item' && !L.dex.items.get(who.item).megaStone) safe(`${who === src ? 'atk' : 'def'} item ${id}`, () => who.setItem(id));
	}
	if (rand.chance(0.2)) { const x = pickAb(); safe(`atk ability ${x}`, () => src.setAbility(x, null, null, true)); }
	if (rand.chance(0.2)) { const x = pickAb(); safe(`def ability ${x}`, () => tgt.setAbility(x, null, null, true)); }
	if (tgtAlly && rand.chance(0.08)) safe('def ally friendguard', () => tgtAlly.setAbility('friendguard', null, null, true));
	if (srcAlly && rand.chance(0.04)) safe('atk ally fairyaura', () => srcAlly.setAbility('fairyaura', null, null, true));
	if (rand.chance(0.25) && !L.dex.items.get(src.item).megaStone) { const x = pickIt(); safe(`atk item ${x}`, () => src.setItem(x)); }
	if (rand.chance(0.25) && !L.dex.items.get(tgt.item).megaStone) { const x = pickIt(); safe(`def item ${x}`, () => tgt.setItem(x)); }
	if (src.canMegaEvo && rand.chance(0.5)) safe('atk mega', () => battle.actions.runMegaEvo(src));
	if (tgt.canMegaEvo && rand.chance(0.5)) safe('def mega', () => battle.actions.runMegaEvo(tgt));
	if (rand.chance(0.3)) { const w = rand.pick(WEATHERS); safe(`weather ${w}`, () => battle.field.setWeather(w, src)); }
	if (rand.chance(0.3)) { const t = rand.pick(TERRAINS); safe(`terrain ${t}`, () => battle.field.setTerrain(t, src)); }
	if (rand.chance(0.05)) safe('gravity', () => battle.field.addPseudoWeather('gravity', src));
	if (rand.chance(0.4)) {
		for (const s of ['atk', 'spa']) src.boosts[s] = rand.range(-2, 5);
		for (const s of ['def', 'spd', 'atk']) tgt.boosts[s] = rand.range(-2, 5);
		notes.push('boosts');
	}
	if (rand.chance(0.2)) {
		const sc = rand.pick(['reflect', 'lightscreen', 'auroraveil']);
		safe(`screen ${sc}`, () => tgt.side.addSideCondition(sc, tgt.side.foe.active[0] || tgt));
	}
	if (rand.chance(0.15)) {
		safe('helping hand', () => src.addVolatile('helpinghand', srcAlly || src));
		if (rand.chance(0.2)) safe('helping hand x2', () => src.addVolatile('helpinghand', srcAlly || src));
	}
	if (rand.chance(0.15)) {
		const st = rand.pick(['brn', 'brn', 'par', 'psn', 'tox', 'slp']);
		safe(`atk status ${st}`, () => { src.setStatus(st, null, null, true); });
	}
	if (rand.chance(0.1)) {
		const st = rand.pick(['brn', 'par', 'psn', 'slp']);
		safe(`def status ${st}`, () => { tgt.setStatus(st, null, null, true); });
	}
	if (rand.chance(0.4)) { src.hp = Math.max(1, rand.int(src.maxhp + 1)); notes.push(`atk hp ${src.hp}/${src.maxhp}`); }
	if (rand.chance(0.4)) { tgt.hp = Math.max(1, rand.int(tgt.maxhp + 1)); notes.push(`def hp ${tgt.hp}/${tgt.maxhp}`); }
	if (rand.chance(0.04)) safe('atk charge', () => src.addVolatile('charge'));
	if (rand.chance(0.04)) safe('def tarshot', () => tgt.addVolatile('tarshot'));
	if (rand.chance(0.04)) safe('def glaiverush', () => tgt.addVolatile('glaiverush'));
	if (rand.chance(0.03)) safe('def magnetrise', () => tgt.addVolatile('magnetrise'));
	if (rand.chance(0.03)) safe('def smackdown', () => tgt.addVolatile('smackdown'));
	if (rand.chance(0.03)) safe('def roost', () => tgt.addVolatile('roost'));
	if (rand.chance(0.03)) safe('atk focusenergy', () => src.addVolatile('focusenergy'));
	if (src.hasAbility('flashfire') && rand.chance(0.5)) safe('atk flashfire', () => src.addVolatile('flashfire'));
	if (src.fainted || tgt.fainted || !src.hp || !tgt.hp) return null;

	// The move.
	let moveid;
	const own = src.moveSlots.map(m => m.id).filter(isDamaging);
	if (own.length && rand.chance(0.75)) moveid = rand.pick(own);
	else moveid = rand.pick(info.moves.filter(isDamaging));
	const snap = snapshotMons(battle);
	const m = activeMove(battle, src, tgt, moveid);
	battle.clearActiveMove();
	restoreMons(snap);
	const spreadType = ['allAdjacent', 'allAdjacentFoes'].includes(m.target);
	const spread = spreadType && rand.chance(0.7);

	const state = L.exportState(battle, [ta, tb]);
	const expect = { normal: [], crit: [] };
	for (let roll = 0; roll < 16; roll++) {
		expect.normal.push(showdownDamage(battle, src, tgt, moveid, spread, false, roll));
		expect.crit.push(showdownDamage(battle, src, tgt, moveid, spread, true, roll));
	}
	const code = p => p.side.n * 2 + p.position;
	return {
		id: k,
		state,
		src: code(src),
		tgt: code(tgt),
		move: moveid,
		spread,
		expect,
		info: {
			atk: `${src.species.name} [${src.ability}] @${src.item || '-'} ${L.STATS.slice(1).map(s => src.boosts[s] || 0).join('/')}`,
			def: `${tgt.species.name} [${tgt.ability}] @${tgt.item || '-'} ${L.STATS.slice(1).map(s => tgt.boosts[s] || 0).join('/')}`,
			field: `${battle.field.weather || '-'} ${battle.field.terrain || '-'}`,
			notes: notes.join(', '),
		},
	};
}

function main() {
	const pool = L.loadPool();
	const info = L.runEngine(['supported'], [])[0];
	const sup = info.teams;
	const rand = new L.Rand(SEED);
	const scenarios = [];
	const gen = (count, focus) => {
		FOCUS = focus;
		let made = 0, attempts = 0;
		while (made < count && attempts < count * 5) {
			attempts++;
			let s = null;
			try {
				s = makeScenario(rand, pool, sup, info, scenarios.length);
			} catch (e) {
				if (attempts < 5) console.error('scenario error:', e.stack);
			}
			if (s) {
				if (focus) s.info.notes = `[focus ${focus}] ` + s.info.notes;
				scenarios.push(s);
				made++;
			}
		}
	};
	if (FOCUS_ALL) {
		for (const a of info.abilities) gen(PER, `ability:${a}`);
		for (const it of info.items) if (it && !L.dex.items.get(it).megaStone) gen(PER, `item:${it}`);
	} else {
		gen(N, FOCUS);
	}
	const results = L.runEngine(['damage'], scenarios.map(s => ({ id: s.id, state: s.state, src: s.src, tgt: s.tgt, move: s.move, spread: s.spread })));
	const byId = new Map(results.map(r => [r.id, r]));
	let ok = 0, bad = 0, skipped = 0, values = 0, badValues = 0;
	const skipReasons = {};
	const badByMove = {};
	const bads = [];
	for (const s of scenarios) {
		const r = byId.get(s.id);
		if (!r || r.err) {
			skipped++;
			const why = r ? r.err.replace(/[0-9]+/g, 'N') : 'missing';
			skipReasons[why] = (skipReasons[why] || 0) + 1;
			continue;
		}
		let mism = 0;
		for (const kind of ['normal', 'crit']) {
			for (let i = 0; i < 16; i++) {
				values++;
				if (s.expect[kind][i] !== r[kind][i]) mism++;
			}
		}
		if (mism) {
			bad++;
			badValues += mism;
			badByMove[s.move] = (badByMove[s.move] || 0) + 1;
			bads.push({ s, r });
		} else {
			ok++;
		}
	}
	console.log(`damage parity: ${scenarios.length} scenarios, ${ok} exact, ${bad} mismatched, ${skipped} skipped (engine could not load)`);
	console.log(`  values compared: ${values}, mismatched values: ${badValues}`);
	if (Object.keys(skipReasons).length) {
		console.log('  skip reasons:');
		for (const [k, v] of Object.entries(skipReasons).sort((a, b) => b[1] - a[1]).slice(0, 15)) console.log(`    ${v}  ${k}`);
	}
	if (bads.length) {
		console.log('  mismatches by move:', Object.entries(badByMove).sort((a, b) => b[1] - a[1]).slice(0, 20).map(([k, v]) => `${k}:${v}`).join(' '));
		for (const { s, r } of bads.slice(0, SHOW)) {
			console.log(`  #${s.id} ${s.move}${s.spread ? ' (spread)' : ''}: ${s.info.atk} -> ${s.info.def} | ${s.info.field} | ${s.info.notes}`);
			console.log(`     showdown normal ${JSON.stringify(s.expect.normal)}`);
			console.log(`     engine   normal ${JSON.stringify(r.normal)}`);
			if (JSON.stringify(s.expect.crit) !== JSON.stringify(r.crit)) {
				console.log(`     showdown crit   ${JSON.stringify(s.expect.crit)}`);
				console.log(`     engine   crit   ${JSON.stringify(r.crit)}`);
			}
		}
	}
	if (FIXTURE) {
		const body = scenarios.map(s => JSON.stringify({ id: s.id, state: s.state, src: s.src, tgt: s.tgt, move: s.move, spread: s.spread, expect: s.expect })).join('\n') + '\n';
		fs.writeFileSync(FIXTURE, FIXTURE.endsWith('.gz') ? zlib.gzipSync(body, { level: 9 }) : body);
		console.log(`wrote fixture ${FIXTURE}`);
	}
	process.exitCode = bad ? 1 : 0;
}

main();
