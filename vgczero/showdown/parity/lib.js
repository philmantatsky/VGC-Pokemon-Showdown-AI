'use strict';
// Shared helpers for the Showdown <-> vgczero parity tests.
//
// - load the compiled team pool and build Showdown battles from it
// - export a Showdown battle's full state into the JSON the engine loads
//   (`Battle::from_state_json`, engine/src/snapshot.rs)
// - extract comparable "outcome features" from a Showdown battle
// - random legal choices, and conversion of a choice to the engine's
//   per-slot action encoding (engine/src/actions.rs)
// - run the engine's `parity` binary

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const cp = require('child_process');

const HERE = __dirname;
const ROOT = path.join(HERE, '..', '..');
const PS_DIR = process.env.PS_DIR || path.join(HERE, '..', 'ps');
const FORMAT = process.env.FORMAT || 'gen9championsvgc2026regmc';
const ENGINE_DIR = path.join(ROOT, 'engine');
const PARITY_BIN = process.env.PARITY_BIN || path.join(ENGINE_DIR, 'target', 'release', 'parity');

const sim = require(path.join(PS_DIR, 'dist', 'sim'));
const { State } = require(path.join(PS_DIR, 'dist', 'sim', 'state'));
const { PRNG } = require(path.join(PS_DIR, 'dist', 'sim', 'prng'));
const { Battle, Dex, toID } = sim;
const dex = Dex.forFormat(FORMAT);

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];

// ---- small utilities --------------------------------------------------------

/** Deterministic PRNG for the harness itself (not the battles). */
function mulberry32(seed) {
	let a = seed >>> 0;
	return function () {
		a = (a + 0x6D2B79F5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
class Rand {
	constructor(seed) { this.r = mulberry32(seed); }
	f() { return this.r(); }
	int(n) { return Math.floor(this.r() * n); }
	range(lo, hi) { return lo + this.int(hi - lo); }
	pick(a) { return a[this.int(a.length)]; }
	chance(p) { return this.r() < p; }
	shuffle(a) {
		const b = a.slice();
		for (let i = b.length - 1; i > 0; i--) {
			const j = this.int(i + 1);
			[b[i], b[j]] = [b[j], b[i]];
		}
		return b;
	}
	seed() { return [this.int(0x10000), this.int(0x10000), this.int(0x10000), this.int(0x10000)]; }
}

function readJsonGz(file) {
	const buf = fs.readFileSync(file);
	return JSON.parse(file.endsWith('.gz') ? zlib.gunzipSync(buf).toString() : buf.toString());
}

let POOL = null;
function loadPool() {
	if (!POOL) POOL = readJsonGz(path.join(ROOT, 'data', 'teams_regmc.json.gz')).teams;
	return POOL;
}

// ---- the engine binary ------------------------------------------------------

function buildEngine() {
	const r = cp.spawnSync('cargo', ['build', '--release', '--bin', 'parity'], { cwd: ENGINE_DIR, stdio: 'inherit' });
	if (r.status !== 0) throw new Error('cargo build failed');
}

/** Run `parity <args...>` with `input` (array of objects -> JSON lines). Returns parsed JSON lines. */
function runEngine(args, input) {
	const body = input.map(x => JSON.stringify(x)).join('\n') + '\n';
	const r = cp.spawnSync(PARITY_BIN, args, { input: body, maxBuffer: 1 << 30 });
	if (r.status !== 0) {
		throw new Error(`parity ${args.join(' ')} failed (${r.status}):\n${r.stderr.toString().slice(0, 4000)}`);
	}
	const err = r.stderr.toString();
	if (err.trim()) process.stderr.write(err);
	return r.stdout.toString().split('\n').filter(Boolean).map(l => JSON.parse(l));
}

let SUPPORTED = null;
/** Pool indices of the teams the engine fully supports. */
function supportedTeams() {
	if (!SUPPORTED) SUPPORTED = runEngine(['supported'], [])[0].teams;
	return SUPPORTED;
}

// ---- Showdown battles -------------------------------------------------------

function toSet(m) {
	const sp = dex.species.get(m.species);
	const evs = {};
	STATS.forEach((s, i) => { evs[s] = m.evs[i]; });
	return {
		name: sp.baseSpecies === sp.name ? sp.name : sp.name,
		species: sp.name,
		item: m.item,
		ability: m.ability,
		moves: m.moves.slice(),
		nature: m.nature,
		gender: m.gender,
		level: m.level,
		evs,
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
	};
}

function newBattle(teamA, teamB, seed) {
	const b = new Battle({ formatid: FORMAT, seed: seed || [1, 2, 3, 4] });
	// vgczIdx: the set's index in the six-Pokemon team. It survives State
	// serialization (deserialized battles only keep the brought sets).
	b.setPlayer('p1', { name: 'p1', team: teamA.mons.map((m, i) => ({ ...toSet(m), vgczIdx: i })) });
	b.setPlayer('p2', { name: 'p2', team: teamB.mons.map((m, i) => ({ ...toSet(m), vgczIdx: i })) });
	return b;
}

/** Index of a Showdown Pokemon in its original six-Pokemon team. */
function teamIdx(p) {
	const i = p.set.vgczIdx;
	if (i === undefined) throw new Error('pokemon set without vgczIdx');
	return i;
}

/** Team preview choice "team abcd" from 4 team indices (first two lead). */
function teamChoice(order) {
	return 'team ' + order.map(i => i + 1).join('');
}

function serialize(battle) {
	return JSON.stringify(State.serializeBattle(battle));
}
function deserialize(str) {
	return State.deserializeBattle(str);
}
function reseed(battle, seed) {
	battle.prng = new PRNG(seed);
}

// ---- state export ------------------------------------------------------------

function slotCode(side, slot) {
	return side * 2 + slot;
}
function slotStrCode(s) {
	// 'p1a' -> 0, 'p1b' -> 1, 'p2a' -> 2, 'p2b' -> 3
	if (!s || typeof s !== 'string') return null;
	const m = /^p([12])([ab])/.exec(s);
	if (!m) return null;
	return (Number(m[1]) - 1) * 2 + (m[2] === 'a' ? 0 : 1);
}

/** Primitive fields of an effect state (volatile, side condition...). */
function effectState(st) {
	const out = {};
	for (const [k, v] of Object.entries(st)) {
		if (['id', 'target', 'effectOrder', 'sourceEffect', 'linkedPokemon', 'linkedStatus'].includes(k)) continue;
		if (k === 'source') {
			if (v && v.getSlot) {
				out.sourceSlot = slotStrCode(v.getSlot());
				out.sourceIdx = v.set.vgczIdx;
			}
			continue;
		}
		if (k === 'sourceSlot') { out.sourceSlot = slotStrCode(v); continue; }
		if (v === null || ['number', 'string', 'boolean'].includes(typeof v)) out[k] = v;
	}
	return out;
}

function exportMon(battle, side, p) {
	const st = {};
	if (p.status === 'slp' || p.status === 'frz') st.turns = p.statusState.time;
	if (p.status === 'tox') st.turns = p.statusState.stage || 0;
	const vols = {};
	for (const [id, s] of Object.entries(p.volatiles)) vols[id] = effectState(s);
	return {
		species: p.species.id,
		is_mega: !!p.species.isMega,
		ability: p.ability,
		base_ability: p.baseAbility,
		ability_state: effectState(p.abilityState),
		item: p.item,
		last_item: p.lastItem || '',
		ate_berry: !!p.ateBerry,
		item_knocked: !!p.itemKnockedOff,
		hp: p.hp,
		maxhp: p.maxhp,
		status: p.status === 'fnt' ? '' : (p.status || ''),
		status_turns: st.turns || 0,
		fainted: !!p.fainted,
		boosts: BOOSTS.map(b => p.boosts[b] || 0),
		types: p.types.slice(),
		added_type: p.addedType || '',
		stats: STATS.map(s => (s === 'hp' ? p.maxhp : p.storedStats[s])),
		weighthg: p.weighthg,
		moves: p.moveSlots.map(m => ({ id: m.id, pp: m.pp, maxpp: m.maxpp })),
		base_moves: p.baseMoveSlots.map(m => m.id),
		// A fainted Pokemon keeps its slot until replaced.
		active: side.active.includes(p),
		position: side.active.indexOf(p),
		volatiles: vols,
		last_move: p.lastMove ? p.lastMove.id : '',
		move_this_turn: p.moveThisTurn || '',
		move_last_turn_result: p.moveLastTurnResult === undefined ? null : p.moveLastTurnResult,
		move_this_turn_result: p.moveThisTurnResult === undefined ? null : p.moveThisTurnResult,
		active_turns: p.activeTurns,
		active_move_actions: p.activeMoveActions,
		times_attacked: p.timesAttacked,
		newly_switched: !!p.newlySwitched,
		stats_raised_this_turn: !!p.statsRaisedThisTurn,
		stats_lowered_this_turn: !!p.statsLoweredThisTurn,
		hurt_this_turn: p.hurtThisTurn === null || p.hurtThisTurn === undefined ? null : p.hurtThisTurn,
		switch_flag: typeof p.switchFlag === 'string' ? p.switchFlag : !!p.switchFlag,
		force_switch_flag: !!p.forceSwitchFlag,
		being_called_back: !!p.beingCalledBack,
		transformed: !!p.transformed,
		illusion: !!p.illusion,
		attacked_by: (p.attackedBy || []).map(a => ({
			source: slotStrCode(a.source.getSlot ? a.source.getSlot() : null),
			move: a.move, damage: a.damage, this_turn: !!a.thisTurn, slot: slotStrCode(a.slot),
			damage_value: a.damageValue === undefined ? null : a.damageValue,
		})),
		last_damage: p.lastDamage || 0,
		can_mega: !!p.canMegaEvo,
	};
}

function exportSide(battle, side) {
	const mons = new Array(6).fill(null);
	for (const p of side.pokemon) mons[teamIdx(p)] = exportMon(battle, side, p);
	const conds = {};
	for (const [id, s] of Object.entries(side.sideConditions)) conds[id] = effectState(s);
	const slotConds = side.slotConditions.map(sc => {
		const o = {};
		for (const [id, s] of Object.entries(sc)) o[id] = effectState(s);
		return o;
	});
	return {
		active: side.active.map(p => (p ? teamIdx(p) : -1)),
		order: side.pokemon.map(teamIdx),
		mons,
		conds,
		slot_conds: slotConds,
		total_fainted: side.totalFainted,
		fainted_last_turn: !!side.faintedLastTurn,
		fainted_this_turn: !!side.faintedThisTurn,
		mega_used: side.pokemon.some(p => p.species.isMega),
	};
}

function exportQueue(battle) {
	const out = [];
	for (const a of battle.queue.list) {
		const e = { choice: a.choice, order: a.order, priority: a.priority, speed: a.speed };
		if (a.pokemon) {
			e.pos = slotCode(a.pokemon.side.n, a.pokemon.position);
			e.mon = a.pokemon.isActive ? teamIdx(a.pokemon) : -1;
			e.mon_team_idx = teamIdx(a.pokemon);
		}
		if (a.choice === 'move') {
			e.move = a.move.id;
			e.target_loc = a.targetLoc;
			e.fractional_priority = a.fractionalPriority || 0;
			e.mega = !!a.mega;
		}
		if (a.choice === 'switch' || a.choice === 'instaswitch') e.target = teamIdx(a.target);
		out.push(e);
	}
	return out;
}

function exportField(battle) {
	const f = battle.field;
	const pw = {};
	for (const [id, s] of Object.entries(f.pseudoWeather)) pw[id] = effectState(s);
	return {
		weather: f.weather || '',
		weather_turns: f.weather ? (f.weatherState.duration || 0) : 0,
		terrain: f.terrain || '',
		terrain_turns: f.terrain ? (f.terrainState.duration || 0) : 0,
		pseudo_weather: pw,
	};
}

/** Full engine-loadable state of a Showdown battle. */
function exportState(battle, teams) {
	let phase = battle.requestState || (battle.ended ? 'ended' : '');
	const switchSlots = battle.sides.map(s => s.active.map(p => !!(p && p.switchFlag)));
	return {
		v: 1,
		format: FORMAT,
		teams: teams.map(t => ({ name: t.name, mons: t.mons })),
		turn: battle.turn,
		phase,
		mid_turn: battle.queue.list.some(a => a.choice === 'residual'),
		switch_slots: switchSlots,
		field: exportField(battle),
		sides: battle.sides.map(s => exportSide(battle, s)),
		queue: exportQueue(battle),
	};
}

// ---- outcome features --------------------------------------------------------

// Volatiles compared between the two simulators (Showdown id -> value fn).
const TRACKED_VOLATILES = {
	confusion: s => 'y',
	taunt: s => s.duration,
	encore: s => `${s.duration}`,
	disable: s => `${s.duration}`,
	yawn: s => s.duration,
	perishsong: s => s.duration,
	substitute: s => s.hp,
	leechseed: s => 'y',
	saltcure: s => 'y',
	curse: s => 'y',
	flashfire: s => 'y',
	torment: s => 'y',
	healblock: s => s.duration,
	magnetrise: s => s.duration,
	aquaring: s => 'y',
	ingrain: s => 'y',
	charge: s => 'y',
	stockpile: s => s.layers,
	attract: s => 'y',
	noretreat: s => 'y',
	laserfocus: s => s.duration,
	tarshot: s => 'y',
	smackdown: s => 'y',
	choicelock: s => s.move,
	stall: s => `${s.counter}/${s.duration}`,
	mustrecharge: s => 'y',
	twoturnmove: s => s.move,
	glaiverush: s => 'y',
	throatchop: s => s.duration,
	imprison: s => 'y',
	destinybond: s => 'y',
	partiallytrapped: s => s.duration,
	unburden: s => 'y',
};

function volString(p) {
	const parts = [];
	for (const [id, s] of Object.entries(p.volatiles)) {
		if (!(id in TRACKED_VOLATILES)) continue;
		parts.push(`${id}=${TRACKED_VOLATILES[id](s)}`);
	}
	// A twoturnmove without its move's volatile is spent (the attack happened;
	// it lingers until the residual).
	if (p.volatiles['twoturnmove'] && !p.volatiles[p.volatiles['twoturnmove'].move]) {
		const k = parts.findIndex(x => x.startsWith('twoturnmove='));
		if (k >= 0) parts.splice(k, 1);
	}
	// A choicelock without a Choice item is stale (removed at the next DisableMove).
	if (p.volatiles['choicelock'] && !p.getItem().isChoice) {
		const k = parts.findIndex(x => x.startsWith('choicelock='));
		if (k >= 0) parts.splice(k, 1);
	}
	// The engine keeps one crit stage for Focus Energy / Dragon Cheer.
	if (p.volatiles['focusenergy'] || p.volatiles['dragoncheer']) parts.push('critstage=y');
	return parts.sort().join(' ');
}

function boostString(boosts) {
	return BOOSTS.filter(b => boosts[b]).map(b => `${b}${boosts[b] > 0 ? '+' : ''}${boosts[b]}`).join(' ');
}

/** Comparable features of a Showdown battle at a decision point. */
function outcome(battle, logStart) {
	const f = {};
	if (battle.ended) {
		f.req = 'end:' + (battle.winner === 'p1' ? 0 : battle.winner === 'p2' ? 1 : 2);
	} else if (battle.requestState === 'switch') {
		const slots = [];
		battle.sides.forEach((s, si) => s.active.forEach((p, i) => { if (p && p.switchFlag) slots.push(`p${si + 1}${'ab'[i]}`); }));
		const mid = battle.queue.list.some(a => a.choice === 'residual');
		f.req = `switch${mid ? ':mid' : ''}:${slots.join(',')}`;
	} else {
		f.req = battle.requestState;
	}
	f.turn = battle.turn;
	for (const side of battle.sides) {
		const sid = side.id;
		for (const p of side.pokemon) {
			const k = `${sid}.${teamIdx(p)}`;
			f[`${k}.hp`] = p.hp;
			f[`${k}.st`] = p.fainted ? 'fnt' : (p.status || '-');
			// (A Pokemon that fainted as the battle ended keeps its old status.)
			if (!p.fainted && (p.status === 'slp' || p.status === 'frz')) f[`${k}.stt`] = p.statusState.time;
			if (!p.fainted && p.status === 'tox') f[`${k}.stt`] = p.statusState.stage || 0;
			f[`${k}.b`] = boostString(p.boosts);
			f[`${k}.pos`] = p.isActive && !p.fainted ? 'ab'[p.position] : '-';
			f[`${k}.it`] = p.item || '-';
			f[`${k}.ab`] = p.ability;
			f[`${k}.sp`] = p.species.id;
			// '???' (Burn Up, Double Shock) only matters when it is the only type.
			const tys = p.types.filter(t => t !== '???');
			f[`${k}.ty`] = tys.length ? tys.join('/') : '???';
			f[`${k}.v`] = p.isActive ? volString(p) : '';
			f[`${k}.pp`] = p.moveSlots.map(m => m.pp).join(',');
		}
		const sc = [];
		for (const [id, s] of Object.entries(side.sideConditions)) {
			const v = s.layers !== undefined ? s.layers : (s.duration !== undefined ? s.duration : 1);
			sc.push(`${id}:${v}`);
		}
		f[`${sid}.sc`] = sc.sort().join(' ');
		const slots = [];
		side.slotConditions.forEach((o, i) => {
			for (const [id, s] of Object.entries(o)) slots.push(`${'ab'[i]}.${id}:${s.duration || 1}`);
		});
		f[`${sid}.slot`] = slots.sort().join(' ');
	}
	const fd = battle.field;
	f['f.w'] = fd.weather ? `${fd.weather}:${fd.weatherState.duration}` : '-';
	f['f.t'] = fd.terrain ? `${fd.terrain}:${fd.terrainState.duration}` : '-';
	f['f.pw'] = Object.entries(fd.pseudoWeather).map(([id, s]) => `${id}:${s.duration}`).sort().join(' ');
	if (logStart !== undefined) {
		const order = [];
		for (const line of battle.log.slice(logStart)) {
			const m = /^\|(move|cant)\|(p[12][ab]):/.exec(line);
			if (m && m[1] === 'move' && (!line.includes('[from]') || line.includes('[from] lockedmove'))) order.push(m[2]);
		}
		f.order = order.join(',');
	}
	return f;
}

// ---- choices -----------------------------------------------------------------

// Move target types that take a chosen target in doubles.
function targetLocs(battle, pokemon, targetType, allyDamage) {
	const side = pokemon.side;
	const foe = side.foe;
	const pos = pokemon.position;
	const locs = [];
	switch (targetType) {
	case 'normal': case 'any': case 'adjacentFoe': {
		for (let i = 0; i < 2; i++) {
			const t = foe.active[i];
			if (t && !t.fainted) locs.push(i + 1);
		}
		if (!locs.length) locs.push(1);
		if (targetType !== 'adjacentFoe' && allyDamage) {
			const ally = side.active[1 - pos];
			if (ally && !ally.fainted) locs.push(-(2 - pos));
		}
		break;
	}
	case 'adjacentAllyOrSelf': {
		locs.push(-(pos + 1));
		const ally = side.active[1 - pos];
		if (ally && !ally.fainted) locs.push(-(2 - pos));
		break;
	}
	case 'adjacentAlly': {
		// The ally slot, even if it fainted (the move then fails).
		locs.push(-(2 - pos));
		break;
	}
	default:
		locs.push(0);
	}
	return locs;
}

/**
 * A random legal choice for a side from its current request, as a list of
 * per-slot structured actions: {kind:'move', slot, moveid, targetLoc, mega},
 * {kind:'switch', to (team idx), pos (side.pokemon position)}, {kind:'pass'}.
 */
function randomChoice(battle, side, rand, opts = {}) {
	const req = side.activeRequest;
	if (!req || req.wait) return null;
	if (req.teamPreview) {
		const order = rand.shuffle([0, 1, 2, 3, 4, 5]).slice(0, 4);
		return { preview: order };
	}
	const used = new Set();
	const acts = [];
	if (req.forceSwitch) {
		const bench = side.pokemon.filter(p => !p.isActive && !p.fainted);
		let avail = bench.length;
		const slotsOut = [{ kind: 'pass' }, { kind: 'pass' }];
		// With fewer replacements than flagged slots, which slot passes is a choice.
		for (const i of rand.shuffle([0, 1])) {
			if (!req.forceSwitch[i]) continue;
			const opts2 = bench.filter(p => !used.has(p));
			if (!opts2.length || avail <= 0) continue;
			const p = rand.pick(opts2);
			used.add(p);
			avail--;
			slotsOut[i] = { kind: 'switch', to: teamIdx(p), pos: p.position };
		}
		return { slots: slotsOut.slice(0, req.forceSwitch.length) };
	}
	let megaUsed = false;
	req.active.forEach((a, i) => {
		const pokemon = side.active[i];
		if (!pokemon || pokemon.fainted || !a) { acts.push({ kind: 'pass' }); return; }
		const options = [];
		const moves = a.moves;
		moves.forEach((m, mi) => {
			// Engine move slot: index in the moveset (0 for Struggle / Recharge).
			const slot = Math.max(0, pokemon.moveSlots.findIndex(x => x.id === m.id));
			// Hidden-disabled moves (Imprison) look enabled in the last active's
			// request but are still rejected.
			if (m.disabled || (pokemon.moveSlots[slot] && pokemon.moveSlots[slot].id === m.id && pokemon.moveSlots[slot].disabled)) return;
			if (m.pp === undefined) {
				// Locked move (charging, recharging, Struggle): no target to choose.
				options.push({ kind: 'move', slot, reqIndex: mi, moveid: m.id, targetLoc: 0, target: 'locked', mega: false });
				return;
			}
			const mv = dex.moves.get(m.id);
			let tt = m.target || mv.target;
			const allyDamage = opts.allyDamage && mv.category !== 'Status' ? true : mv.category === 'Status';
			for (const loc of targetLocs(battle, pokemon, tt, allyDamage)) {
				options.push({ kind: 'move', slot, reqIndex: mi, moveid: m.id, targetLoc: loc, target: tt, mega: false });
				if (a.canMegaEvo) options.push({ kind: 'move', slot, reqIndex: mi, moveid: m.id, targetLoc: loc, target: tt, mega: true });
			}
		});
		if (!a.trapped && !(opts.noSwitch)) {
			for (const p of side.pokemon) {
				if (!p.isActive && !p.fainted) options.push({ kind: 'switch', to: teamIdx(p), pos: p.position });
			}
		}
		let pool = options.filter(o => !(o.kind === 'switch' && used.has(o.to)) && !(o.mega && megaUsed));
		// Prefer moves (switching every turn makes for short, dull scenarios).
		const movesOnly = pool.filter(o => o.kind === 'move');
		if (movesOnly.length && rand.chance(opts.moveBias === undefined ? 0.85 : opts.moveBias)) pool = movesOnly;
		if (!pool.length) { acts.push({ kind: 'pass' }); return; }
		const c = rand.pick(pool);
		if (c.kind === 'switch') used.add(c.to);
		if (c.mega) megaUsed = true;
		acts.push(c);
	});
	return { slots: acts };
}

/** Showdown choice string from a structured choice. */
function choiceString(side, choice) {
	if (choice.preview) return teamChoice(choice.preview);
	return choice.slots.map(a => {
		if (a.kind === 'pass') return 'pass';
		if (a.kind === 'switch') {
			const p = side.pokemon.find(q => teamIdx(q) === a.to);
			return `switch ${p.position + 1}`;
		}
		let s = `move ${(a.reqIndex === undefined ? a.slot : a.reqIndex) + 1}`;
		if (a.targetLoc) s += ` ${a.targetLoc}`;
		if (a.mega) s += ' mega';
		return s;
	}).join(', ');
}

/** Engine action codes [slot0, slot1] (engine/src/actions.rs) from a structured choice. */
function engineAction(side, choice, slotIdx) {
	const a = choice.slots[slotIdx];
	if (a.kind === 'pass') return 0;
	if (a.kind === 'switch') return 1 + a.to;
	let t = 0;
	const loc = a.targetLoc || 0;
	if (['normal', 'any', 'adjacentFoe'].includes(a.target)) {
		t = loc > 0 ? loc - 1 : (loc < 0 ? 2 : 0);
	} else if (a.target === 'adjacentAllyOrSelf') {
		t = loc < 0 && (-loc - 1) !== slotIdx ? 2 : 0;
	}
	return 7 + (a.slot * 3 + t) * 2 + (a.mega ? 1 : 0);
}

function engineChoice(side, choice) {
	if (!choice) return { preview: 0, slots: [0, 0] };
	if (choice.preview) {
		const [a, b, c, d] = choice.preview;
		return { preview_order: [a, b, c, d], slots: [0, 0] };
	}
	return { slots: [engineAction(side, choice, 0), engineAction(side, choice, 1)] };
}

module.exports = {
	HERE, ROOT, PS_DIR, FORMAT, ENGINE_DIR, PARITY_BIN,
	sim, State, PRNG, Battle, Dex, dex, toID,
	STATS, BOOSTS, Rand, readJsonGz, loadPool, buildEngine, runEngine, supportedTeams,
	toSet, newBattle, teamIdx, teamChoice, serialize, deserialize, reseed,
	exportState, exportMon, outcome, randomChoice, choiceString, engineChoice, engineAction,
	slotStrCode, TRACKED_VOLATILES,
};
