#!/usr/bin/env node
/*
 * Persistent JSON-lines bridge to Pokemon Showdown's exact simulator.
 *
 * This deliberately accepts and returns Showdown's own serialized Battle state.
 * It does not attempt to translate a poke-env snapshot: that translation must pass
 * parity tests before live search is enabled. Keeping the boundary this strict avoids
 * quietly treating an approximate reconstruction as an exact state.
 */

'use strict';

const readline = require('readline');
const path = require('path');

const showdownRoot = path.resolve(__dirname, '..', 'pokemon-showdown');
const {Battle, Teams} = require(path.join(showdownRoot, 'dist', 'sim'));

function resultFor(battle, logStart = battle.log.length, emitted = []) {
	return {
		state: battle.toJSON(),
		requests: battle.sides.map(side => side.activeRequest),
		request_state: battle.requestState,
		turn: battle.turn,
		ended: battle.ended,
		winner: battle.winner || null,
		log: battle.log.slice(logStart),
		emitted,
	};
}

function parseTeam(request, side) {
	const text = request[`${side}_team_text`];
	if (typeof text === 'string') {
		const team = Teams.import(text);
		if (!team) throw new Error(`Could not parse ${side}_team_text`);
		return team;
	}
	const team = request[`${side}_team`];
	if (Array.isArray(team) || typeof team === 'string') return team;
	throw new Error(`Missing ${side}_team or ${side}_team_text`);
}

function create(request) {
	const battle = new Battle({
		formatid: request.formatid,
		seed: request.seed || [1, 2, 3, 4],
		strictChoices: true,
		p1: {name: request.p1_name || 'p1', team: parseTeam(request, 'p1')},
		p2: {name: request.p2_name || 'p2', team: parseTeam(request, 'p2')},
	});
	if (request.p1_preview || request.p2_preview) {
		if (!request.p1_preview || !request.p2_preview) {
			throw new Error('Both preview choices are required');
		}
		battle.makeChoices(request.p1_preview, request.p2_preview);
	}
	return resultFor(battle);
}

function simulateOne(state, p1Choice, p2Choice, rngSeed = null) {
	// State.deserialize mutates nested arrays/objects while restoring class
	// instances. simulate_batch reuses one parent state for many counterfactuals, so
	// deserializing it directly lets an earlier branch leak switches and effects into
	// later branches. A branch-local clone is mandatory for actual counterfactuals.
	const battle = Battle.fromJSON(structuredClone(state));
	if (rngSeed !== null) battle.resetRNG(rngSeed);
	const emitted = [];
	// A deserialized battle is intentionally inert until restart() supplies output.
	battle.restart((type, data) => emitted.push({type, data}));
	const before = battle.log.length;
	battle.makeChoices(p1Choice, p2Choice);
	return resultFor(battle, before, emitted);
}

function simulate(request) {
	if (!request.state) throw new Error('Missing serialized battle state');
	if (typeof request.p1_choice !== 'string' || typeof request.p2_choice !== 'string') {
		throw new Error('Both p1_choice and p2_choice are required');
	}
	return simulateOne(
		request.state, request.p1_choice, request.p2_choice, request.rng_seed ?? null
	);
}

function simulateBatch(request) {
	if (!request.state) throw new Error('Missing serialized battle state');
	if (!Array.isArray(request.branches)) throw new Error('Missing branches array');
	return request.branches.map(branch => {
		if (typeof branch.p1_choice !== 'string' || typeof branch.p2_choice !== 'string') {
			throw new Error('Every branch requires p1_choice and p2_choice');
		}
		return simulateOne(
			request.state, branch.p1_choice, branch.p2_choice, branch.rng_seed ?? null
		);
	});
}

function id(value) {
	return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function conditionState(battle, identifier, target, previous, data = {}) {
	const state = previous || battle.initEffectState({id: identifier, target});
	if (data.duration !== null && data.duration !== undefined) {
		state.duration = Number(data.duration);
	}
	if (data.move) state.move = id(data.move);
	if (data.target_loc !== null && data.target_loc !== undefined) {
		state.targetLoc = Number(data.target_loc);
	}
	return state;
}

function applyConditions(battle, target, current, requested = {}) {
	for (const identifier of Object.keys(current)) {
		if (!(identifier in requested)) delete current[identifier];
	}
	for (const [identifier, data] of Object.entries(requested)) {
		current[identifier] = conditionState(
			battle, identifier, target, current[identifier], data
		);
	}
}

function findPokemon(side, record) {
	const nickname = id(record.nickname);
	const species = id(record.base_species || record.species);
	let matches = side.pokemon.filter(pokemon => id(pokemon.set?.name) === nickname);
	if (!matches.length) {
		matches = side.pokemon.filter(pokemon => (
			id(pokemon.baseSpecies?.id || pokemon.set?.species) === species
		));
	}
	// A divergent shadow can temporarily contain two matching party records after
	// its switch order differs from the live battle.  The live active slot is the
	// authoritative identity in that case.  Using it lets reconciliation repair the
	// roster instead of discarding every determinization and spending the turn on a
	// champion fallback.
	if (matches.length > 1 && record.active_slot !== null && record.active_slot !== undefined) {
		const active = side.active[Number(record.active_slot)];
		if (active && matches.includes(active)) return active;
	}
	if (matches.length !== 1) {
		throw new Error(`Could not uniquely map snapshot Pokemon ${nickname || species}`);
	}
	return matches[0];
}

function applyPokemon(battle, pokemon, record) {
	if (record.species && id(pokemon.species?.id) !== id(record.species)) {
		pokemon.setSpecies(battle.dex.species.get(record.species), null, true);
	}
	if (record.maxhp !== null && record.maxhp !== undefined) {
		pokemon.maxhp = Number(record.maxhp);
		pokemon.baseMaxhp = Number(record.maxhp);
	}
	if (record.hp !== null && record.hp !== undefined) {
		pokemon.hp = Math.max(0, Math.min(pokemon.maxhp, Number(record.hp)));
	} else if (record.hp_fraction !== null && record.hp_fraction !== undefined) {
		pokemon.hp = Math.max(
			0, Math.min(pokemon.maxhp, Math.round(Number(record.hp_fraction) * pokemon.maxhp))
		);
	}
	pokemon.fainted = !!record.fainted || pokemon.hp <= 0;
	pokemon.faintQueued = false;
	pokemon.forceSwitchFlag = false;
	pokemon.switchFlag = false;
	pokemon.beingCalledBack = false;
	pokemon.status = record.status || '';
	pokemon.boosts = {...pokemon.boosts, ...(record.boosts || {})};
	if (record.ability) pokemon.ability = id(record.ability);
	if (record.item !== null && record.item !== undefined) pokemon.item = id(record.item);
	pokemon.activeTurns = record.first_turn ? 0 : Math.max(1, pokemon.activeTurns || 1);
	pokemon.activeMoveActions = record.first_turn ? 0 : Math.max(1, pokemon.activeMoveActions || 1);
	if (record.last_move) {
		// Encore and several other mechanics expect an ActiveMove, not the immutable
		// Dex Move returned by moves.get().  A plain Move survives one request but
		// loses ActiveMove-only fields when the reconciled state is serialized.
		const lastMove = battle.dex.getActiveMove(record.last_move);
		pokemon.lastMove = lastMove;
		pokemon.lastMoveUsed = lastMove;
	} else {
		pokemon.lastMove = null;
		pokemon.lastMoveUsed = null;
	}
	applyConditions(battle, pokemon, pokemon.volatiles, record.effects || {});
	const twoTurn = record.effects?.twoturnmove;
	if (twoTurn?.target_loc !== null && twoTurn?.target_loc !== undefined) {
		pokemon.lastMoveTargetLoc = Number(twoTurn.target_loc);
		const moveVolatile = pokemon.volatiles[id(twoTurn.move)];
		if (moveVolatile) moveVolatile.targetLoc = Number(twoTurn.target_loc);
	}
	for (const move of record.moves || []) {
		const slot = pokemon.moveSlots.find(candidate => candidate.id === id(move.id));
		if (!slot) continue;
		if (move.pp !== null && move.pp !== undefined) slot.pp = Number(move.pp);
		if (move.disabled !== undefined) slot.disabled = !!move.disabled;
	}
}

function reconcile(request) {
	if (!request.state || !request.snapshot) throw new Error('Missing state or snapshot');
	const battle = Battle.fromJSON(structuredClone(request.state));
	battle.restart(() => {});
	const snapshot = request.snapshot;
	battle.turn = Number(snapshot.turn || 0);
	for (let sideIndex = 0; sideIndex < 2; sideIndex++) {
		const side = battle.sides[sideIndex];
		const sideSnapshot = snapshot.sides[sideIndex];
		const activeRecords = sideSnapshot.pokemon
			.filter(record => record.active_slot !== null && record.active_slot !== undefined)
			.sort((a, b) => a.active_slot - b.active_slot);
		for (const pokemon of side.pokemon) pokemon.isActive = false;
		for (const record of activeRecords) {
			const pokemon = findPokemon(side, record);
			const slot = Number(record.active_slot);
			const previous = side.active[slot];
			if (previous !== pokemon) {
				const targetPosition = previous ? previous.position : slot;
				const oldPosition = pokemon.position;
				side.pokemon[targetPosition] = pokemon;
				pokemon.position = targetPosition;
				if (previous) {
					side.pokemon[oldPosition] = previous;
					previous.position = oldPosition;
				}
			}
			side.active[slot] = pokemon;
			pokemon.isActive = true;
		}
		for (const record of sideSnapshot.pokemon) {
			applyPokemon(battle, findPokemon(side, record), record);
		}
		// Once-per-battle mechanics belong to the side, not merely the Pokemon
		// whose current forme happens to reveal their use. A divergent shadow may
		// still have Charizard.canMegaEvo after live Floette already Mega Evolved;
		// clear every related request capability before makeRequest() rebuilds the
		// legal choices. The same repair keeps this bridge safe for formats with Z,
		// Dynamax, or Tera resources.
		const mechanicUsage = sideSnapshot.mechanic_usage || {};
		if (mechanicUsage.mega_used) {
			for (const pokemon of side.pokemon) {
				pokemon.canMegaEvo = false;
				pokemon.canMegaEvoX = false;
				pokemon.canMegaEvoY = false;
				pokemon.canUltraBurst = null;
			}
		}
		if (mechanicUsage.z_move_used) side.zMoveUsed = true;
		if (mechanicUsage.dynamax_used) side.dynamaxUsed = true;
		if (mechanicUsage.tera_used) {
			for (const pokemon of side.pokemon) pokemon.canTerastallize = null;
		}
		if (snapshot.request_state === 'switch') {
			const explicit = sideSnapshot.force_switch;
			const canSwitch = side.pokemon.some(pokemon => !pokemon.isActive && !pokemon.fainted);
			for (let slot = 0; slot < side.active.length; slot++) {
				const pokemon = side.active[slot];
				if (!pokemon) continue;
				pokemon.switchFlag = Array.isArray(explicit) ?
					!!explicit[slot] : (!!pokemon.fainted && canSwitch);
			}
		}
		side.pokemonLeft = side.pokemon.filter(pokemon => !pokemon.fainted).length;
		side.totalFainted = side.pokemon.length - side.pokemonLeft;
		applyConditions(
			battle, side, side.sideConditions, sideSnapshot.side_conditions || {}
		);
	}
	battle.field.weather = snapshot.weather || '';
	battle.field.terrain = snapshot.terrain || '';
	applyConditions(
		battle, battle.field, battle.field.pseudoWeather, snapshot.pseudo_weather || {}
	);
	// A serialized state includes request-time caches.  They describe the shadow's
	// old RNG branch and makeRequest() does not recompute them by itself.  Rebuild
	// the public choice state after HP, boosts, volatiles, abilities, and fields have
	// all been repaired.  This catches dynamic DisableMove rules (for example
	// Strength Sap with no legal Attack drop) and removes stale Solar Beam trapping.
	for (const side of battle.sides) {
		for (const pokemon of side.active) {
			if (!pokemon || pokemon.fainted) continue;
			pokemon.trapped = false;
			pokemon.maybeTrapped = false;
			pokemon.maybeDisabled = false;
			pokemon.maybeLocked = false;
			for (const moveSlot of pokemon.moveSlots) {
				moveSlot.disabled = false;
				moveSlot.disabledSource = '';
			}
			battle.runEvent('DisableMove', pokemon);
			for (const moveSlot of pokemon.moveSlots) {
				const activeMove = battle.dex.getActiveMove(moveSlot.id);
				battle.singleEvent('DisableMove', activeMove, null, pokemon);
				if (activeMove.flags['cantusetwice'] && pokemon.lastMove?.id === moveSlot.id) {
					pokemon.disableMove(pokemon.lastMove.id);
				}
			}
			battle.runEvent('TrapPokemon', pokemon);
			if (!pokemon.knownType || battle.dex.getImmunity('trapped', pokemon)) {
				battle.runEvent('MaybeTrapPokemon', pokemon);
			}
		}
	}
	// The controlled player's request is the final authority for information the
	// client receives directly. Reapply its disabled/trapping flags after dynamic
	// event recomputation, which otherwise clears Shadow Tag and forced-move details.
	for (let sideIndex = 0; sideIndex < battle.sides.length; sideIndex++) {
		const side = battle.sides[sideIndex];
		const sideSnapshot = snapshot.sides[sideIndex];
		for (const record of sideSnapshot.pokemon || []) {
			if (record.active_slot === null || record.active_slot === undefined) continue;
			const pokemon = findPokemon(side, record);
			if (record.trapped === true) pokemon.trapped = true;
			for (const move of record.moves || []) {
				if (move.disabled === undefined) continue;
				const slot = pokemon.moveSlots.find(candidate => candidate.id === id(move.id));
				if (slot) slot.disabled = !!move.disabled;
			}
		}
	}
	battle.queue.clear();
	battle.faintQueue = [];
	battle.ended = false;
	battle.winner = '';
	battle.makeRequest(snapshot.request_state || 'move');
	// The exact state now contains the reconciled mechanics, while its historical log
	// still describes the shadow's old RNG outcomes. Preserve the public snapshot as
	// a private bridge marker so the policy adapter sees the same repaired view.
	battle.log.push(`|vgcsnapshot|${JSON.stringify(snapshot)}`);
	return resultFor(battle);
}

function switchAtoms(request) {
	const side = request.side;
	if (!side || !Array.isArray(side.pokemon)) return [];
	const out = [];
	for (let i = 0; i < side.pokemon.length; i++) {
		const pokemon = side.pokemon[i];
		if (!pokemon.active && !String(pokemon.condition || '').includes(' fnt')) {
			out.push(`switch ${i + 1}`);
		}
	}
	return out;
}

function moveAtoms(active, slot, switches, pokemon = null) {
	// Showdown can retain an `active` request entry for a slot whose Pokemon fainted
	// when that side has no replacement left. The only legal input for that slot is
	// pass; enumerating the stale moves makes every joint choice fail validation.
	if (!active || !pokemon || pokemon.fainted) return ['pass'];
	const out = [];
	const events = [''];
	if (active.canMegaEvo) events.push(' mega');
	if (active.canUltraBurst) events.push(' ultra');
	if (active.canDynamax) events.push(' dynamax');
	if (active.canTerastallize) events.push(' terastallize');
	for (let i = 0; i < (active.moves || []).length; i++) {
		const move = active.moves[i];
		if (move.disabled) continue;
		let targets = [''];
		if (move.target === 'normal' || move.target === 'any') {
			targets = [slot === 0 ? ' -2' : ' -1', ' +1', ' +2'];
		} else if (move.target === 'adjacentFoe') {
			targets = [' +1', ' +2'];
		} else if (move.target === 'adjacentAlly') {
			targets = [slot === 0 ? ' -2' : ' -1'];
		} else if (move.target === 'adjacentAllyOrSelf') {
			targets = [' -1', ' -2'];
		}
		const moveEvents = [...events];
		if (Array.isArray(active.canZMove) && active.canZMove[i]) moveEvents.push(' zmove');
		for (const target of targets) {
			for (const event of moveEvents) out.push(`move ${i + 1}${target}${event}`);
		}
	}
	if (!active.trapped) out.push(...switches);
	return out.length ? out : ['pass'];
}

function enumerateChoices(request) {
	if (!request.state) throw new Error('Missing serialized battle state');
	if (request.side !== 'p1' && request.side !== 'p2') {
		throw new Error('side must be p1 or p2');
	}
	const battle = Battle.fromJSON(structuredClone(request.state));
	battle.restart(() => {});
	const side = battle.getSide(request.side);
	const activeRequest = side.activeRequest;
	if (battle.ended) return [];
	// During a one-sided forced replacement the other side submits no choice. The
	// empty string tells makeChoices to leave that side alone while still allowing a
	// complete (p1, p2) branch tuple in the planner.
	if (!activeRequest || activeRequest.wait) return [''];
	if (activeRequest.teamPreview) {
		const size = activeRequest.maxTeamSize || side.pokemon.length;
		const choices = [];
		const walkTeam = (picked) => {
			if (picked.length === size) {
				const input = `team ${picked.join(', ')}`;
				try {
					if (side.choose(input) && side.isChoiceDone()) choices.push(side.getChoice());
				} finally {
					side.clearChoice();
				}
				return;
			}
			for (let i = 1; i <= side.pokemon.length; i++) {
				if (!picked.includes(i)) walkTeam([...picked, i]);
			}
		};
		walkTeam([]);
		return [...new Set(choices)];
	}

	const switches = switchAtoms(activeRequest);
	let perSlot;
	if (activeRequest.forceSwitch) {
		perSlot = activeRequest.forceSwitch.map(forced => forced ? [...switches, 'pass'] : ['pass']);
	} else if (activeRequest.active) {
		perSlot = side.active.map((pokemon, slot) => (
			moveAtoms(activeRequest.active[slot], slot, switches, pokemon)
		));
	} else {
		return [];
	}

	const inputs = [];
	const walk = (slot, chosen) => {
		if (slot === perSlot.length) {
			inputs.push(chosen.join(', '));
			return;
		}
		for (const atom of perSlot[slot]) walk(slot + 1, [...chosen, atom]);
	};
	walk(0, []);

	const legal = new Set();
	for (const input of inputs) {
		try {
			if (side.choose(input) && side.isChoiceDone()) legal.add(side.getChoice());
		} catch (_) {
			// strictChoices throws for invalid targets, duplicate switches, and using
			// two once-per-battle mechanics. Those are precisely what we are filtering.
		} finally {
			side.clearChoice();
		}
	}
	return [...legal];
}

function handle(request) {
	switch (request.op) {
	case 'ping':
		return {ok: true, backend: 'pokemon-showdown', exact: true};
	case 'create':
		return create(request);
	case 'simulate':
		return simulate(request);
	case 'simulate_batch':
		return simulateBatch(request);
	case 'choices':
		return enumerateChoices(request);
	case 'reconcile':
		return reconcile(request);
	default:
		throw new Error(`Unknown operation: ${request.op}`);
	}
}

const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
rl.on('line', line => {
	if (!line.trim()) return;
	let id = null;
	try {
		const request = JSON.parse(line);
		id = request.id ?? null;
		process.stdout.write(`${JSON.stringify({id, ok: true, result: handle(request)})}\n`);
	} catch (error) {
		process.stdout.write(`${JSON.stringify({
			id,
			ok: false,
			// Preserve the stack in protocol errors. Rare reconciliation failures are
			// otherwise impossible to localize because the persistent worker owns all
			// Showdown internals and only its first-line message reaches Python.
			error: error instanceof Error ? (error.stack || error.message) : String(error),
		})}\n`);
	}
});
