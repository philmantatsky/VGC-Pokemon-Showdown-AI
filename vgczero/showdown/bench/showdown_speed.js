#!/usr/bin/env node
// Time upstream Showdown on random-player Reg M-C doubles games (in-process,
// no network), for comparison with the vgczero engine's `bench` binary.
// Usage: node vgczero/showdown/bench/showdown_speed.js [games]
'use strict';
const path = require('path');
const zlib = require('zlib');
const fs = require('fs');
const PS = process.env.PS_DIR || path.join(__dirname, '..', 'ps');
const { Teams } = require(path.join(PS, 'dist', 'sim'));
const BattleStreams = require(path.join(PS, 'dist', 'sim', 'battle-stream'));
const { RandomPlayerAI } = require(path.join(PS, 'dist', 'sim', 'tools', 'random-player-ai'));
const FORMAT = 'gen9championsvgc2026regmc';

const pool = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(__dirname, '..', '..', 'data', 'teams_regmc.json.gz'))));
const texts = fs.readdirSync(path.join(__dirname, '..', '..', '..', 'teams', 'reg_mc')).slice(0, 400)
	.map(f => Teams.pack(Teams.import(fs.readFileSync(path.join(__dirname, '..', '..', '..', 'teams', 'reg_mc', f), 'utf8'))));

async function game(i) {
	const streams = BattleStreams.getPlayerStreams(new BattleStreams.BattleStream());
	const p1 = new RandomPlayerAI(streams.p1, { seed: [i, 1, 2, 3] });
	const p2 = new RandomPlayerAI(streams.p2, { seed: [i, 4, 5, 6] });
	void p1.start();
	void p2.start();
	let turns = 0;
	const done = (async () => {
		for await (const chunk of streams.omniscient) {
			for (const line of chunk.split('\n')) {
				if (line.startsWith('|turn|')) turns = +line.slice(6);
			}
		}
	})();
	const spec = { formatid: FORMAT, seed: [i, 7, 8, 9] };
	void streams.omniscient.write(`>start ${JSON.stringify(spec)}\n` +
		`>player p1 ${JSON.stringify({ name: 'a', team: texts[i % texts.length] })}\n` +
		`>player p2 ${JSON.stringify({ name: 'b', team: texts[(i * 7 + 3) % texts.length] })}`);
	await done;
	return turns;
}

(async () => {
	const n = +(process.argv[2] || 200);
	const t0 = Date.now();
	let turns = 0;
	for (let i = 0; i < n; i++) turns += await game(i);
	const dt = (Date.now() - t0) / 1000;
	console.log(`showdown: ${n} games in ${dt.toFixed(2)}s = ${(n / dt).toFixed(1)} games/s, ${(turns / n).toFixed(1)} turns/game (1 thread)`);
	void pool;
})();
