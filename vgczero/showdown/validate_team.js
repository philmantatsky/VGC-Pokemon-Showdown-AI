#!/usr/bin/env node
// Validate teams (packed strings) with Showdown's TeamValidator.
// Input: an evolve.py population JSON (rows with team.packed) or a
// compiled teams file. Usage: node showdown/validate_team.js FILE
'use strict';
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const PS = process.env.PS_DIR || path.join(__dirname, 'ps');
const { Teams, TeamValidator } = require(path.join(PS, 'dist', 'sim'));
const FORMAT = process.env.FORMAT || 'gen9championsvgc2026regmc';
const file = process.argv[2];
let raw = fs.readFileSync(file);
if (file.endsWith('.gz')) raw = zlib.gunzipSync(raw);
const data = JSON.parse(raw);
const rows = Array.isArray(data) ? data.map(r => r.team || r) : data.teams;
const v = TeamValidator.get(FORMAT);
let bad = 0;
for (const t of rows) {
	const team = Teams.unpack(t.packed);
	const errors = v.validateTeam(team);
	if (errors) {
		bad++;
		console.log(`INVALID ${t.name}: ${errors.join('; ')}`);
	}
}
console.log(`${rows.length - bad}/${rows.length} valid`);
