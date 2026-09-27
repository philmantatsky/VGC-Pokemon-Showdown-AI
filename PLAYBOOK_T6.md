# T6 playbook — draft 1 (2026-09-27; approved by the user: "i like them all, we can edit and make tweaks later")

Machine form: `data/playbook_t6.json`, read by `vgc_bench/src/playbook.py`
(cards in priority order: Fast Sun, Sun Room, Support Room, Water Room = default).
Tuning after replaying the planner over our 135 ladder opponents: Support Room
needs 4+ heavy physical attackers or 2+ Fake Out users (3+ flagged 65% of teams);
Fast Sun is marked experimental -- kept, but not picked until a local test backs it
(1-6 as a lead; the 7 teams its rule flags were won 5-2 with other leads).
**Draft 1.1 (2026-09-27 01:45, before any playbook result):** Sun Room now also
avoids Tailwind teams, as Plan A already said (Water Room is also the pick against
Tailwind teams) -- the machine rule had missed it, so 18 of our 25 ladder Tailwind
teams got Sun Room (flagged by the openings research, `OPENINGS_RESEARCH_T6.md`).
Turn-1 Fake Out now goes only where it can land: not a Ghost, not a grounded foe
under Psychic Terrain, not into Armor Tail / Dazzling / Queenly Majesty. The
planner's log names the cards it passed over and why.

The team's own game plans, written as a pilot would think about them. Each card:
when to pick it, the four and the leads, every Pokémon's job, the first turns, the
win condition, what beats it, and the fallback. Numbers in **[brackets]** are our
own record: ladder = the 135 T6-era ladder games (2026-09-23 → 09-26, all brains),
local = the 12,408 held-out battery games of 2026-09-26. Small ladder cells (under
~15 games) are hints, not proof.

This is the input the preview planner will use: it picks a card by its **pick when /
avoid when** rules plus a matchup check, and passes the card (roles, targets,
turn-1 script, fallback) to the battle brain.

## The team

| Pokémon | Set | Job | Speed (Champions) |
|---|---|---|---|
| Farigiraf | Sitrus, Armor Tail — Trick Room, Psychic, Helping Hand, Rain Dance | Trick Room setter; Armor Tail blocks their priority (Fake Out, Sucker Punch, Extreme Speed, Prankster moves at us) | 80 |
| Blastoise (Mega) | Fake Out, Water Spout, Water Pulse (Mega Launcher), Ice Beam | Turn-1 Fake Out, then the main Trick Room attacker | 88 |
| Torkoal | Charcoal, Drought — Eruption, Heat Wave, Weather Ball, Protect | Sun + the slowest Trick Room nuke | 36 |
| Incineroar | Passho, Intimidate — Fake Out, Flare Blitz, Parting Shot, Throat Chop | Intimidate / Fake Out / pivot support | 80 |
| Charizard (Mega Y) | Drought — Heat Wave, Solar Beam, Ancient Power, Protect | Fast sun attacker (4x weak to Rock) | 152 |
| Venusaur | Focus Sash, Chlorophyll — Sludge Bomb, Leaf Storm, Sleep Powder, Protect | Fast in sun (264); sleep + cleanup when Trick Room is over | 132 |

Under Trick Room we move in the order Torkoal → Farigiraf/Incineroar → Blastoise →
Venusaur → Charizard. Outside it, almost everything outspeeds Torkoal, Farigiraf,
Incineroar and Blastoise.

**Weather clash:** sun (Torkoal / Charizard) halves Blastoise's water moves; rain
(Farigiraf's Rain Dance, or theirs) halves our fire moves. Never have both on the
field fighting each other: sequence them (Blastoise first, Torkoal when Blastoise
is down or when switching modes).

---

## Plan A — Water Room (the default)

**Pick when:** no reason for another card. Also the pick against sand, Wide Guard,
Imprison, Psychic Terrain and Tailwind teams. [ladder 35-30 as a lead, 54% against
both "clean" teams and teams hostile to a Torkoal lead; local 94%]

- **Bring:** Blastoise, Farigiraf + Incineroar + Torkoal (Venusaur instead of
  Torkoal against sand or Wide Guard).
- **Lead:** Blastoise + Farigiraf.
- **Jobs:** Farigiraf sets and resets Trick Room, then Helping Hand / Rain Dance.
  Blastoise Mega-evolves turn 1 and is the main attacker. Incineroar is the pivot
  and Intimidate. Torkoal is the second nuke once Blastoise is down.
- **Turn 1:** Mega Blastoise Fake Out on their biggest threat to Farigiraf. If they
  lead a Trick Room / Tailwind setter, Fake Out THE SETTER instead. Farigiraf sets
  Trick Room (Armor Tail stops their Fake Out on it). Fake Out only a foe it can
  reach (not a Ghost, not grounded under Psychic Terrain, no Armor Tail / Dazzling /
  Queenly Majesty on their side); if neither, Blastoise attacks instead.
- **Turns 2–4:** Water Spout while Blastoise is healthy (Rain Dance only if rain is
  not up already, Helping Hand for a KO). Below ~half HP use Water Pulse / Ice Beam
  instead, unless Water Spout still finishes a low foe and chips the other.
- **Win condition:** two or three spread nukes under Trick Room, then Torkoal's
  Eruption to finish.
- **Beaten by:** fast Grass/Electric attackers before Trick Room (Rillaboom,
  Raichu, Archaludon's Electro Shot in rain), Kingambit's Throat Chop on Farigiraf,
  Wide Guard (switch to single-target moves once it is shown).
- **Fallback:** Trick Room blocked (Imprison / Taunt) → Blastoise fights at normal
  speed with Water Pulse / Ice Beam; Incineroar Parting Shots to bring the right
  attacker in.

## Plan B — Sun Room (narrow)

**Pick when:** a "clean" team — no sand setter, no Wide Guard, no Imprison, no
Psychic Terrain — and their team is weak to fire. [ladder 14-10 there]
**Avoid when:** any of those four. [7-19 against them — the worst record of any
opening we use] Also a Tailwind team: Water Room's job (Plan A).

- **Bring:** Farigiraf, Torkoal + Blastoise + Incineroar.
- **Lead:** Farigiraf + Torkoal.
- **Turn 1:** Trick Room + Torkoal Protect if they can hit it hard before it moves
  (their Fake Out is blocked by Armor Tail); Eruption otherwise.
- **Turns 2–4:** Eruption at full HP (Helping Hand for a KO), Heat Wave /
  Weather Ball once Torkoal is below ~half.
- **Win condition:** Eruption spam under Trick Room in sun.
- **Beaten by:** rain / sand taking the weather, Wide Guard, Rock Slide, Imprison.
- **Fallback:** Blastoise comes in when their water team takes the weather.

## Plan C — Support Room

**Pick when:** they lead or bring heavy physical attackers or Fake Out users
(Kingambit, Sneasler, Arcanine, Rillaboom, their Incineroar), or a Trick Room /
Tailwind setter we must Fake Out on turn 1 while Farigiraf sets up. [ladder 5-5;
local 88%, 96% against grassy Fake Out teams]

- **Bring:** Incineroar, Farigiraf + Blastoise + Torkoal.
- **Lead:** Incineroar + Farigiraf.
- **Turn 1:** Incineroar Fake Out on their setter or biggest threat, Farigiraf
  Trick Room; Intimidate softens their physical attackers.
- **Turn 2:** Parting Shot into Blastoise or Torkoal (whichever their team fears
  more) so the nuke comes in under Trick Room.
- **Win condition:** a safe Trick Room, then Blastoise / Torkoal spread damage.
- **Beaten by:** special attackers (Intimidate does nothing), Throat Chop users.
- **Fallback:** Incineroar Throat Chops sound users; Flare Blitz Steel / Grass.

## Plan D — Fast Sun (anti–Trick Room, experimental)

**Pick when:** THEY are the Trick Room team (two slow sweepers plus a setter) and
have no Rock Slide / Tailwind: play fast instead of fighting over the room.
[ladder 1-6 — including 0-4 against Tailwind and 0-3 against Wide Guard; keep
narrow until it proves itself]

- **Bring:** Charizard, Venusaur + Farigiraf + Torkoal.
- **Lead:** Charizard + Venusaur.
- **Turn 1:** Mega Charizard Y (sun) + Sleep Powder on their Trick Room setter;
  Protect Charizard if they have Rock Slide.
- **Win condition:** Heat Wave / Solar Beam / Sludge Bomb before their room goes up.
- **Beaten by:** Rock Slide (Charizard 4x), Tailwind, rain, Wide Guard.
- **Fallback:** if their Trick Room goes up anyway, Farigiraf reverses it.

---

## Rules for every plan (from our own mistakes)

1. **Their Trick Room:** Fake Out their setter on turn 1 instead of setting ours into
   theirs (ours cancelled theirs in two ladder games). If their room is already up
   and our slow team benefits, do not reverse it.
2. **Trick Room's last turn:** expect double Protect (it happened in ladder games 2
   and 9 last night). Don't throw attacks into it — use the turn to line up the
   next phase: switch in the attacker that is fast AFTER Trick Room (Venusaur in
   sun) or keep Farigiraf healthy to reset.
3. **Wide Guard shown or likely** (Aerodactyl, Pelipper, …): single-target moves
   only (Water Pulse, Weather Ball, Ice Beam).
4. **HP-based moves:** Water Spout / Eruption only at high HP, or to finish a low foe
   while chipping the other.
5. **Weather:** never Rain Dance into rain; don't put our sun and our rain on the
   field together; take the weather back when theirs hurts our current attacker.
6. **Doomed Pokémon:** when Trick Room is down and a slow Pokémon will be knocked
   out before it moves, Protect or switch instead of clicking an attack.
7. **Their setup:** a foe at +2 or more is the target, before anything else.

## What the planner reads at team preview

From their six (species, plus sets on an open sheet; usage data otherwise):
their weather setters (rain / sand / sun), Trick Room setters, Tailwind users,
Wide Guard users, Imprison users, Psychic Terrain, physical Fake Out / Intimidate
leads, and our type matchups against each of their Pokémon. The pick rules above
choose the card; the human-trained model's guess of THEIR lead decides turn-1
targets (Fake Out the setter, etc.).
