# T6 openings research (2026-09-27)

The user: "research different openings for our bot to play + first move strategies
that go alongside it or like against which opponents". This is the evidence for a
draft 2 of `PLAYBOOK_T6.md`. Nothing here is wired into the bot or changes the
running playbook evaluation (pre-registered 00:55); every new opening needs practice
and a local test before ladder (section 8).

> **Update, 2026-09-27 evening (the playbook session's results, PROJECT_STATUS 02:53
> → 18:05):** locally the planner's own cards lost — Support Room -14.6pp, Sun Room
> -5.2, Water Room -2.7 — so the "widen Support Room" proposal below is contradicted
> as a general card. After a practice cycle, Water Room + the turn-1 script is even
> with T6tac, the only playbook configuration that holds up; without the script the
> practised brain's own turn 1 is worse. Ladder: T6tac + Water Room 11-4 (played out
> 4-4); practised brain + Water Room 6-9. Team and moveset follow-up:
> `TEAM_REVIEW_T6.md`.

**Evidence** (numbers in [brackets]: W-L, 95% Wilson interval when shown):
- **Human games:** 7,432 Reg M-C Showdown games (Bo1 + Bo3, 09-09 → 09-20,
  `battle_logs_top_mc_*`) + 268 from T6-like pilots. Their ratings (median ~1200,
  90th pct ~1390) match our ladder band (our opponents: median ~1175). "vsElo" =
  score above what the two ratings predict (removes most of the skill gap).
- **Our ladder:** the 134 T6-era games (09-23 → 09-26, all brains; 84 since the
  human-opening retraining), `ladder_replays_mc_deployed_T6*`.
- **Web (two research agents, read-only):** Baltimore Regional (Reg M-C, 1,081
  Masters, 09-18..20; Limitless), Pikalytics tournament sheets, 1,248 top-ladder
  replays (1429-1678), the replays of the two strongest pilots of our exact six
  (thruxy ~1400-1650, dksnnfud ~1400-1600), Champions write-ups (Kim's PJCS 2026
  qualifier report, 3rdAubergine, StarRaikou, Game8), Smogon's Champions mechanics
  thread. Links in section 9.
- **Engine:** the Champions mod of our bundled simulator (`data/mods/champions`).

Caveats: records by lead are correlational (a lead is chosen for the matchup); cells
under ~30 games are hints; uploaded replays lean toward wins; none of this is a
win-rate claim for our bot. Scripts and full tables: `results_analysis/openings_20260927/`.

---

## 1. Findings

1. **Keep Water Room (Blastoise + Farigiraf) as the default, with the experts'
   script:** turn 1 Mega + Fake Out the most dangerous *reachable* foe + Trick Room;
   turn 2 **Rain Dance + Water Spout** (or an attack + Ice Beam on Rillaboom /
   Salamence); Torkoal comes in on the first faint while the room is still up.
   [ours 35-30: 12-4 vs Trick Room, 8-3 vs Tailwind]; [humans with this lead 15-10,
   +8.9pp, and 10-3 with Fake Out + Trick Room]; thruxy and dksnnfud (dksnnfud's
   Blastoise / Farigiraf sets are exactly ours) lead it by default and went 10-1 with
   the line in their uploaded replays; three independent Champions write-ups give
   the same turn 1 / turn 2. Our Blastoise: Fake Out turn 1 [20-11], Water Spout
   turn 1 [9-13] (confounded: Water Spout is its move when Fake Out can't land).
2. **It loses three matchups:** Psychic Terrain [4-8 with Water Room; 5-11 when they
   lead Indeedee], sand [1-9 when Tyranitar leads], rain [8-14];
   also Wide Guard rosters [12-25]. **Rain + Archaludon is structural, not
   piloting:** Electro Shot (instant in rain) one-shots Mega Blastoise and
   Archaludon resists Water; the expert pilots went 1-3 against it and no source
   has a winning line for this team.
3. **Fake Out cannot reach most Trick Room leads:** Farigiraf (Armor Tail shields its
   partner too), Indeedee (its Psychic Terrain), Ghost setters (Sinistcha,
   Cofagrigus, Dusclops). On the top ladder Armor Tail blanked 14 Fake Outs and
   Psychic Terrain 20. "Fake Out their setter" only works on Hatterene, Armarouge,
   Oranguru, Porygon2 or Gardevoir *without* Indeedee beside them.
4. **Trick Room mirrors cancel:** both clicking on turn 1 = no room. Our Farigiraf
   (80) moves after Indeedee / Armarouge / Sinistcha / Gardevoir setters and before
   Hatterene / Cofagrigus / Dusclops / Slowking. Humans who got the room alone
   [72-56, +7.4pp]; our local mirror: yielding the room loses (39.1%); we are [12-4]
   vs Trick Room teams while contesting. Untested third line: Kim (PJCS qualifier,
   9-1, Reg M-A) led Blastoise + Farigiraf vs Trick Room teams with **Helping Hand +
   Water Spout, no Trick Room** (Armor Tail blanks their Fake Out; the nuke lands
   before the room matters).
5. **Trick Room beats Tailwind on turn 1:** when one side set the room and the other
   Tailwind on turn 1, the room side went [17-2]. With Armor Tail blocking Fake Out
   and Prankster Taunt / Encore, the room is nearly unstoppable there.
6. **Two untried openings have real human support** (both need practice first):
   **Charizard + Farigiraf** [humans 24-15, +7.2pp; 7-2 vs Tailwind] and
   **Charizard + Incineroar** [23-18, +5.4pp; 8-4 vs Trick Room; reportedly the
   lead of Arsal Puri's winning Indianapolis team, Reg M-A]. gankyburner (a Salamence
   version of our roster, [36-8]) led **Farigiraf + Incineroar** most [9-2].
7. **Torkoal is a closer, not a lead:** the two strong pilots of our six never led
   it; it comes in on the first faint under Trick Room, and opponents Protect its
   first Eruption.
8. **Flag for the playbook evaluation — addressed in draft 1.1 (`c7b76b2`):**
   replayed over our 134 ladder opponents, draft 1 picked Sun Room (Farigiraf +
   Torkoal) for 39 (29%), including 18 of the 25 Tailwind teams, although since the
   human-opening retraining the brain has led Farigiraf + Torkoal 0 times in 84
   ladder games and unpractised leads collapsed before (section 8). Draft 1.1 adds
   Tailwind to Sun Room's avoid list (the approved Plan A text already said so) and
   switches off, for the ladder trial, any card whose battery cells lose >= 5pp over
   500+ games. Now: Water 74 / Support 44 / Sun 16 of our 134 opponents.
9. **An opponent to expect:** in a 15-hour mid-ladder sample (1150-1399) ending
   09-27, four accounts (SC-Control, SC-SME, Scorecard-Pokemon, kevdan42; 131-164
   uploads each) were 31% of team-sides, all with the Baltimore runner-up team:
   Charizard-Y, Mega Golisopod, Politoed, Archaludon, Grimmsnarl (screens),
   Farigiraf (Trick Room). That is rain + Archaludon plus a Farigiraf that blanks our
   Fake Out. Not met in our 134 games yet.
10. **Follow Me leads punish focus fire:** vs Gardevoir + Indeedee-F, sides that
    double-targeted one foe went [6-35]; any spread move [74-70] vs none [101-146].

## 2. The field

### 2a. Archetypes (corpus rosters / our 134 opponents)

| Archetype | Humans | Ours | Our record |
|---|---|---|---|
| Tailwind offense | 24.5% | 18.7% | Water Room 8-3, Sun Room 5-5 |
| Trick Room | 22.5% | 19.4% | Water Room 12-4 |
| Rain | 10.9% | 12.7% | rain setter on roster 8-14 |
| Sand | 10.6% | 14.9% | sand setter 6-16; sand setter led 1-9 |
| Psychic Terrain | 9.9% | 11.2% | Indeedee led 5-11 |
| Balance | 9.6% | 11.2% | Water Room 2-4 |
| Sun | 8.6% | 5.2% | sun setter 8-10 |
| Snow | 2.8% | 3.7% | 5-1 |

On rosters: a Tailwind user 63% / 59%, a Trick Room setter 43% / 55%, Rock Slide
36% / 39%, **Wide Guard 18% / 28%, Imprison 5% / 13%**. Baltimore Day 2 and the top
ladder agree on the top dozen (Rillaboom ~50%, Sneasler ~45%, Mega Salamence ~35%,
Gholdengo, Incineroar, Kingambit, Arcanine-H, Mega Raichu-Y, Indeedee ~25%); sand
won Baltimore (Mega Salamence + Mega Tyranitar + Excadrill + Scarf Indeedee).
Top-ladder turn-1 base rates: Fake Out on turn 1 in 30% of games, Trick Room in 14%,
Tailwind in 10%; Protect on 19% of sides; Psychic or Grassy Terrain up at turn 0 in
~40% of games; a Mega led Mega-Evolves on turn 1 77% of the time. Facing a Farigiraf
lead, opponents most often Protect (13%) or Scarf-Trick it (6%), and 67% of their
attacks go into Farigiraf. Leads are spread thin (the top pair is 2.5% of sides),
so plan by archetype, not by memorised lead pairs.

### 2b. The leads we meet and their turn 1 (share of the times they lead)

| Lead | Turn 1 | For us |
|---|---|---|
| Sneasler | Fake Out 27%, Close Combat 19%, Dire Claw 13% | Fake Out / priority blanked by Armor Tail |
| Incineroar | Fake Out 36%, Parting Shot 16% | same |
| Rillaboom | Fake Out 44%, Wood Hammer 14% | Grassy Glide (priority) blanked by Armor Tail; Ice Beam 2x |
| Mega Salamence | Hyper Voice 25%, Tailwind 17%, Protect 16% | Fake Out stops Tailwind; Ice Beam 4x |
| Indeedee-F | Follow Me 37%, Trick Room 20%, Helping Hand 14% | no Fake Out; spread, don't focus |
| Indeedee (M) | Expanding Force 48%, Imprison 13% | Imprison kills our Trick Room for the game |
| Kingambit | Kowtow Cleave 32%, Iron Head 12% | Sucker Punch blanked by Armor Tail |
| Gholdengo | Nasty Plot 29%, Shadow Ball 23% | Fake Out it to stop the Plot |
| Mega Raichu-Y | Fake Out 35%, Zap Cannon 33% (No Guard) | 2x on Blastoise; Water leads did worst vs Gholdengo + Raichu [-21pp] |
| Farigiraf | Trick Room 48% | its Armor Tail blanks OUR Fake Out on both |
| Tyranitar | Rock Slide 21%, Protect 19%, Knock Off 18% | Knock Off goes at Farigiraf; Mega re-sets sand |
| Whimsicott | Tailwind 43-55% | Fake Out stops Prankster Tailwind |
| Archaludon | Electro Shot 30%, Protect 19% | 2x on Blastoise, instant in rain |
| Grimmsnarl | Light Screen 34%, Reflect 28% | Fake Out it [19-7 for sides that did, vs Grimmsnarl + Archaludon] |
| Mega Gengar | Protect 34%, Perish Song 24% | Ghost: no Fake Out; Fake Out its partner |

## 3. The openings

Champions Speeds (base + points + 20, nature): Charizard-Y 152, Venusaur 132 (264 in
sun), Mega Blastoise 88, Farigiraf 80, Incineroar 80, Torkoal 36. One Mega per game,
so Blastoise and Charizard openings are exclusive.

**A. Water Room — Blastoise + Farigiraf (default).** Back Incineroar + Torkoal;
Incineroar + Venusaur against sand / Wide Guard (Torkoal's sun would end our rain).
Turn 1: Farigiraf Trick Room; Blastoise Mega + Fake Out, target order: their
Tailwind setter → screens setter → a reachable Trick Room setter → Tyranitar → their
Mega / fastest attacker (what the experts hit: Charizard-Y, Salamence, Garchomp-Z,
Raichu-Y, Aerodactyl). Nothing reachable: Water Spout (Water Pulse / Ice Beam into a
shown or likely Wide Guard). Turn 2: Rain Dance + Water Spout, or Psychic + Ice Beam
on a Grass / Dragon target; Torkoal on the first faint under the room.

**B. Support Room — Farigiraf + Incineroar (widen to Psychic Terrain).** Back
Blastoise + Torkoal. [humans 91-75, +4.8pp: vs Tailwind 28-14, vs Psychic Terrain
11-6, vs Trick Room 19-17]; [ours 5-5]. Incineroar is Dark: immune to Expanding
Force, 2x on Indeedee / Armarouge (Dark-type leads [+12.8pp] vs Armarouge +
Indeedee-F, but [-9.7pp] vs Gardevoir + Indeedee-F — Fairy STAB), so key it on the
Psychic-type attacker, not the terrain. Turn 1: Trick Room + Fake Out (reachable) or
Parting Shot into the attacker [humans: TR + Parting Shot 7-4, +16.6pp]. Our
Incineroar has never used Fake Out from this lead (Parting Shot 7, Flare Blitz 3).

**C. Sun Room — Farigiraf + Torkoal (narrow).** [ours 21-29, fixed-opening era:
sand 1-7, sun 1-5, Psychic Terrain 0-2]; [humans 7-4]; strong pilots don't lead
Torkoal. Keep to clean, Fire-weak, non-Tailwind teams until the battery says more.

**D. Fast Sun — Charizard + Venusaur (experimental).** [ours 1-6]; [humans 81-70,
+2.8pp; vs rain 6-13; turn 1 Heat Wave + Sleep Powder 18-11, Protect + Sleep Powder
2-7]. If used: attack with Charizard on turn 1 (human Charizard leads: Heat Wave
turn 1 +9.0pp vsElo, Protect -1.7pp).

**E. NEW — Sun Shield: Charizard + Farigiraf.** Back Torkoal + Incineroar. [humans
24-15, +7.2pp; vs Tailwind 7-2, rain 5-2, balance 6-0; vs Trick Room 3-4,
Psychic Terrain 1-3]. Fake Out is the top move thrown at a Charizard lead in every
archetype; Armor Tail blanks it (and Sucker Punch, Grassy Glide, Prankster).
Turn 1: Mega + Heat Wave; Farigiraf Trick Room vs Tailwind teams (under the room
Charizard moves before Tailwinded attackers), else Helping Hand / Psychic.

**F. NEW — Fire Pivot: Charizard + Incineroar.** Back Farigiraf + Torkoal. [humans
23-18, +5.4pp; T6-like rosters 9-4; vs Trick Room 8-4, Psychic Terrain 5-2, rain
3-5]. Intimidate covers Charizard vs Rock Slide users; turn 1 Fake Out the biggest
threat to Charizard (or Parting Shot) + Mega Heat Wave.

For E / F / D (expert line, Aaron Zheng via Game8): Mega Charizard Y on turn 1
unless their weather setter waits in the back; then hold the Mega until they have
spent their weather — Drought fires on Mega Evolution and the last weather wins.

Not recommended: Blastoise + Incineroar (humans 26-12, but with a Shell Smash
Blastoise, not our set); Torkoal + Venusaur [ours 0-2]; Blastoise with Torkoal or
Charizard (weather / Mega clash).

## 4. Opening book by opponent (proposal for draft 2)

Chosen at team preview from their six (the planner's features); turn-1 details from
their actual leads.

| Their team | Opening | Turn 1 → 2 | Evidence |
|---|---|---|---|
| Tailwind offense | A or B (draft 1.1: B 16, A 9 of our 25; test E) | Fake Out their Tailwind setter + TR → Rain Dance + Water Spout | ours A 8-3, C 5-5; humans B 28-14; TR vs Tailwind 17-2 |
| Trick Room, setter we can Fake Out | A | Fake Out the setter + TR (`playbook_opening` aims only at reachable foes since draft 1.1) | unresolved: humans facing a setter lead who Faked Out the setter went 21-18, the partner 56-27 (confounded by reachability) |
| Trick Room, setter we can't (Farigiraf / Indeedee / Ghost) | A | contest with TR (current); test Kim's Helping Hand + Water Spout; after Imprison shows, never TR | ours 12-4; mirror 39.1% for yielding |
| Psychic Terrain + Psychic attacker (Indeedee, Armarouge, Hatterene) | **B** | TR + Parting Shot / Throat Chop; spread damage | Dark leads +12.8pp vs Armarouge + Indeedee-F |
| Psychic Terrain + Gardevoir | A | Water Spout (spread); never double-target | focus fire 6-35 |
| Sand (Tyranitar / Hippowdon) | A, back **Incineroar + Venusaur** | Fake Out Tyranitar (or its Tailwind partner) + TR → **Rain Dance** (ends sand and Sand Rush, 1.5x Water Spout) | ours 1-9; TR turn 1 +14pp vs Tyranitar + Excadrill; dksnnfud and thruxy won with turn-2 Rain Dance |
| Rain + Archaludon (incl. the Baltimore runner-up clones) | A; the weather specialist if it passes; test back Incineroar + Venusaur (resists Electro Shot and Water) vs + Torkoal (resets sun on entry) | Fake Out Grimmsnarl / Pelipper / Archaludon + TR; no Rain Dance | ours 8-14; experts 1-3; no known winning line |
| Sun (Charizard-Y / Torkoal) | A | Fake Out their Charizard + TR → Rain Dance + Water Spout | ours A 5-2; dksnnfud D1 |
| Wide Guard on roster | A, back Incineroar + Venusaur | Water Pulse / Ice Beam once shown | ours 12-25 |
| Balance / Fake Out goodstuffs | A or B | as A (their Fake Out and priority are blanked) | |

## 5. Turn-1 facts (mechanics, not statistics)

- **Fake Out can't land on:** Ghost types (incl. Mega Absol-Z, Dark/Ghost);
  anything beside an Armor Tail / Dazzling / Queenly Majesty holder; grounded foes
  under Psychic Terrain (Flying / Levitate ones are fair game — Charizard, Mega
  Garchomp-Z); Inner Focus (no flinch). **Our** Armor Tail blanks their Fake Out,
  Sucker Punch, Grassy Glide, Extreme Speed, Aqua Jet and Prankster moves aimed at
  us (Taunt, Encore) — not Prankster moves on their own side (Tailwind, screens) —
  and **Mold Breaker ignores it** (Tinkaton's Fake Out flinched Farigiraf, thruxy L).
- **Trick Room:** -7 priority, faster setter first; a second Trick Room the same turn
  cancels it. Setter Speeds (0 points, neutral / -Spe): Indeedee 115/103, Indeedee-F
  105/94, Gardevoir 100/90, Armarouge 95/85, Sinistcha 90/81, **our Farigiraf 80**,
  Farigiraf / Oranguru / Porygon2 80/72, Slowking / Cofagrigus / Reuniclus 50/45,
  Hatterene 49/44, Dusclops 45/40. Imprison (Indeedee-M, Oranguru, 10% of Farigiraf)
  blocks ours for the game; it blocked our Farigiraf 4 times on ladder.
- **Weather:** turn order is switches → Mega Evolution → moves; Drought / Sand Stream /
  Snow Warning fire again on Mega Evolution (Mega Tyranitar took sand back on turn 1
  in 5 of our games); the setter that enters or evolves last wins; an ability can't
  re-set an active weather (Torkoal + Charizard-Y never extend sun).
- **Encore after a Fake Out forces Struggle** in Champions (Fake Out is disabled
  after the first turn): Whimsicott carries Encore on 73% of sheets — Prankster
  Encore is blanked while Farigiraf stands beside Blastoise.
- **Champions changes:** sleep = asleep turn 1, 1/3 wake on turn 2, always wakes
  turn 3 (Sleep Powder is worth less than in Scarlet/Violet); full paralysis 1/8;
  Iron Head flinch 20%; Protect 8 PP; Megas get their new Speed and ability on the
  turn they evolve; no Trick Room speed overflow.

## 6. Set note (a team question, not an opening)

Our Blastoise / Farigiraf are dksnnfud's sets. Most tournament Mega Blastoise run
Shell Smash (69%) and Dark Pulse (58%); MaxwellBaker's Water Room Blastoise runs
**Dark Pulse + Aura Sphere** (Mega Launcher 1.5x) instead of our Water Pulse + Ice
Beam. Aura Sphere is 4x on Tyranitar / Kingambit and 2x on Archaludon; Dark Pulse
2x on Indeedee / Armarouge / Farigiraf / Gholdengo — our three weak matchups. Ice
Beam's 4x on Salamence / Garchomp and 2x on Rillaboom (five KOs in the expert
replays) would go. A set change means retraining and re-gating: the user's call.

## 7. What the planner picks (replayed over our 134 opponents)

Draft 1: Trick Room teams Water 19 / Sun 5 / Support 2; Tailwind **Sun 18** / Water
4 / Support 3; sand Support 13 / Water 7; rain Water 12 / Support 5; Psychic Terrain
**Water 13** / Support 2; balance Sun 7 / Support 4 / Water 4.
Draft 1.1 (`c7b76b2`, after finding 8): Water 74 / Support 44 / Sun 16 overall;
Tailwind Support 16 / Water 9; Trick Room Water 21 / Support 3 / Sun 2; Psychic
Terrain still **Water 13** / Support 2 — the remaining disagreement (section 4).

## 8. How to test (proposal, nothing run)

1. **Practice first.** Unpractised openings collapse (2026-09-23 control arm:
   Farigiraf + Incineroar 62% and Charizard + Venusaur 72% where the brain's own
   practised lead scored 90%+ on the same matchups).
   New openings (B widened, E, F) must enter the training preview distribution —
   the way `training/human_preview.py` sampled the human model — before judging.
2. **Opening study by archetype:** `evaluation/opening_study.py --plans <plans file>`
   on held-out rosters of each archetype, same turn policy and opponent preview,
   ≥300 games per arm, against the current preview.
3. **Turn-1 facts as a teacher, not guards:** Fake Out reachability (incl. Mold
   Breaker), Trick Room order / cancellation and Imprison are exact; they fit
   `training/tactical_teacher.py` next to "priority into Psychic Terrain is
   useless" and are judged by the 2,000-game mirror.
4. Then the planner picks cards by these rules; ladder is the final arbiter.

## 9. Sources

- Baltimore Regional: https://standings.limitlessvgc.com/0037/pokemon (+ `?conversion`),
  https://limitlessvgc.com/tournaments/441, https://victoryroad.pro/2027-baltimore/
- Pikalytics tournament sheets: https://www.pikalytics.com/pokedex/championstournaments
  (Blastoise-Mega, Farigiraf, Torkoal pages); ladder: https://www.pikalytics.com/pokedex/gen9championsvgc2026regmc
- Expert replays (our six): thruxy, e.g. https://replay.pokemonshowdown.com/gen9championsvgc2026regmc-2687410790,
  -2687392303, -2687319909, -2683895205 (Mold Breaker loss); dksnnfud
  -2683572822, -2683561174 (rain loss), -2682817857; Reg M-B
  https://replay.pokemonshowdown.com/gen9championsvgc2026regmb-2661181949
- Kim, PJCS 2026 qualifier (9-1): https://note.com/kimupoke05/n/nb1b030e71e04 ;
  3rdAubergine: https://note.com/3rd_aubergine/n/nacc8efd16b62 ; StarRaikou:
  https://www.gamecards.gg/teams/blastoise-mega-trick-room-rain ; Game8 Trick Room:
  https://game8.co/games/Pokemon-Champions/archives/594400 ; Mega Charizard Y timing:
  https://game8.co/games/Pokemon-Champions/archives/594157
- Mechanics: https://www.smogon.com/forums/threads/champions-battle-mechanics-research.3780372/ ,
  https://champdex.com/guides/format-rules , local `pokemon-showdown/data/mods/champions/`
- Reg M-C discussion: https://www.smogon.com/forums/threads/vgc-reg-m-c-metagame-discussion-thread.3788116/
