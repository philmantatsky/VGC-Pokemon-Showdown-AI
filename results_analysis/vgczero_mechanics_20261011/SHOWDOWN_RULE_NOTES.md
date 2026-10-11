# Rule notes for eight mechanics the vgczero engine has not built

Written 2026-10-11 for the vgczero session, read-only from Pokemon Showdown's source at
`~/vgczero-build/showdown_ps_head`, commit `942cda71d` (2026-10-09, "Fix virtual
formats"). Every `file:line` below is in that checkout. Nothing here was played: the
log lines are what the code writes, and each "scenario" is a test to run against the
real simulator, not a result.

The eight, by how many of the 640 unsupported teams they block: Ally Switch (50),
Illusion (50), Dragon Darts (48), Bulletproof (44), Leek (39), Disguise (35), Imposter
(26), Transform (26).

## What the Champions mod changes that these touch

`data/mods/champions/` overrides none of the eight entries except the Leek's legality
(`items.ts:522-525`: `inherit: true, isNonstandard: null`; the handler is the base one).
Its `scripts.ts` does replace core routines that several of them run through:

| routine | where | what matters here |
|---|---|---|
| `init` and `calculatePP` | `scripts.ts:3-9`, `41-43` | base PP is capped at 20 and max PP is `(pp / 5 + 1) * 4` (5 -> 8, 10 -> 12, 15 -> 16, 20 -> 20). Ally Switch has 16, Dragon Darts 12, Transform 12. |
| `pokemon.formeChange` | `scripts.ts:56-121` | the permanent forme change Disguise uses; Illusion's apparent species in `-mega` and `detailschange`. |
| `pokemon.clearVolatile` | `scripts.ts:123-176` | undoes a transformation on switch-out; resets `timesAttacked`. |
| `actions.spreadMoveHit` | `scripts.ts:314-423` | one hit: substitute check, damage, `DamagingHit` (Illusion breaks here, a busted Disguise still triggers it). |
| `actions.hitStepMoveHitLoop` | `scripts.ts:426-590` | the multi-hit loop with the Dragon Darts branches. |
| `actions.modifyDamage` | `scripts.ts:196-312` | spread modifier keyed on `move.spreadHit` (Dragon Darts never sets it), crit x1.5. |

`trySpreadMoveHit` and hit steps 0 to 6 (`sim/battle-actions.ts:550-770`), target
resolution (`sim/pokemon.ts:800-850`, `sim/battle.ts:2441-2500`), switching
(`sim/battle-actions.ts:62-196`) and `transformInto` (`sim/pokemon.ts:1270-1380`) are
the base ones.

## Shared machinery

**A move's targets, in this order** (`sim/pokemon.ts:822-843`): the chosen location is
turned into a Pokemon (`sim/battle.ts:2441-2500`: by location, so whoever stands there
when the move executes); a fainted foe target is replaced by a random legal one
(`824-829`); `RedirectTarget` runs (`830-831`: Follow Me, Rage Powder, Lightning Rod,
Storm Drain); then a smart-target move expands to two targets (`832-834`).

**The hit steps** (`sim/battle-actions.ts:553-578`), each over the whole target list:
0 semi-invulnerability, 1 `TryHit` (Protect and its family, abilities such as
Bulletproof), 2 type immunity, 3 move-specific immunity, 4 accuracy, 5 breaking
protection, 6 stealing boosts, 7 the hit loop. Before them the move's own `Try` and
`PrepareHit` (`590-599`); a `false` there writes `|-fail|USER` and marks the move line
`[still]`.

**Switching in** (`sim/battle-actions.ts:62-165`): the outgoing Pokemon's ability and
item `End` (`103-104`); the party array is updated -- the incoming takes index `pos`,
the outgoing takes the incoming's old index (`125-133`); `BeforeSwitchIn` runs (`148`);
the `|switch|` line is written (`149-153`); then `runSwitch` (`179-196`) gathers every
Pokemon that came in together and fires one `SwitchIn` field event for them. Its
handlers are sorted by speed with the order of the actives fixed beforehand
(`sim/battle.ts:1011-1017`), and an ability or item without an `onSwitchIn` of its own
has its `onStart` run there (`sim/battle.ts:1021-1036`).

**An ability is ignored** (`sim/pokemon.ts:858-878`) when its holder is not active,
when it has the `notransform` flag and the holder is transformed (`862`), under Gastro
Acid or Neutralizing Gas unless it has `cantsuppress` (`863-875`). `breakable`
abilities are also skipped while a Mold Breaker-type user's move is being resolved.

---

## 1. Ally Switch (move, 50 teams)

**Where.** `data/moves.ts:302-356`. Status, Psychic, priority +2, target `self`,
flags `metronome` only. Swap: `sim/battle.ts:1592-1611`.

**Handlers in order.**
1. `onPrepareHit` (`311-313`): `pokemon.addVolatile('allyswitch')`. A first use starts
   the volatile with `counter = 3`, duration 2 (`332-337`). A use while the volatile is
   still there calls `onRestart` (`338-351`; `sim/pokemon.ts:1983-1986` returns its
   result): success with chance `1 / counter`; on success `counter *= 3` up to 729 and
   the duration is set back to 2; on failure the volatile is deleted and the move fails
   here (`|-fail|USER`, the move line marked `[still]`).
2. `onHit` (`314-331`): fails unless the game is doubles or triples, the other active
   slot holds a Pokemon and that Pokemon is not fainted. Failure writes
   `|-fail|USER|move: Ally Switch` and returns "not a fail" (so nothing else reports).
   Success calls `swapPosition(pokemon, newPosition, '[from] move: Ally Switch')`.
3. `swapPosition` (`sim/battle.ts:1592-1611`): writes `|swap|USER|NEWPOSITION|[from]
   move: Ally Switch`, exchanges the two entries of `side.pokemon` and of
   `side.active`, exchanges the two `position` values, then runs a `Swap` event for
   each of the two.

**Doubles.**
- Every move already chosen with a target location lands on whoever stands in that
  location when it executes (`sim/battle.ts:2441-2500`). An opponent's single-target
  move aimed at the slot the user was in hits the partner, and the reverse. Moves
  aimed by our own side at "my partner" are relative locations and follow the same
  rule. A user with Stalwart or Propeller Tail, or a `tracksTarget` move, keeps the
  Pokemon it aimed at (`sim/battle.ts:2444-2451`).
- The queue holds Pokemon, not slots: both still act, in the order already decided.
- Slot conditions stay with the slot: Wish (`data/moves.ts:20924`), Future Sight and
  Doom Desire (`3857`, `6403`), Healing Wish and Lunar Dance (`8365`, `10580`, the only
  listeners of the `Swap` event, `8370`, `10585`: they heal the Pokemon swapped into
  the slot if it is hurt or statused), Revival Blessing (`15125`).
- Dragon Darts can end up aimed at its user's own position (`sim/battle.ts:2453-2458`).
- Illusion reads the party order (section 2): a swap changes who is "to the right".

**Request and log.** After the swap the owner's request lists the two actives in the
new order (`active[0]` is the Pokemon now in slot a) and `side.pokemon[0..1]` likewise;
switch choices keep indexing the party array. The opponent sees only the `|swap|` line.

**Hidden state.** The `allyswitch` volatile on the user: `counter` (3, 9, 27 ... 729)
and `duration` (2: it is gone at the end of the turn after the last use). It is a
separate volatile from the `stall` of Protect and its family: neither affects the
other. Public from the log except the roll itself.

**Scenarios.**
1. Our slot-a Pokemon uses Ally Switch; an opponent has chosen a single-target move
   into slot a. The swap comes first (+2); the move hits the Pokemon that was in slot b.
2. Ally Switch on two turns running: the second succeeds one time in three. On a
   failure the log has `|-fail|` and a `[still]` move line, no `|swap|`, and the
   volatile is gone -- a third use next turn succeeds for certain (counter back to 3).
3. The partner fainted earlier in the turn (slot not yet refilled):
   `|-fail|USER|move: Ally Switch`; the volatile was still started or advanced by
   step 1, so the use counts toward the next roll.
4. Wish used from slot a, then Ally Switch next turn: the Wish heals whoever is in
   slot a when it resolves.
5. After a swap, the next request's `active` array and the first two party entries
   are exchanged; a choice written for "slot 1" now moves the other Pokemon.

---

## 2. Illusion (ability, 50 teams: Zoroark, Zoroark-Hisui)

**Where.** `data/abilities.ts:2056-2097`; flags `failroleplay, noreceiver, noentrain,
notrace, failskillswap`. The field is `pokemon.illusion` (`sim/pokemon.ts:137`).

**Handlers in order.**
1. `onBeforeSwitchIn` (`2057-2071`), on every entry (leads, switches, replacements,
   drags), after the party array has been updated: clear `illusion`, then scan
   `side.pokemon` from the LAST index down to the holder's own position + 1; the first
   Pokemon that is not fainted becomes the disguise. Nobody there: no disguise.
2. The `|switch|` line is written right after with the disguise applied
   (`sim/battle-actions.ts:148-153`).
3. `onDamagingHit` (`2073-2077`): after a hit of a damaging move that dealt damage to
   the holder (the event runs in `scripts.ts:407-410`), the ability's `End` runs.
4. `onEnd` (`2078-2090`): if disguised and not being called back, clear it and write
   `|replace|SLOT: REALNAME|REAL DETAILS` then `|-end|POKEMON|Illusion`.
   `End` also runs when the ability is replaced (`sim/pokemon.ts:1923`) or suppressed
   by Gastro Acid (`data/moves.ts:6450`), which reveals it, and on switch-out, where
   `beingCalledBack` keeps it silent (`sim/battle-actions.ts:75`, `103`).
5. `onFaint` (`2091-2093`): clears the field with no message. The `|faint|` line is
   written BEFORE this (`sim/battle.ts:2553-2558`), so a holder that faints while
   still disguised is logged as the disguise fainting, and no `replace` follows.

**What the disguise changes.**
- The name in every log line: `toString` uses the disguise's name with the real slot
  letter (`sim/pokemon.ts:531-534`).
- The details in `|switch|` and `|drag|`: the disguise's species, gender and shininess;
  the level is the holder's own unless the Illusion Level Mod rule is on
  (`sim/pokemon.ts:543-553`). The HP shown is the holder's real HP.
- The types the opponent is told (`sim/battle.ts:1716-1728`) and the "might it trap
  me" check (`1737-1745`) use the disguise's species.
- If a holder can Mega Evolve, doing so while disguised keeps the disguise: `-mega`
  and `detailschange` name the apparent species (`scripts.ts:64-72`, `91`, `106-109`).
- It does NOT change anything the holder does: moves are named truthfully in the log,
  damage and stats are the holder's, the ability line is the holder's.

**Who the disguise is, in a bring-four doubles game.** Party indices 0 and 1 are the
actives, 2 and 3 the bench, and a switch exchanges the incoming's index with the
outgoing's (`sim/battle-actions.ts:125-133`).
- A lead takes the last of the four chosen (index 3) if it is alive, else index 2.
- Coming in from index 3 for a healthy Pokemon, the holder disguises as the Pokemon it
  just replaced (now at index 3).
- In slot a with both bench Pokemon fainted it can disguise as its partner in slot b
  (index 1 is to its right); in slot b with both bench Pokemon fainted it has nobody to
  its right and appears as itself.

**Doubles.** Nothing targets differently. Transform and Imposter fail on a disguised
Pokemon and a disguised Pokemon cannot transform (`sim/pokemon.ts:1273`). A hit that
lands on a Substitute does not break the disguise (no `DamagingHit` for the holder).

**Request and log.** The owner's request shows the truth (`ident`, `details`:
`sim/pokemon.ts:1153-1190`) and carries no field saying a disguise is up; the owner's
own log stream shows the disguised `|switch|` line too. The opponent's stream is the
disguise until `|replace|`.

**Hidden state.** `illusion`: a reference to one of the holder's own party members.
It is fully determined by public party order and faint history for the owner; for the
opponent it is hidden until the first damaging hit.

**Scenarios.**
1. Lead Zoroark with the four chosen as [Zoroark, X, Y, Z]: the opponent's `|switch|`
   line names Z (species, gender, shiny) with Zoroark's HP. A damaging hit writes
   `|replace|` then `|-end|...|Illusion`, after the `|-damage|` of that hit.
2. A status move, Sandstorm damage or poison does not break it. Fainting to such
   damage writes `|faint|p2a: Z` and nothing else: the opponent's record now has Z
   fainted while the real Z is alive on the bench.
3. Zoroark at party index 3 switches in for a healthy slot-a Pokemon: it appears as
   that Pokemon. With that Pokemon fainted instead, it appears as the bench Pokemon at
   index 2 if alive, else as its slot-b partner, else as itself.
4. Transform aimed at a disguised Zoroark fails; a Ditto with Imposter switching in
   across from one stays Ditto.
5. The disguised holder uses Night Daze: the move line reads `|move|p2a: Z|Night
   Daze|...` -- a move Z cannot have, with no reveal.

---

## 3. Dragon Darts (move, 48 teams: Dragapult)

**Where.** `data/moves.ts:4118-4132`: 50 power, 100 accuracy, Physical, Dragon,
`multihit: 2`, `smartTarget: true`, target `normal`, flags `protect, mirror,
metronome, noparentalbond` (no contact). The flag is read at `sim/pokemon.ts:756-768`
and `832-834`, `sim/battle.ts:2453-2458`, `sim/battle-actions.ts:551`, `607`,
`633-634`, `661`, `739-740`, and in the Champions hit loop `scripts.ts:465-478`,
`519-523`, `529`, `546-549`, `556-558`, `564`.

**Handlers in order.**
1. The choice names one target location, as for any `normal` move: an adjacent foe or
   the ally, never the user (`sim/battle.ts:2420-2424`).
2. Target resolution: a fainted chosen foe is replaced; redirection runs; then
   `getSmartTargets` (`sim/pokemon.ts:756-768`): with T the target and T2 T's adjacent
   ally -- if T2 does not exist, is fainted or is the user, the flag is turned off and
   the list is [T]; if T is fainted, off and [T2]; otherwise [T, T2].
   Follow Me and Rage Powder turn the flag off as they redirect
   (`data/moves.ts:6062-6064`, `14617-14622`), so both darts go to the redirector.
3. The hit steps run over the list. It is not a spread move (`551`): no 0.75 modifier,
   no `[spread]` tag, Wide Guard does not stop it. Any target that fails a step is
   dropped, and as soon as one has failed the flag is turned off (`607`), so the
   survivor takes both darts. A dropped target's own message is not written while the
   flag is still on: no `-miss` (`633-634`, `739-740`), no `-activate|...|Protect`
   (`data/moves.ts:13984-13990`; the same in Baneful Bunker `1008`, Burning Bulwark
   `2043`, King's Shield `9929`, Obstruct `12897`, Silk Trap `16428`, Spiky Shield
   `17555`), no `-immune` for a Fairy (`661`). Wonder Guard `data/abilities.ts:5557`.
   Targets are processed in list order inside a step. In the invulnerability, `TryHit`
   and accuracy steps the first failure turns the flag off on the spot, so a second
   failure in the same step is reported normally. In the type-immunity step the flag
   is only turned off after the step (`607`), so two immune targets are both silent.
4. The hit loop (`scripts.ts:459-541`). With the flag on and two targets, hit 1 is
   spread over `[targets[0]]` only and hit 2 over `[targets[1]]` only (`465-470`); the
   second dart is announced with `|-anim|USER|Dragon Darts|T2` (`472-478`). With one
   target it is an ordinary two-hit move: both on it, stopping if it faints.
   Each dart rolls damage and a critical hit on its own and is a separate hit for
   Focus Sash, Disguise, Stamina and the like.
5. After the loop no `|-hitcount|` line is written for this move (`546-549`: the
   condition is that the flag is not a boolean, and here it is `true` or `false`).
   Each target hit counts once in `timesAttacked` (`564`).

**Doubles.** All of it is doubles-only: in singles the list is always [T].

**Request and log.** The request is that of a single-target move. The log: one
`|move|` line naming the chosen target (retargeted to T when hit 1 lands, `476`), a
`|-damage|` per dart, `|-anim|` before the second when it goes to the other Pokemon.

**Hidden state.** None that persists; the flag lives on the move being resolved.

**Scenarios.**
1. Both foes healthy and unprotected: each takes one dart; the chosen one first; an
   `|-anim|` line before the second; no `|-hitcount|`.
2. The chosen target uses Protect: both darts hit its partner and the log has no
   `-activate` for Protect. The partner protects instead: both hit the chosen one.
   Both protect: the first is dropped silently, the second writes its `-activate`.
3. One foe is a Fairy: both darts hit the other, with no `-immune` line.
4. The partner of the chosen target used Follow Me: both darts hit the Follow Me
   user (the flag is off before the list is built).
5. One foe left: two hits on it; a knock-out on the first ends the move.
6. The chosen target is at full HP holding a Focus Sash and its partner is healthy:
   one dart each, so this move alone cannot knock the Sash holder out. With the
   partner protecting, both darts hit the holder: the first leaves 1 HP, the second
   knocks it out.
7. Aimed at the user's own ally: both darts hit the ally (T2 would be the user).
8. Both foes are Fairies: nothing is written after the move line (both dropped
   silently in the immunity step).

---

## 4. Bulletproof (ability, 44 teams: Kommo-o; Chesnaught-Mega; hidden on Chesnaught, Dubwool, Applin)

**Where.** `data/abilities.ts:480-491`, flag `breakable`. Holders:
`data/pokedex.ts:12038` (Chesnaught, hidden), `12050-12056` (Chesnaught-Mega, its only
ability: this is the "as a Mega" entry of the engine's list), `15156` (Kommo-o, slot
0), `16029` (Dubwool, hidden), `16163` (Applin, hidden).

**Handler.** `onTryHit(pokemon, target, move)` (`481-486`): if the incoming move has
the `bullet` flag, write `|-immune|HOLDER|[from] ability: Bulletproof` and return
`null`. That is hit step 1 (`sim/battle-actions.ts:643-653`): the holder is dropped
from the targets; no `|-fail|` line is added (the `-fail` there is for a literal
`false`). Step 1 comes before type immunity (step 2), so a Ghost bullet move into a
Normal-type holder reports Bulletproof, not the type. Protect's own `TryHit` has
priority 3 and runs first when the holder is also protecting.

**The bullet moves** (26 in `data/moves.ts`; an asterisk is "nonstandard" in the base
data; Mist Ball is also "Past" in Champions): Acid Spray, Aura Sphere, Barrage*, Beak
Blast, Bullet Seed, Egg Bomb*, Electro Ball, Energy Ball, Focus Blast, Gyro Ball, Ice
Ball*, Magnet Bomb*, Mist Ball, Mud Bomb*, Octazooka*, Pollen Puff, Pyro Ball, Rock
Blast, Rock Wrecker, Searing Shot*, Seed Bomb, Shadow Ball, Sludge Bomb, Syrup Bomb,
Weather Ball, Zap Cannon. Take the flag from the dex export, not from this list.

**Doubles.** None of these moves hits more than one target. The one ally interaction:
Pollen Puff aimed at a Bulletproof ally is blocked like any other (no heal).
A Mold Breaker, Teravolt or Turboblaze user ignores the ability (`breakable`).

**Request and log.** Nothing in the request. The `-immune` line names the ability, so
it reveals a hidden Bulletproof (Chesnaught and Dubwool have other abilities).

**Hidden state.** None beyond which ability the Pokemon has. Chesnaught gains it on
Mega Evolution (the forme change sets the Mega's ability: `scripts.ts:110-114`), which
happens before moves in the turn.

**Scenarios.**
1. Shadow Ball into Kommo-o with Bulletproof: `|-immune|...|[from] ability:
   Bulletproof`, no damage, no Special Defense drop, the user's PP spent.
2. The same from a Mold Breaker user: it hits.
3. Rock Blast or Bullet Seed: stopped once, before the hit loop -- no hits, no
   `|-hitcount|`.
4. An ally's Pollen Puff onto the holder: blocked, no heal.
5. Chesnaught holding its stone, hit by Sludge Bomb on the turn it Mega Evolves:
   immune (the ability changed before the move); on a turn it does not evolve and has
   Overgrow: hit.

---

## 5. Leek (item, 39 teams: Sirfetch'd, Farfetch'd)

**Where.** `data/items.ts:3322-3336`; Champions `data/mods/champions/items.ts:522-525`
only makes it legal.

**Handler.** `onModifyCritRatio(critRatio, user)` (`3328-3332`): +2 stages when the
holder's base species is Farfetch'd or Sirfetch'd (`user.baseSpecies.baseSpecies`, so
Farfetch'd-Galar counts; a Ditto transformed into one does not; a Farfetch'd
transformed into something else still does).

**How a stage becomes a chance** (`sim/battle-actions.ts:1614-1637`): the move's own
ratio (1, or 2 for high-crit moves) goes through the `ModifyCritRatio` event of the
attacker, is clamped to 0..4 and indexes `[0, 24, 8, 2, 1]` as "one in". With the Leek
an ordinary move is at stage 3: one in two. A high-crit move, or an ordinary one with
Focus Energy, is at stage 4: always. `move.willCrit` set by a move overrides the roll
(`1629-1630`); then the target's `CriticalHit` event may cancel it (`1636`: Battle
Armor and Shell Armor `data/abilities.ts:356`, `4224`; an intact Disguise `979`).
A critical hit multiplies by 1.5 (`scripts.ts:218-221`).

**Doubles.** A spread move rolls the crit per target (the damage is computed per
target). Nothing else.

**Request and log.** `|-crit|TARGET` after the hit. Nothing in the request.

**Hidden state.** None.

**Scenarios.**
1. Sirfetch'd with the Leek using an ordinary move, over many seeds: half the hits
   are critical (compare 1 in 24 without the item).
2. The same with Leaf Blade or Night Slash: every hit is critical.
3. The Leek on another species: 1 in 24.
4. Into Shell Armor: never critical.

---

## 6. Disguise (ability, 35 teams: Mimikyu)

**Where.** `data/abilities.ts:970-1016`; flags `failroleplay, noreceiver, noentrain,
notrace, failskillswap, cantsuppress, breakable, notransform`.

**Handlers in order, for one damaging hit on an intact Mimikyu.**
1. In the damage calculation: `onEffectiveness` (`990-1001`) returns 0, so the hit is
   computed as neutral; `onCriticalHit` (`979-989`) returns false. Both step aside when
   the move will hit a Substitute or the target is immune to the move's type.
2. `onDamage`, priority 1 (`971-978`), in the `Damage` event
   (`sim/battle.ts:2125`): only when the damage comes from an effect whose type is
   `Move` and the species id is exactly `mimikyu` (or the totem). It writes
   `|-activate|TARGET|ability: Disguise`, sets `busted` on the ability's state and
   returns 0. The `|-damage|` line is still written, with the HP unchanged
   (`sim/battle.ts:2146-2163`).
3. The hit still counts as a damaging hit of 0 (`scripts.ts:398-410`): `DamagingHit`
   handlers run (Rocky Helmet, Rough Skin and the like fire on the busting hit).
4. `onUpdate` (`1002-1008`) at the next `Update` -- after each hit of the loop
   (`scripts.ts:536`): the forme changes permanently to Mimikyu-Busted
   (`detailschange`; the Champions `formeChange` keeps the ability for this source,
   `scripts.ts:105`) and the holder loses `baseMaxhp / 8` (rounded down, at least 1),
   logged as `|-damage|...|[from] pokemon: Mimikyu-Busted`.

**What counts as a move.** Any damaging move, a fixed-damage move, and the confusion
self-hit (`data/conditions.ts:193` builds it with effect type `Move`). Not weather,
status damage, entry hazards, recoil, Rocky Helmet or Life Orb.

**What gets around it.** A Mold Breaker-type attacker (`breakable`): full damage and
the Disguise stays intact. Not Gastro Acid or Neutralizing Gas (`cantsuppress`,
`sim/pokemon.ts:863`). A Ditto transformed into an intact Mimikyu copies the ability
but it is ignored (`notransform`, `sim/pokemon.ts:862`): the copy takes the first hit.

**Doubles.** A spread move busts it and damages the partner normally. A multi-hit
move busts with its first hit and damages with the rest, because the forme changes at
the `Update` between hits. Dragon Darts spends one dart on it when the darts are
split, and both -- the second one damaging -- when they are not.

**Request and log.** The busted forme is public (`detailschange`). The owner's request
then shows `Mimikyu-Busted` in `details`. Busted is permanent: it survives switching
(`scripts.ts:68` moves the base species).

**Hidden state.** `abilityState.busted`, alive only between the hit and the next
`Update`; afterwards the species id says it.

**Scenarios.**
1. First damaging hit: `|-activate|...|ability: Disguise`, a `|-damage|` at unchanged
   HP, `|detailschange|...|Mimikyu-Busted...`, then a loss of an eighth of max HP.
   No "super effective" or `|-crit|` line whatever the move, and a Weakness Policy on
   it is not triggered by that hit.
2. A two-hit move: the first hit busts, the second deals normal damage (now with real
   effectiveness and possible crits).
3. A Mold Breaker user: normal damage, no bust; a later hit from another Pokemon
   busts.
4. Confusion self-hit busts it; Sandstorm, poison and Stealth Rock do not.
5. A contact move that only busts it still takes Rocky Helmet damage
   (`data/items.ts:5302-5303` does not look at the amount); a Weakness Policy on the
   Mimikyu is not used by the busting hit (`data/items.ts:7602-7603` needs a positive
   type modifier, and step 1 made it 0).
6. Busted, switched out and back in: still busted, no second free hit.

---

## 7. Imposter (ability, 26 teams: Ditto)

**Where.** `data/abilities.ts:2116-2131`; flags `failroleplay, noreceiver, noentrain,
notrace`. The work is `transformInto` (section 8).

**Handler.** `onSwitchIn(pokemon)` (`2117-2126`): the target is
`pokemon.side.foe.active[foe.active.length - 1 - pokemon.position]` -- the Pokemon
directly across. In doubles our slot a faces their slot b and our slot b their slot a.
If there is one, `pokemon.transformInto(target, Imposter)`. There is no second choice:
an empty slot across, or a refused transformation, leaves Ditto as Ditto.

**When it runs.** In the `SwitchIn` field event (`sim/battle-actions.ts:179-196`),
sorted by speed among everything that came in together, using Ditto's own speed.
All Pokemon of the batch are already on the field, so a Ditto sent in as a
replacement at the end of a turn copies the opponent's replacement across from it.
Faster Pokemon's entry abilities (Intimidate, weather) have already fired.
It does not run when the ability is gained later or when Neutralizing Gas leaves
(`2118-2119`).

**Doubles.** The "across" rule above; nothing else.

**Request and log.** `|-transform|USER|TARGET|[from] ability: Imposter`
(`sim/pokemon.ts:1343-1347`); see section 8 for the request.

**Hidden state.** That of a transformation (section 8).

**Scenarios.**
1. Ditto in our slot a, the opponent has different Pokemon in a and b: Ditto becomes
   the one in THEIR slot b.
2. The Pokemon across is behind a Substitute, is a disguised Zoroark, or is itself
   transformed: no transformation; Ditto's request shows its own moves.
3. The Pokemon across has Intimidate: Ditto's copy starts on transformation
   (`sim/pokemon.ts:1941-1944`) and lowers the opponents' Attack then, in Ditto's turn
   of the switch-in order.
4. Ditto switches out and back in: it is Ditto again on the bench (section 8) and
   transforms anew, possibly into another Pokemon, with fresh 5 PP.
5. The Pokemon across has +2 in a stat: Ditto has it too.

---

## 8. Transform (move, 26 teams)

**Where.** `data/moves.ts:19828-19843`: Status, Normal, accuracy `true`, target
`normal` (an adjacent foe or the ally), flags `allyanim, failencore, noassist,
failcopycat, failmimic, failinstruct`. No `protect` flag: it goes through Protect
(`sim/battle.ts:1304-1313`). `onHit` returns `pokemon.transformInto(target)`; `false`
is a failed move (`|-fail|USER`).

**`transformInto`** (`sim/pokemon.ts:1270-1380`).
- Refused (`1272-1280`) when the target is fainted, either Pokemon is disguised by
  Illusion, the target is behind a Substitute, the target is already transformed, or
  the user is already transformed.
- Copied: the species (`1289`, with its types, added type and weight: `1292-1298`);
  the five battle stats as the target has them stored (`1300-1304`), so the target's
  nature and stat points, not the user's; every stat stage, accuracy and evasion
  included (`1327-1330`); the target's moves as new slots of `min(5, base PP)` PP each
  with max PP the same (`1305-1326`); the crit volatiles Focus Energy, Dragon Cheer and
  Laser Focus (`1331-1342`); `timesAttacked` (`1308`); the target's CURRENT ability
  (`1352`).
- Not copied: HP and max HP, level, gender, status, item, other volatiles, the
  Substitute, the Tera type.
- The ability is set without the usual checks or messages (`sim/pokemon.ts:1908-1946`
  with `isTransform`), and its `Start` runs if it differs from the old one
  (`1941-1944`). An ability with `notransform` (Disguise, and the forme-changing ones)
  is copied but ignored (`sim/pokemon.ts:862`).
- Log: `|-transform|USER|TARGET` (`1343-1347`). No `detailschange`.
- Undone by `clearVolatile` on switch-out or faint (`scripts.ts:123-176`): moves,
  ability, species and stats return, `transformed` is cleared.

**Doubles.** It may target the ally. A user transformed into a Mega forme has that
forme's stats and ability but cannot Mega Evolve itself: the check is keyed on the
user's own base species and item (`scripts.ts:182-194`).

**Request and log.** The owner's move request lists the copied moves at 5 / 5 PP.
In `side.pokemon` the entry keeps the user's own `details` and its own base stats
(`sim/pokemon.ts:1156-1165`) while `moves` and `ability` are the copied ones
(`1166-1180`). So the transformation hands the owner the target's four moves and its
ability -- hidden information in a closed-sheet game -- and the engine's view for
that side has to carry them.

**Hidden state.** `transformed`; the copied move slots (each with its own PP count);
the copied stored stats, types, ability and stat stages; and everything needed to
undo it: base move slots, base ability, base stored stats, base species.

**Scenarios.**
1. Transform into a target that used Protect this turn: it works.
2. Into a target behind a Substitute, a disguised Zoroark, or a Ditto that has
   already transformed: `|-fail|`. A transformed user using Transform again: fails.
3. After transforming: every move has 5 PP; the user's HP is unchanged; damage it
   deals uses the target's Attack and the user's own level.
4. Transform into an intact Mimikyu: the user's Disguise does nothing; the first hit
   damages it normally.
5. Transform into an ally at +2 Attack after Swords Dance: the stages are copied; the
   item (say a Choice Scarf) is still the user's.
6. Switch out and in: the original moves with their earlier PP are back.

---

## Checks that cover several at once

- The opponent's view after Illusion faints to indirect damage (2.2) and after an
  Ally Switch (1.5) are the two places where the public record disagrees with a naive
  "slot holds this Pokemon" model.
- Dragon Darts into a side with an intact Mimikyu and a Protecting partner: both
  darts hit Mimikyu, the first busts, the second damages (3 and 6).
- Imposter across from a Mega-evolved Chesnaught: the copy has Bulletproof and is
  immune to bullet moves (4, 7, 8).
- A Leek holder's critical hit cannot land on an intact Disguise (5 and 6).
