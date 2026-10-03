//! Battle state. Everything here is `Copy` with fixed-size arrays, so cloning
//! a battle (for search) is a plain memory copy.

use crate::dex::{AbilityId, Boosts, ItemId, MoveId, SpeciesId, Status, Terrain, Weather, NONE};
use crate::teams::SetSpec;
use crate::types::Type;

pub const NO_MON: u8 = 255;

/// What a side has seen of an opposing Pokemon (for observations).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Reveal {
    /// Has been on the field (so the opponent knows it was brought).
    pub seen: bool,
    pub moves: u8,
    pub ability: bool,
    pub item: bool,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ProtectKind {
    #[default]
    None,
    Protect,
    SpikyShield,
    BanefulBunker,
    KingsShield,
    SilkTrap,
    BurningBulwark,
}

/// Two-turn move state.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum SemiInv {
    #[default]
    None,
    Air,
    Underground,
    Underwater,
    Shadow,
}

/// Volatile state, cleared when the Pokemon leaves the field.
#[derive(Copy, Clone, Debug, Default)]
pub struct Vol {
    pub flinch: bool,
    pub protect: ProtectKind,
    pub endure: bool,
    /// Showdown's 'stall' volatile: counter (3, 9, ... 729) and remaining duration.
    pub stall_counter: u16,
    pub stall_dur: u8,
    pub helping_hand: u8,
    /// 0 none, 1 Follow Me, 2 Rage Powder (this turn).
    pub follow_me: u8,
    pub confusion: u8,
    pub taunt: u8,
    pub encore: u8,
    pub encore_slot: u8,
    pub disable: u8,
    pub disable_slot: u8,
    pub throat_chop: u8,
    pub yawn: u8,
    pub perish: u8,
    pub substitute: u16,
    /// Charging a two-turn move: slot + 1 (0 = not charging).
    pub charging: u8,
    pub charge_target: i8,
    pub semi_inv: SemiInv,
    pub must_recharge: bool,
    pub crit_stage: u8,
    /// Choice item lock: move slot + 1.
    pub choice_lock: u8,
    pub glaive_rush: u8,
    pub roost: bool,
    pub imprison: bool,
    pub leech_seed: bool,
    /// Position (side * 2 + slot) that seeded us.
    pub leech_seed_src: u8,
    pub salt_cure: bool,
    pub curse: bool,
    pub destiny_bond: bool,
    pub flash_fire: bool,
    pub unburden: bool,
    pub torment: bool,
    pub heal_block: u8,
    pub magnet_rise: u8,
    pub aqua_ring: bool,
    pub ingrain: bool,
    pub charge: bool,
    pub stockpile: u8,
    pub attract: bool,
    pub no_retreat: bool,
    pub laser_focus: u8,
    pub tar_shot: bool,
    pub smacked_down: bool,
    pub partial_trap: u8,
    pub type_changed: bool,
    pub ability_changed: bool,
    pub transformed: bool,
    pub stall_used_this_turn: bool,
    /// Turns on the field (0 on the turn it switched in).
    pub active_turns: u16,
    /// Moves executed since switching in (Showdown's activeMoveActions).
    pub move_actions: u16,
    pub last_move: MoveId,
    pub last_move_slot: u8,
    /// moveLastTurnResult === false (Stomping Tantrum, Temper Flare).
    pub last_move_failed: bool,
    pub this_move_failed: bool,
    pub moved_this_turn: bool,
    pub hurt_this_turn: bool,
    pub stats_raised_this_turn: bool,
    pub stats_lowered_this_turn: bool,
    pub times_attacked: u8,
    pub newly_switched: bool,
    /// Damage taken this turn from the last attacker (Counter / Mirror Coat / Metal Burst).
    pub last_damage_taken: u16,
    pub last_damage_physical: bool,
    pub last_damage_src: u8,
    /// Switch out at the end of the current action (U-turn, Emergency Exit...).
    pub switch_flag: bool,
    pub switch_copyvolatile: bool,
    pub switch_shedtail: bool,
    /// Forced out by Red Card / Roar (random replacement).
    pub force_switch: bool,
    /// Already moved / Quash ordering.
    pub quashed: bool,
    pub sky_drop: bool,
    pub focus_punch: bool,
    pub destiny_bond_pending: bool,
    /// Supreme Overlord: fainted allies when it entered (max 5).
    pub fallen: u8,
}

#[derive(Copy, Clone, Debug)]
pub struct Mon {
    pub species: SpeciesId,
    pub base_species: SpeciesId,
    pub types: [Type; 2],
    pub base_types: [Type; 2],
    pub ability: AbilityId,
    pub base_ability: AbilityId,
    pub item: ItemId,
    /// Item consumed this battle (for Recycle / Harvest / Unburden).
    pub last_item: ItemId,
    pub stats: [u16; 6],
    pub weight_hg: u32,
    pub level: u8,
    pub gender: u8,
    pub moves: [MoveId; 4],
    pub pp: [u8; 4],
    pub max_pp: [u8; 4],
    pub n_moves: u8,
    pub hp: u16,
    pub max_hp: u16,
    pub status: Status,
    /// Sleep/freeze turns remaining, or toxic counter.
    pub status_turns: u8,
    pub boosts: Boosts,
    pub vol: Vol,
    pub fainted: bool,
    /// Brought to battle (chosen at team preview).
    pub brought: bool,
    /// Slot (0/1) if on the field.
    pub slot: i8,
    pub can_mega: bool,
    pub is_mega: bool,
    pub mega_species: SpeciesId,
    pub mega_ability: AbilityId,
    pub mega_types: [Type; 2],
    pub mega_stats: [u16; 6],
    pub mega_weight_hg: u32,
    /// Index in the original six-Pokemon team.
    pub team_idx: u8,
    /// What the opponent has seen of this Pokemon.
    pub reveal: Reveal,
    /// Berry eaten this battle (Cud Chew, Belch) / item knocked off.
    pub ate_berry: bool,
    pub item_knocked: bool,
    pub illusion: bool,
}

impl Mon {
    pub fn from_set(set: &SetSpec, team_idx: u8) -> Mon {
        let d = crate::dex::dex();
        let mut pp = [0u8; 4];
        for i in 0..set.n_moves as usize {
            // PP ups are maxed: pp * 8 / 5 (Champions caps base PP at 20 already).
            let base = d.mv(set.moves[i]).pp as u16;
            pp[i] = (base * 8 / 5).min(255) as u8;
        }
        let (can_mega, mega) = match set.mega {
            Some(m) => (true, m),
            None => (false, Default::default()),
        };
        Mon {
            species: set.species,
            base_species: set.species,
            types: set.types,
            base_types: set.types,
            ability: set.ability,
            base_ability: set.ability,
            item: set.item,
            last_item: NONE,
            stats: set.stats,
            weight_hg: set.weight_hg,
            level: set.level,
            gender: set.gender,
            moves: set.moves,
            pp,
            max_pp: pp,
            n_moves: set.n_moves,
            hp: set.stats[0],
            max_hp: set.stats[0],
            status: Status::None,
            status_turns: 0,
            boosts: [0; 7],
            vol: Vol::default(),
            fainted: false,
            brought: false,
            slot: -1,
            can_mega,
            is_mega: false,
            mega_species: mega.species,
            mega_ability: mega.ability,
            mega_types: mega.types,
            mega_stats: mega.stats,
            mega_weight_hg: mega.weight_hg,
            team_idx,
            reveal: Reveal::default(),
            ate_berry: false,
            item_knocked: false,
            illusion: false,
        }
    }

    #[inline]
    pub fn alive(&self) -> bool {
        !self.fainted && self.hp > 0
    }

    #[inline]
    pub fn active(&self) -> bool {
        self.slot >= 0
    }

    #[inline]
    pub fn has_type(&self, t: Type) -> bool {
        self.types[0] == t || self.types[1] == t
    }

    pub fn move_slot(&self, m: MoveId) -> Option<usize> {
        (0..self.n_moves as usize).find(|&i| self.moves[i] == m)
    }

    pub fn hp_fraction(&self) -> f32 {
        if self.max_hp == 0 {
            0.0
        } else {
            self.hp as f32 / self.max_hp as f32
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct SideConds {
    pub tailwind: u8,
    pub reflect: u8,
    pub light_screen: u8,
    pub aurora_veil: u8,
    pub safeguard: u8,
    pub mist: u8,
    pub wide_guard: bool,
    pub quick_guard: bool,
    pub stealth_rock: bool,
    pub spikes: u8,
    pub toxic_spikes: u8,
    pub sticky_web: bool,
}

#[derive(Copy, Clone, Debug)]
pub struct Side {
    pub mons: [Mon; 6],
    /// Mon index (into `mons`) in each active slot, or NO_MON.
    pub active: [u8; 2],
    pub conds: SideConds,
    pub mega_used: bool,
    pub total_fainted: u8,
    pub fainted_this_turn: bool,
    pub fainted_last_turn: bool,
    /// Healing Wish waiting for the next Pokemon into each slot.
    pub healing_wish: [bool; 2],
    /// Wish: turns until it lands and HP it restores, per slot.
    pub wish: [(u8, u16); 2],
    /// Open team sheets: the opponent sees moves, items and abilities.
    pub sheet_open: bool,
}

impl Side {
    #[inline]
    pub fn mon(&self, slot: usize) -> Option<&Mon> {
        let i = self.active[slot];
        if i == NO_MON {
            None
        } else {
            Some(&self.mons[i as usize])
        }
    }

    /// Brought, alive, not on the field.
    pub fn can_switch_to(&self, i: usize) -> bool {
        let m = &self.mons[i];
        m.brought && m.alive() && !m.active()
    }

    pub fn bench_available(&self) -> u8 {
        (0..6).filter(|&i| self.can_switch_to(i)).count() as u8
    }

    pub fn alive_brought(&self) -> u8 {
        self.mons.iter().filter(|m| m.brought && m.alive()).count() as u8
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Field {
    pub weather: Weather,
    pub weather_turns: u8,
    pub terrain: Terrain,
    pub terrain_turns: u8,
    pub trick_room: u8,
    pub gravity: u8,
}

/// Position of a Pokemon on the field: side and slot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pos {
    pub side: u8,
    pub slot: u8,
}

impl Pos {
    #[inline]
    pub fn new(side: usize, slot: usize) -> Pos {
        Pos { side: side as u8, slot: slot as u8 }
    }
    #[inline]
    pub fn s(self) -> usize {
        self.side as usize
    }
    #[inline]
    pub fn i(self) -> usize {
        self.slot as usize
    }
    #[inline]
    pub fn ally(self) -> Pos {
        Pos { side: self.side, slot: 1 - self.slot }
    }
    #[inline]
    pub fn foe(self, slot: usize) -> Pos {
        Pos { side: 1 - self.side, slot: slot as u8 }
    }
    #[inline]
    pub fn code(self) -> u8 {
        self.side * 2 + self.slot
    }
    #[inline]
    pub fn from_code(c: u8) -> Pos {
        Pos { side: c / 2, slot: c % 2 }
    }
}
