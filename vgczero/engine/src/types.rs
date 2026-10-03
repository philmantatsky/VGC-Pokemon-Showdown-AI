//! Pokemon types and the type chart.

use serde::{Deserialize, Serialize};

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Type {
    Bug = 0,
    Dark,
    Dragon,
    Electric,
    Fairy,
    Fighting,
    Fire,
    Flying,
    Ghost,
    Grass,
    Ground,
    Ice,
    #[default]
    Normal,
    Poison,
    Psychic,
    Rock,
    Steel,
    Water,
    /// Typeless (Struggle, confusion self-hit) and "no second type".
    None,
}

pub const N_TYPES: usize = 18;

pub const ALL_TYPES: [Type; N_TYPES] = [
    Type::Bug,
    Type::Dark,
    Type::Dragon,
    Type::Electric,
    Type::Fairy,
    Type::Fighting,
    Type::Fire,
    Type::Flying,
    Type::Ghost,
    Type::Grass,
    Type::Ground,
    Type::Ice,
    Type::Normal,
    Type::Poison,
    Type::Psychic,
    Type::Rock,
    Type::Steel,
    Type::Water,
];

impl Type {
    pub fn from_name(s: &str) -> Option<Type> {
        ALL_TYPES.iter().copied().find(|t| t.name().eq_ignore_ascii_case(s))
    }

    pub fn name(self) -> &'static str {
        match self {
            Type::Bug => "Bug",
            Type::Dark => "Dark",
            Type::Dragon => "Dragon",
            Type::Electric => "Electric",
            Type::Fairy => "Fairy",
            Type::Fighting => "Fighting",
            Type::Fire => "Fire",
            Type::Flying => "Flying",
            Type::Ghost => "Ghost",
            Type::Grass => "Grass",
            Type::Ground => "Ground",
            Type::Ice => "Ice",
            Type::Normal => "Normal",
            Type::Poison => "Poison",
            Type::Psychic => "Psychic",
            Type::Rock => "Rock",
            Type::Steel => "Steel",
            Type::Water => "Water",
            Type::None => "???",
        }
    }

    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
}

/// Effectiveness of one attacking type against one defending type, in
/// Showdown's damageTaken encoding.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Eff {
    #[default]
    Neutral = 0,
    Super = 1,
    Resist = 2,
    Immune = 3,
}

/// `chart[atk][def]`.
#[derive(Clone, Debug)]
pub struct TypeChart {
    pub chart: [[Eff; N_TYPES]; N_TYPES],
}

impl TypeChart {
    #[inline]
    pub fn eff(&self, atk: Type, def: Type) -> Eff {
        if atk == Type::None || def == Type::None {
            return Eff::Neutral;
        }
        self.chart[atk.idx()][def.idx()]
    }

    /// Showdown's log2 effectiveness of `atk` into one defending type
    /// (ignoring immunity, which is checked separately).
    #[inline]
    pub fn log2(&self, atk: Type, def: Type) -> i32 {
        match self.eff(atk, def) {
            Eff::Super => 1,
            Eff::Resist => -1,
            _ => 0,
        }
    }

    #[inline]
    pub fn immune(&self, atk: Type, def: Type) -> bool {
        self.eff(atk, def) == Eff::Immune
    }
}
