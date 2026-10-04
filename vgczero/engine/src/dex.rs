//! Static game data, loaded once from the JSON exported by
//! `showdown/export_dex.js`. Everything is indexed by small integer ids so
//! battle state stays `Copy` and cheap to clone.

use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

use serde_json::Value;

use crate::kinds::{Ab, It, MoveFx};
use crate::types::{Eff, Type, TypeChart, N_TYPES};

pub type SpeciesId = u16;
pub type MoveId = u16;
pub type ItemId = u16;
pub type AbilityId = u16;

/// Id 0 in every table is a reserved "none" entry.
pub const NONE: u16 = 0;

// Stat indices.
pub const HP: usize = 0;
pub const ATK: usize = 1;
pub const DEF: usize = 2;
pub const SPA: usize = 3;
pub const SPD: usize = 4;
pub const SPE: usize = 5;
// Boost indices (boosts[0..5] = atk..spe, then accuracy, evasion).
pub const B_ATK: usize = 0;
pub const B_DEF: usize = 1;
pub const B_SPA: usize = 2;
pub const B_SPD: usize = 3;
pub const B_SPE: usize = 4;
pub const B_ACC: usize = 5;
pub const B_EVA: usize = 6;

pub type Boosts = [i8; 7];

fn boost_index(name: &str) -> Option<usize> {
    Some(match name {
        "atk" => B_ATK,
        "def" => B_DEF,
        "spa" => B_SPA,
        "spd" => B_SPD,
        "spe" => B_SPE,
        "accuracy" => B_ACC,
        "evasion" => B_EVA,
        _ => return None,
    })
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Category {
    Physical,
    Special,
    #[default]
    Status,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Target {
    #[default]
    Normal,
    Any,
    AdjacentAlly,
    AdjacentAllyOrSelf,
    AdjacentFoe,
    AllAdjacent,
    AllAdjacentFoes,
    Allies,
    AllySide,
    AllyTeam,
    All,
    FoeSide,
    RandomNormal,
    Scripted,
    User,
}

impl Target {
    fn parse(s: &str) -> Target {
        match s {
            "normal" => Target::Normal,
            "any" => Target::Any,
            "adjacentAlly" => Target::AdjacentAlly,
            "adjacentAllyOrSelf" => Target::AdjacentAllyOrSelf,
            "adjacentFoe" => Target::AdjacentFoe,
            "allAdjacent" => Target::AllAdjacent,
            "allAdjacentFoes" => Target::AllAdjacentFoes,
            "allies" => Target::Allies,
            "allySide" => Target::AllySide,
            "allyTeam" => Target::AllyTeam,
            "all" => Target::All,
            "foeSide" => Target::FoeSide,
            "randomNormal" => Target::RandomNormal,
            "scripted" => Target::Scripted,
            _ => Target::User,
        }
    }

    /// The player picks a target slot for these.
    pub fn needs_choice(self) -> bool {
        matches!(self, Target::Normal | Target::Any | Target::AdjacentFoe | Target::AdjacentAllyOrSelf)
    }

    pub fn is_spread(self) -> bool {
        matches!(self, Target::AllAdjacent | Target::AllAdjacentFoes)
    }
}

/// Move flags (subset of Showdown's `flags`).
pub mod flag {
    pub const CONTACT: u32 = 1 << 0;
    pub const PROTECT: u32 = 1 << 1;
    pub const SOUND: u32 = 1 << 2;
    pub const PUNCH: u32 = 1 << 3;
    pub const BITE: u32 = 1 << 4;
    pub const SLICING: u32 = 1 << 5;
    pub const PULSE: u32 = 1 << 6;
    pub const BULLET: u32 = 1 << 7;
    pub const POWDER: u32 = 1 << 8;
    pub const WIND: u32 = 1 << 9;
    pub const DANCE: u32 = 1 << 10;
    pub const HEAL: u32 = 1 << 11;
    pub const RECHARGE: u32 = 1 << 12;
    pub const CHARGE: u32 = 1 << 13;
    pub const DEFROST: u32 = 1 << 14;
    pub const BYPASSSUB: u32 = 1 << 15;
    pub const REFLECTABLE: u32 = 1 << 16;
    pub const MIRROR: u32 = 1 << 17;
    pub const GRAVITY: u32 = 1 << 18;
    pub const FAILENCORE: u32 = 1 << 19;
    pub const FAILINSTRUCT: u32 = 1 << 20;
    pub const CANTUSETWICE: u32 = 1 << 21;
    pub const NOSLEEPTALK: u32 = 1 << 22;
    pub const SNATCH: u32 = 1 << 23;
    pub const ALLYANIM: u32 = 1 << 24;
    pub const FUTUREMOVE: u32 = 1 << 25;
    pub const MUSTPRESSURE: u32 = 1 << 26;
    pub const NOPARENTALBOND: u32 = 1 << 27;

    pub fn parse(name: &str) -> u32 {
        match name {
            "contact" => CONTACT,
            "protect" => PROTECT,
            "sound" => SOUND,
            "punch" => PUNCH,
            "bite" => BITE,
            "slicing" => SLICING,
            "pulse" => PULSE,
            "bullet" => BULLET,
            "powder" => POWDER,
            "wind" => WIND,
            "dance" => DANCE,
            "heal" => HEAL,
            "recharge" => RECHARGE,
            "charge" => CHARGE,
            "defrost" => DEFROST,
            "bypasssub" => BYPASSSUB,
            "reflectable" => REFLECTABLE,
            "mirror" => MIRROR,
            "gravity" => GRAVITY,
            "failencore" => FAILENCORE,
            "failinstruct" => FAILINSTRUCT,
            "cantusetwice" => CANTUSETWICE,
            "nosleeptalk" => NOSLEEPTALK,
            "snatch" => SNATCH,
            "allyanim" => ALLYANIM,
            "futuremove" => FUTUREMOVE,
            "mustpressure" => MUSTPRESSURE,
            "noparentalbond" => NOPARENTALBOND,
            _ => 0,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Status {
    #[default]
    None = 0,
    Brn,
    Par,
    Slp,
    Frz,
    Psn,
    Tox,
}

impl Status {
    fn parse(s: &str) -> Status {
        match s {
            "brn" => Status::Brn,
            "par" => Status::Par,
            "slp" => Status::Slp,
            "frz" => Status::Frz,
            "psn" => Status::Psn,
            "tox" => Status::Tox,
            _ => Status::None,
        }
    }
}

/// Volatile statuses a move (or secondary) can inflict, by name.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VolKind {
    Flinch,
    Confusion,
    MustRecharge,
    Taunt,
    Encore,
    Disable,
    Yawn,
    FocusEnergy,
    Substitute,
    LeechSeed,
    HelpingHand,
    FollowMe,
    RagePowder,
    Protect,
    Endure,
    Roost,
    Imprison,
    ThroatChop,
    GlaiveRush,
    Curse,
    SaltCure,
    DestinyBond,
    Torment,
    HealBlock,
    MagnetRise,
    AquaRing,
    Ingrain,
    Charge,
    Stockpile,
    Attract,
    PartiallyTrapped,
    NoRetreat,
    LaserFocus,
    DragonCheer,
    TarShot,
    SmackDown,
    Unsupported,
}

impl VolKind {
    fn parse(s: &str) -> VolKind {
        match s {
            "flinch" => VolKind::Flinch,
            "confusion" => VolKind::Confusion,
            "mustrecharge" => VolKind::MustRecharge,
            "taunt" => VolKind::Taunt,
            "encore" => VolKind::Encore,
            "disable" => VolKind::Disable,
            "yawn" => VolKind::Yawn,
            "focusenergy" => VolKind::FocusEnergy,
            "substitute" => VolKind::Substitute,
            "leechseed" => VolKind::LeechSeed,
            "helpinghand" => VolKind::HelpingHand,
            "followme" => VolKind::FollowMe,
            "ragepowder" => VolKind::RagePowder,
            "protect" | "spikyshield" | "banefulbunker" | "kingsshield" | "silktrap"
            | "burningbulwark" | "obstruct" => VolKind::Protect,
            "endure" => VolKind::Endure,
            "roost" => VolKind::Roost,
            "imprison" => VolKind::Imprison,
            "throatchop" => VolKind::ThroatChop,
            "glaiverush" => VolKind::GlaiveRush,
            "curse" => VolKind::Curse,
            "saltcure" => VolKind::SaltCure,
            "destinybond" => VolKind::DestinyBond,
            "torment" => VolKind::Torment,
            "healblock" => VolKind::HealBlock,
            "magnetrise" => VolKind::MagnetRise,
            "aquaring" => VolKind::AquaRing,
            "ingrain" => VolKind::Ingrain,
            "charge" => VolKind::Charge,
            "stockpile" => VolKind::Stockpile,
            "attract" => VolKind::Attract,
            "partiallytrapped" => VolKind::PartiallyTrapped,
            "noretreat" => VolKind::NoRetreat,
            "laserfocus" => VolKind::LaserFocus,
            "dragoncheer" => VolKind::DragonCheer,
            "tarshot" => VolKind::TarShot,
            "smackdown" => VolKind::SmackDown,
            _ => VolKind::Unsupported,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SideCond {
    Tailwind,
    Reflect,
    LightScreen,
    AuroraVeil,
    WideGuard,
    QuickGuard,
    Safeguard,
    Mist,
    StealthRock,
    Spikes,
    ToxicSpikes,
    StickyWeb,
    CraftyShield,
    MatBlock,
    Unsupported,
}

impl SideCond {
    fn parse(s: &str) -> SideCond {
        match s {
            "tailwind" => SideCond::Tailwind,
            "reflect" => SideCond::Reflect,
            "lightscreen" => SideCond::LightScreen,
            "auroraveil" => SideCond::AuroraVeil,
            "wideguard" => SideCond::WideGuard,
            "quickguard" => SideCond::QuickGuard,
            "safeguard" => SideCond::Safeguard,
            "mist" => SideCond::Mist,
            "stealthrock" => SideCond::StealthRock,
            "spikes" => SideCond::Spikes,
            "toxicspikes" => SideCond::ToxicSpikes,
            "stickyweb" => SideCond::StickyWeb,
            "craftyshield" => SideCond::CraftyShield,
            "matblock" => SideCond::MatBlock,
            _ => SideCond::Unsupported,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Weather {
    #[default]
    None = 0,
    Rain,
    Sun,
    Sand,
    Snow,
}

impl Weather {
    fn parse(s: &str) -> Weather {
        match s {
            "RainDance" | "raindance" => Weather::Rain,
            "SunnyDay" | "sunnyday" => Weather::Sun,
            "Sandstorm" | "sandstorm" => Weather::Sand,
            "Snowscape" | "snowscape" | "snow" => Weather::Snow,
            _ => Weather::None,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Terrain {
    #[default]
    None = 0,
    Grassy,
    Psychic,
    Electric,
    Misty,
}

impl Terrain {
    fn parse(s: &str) -> Terrain {
        match s {
            "grassyterrain" => Terrain::Grassy,
            "psychicterrain" => Terrain::Psychic,
            "electricterrain" => Terrain::Electric,
            "mistyterrain" => Terrain::Misty,
            _ => Terrain::None,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PseudoWeather {
    TrickRoom,
    Gravity,
    WonderRoom,
    MagicRoom,
    Unsupported,
}

impl PseudoWeather {
    fn parse(s: &str) -> PseudoWeather {
        match s {
            "trickroom" => PseudoWeather::TrickRoom,
            "gravity" => PseudoWeather::Gravity,
            "wonderroom" => PseudoWeather::WonderRoom,
            "magicroom" => PseudoWeather::MagicRoom,
            _ => PseudoWeather::Unsupported,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Secondary {
    /// Percent chance (100 = always).
    pub chance: u8,
    pub status: Status,
    pub volatile: Option<VolKind>,
    pub boosts: Option<Boosts>,
    pub self_boosts: Option<Boosts>,
    /// Has a scripted onHit (Dire Claw, Tri Attack, Throat Chop...): the
    /// move's `MoveFx` applies it.
    pub scripted: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MoveData {
    pub id: String,
    pub name: String,
    pub num: i32,
    pub ty: Type,
    pub category: Category,
    pub base_power: u16,
    /// 0 means "always hits" (Showdown's `accuracy: true`).
    pub accuracy: u8,
    pub pp: u8,
    pub priority: i8,
    pub target: Target,
    pub flags: u32,
    pub crit_ratio: u8,
    pub will_crit: bool,
    pub multihit: (u8, u8),
    pub multiaccuracy: bool,
    pub drain: (u16, u16),
    pub recoil: (u16, u16),
    pub heal: (u16, u16),
    pub secondaries: Vec<Secondary>,
    pub self_boosts: Option<Boosts>,
    /// `selfBoost`: applied once after the move succeeds (Clanging Scales,
    /// Scale Shot), not removed by Sheer Force.
    pub self_boost_after: Option<Boosts>,
    pub self_volatile: Option<VolKind>,
    pub boosts: Option<Boosts>,
    pub status: Status,
    pub volatile: Option<VolKind>,
    pub side_condition: Option<SideCond>,
    pub pseudo_weather: Option<PseudoWeather>,
    pub weather: Weather,
    pub terrain: Terrain,
    /// 0 none, 1 switch (U-turn), 2 copyvolatile (Baton Pass), 3 shedtail.
    pub self_switch: u8,
    pub force_switch: bool,
    /// 0 none, 1 always (Explosion), 2 ifHit, 3 always but after (Healing Wish style).
    pub selfdestruct: u8,
    pub breaks_protect: bool,
    pub ignore_evasion: bool,
    pub ignore_defensive: bool,
    pub ignore_immunity: bool,
    pub ignore_ability: bool,
    pub ohko: bool,
    pub crash_damage: bool,
    pub mind_blown_recoil: bool,
    pub struggle_recoil: bool,
    pub stalling: bool,
    pub thaws_target: bool,
    pub sleep_usable: bool,
    pub smart_target: bool,
    pub override_offensive_stat: Option<usize>,
    pub override_defensive_stat: Option<usize>,
    pub foul_play: bool,
    pub has_sheer_force_boost: bool,
    pub fx: MoveFx,
    pub callbacks: Vec<String>,
    /// Why the engine cannot run this move faithfully (empty if it can).
    pub unsupported: Vec<String>,
}

impl MoveData {
    #[inline]
    pub fn has(&self, f: u32) -> bool {
        self.flags & f != 0
    }
    pub fn is_damaging(&self) -> bool {
        self.category != Category::Status
    }
    pub fn supported(&self) -> bool {
        self.unsupported.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpeciesData {
    pub id: String,
    pub name: String,
    pub num: i32,
    pub base_species: String,
    pub types: [Type; 2],
    pub base: [u16; 6],
    pub weight_hg: u32,
    pub abilities: Vec<String>,
    pub is_mega: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ItemData {
    pub id: String,
    pub name: String,
    pub num: i32,
    pub kind: It,
    pub is_berry: bool,
    pub is_mega_stone: bool,
    pub fling_bp: u16,
    pub callbacks: Vec<String>,
    pub unsupported: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct AbilityData {
    pub id: String,
    pub name: String,
    pub num: i32,
    pub kind: Ab,
    pub breakable: bool,
    pub cantsuppress: bool,
    pub callbacks: Vec<String>,
    pub unsupported: Vec<String>,
}

pub struct Dex {
    pub format: String,
    pub chart: TypeChart,
    pub species: Vec<SpeciesData>,
    pub moves: Vec<MoveData>,
    pub items: Vec<ItemData>,
    pub abilities: Vec<AbilityData>,
    pub species_by_id: HashMap<String, SpeciesId>,
    pub move_by_id: HashMap<String, MoveId>,
    pub item_by_id: HashMap<String, ItemId>,
    pub ability_by_id: HashMap<String, AbilityId>,
    /// Struggle (not in the legal move list, always added).
    pub struggle: MoveId,
    /// Confusion self-hit pseudo-move.
    pub confusion_hit: MoveId,
    /// Nature id -> (raised stat, lowered stat) as stat indices.
    pub natures: HashMap<String, (Option<usize>, Option<usize>)>,
}

static DEX: OnceLock<Dex> = OnceLock::new();

/// The process-wide dex. Panics if `load_dex` was never called.
#[inline]
pub fn dex() -> &'static Dex {
    DEX.get().expect("vgczero: dex not loaded (call dex::load_dex first)")
}

pub fn dex_loaded() -> bool {
    DEX.get().is_some()
}

/// Load the dex from a `.json` or `.json.gz` file (first call wins).
pub fn load_dex(path: &str) -> Result<&'static Dex, String> {
    if let Some(d) = DEX.get() {
        return Ok(d);
    }
    let v = read_json(path)?;
    let d = Dex::from_json(&v)?;
    let _ = DEX.set(d);
    Ok(DEX.get().unwrap())
}

pub fn read_json(path: &str) -> Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    let text = if path.ends_with(".gz") {
        let mut s = String::new();
        flate2::read::GzDecoder::new(&bytes[..])
            .read_to_string(&mut s)
            .map_err(|e| format!("gunzip {path}: {e}"))?;
        s
    } else {
        String::from_utf8(bytes).map_err(|e| format!("{path}: {e}"))?
    };
    serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))
}

fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
}
fn n(v: &Value, k: &str) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0)
}
fn b(v: &Value, k: &str) -> bool {
    v.get(k).map(|x| x.as_bool().unwrap_or(!x.is_null())).unwrap_or(false)
}
fn strs(v: Option<&Value>) -> Vec<String> {
    v.and_then(|x| x.as_array())
        .map(|a| a.iter().filter_map(|y| y.as_str().map(String::from)).collect())
        .unwrap_or_default()
}
fn frac(v: &Value, k: &str) -> (u16, u16) {
    match v.get(k).and_then(|x| x.as_array()) {
        Some(a) if a.len() == 2 => (a[0].as_u64().unwrap_or(0) as u16, a[1].as_u64().unwrap_or(1) as u16),
        _ => (0, 0),
    }
}
fn parse_boosts(v: Option<&Value>) -> Option<Boosts> {
    let o = v?.as_object()?;
    let mut out = [0i8; 7];
    let mut any = false;
    for (k, x) in o {
        if let Some(i) = boost_index(k) {
            out[i] = x.as_i64().unwrap_or(0) as i8;
            any = true;
        }
    }
    if any {
        Some(out)
    } else {
        None
    }
}
fn stat_index(name: &str) -> Option<usize> {
    Some(match name {
        "atk" => ATK,
        "def" => DEF,
        "spa" => SPA,
        "spd" => SPD,
        "spe" => SPE,
        _ => return None,
    })
}

impl Dex {
    pub fn from_json(v: &Value) -> Result<Dex, String> {
        let format = v.get("meta").map(|m| s(m, "format")).unwrap_or_default();

        // Type chart.
        let mut chart = TypeChart { chart: [[Eff::Neutral; N_TYPES]; N_TYPES] };
        let tc = v.get("typechart").and_then(|x| x.as_object()).ok_or("dex: no typechart")?;
        for (def, row) in tc {
            let dt = Type::from_name(def).ok_or(format!("bad type {def}"))?;
            for (atk, code) in row.as_object().ok_or("bad typechart row")? {
                let at = Type::from_name(atk).ok_or(format!("bad type {atk}"))?;
                chart.chart[at.idx()][dt.idx()] = match code.as_u64().unwrap_or(0) {
                    1 => Eff::Super,
                    2 => Eff::Resist,
                    3 => Eff::Immune,
                    _ => Eff::Neutral,
                };
            }
        }

        let none_species = SpeciesData { id: String::new(), name: "(none)".into(), ..Default::default() };
        let mut species = vec![none_species];
        for sp in v.get("species").and_then(|x| x.as_array()).ok_or("dex: no species")? {
            let tys = strs(sp.get("types"));
            let t0 = tys.first().and_then(|t| Type::from_name(t)).unwrap_or(Type::Normal);
            let t1 = tys.get(1).and_then(|t| Type::from_name(t)).unwrap_or(Type::None);
            let bs = sp.get("baseStats").ok_or("species without baseStats")?;
            let base = [n(bs, "hp"), n(bs, "atk"), n(bs, "def"), n(bs, "spa"), n(bs, "spd"), n(bs, "spe")]
                .map(|x| x as u16);
            let abilities = sp
                .get("abilities")
                .and_then(|x| x.as_object())
                .map(|o| o.values().filter_map(|y| y.as_str().map(String::from)).collect())
                .unwrap_or_default();
            species.push(SpeciesData {
                id: s(sp, "id"),
                name: s(sp, "name"),
                num: n(sp, "num") as i32,
                base_species: s(sp, "baseSpecies"),
                types: [t0, t1],
                base,
                weight_hg: (n(sp, "weightkg") * 10.0).round() as u32,
                abilities,
                is_mega: b(sp, "isMega"),
            });
        }

        let mut moves = vec![MoveData { name: "(none)".into(), ..Default::default() }];
        for mv in v.get("moves").and_then(|x| x.as_array()).ok_or("dex: no moves")? {
            moves.push(parse_move(mv));
        }
        // Struggle and the confusion self-hit are never legal choices but the
        // engine uses them.
        let struggle = moves.len() as MoveId;
        moves.push(MoveData {
            id: "struggle".into(),
            name: "Struggle".into(),
            ty: Type::None,
            category: Category::Physical,
            base_power: 50,
            accuracy: 0,
            pp: 1,
            target: Target::RandomNormal,
            flags: flag::CONTACT | flag::PROTECT,
            crit_ratio: 1,
            struggle_recoil: true,
            fx: MoveFx::Other,
            ..Default::default()
        });
        let confusion_hit = moves.len() as MoveId;
        moves.push(MoveData {
            id: "confused".into(),
            name: "confusion".into(),
            ty: Type::None,
            category: Category::Physical,
            base_power: 40,
            accuracy: 0,
            pp: 1,
            target: Target::User,
            crit_ratio: 0,
            ..Default::default()
        });

        let mut items = vec![ItemData { name: "(none)".into(), ..Default::default() }];
        for it in v.get("items").and_then(|x| x.as_array()).ok_or("dex: no items")? {
            let id = s(it, "id");
            let kind = It::from_id(&id);
            let callbacks = strs(it.get("callbacks"));
            let is_mega_stone = it.get("megaStone").map(|x| !x.is_null()).unwrap_or(false);
            let mut unsupported = vec![];
            if kind == It::Other && !callbacks.is_empty() && !is_mega_stone {
                unsupported.push(format!("item {id}"));
            }
            items.push(ItemData {
                name: s(it, "name"),
                num: n(it, "num") as i32,
                kind,
                is_berry: b(it, "isBerry"),
                is_mega_stone,
                fling_bp: it.get("fling").map(|f| n(f, "basePower") as u16).unwrap_or(0),
                callbacks,
                unsupported,
                id,
            });
        }

        let mut abilities = vec![AbilityData { name: "(none)".into(), ..Default::default() }];
        for ab in v.get("abilities").and_then(|x| x.as_array()).ok_or("dex: no abilities")? {
            let id = s(ab, "id");
            let kind = Ab::from_id(&id);
            let callbacks = strs(ab.get("callbacks"));
            let fl = ab.get("flags");
            let mut unsupported = vec![];
            if kind == Ab::Other && !callbacks.is_empty() {
                unsupported.push(format!("ability {id}"));
            }
            abilities.push(AbilityData {
                name: s(ab, "name"),
                num: n(ab, "num") as i32,
                kind,
                breakable: fl.map(|f| b(f, "breakable")).unwrap_or(false),
                cantsuppress: fl.map(|f| b(f, "cantsuppress")).unwrap_or(false),
                callbacks,
                unsupported,
                id,
            });
        }

        let index = |names: Vec<&String>| -> HashMap<String, u16> {
            names.into_iter().enumerate().filter(|(_, n)| !n.is_empty()).map(|(i, n)| (n.clone(), i as u16)).collect()
        };
        let species_by_id = index(species.iter().map(|x| &x.id).collect());
        let move_by_id = index(moves.iter().map(|x| &x.id).collect());
        let item_by_id = index(items.iter().map(|x| &x.id).collect());
        let ability_by_id = index(abilities.iter().map(|x| &x.id).collect());

        let mut natures = HashMap::new();
        for n in v.get("natures").and_then(|x| x.as_array()).into_iter().flatten() {
            natures.insert(s(n, "id"), (stat_index(&s(n, "plus")), stat_index(&s(n, "minus"))));
        }

        Ok(Dex {
            natures,
            format,
            chart,
            species,
            moves,
            items,
            abilities,
            species_by_id,
            move_by_id,
            item_by_id,
            ability_by_id,
            struggle,
            confusion_hit,
        })
    }

    #[inline]
    pub fn mv(&self, id: MoveId) -> &MoveData {
        &self.moves[id as usize]
    }
    #[inline]
    pub fn sp(&self, id: SpeciesId) -> &SpeciesData {
        &self.species[id as usize]
    }
    #[inline]
    pub fn item(&self, id: ItemId) -> &ItemData {
        &self.items[id as usize]
    }
    #[inline]
    pub fn ability(&self, id: AbilityId) -> &AbilityData {
        &self.abilities[id as usize]
    }
    #[inline]
    pub fn it(&self, id: ItemId) -> It {
        self.items[id as usize].kind
    }
    #[inline]
    pub fn ab(&self, id: AbilityId) -> Ab {
        self.abilities[id as usize].kind
    }
    pub fn move_id(&self, s: &str) -> Option<MoveId> {
        self.move_by_id.get(s).copied()
    }
    pub fn species_id(&self, s: &str) -> Option<SpeciesId> {
        self.species_by_id.get(s).copied()
    }
    pub fn item_id(&self, s: &str) -> Option<ItemId> {
        if s.is_empty() {
            return Some(NONE);
        }
        self.item_by_id.get(s).copied()
    }
    pub fn ability_id(&self, s: &str) -> Option<AbilityId> {
        self.ability_by_id.get(s).copied()
    }
}

/// Data fields the generic move pipeline does not handle; a move that has any
/// of them must have a `MoveFx` that does.
const UNHANDLED_FIELDS: &[&str] = &[
    "damage", "slotCondition", "callsMove", "tracksTarget", "multihitType",
    "overrideDefensivePokemon", "pressureTarget", "nonGhostTarget", "basePowerCallback",
    "smartTarget", "isFutureMove",
];

fn parse_move(mv: &Value) -> MoveData {
    let id = s(mv, "id");
    let fx = MoveFx::from_id(&id);
    let mut flags = 0u32;
    if let Some(o) = mv.get("flags").and_then(|x| x.as_object()) {
        for (k, x) in o {
            if x.as_i64().unwrap_or(0) != 0 {
                flags |= flag::parse(k);
            }
        }
    }
    let mut unsupported: Vec<String> = vec![];
    let callbacks = strs(mv.get("callbacks"));
    let cond_callbacks = strs(mv.get("conditionCallbacks"));

    let mut secondaries = vec![];
    if let Some(arr) = mv.get("secondaries").and_then(|x| x.as_array()) {
        for sec in arr {
            let scripted = mv
                .get("secondary")
                .and_then(|x| x.get("callbacks"))
                .map(|c| !c.as_array().map(|a| a.is_empty()).unwrap_or(true))
                .unwrap_or(false)
                || sec.get("callbacks").is_some();
            let volatile = sec.get("volatileStatus").and_then(|x| x.as_str()).map(VolKind::parse);
            if volatile == Some(VolKind::Unsupported) {
                unsupported.push(format!("secondary volatile {}", s(sec, "volatileStatus")));
            }
            secondaries.push(Secondary {
                chance: sec.get("chance").and_then(|x| x.as_u64()).unwrap_or(100) as u8,
                status: Status::parse(&s(sec, "status")),
                volatile,
                boosts: parse_boosts(sec.get("boosts")),
                self_boosts: sec.get("self").and_then(|x| parse_boosts(x.get("boosts"))),
                scripted,
            });
        }
    }
    let self_obj = mv.get("self");
    let self_volatile = self_obj.and_then(|x| x.get("volatileStatus")).and_then(|x| x.as_str()).map(VolKind::parse);
    if self_volatile == Some(VolKind::Unsupported) {
        unsupported.push("self volatile".into());
    }
    let volatile = mv.get("volatileStatus").and_then(|x| x.as_str()).map(VolKind::parse);
    if volatile == Some(VolKind::Unsupported) {
        unsupported.push(format!("volatile {}", s(mv, "volatileStatus")));
    }
    let side_condition = mv.get("sideCondition").and_then(|x| x.as_str()).map(SideCond::parse);
    if side_condition == Some(SideCond::Unsupported) {
        unsupported.push(format!("side condition {}", s(mv, "sideCondition")));
    }
    let pseudo_weather = mv.get("pseudoWeather").and_then(|x| x.as_str()).map(PseudoWeather::parse);
    if matches!(pseudo_weather, Some(PseudoWeather::Unsupported | PseudoWeather::WonderRoom | PseudoWeather::MagicRoom)) {
        unsupported.push(format!("pseudo weather {}", s(mv, "pseudoWeather")));
    }
    let multihit = match mv.get("multihit") {
        Some(Value::Number(x)) => {
            let h = x.as_u64().unwrap_or(1) as u8;
            (h, h)
        }
        Some(Value::Array(a)) if a.len() == 2 => (a[0].as_u64().unwrap_or(2) as u8, a[1].as_u64().unwrap_or(5) as u8),
        _ => (0, 0),
    };
    let self_switch = match mv.get("selfSwitch") {
        Some(Value::Bool(true)) => 1,
        Some(Value::String(x)) if x == "copyvolatile" => 2,
        Some(Value::String(x)) if x == "shedtail" => 3,
        _ => 0,
    };
    let selfdestruct = match mv.get("selfdestruct") {
        Some(Value::String(x)) if x == "always" => 1,
        Some(Value::String(x)) if x == "ifHit" => 2,
        Some(Value::Bool(true)) => 1,
        _ => 0,
    };

    // Support check: every scripted callback must be covered by a MoveFx.
    let scripted_secondary = secondaries.iter().any(|x| x.scripted);
    let self_scripted = self_obj.and_then(|x| x.get("callbacks")).is_some();
    if (!callbacks.is_empty() || !cond_callbacks.is_empty() || scripted_secondary || self_scripted)
        && fx == MoveFx::Other
        && !crate::kinds::move_callbacks_are_generic(&id)
    {
        unsupported.push(format!("scripted move {id}: {:?} cond {:?}", callbacks, cond_callbacks));
    }
    if let Some(sb) = mv.get("selfBoost").and_then(|x| x.as_object()) {
        if sb.keys().any(|k| k != "boosts") {
            unsupported.push("field selfBoost".into());
        }
    }
    for f in UNHANDLED_FIELDS {
        if mv.get(*f).is_some() && fx == MoveFx::Other {
            unsupported.push(format!("field {f}"));
        }
    }
    let accuracy = match mv.get("accuracy") {
        Some(Value::Bool(true)) => 0,
        Some(x) => x.as_u64().unwrap_or(100) as u8,
        None => 100,
    };
    let category = match s(mv, "category").as_str() {
        "Physical" => Category::Physical,
        "Special" => Category::Special,
        _ => Category::Status,
    };
    let ty = Type::from_name(&s(mv, "type")).unwrap_or(Type::Normal);
    MoveData {
        name: s(mv, "name"),
        num: n(mv, "num") as i32,
        ty,
        category,
        base_power: n(mv, "basePower") as u16,
        accuracy,
        pp: n(mv, "pp") as u8,
        priority: n(mv, "priority") as i8,
        target: Target::parse(&s(mv, "target")),
        flags,
        crit_ratio: mv.get("critRatio").and_then(|x| x.as_u64()).unwrap_or(1) as u8,
        will_crit: b(mv, "willCrit"),
        multihit,
        multiaccuracy: b(mv, "multiaccuracy"),
        drain: frac(mv, "drain"),
        recoil: frac(mv, "recoil"),
        heal: frac(mv, "heal"),
        secondaries,
        self_boosts: self_obj.and_then(|x| parse_boosts(x.get("boosts"))),
        self_boost_after: mv.get("selfBoost").and_then(|x| parse_boosts(x.get("boosts"))),
        self_volatile,
        boosts: parse_boosts(mv.get("boosts")),
        status: Status::parse(&s(mv, "status")),
        volatile,
        side_condition,
        pseudo_weather,
        weather: Weather::parse(&s(mv, "weather")),
        terrain: Terrain::parse(&s(mv, "terrain")),
        self_switch,
        force_switch: b(mv, "forceSwitch"),
        selfdestruct,
        breaks_protect: b(mv, "breaksProtect"),
        ignore_evasion: b(mv, "ignoreEvasion"),
        ignore_defensive: b(mv, "ignoreDefensive"),
        ignore_immunity: mv.get("ignoreImmunity").map(|x| x.as_bool().unwrap_or(true)).unwrap_or(false),
        ignore_ability: b(mv, "ignoreAbility"),
        ohko: b(mv, "ohko"),
        crash_damage: b(mv, "hasCrashDamage"),
        mind_blown_recoil: b(mv, "mindBlownRecoil"),
        struggle_recoil: b(mv, "struggleRecoil"),
        stalling: b(mv, "stallingMove"),
        thaws_target: b(mv, "thawsTarget"),
        sleep_usable: b(mv, "sleepUsable"),
        smart_target: b(mv, "smartTarget"),
        override_offensive_stat: stat_index(&s(mv, "overrideOffensiveStat")),
        override_defensive_stat: stat_index(&s(mv, "overrideDefensiveStat")),
        foul_play: s(mv, "overrideOffensivePokemon") == "target",
        has_sheer_force_boost: b(mv, "hasSheerForceBoost"),
        fx,
        callbacks,
        unsupported,
        id,
    }
}
