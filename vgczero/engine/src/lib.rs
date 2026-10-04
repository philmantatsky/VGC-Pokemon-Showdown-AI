//! vgczero engine: a fast, copyable Gen 9 Champions VGC doubles simulator
//! for self-play reinforcement learning and search.
//!
//! Data (dex, teams) is exported from Pokemon Showdown, which remains the
//! source of truth; `tests/` and `../showdown/parity` check the engine against it.

pub mod actions;
pub mod baseline;
pub mod battle;
pub mod dex;
pub mod kinds;
pub mod obs;
pub mod rng;
pub mod snapshot;
pub mod state;
pub mod teams;
pub mod types;
pub mod vecenv;
pub mod worlds;

#[cfg(feature = "python")]
mod py;

pub use actions::{Choice, SlotAction, N_PREVIEW_ACTIONS, N_SLOT_ACTIONS};
pub use battle::{Battle, BattleConfig, Phase, Request, RequestKind};
pub use dex::{dex, load_dex};
pub use teams::{load_teams, TeamSpec};
