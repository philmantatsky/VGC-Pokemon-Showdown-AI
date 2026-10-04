//! Batched self-play environment: many battles stepped in parallel, with
//! observations written straight into flat buffers.
//!
//! Every env has two players. Each step takes both players' choices
//! (`[preview, slot0, slot1]` per player); players with nothing to choose are
//! ignored. Finished battles reset automatically with freshly sampled teams.

use rayon::prelude::*;

use crate::actions::{Choice, N_SLOT_ACTIONS};
use crate::battle::{Battle, BattleConfig, RequestKind};
use crate::obs;
use crate::rng::Rng;
use crate::teams::TeamSpec;

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub turn_limit: u16,
    /// Probability that each side's team sheet is open to the opponent.
    pub open_sheet_prob: f32,
    pub ally_damage_targeting: bool,
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig { turn_limit: 100, open_sheet_prob: 0.5, ally_damage_targeting: false }
    }
}

/// Weighted team sampler (cumulative weights).
#[derive(Clone, Debug)]
pub struct TeamSampler {
    cum: Vec<f64>,
}

impl TeamSampler {
    pub fn uniform(n: usize) -> TeamSampler {
        TeamSampler::weighted(&vec![1.0; n])
    }
    pub fn weighted(w: &[f64]) -> TeamSampler {
        let mut cum = Vec::with_capacity(w.len());
        let mut s = 0.0;
        for &x in w {
            s += x.max(0.0);
            cum.push(s);
        }
        TeamSampler { cum }
    }
    pub fn sample(&self, rng: &mut Rng) -> usize {
        let total = *self.cum.last().unwrap_or(&0.0);
        if total <= 0.0 {
            return rng.below(self.cum.len().max(1) as u32) as usize;
        }
        let x = rng.f32() as f64 * total;
        match self.cum.binary_search_by(|c| c.partial_cmp(&x).unwrap()) {
            Ok(i) => (i + 1).min(self.cum.len() - 1),
            Err(i) => i.min(self.cum.len() - 1),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EpisodeInfo {
    pub team: [u32; 2],
    pub winner: i8,
    pub turns: u16,
    pub done: bool,
}

pub struct Env {
    pub battle: Battle,
    pub team: [u32; 2],
    pub rng: Rng,
}

pub struct VecEnv {
    pub teams: Vec<TeamSpec>,
    pub envs: Vec<Env>,
    pub cfg: EnvConfig,
    pub samplers: [TeamSampler; 2],
    pub last_info: Vec<EpisodeInfo>,
}

impl VecEnv {
    pub fn new(teams: Vec<TeamSpec>, n: usize, seed: u64, cfg: EnvConfig) -> VecEnv {
        assert!(!teams.is_empty(), "VecEnv needs at least one team");
        let nt = teams.len();
        let mut root = Rng::new(seed);
        let samplers = [TeamSampler::uniform(nt), TeamSampler::uniform(nt)];
        let mut v = VecEnv { teams, envs: Vec::with_capacity(n), cfg, samplers, last_info: vec![EpisodeInfo::default(); n] };
        for _ in 0..n {
            let mut rng = root.fork();
            let (battle, team) = v.new_battle(&mut rng);
            v.envs.push(Env { battle, team, rng });
        }
        v
    }

    fn new_battle(&self, rng: &mut Rng) -> (Battle, [u32; 2]) {
        let a = self.samplers[0].sample(rng);
        let b = self.samplers[1].sample(rng);
        let cfg = BattleConfig {
            turn_limit: self.cfg.turn_limit,
            ally_damage_targeting: self.cfg.ally_damage_targeting,
            log: false,
            sheets: [rng.f32() < self.cfg.open_sheet_prob, rng.f32() < self.cfg.open_sheet_prob],
        };
        let seed = rng.next_u64();
        (Battle::new([&self.teams[a], &self.teams[b]], seed, cfg), [a as u32, b as u32])
    }

    pub fn set_team_weights(&mut self, side: usize, w: &[f64]) {
        assert_eq!(w.len(), self.teams.len());
        self.samplers[side] = TeamSampler::weighted(w);
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Write observations for both players of every env.
    /// Buffers are [n_envs, 2, ...] flattened.
    pub fn observe(&self, ints: &mut [i32], floats: &mut [f32], field: &mut [f32], mask: &mut [u8], req: &mut [i32]) {
        let (ni, nf, nfield, nm) = obs::sizes();
        let n = self.envs.len();
        assert_eq!(ints.len(), n * 2 * ni);
        assert_eq!(floats.len(), n * 2 * nf);
        assert_eq!(field.len(), n * 2 * nfield);
        assert_eq!(mask.len(), n * 2 * nm);
        assert_eq!(req.len(), n * 2 * 3);
        ints.par_chunks_mut(2 * ni)
            .zip(floats.par_chunks_mut(2 * nf))
            .zip(field.par_chunks_mut(2 * nfield))
            .zip(mask.par_chunks_mut(2 * nm))
            .zip(req.par_chunks_mut(6))
            .zip(self.envs.par_iter())
            .for_each(|(((((ints, floats), field), mask), req), env)| {
                for s in 0..2 {
                    obs::encode(
                        &env.battle,
                        s,
                        &mut ints[s * ni..(s + 1) * ni],
                        &mut floats[s * nf..(s + 1) * nf],
                        &mut field[s * nfield..(s + 1) * nfield],
                        &mut mask[s * nm..(s + 1) * nm],
                    );
                    let r = env.battle.request(s);
                    req[s * 3] = match r.kind {
                        RequestKind::Wait => 0,
                        RequestKind::TeamPreview => 1,
                        RequestKind::Move => 2,
                        RequestKind::Switch => 3,
                    };
                    req[s * 3 + 1] = r.slots[0] as i32;
                    req[s * 3 + 2] = r.slots[1] as i32;
                }
            });
    }

    /// Step every env with `actions` = [n_envs, 2 players, 3] (preview, slot0, slot1).
    /// Writes rewards [n_envs, 2], dones [n_envs]; finished envs reset.
    /// Illegal choices end the episode as a loss for the offending side.
    pub fn step(&mut self, actions: &[u8], rewards: &mut [f32], dones: &mut [u8]) -> usize {
        let n = self.envs.len();
        assert_eq!(actions.len(), n * 6);
        assert_eq!(rewards.len(), n * 2);
        assert_eq!(dones.len(), n);
        let errors = std::sync::atomic::AtomicUsize::new(0);
        let teams = &self.teams;
        let cfg = &self.cfg;
        let samplers = &self.samplers;
        self.envs
            .par_iter_mut()
            .zip(actions.par_chunks(6))
            .zip(rewards.par_chunks_mut(2))
            .zip(dones.par_iter_mut())
            .zip(self.last_info.par_iter_mut())
            .for_each(|((((env, a), r), d), info)| {
                r[0] = 0.0;
                r[1] = 0.0;
                *d = 0;
                info.done = false;
                let choices = [
                    Choice { preview: a[0], slots: [a[1], a[2]] },
                    Choice { preview: a[3], slots: [a[4], a[5]] },
                ];
                let mut forfeit: Option<usize> = None;
                if let Err(e) = env.battle.step(choices) {
                    errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // Attribute the error to the side whose choice is illegal.
                    forfeit = Some(if e.0.starts_with("side 1") { 1 } else { 0 });
                }
                let winner: Option<i8> = match forfeit {
                    Some(s) => Some(1 - s as i8),
                    None => env.battle.winner.map(|w| w as i8),
                };
                if let Some(w) = winner {
                    match w {
                        0 => {
                            r[0] = 1.0;
                            r[1] = -1.0;
                        }
                        1 => {
                            r[0] = -1.0;
                            r[1] = 1.0;
                        }
                        _ => {}
                    }
                    *d = 1;
                    *info = EpisodeInfo { team: env.team, winner: w, turns: env.battle.turn, done: true };
                    // Reset with new teams.
                    let ta = samplers[0].sample(&mut env.rng);
                    let tb = samplers[1].sample(&mut env.rng);
                    let bc = BattleConfig {
                        turn_limit: cfg.turn_limit,
                        ally_damage_targeting: cfg.ally_damage_targeting,
                        log: false,
                        sheets: [env.rng.f32() < cfg.open_sheet_prob, env.rng.f32() < cfg.open_sheet_prob],
                    };
                    let seed = env.rng.next_u64();
                    env.battle = Battle::new([&teams[ta], &teams[tb]], seed, bc);
                    env.team = [ta as u32, tb as u32];
                }
            });
        errors.into_inner()
    }
}

/// Number of observation floats etc. per player, for Python.
pub fn obs_dims() -> [usize; 6] {
    [obs::N_MONS, obs::MON_INT, obs::MON_FLOAT, obs::FIELD_FLOAT, 2, N_SLOT_ACTIONS]
}
