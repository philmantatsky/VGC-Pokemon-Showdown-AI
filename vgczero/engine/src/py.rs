//! Python bindings (feature "python"; build with maturin).
//!
//!   import vgczero_engine as E
//!   E.load(dex_path, teams_path)
//!   env = E.VecEnv(n_envs=256, seed=0, team_indices=[...])
//!   ints, floats, field, mask, req = env.observe()
//!   rewards, dones, errors = env.step(actions)   # actions: uint8 [n, 2, 3]

use std::sync::OnceLock;

use numpy::ndarray::{ArrayView1, ArrayView3};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

use crate::actions::{preview_table, Choice, N_PREVIEW_ACTIONS, N_SLOT_ACTIONS};
use crate::battle::{Battle, BattleConfig, Phase, RequestKind};
use crate::obs;
use crate::rng::Rng;
use crate::teams::TeamSpec;
use crate::vecenv::{EnvConfig, VecEnv};
use crate::worlds;

static TEAMS: OnceLock<Vec<TeamSpec>> = OnceLock::new();

fn teams() -> PyResult<&'static Vec<TeamSpec>> {
    TEAMS.get().ok_or_else(|| PyRuntimeError::new_err("call vgczero_engine.load(dex, teams) first"))
}

fn to_np<'py, T: numpy::Element>(py: Python<'py>, v: Vec<T>, shape: &[usize]) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(v.into_pyarray(py).reshape(shape.to_vec())?)
}

/// Load the dex and team pool (once per process). Returns the number of teams.
#[pyfunction]
fn load(dex_path: &str, teams_path: &str) -> PyResult<usize> {
    crate::dex::load_dex(dex_path).map_err(PyRuntimeError::new_err)?;
    if TEAMS.get().is_none() {
        let t = crate::teams::load_teams(teams_path).map_err(PyRuntimeError::new_err)?;
        let _ = TEAMS.set(t);
    }
    let t = teams()?;
    worlds::init_set_pool(t);
    Ok(t.len())
}

#[pyfunction]
fn team_names() -> PyResult<Vec<String>> {
    Ok(teams()?.iter().map(|t| t.name.clone()).collect())
}

#[pyfunction]
fn team_supported() -> PyResult<Vec<bool>> {
    Ok(teams()?.iter().map(|t| t.supported()).collect())
}

#[pyfunction]
fn team_unsupported_reasons(i: usize) -> PyResult<Vec<String>> {
    Ok(teams()?.get(i).ok_or_else(|| PyValueError::new_err("bad team"))?.unsupported.clone())
}

/// Showdown export text of a team (for laddering with it).
#[pyfunction]
fn team_species(i: usize) -> PyResult<Vec<String>> {
    let d = crate::dex::dex();
    let t = teams()?.get(i).ok_or_else(|| PyValueError::new_err("bad team"))?;
    Ok(t.mons.iter().map(|m| d.sp(m.species).name.clone()).collect())
}

/// Dimensions: {n_mons, mon_int, mon_float, field_float, n_slots, n_slot_actions, n_preview_actions}.
#[pyfunction]
fn dims() -> (usize, usize, usize, usize, usize, usize, usize) {
    (obs::N_MONS, obs::MON_INT, obs::MON_FLOAT, obs::FIELD_FLOAT, 2, N_SLOT_ACTIONS, N_PREVIEW_ACTIONS)
}

/// Vocabulary sizes: (species, moves, items, abilities, statuses, positions).
#[pyfunction]
fn vocab() -> (usize, usize, usize, usize, usize, usize) {
    let v = obs::vocab();
    (v.species, v.moves, v.items, v.abilities, 7, obs::N_POSITIONS)
}

/// Static feature tables: (species [Ns, Fs], moves [Nm, Fm], items [Ni, Fi]).
#[pyfunction]
fn static_tables<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
    let v = obs::vocab();
    let sp = to_np(py, obs::species_table(), &[v.species, obs::SPECIES_FEAT])?;
    let mv = to_np(py, obs::move_table(), &[v.moves, obs::MOVE_FEAT])?;
    let it = to_np(py, obs::item_table(), &[v.items, obs::ITEM_FEAT])?;
    PyTuple::new(py, [sp.into_any(), mv.into_any(), it.into_any()])
}

/// The 90 team-preview options as (lead, lead, back, back) team indices.
#[pyfunction]
fn preview_options() -> Vec<Vec<u32>> {
    preview_table().iter().map(|r| r.iter().map(|&x| x as u32).collect()).collect()
}

#[pyfunction]
fn names(kind: &str) -> PyResult<Vec<String>> {
    let d = crate::dex::dex();
    Ok(match kind {
        "species" => d.species.iter().map(|x| x.name.clone()).collect(),
        "moves" => d.moves.iter().map(|x| x.name.clone()).collect(),
        "items" => d.items.iter().map(|x| x.name.clone()).collect(),
        "abilities" => d.abilities.iter().map(|x| x.name.clone()).collect(),
        _ => return Err(PyValueError::new_err("kind: species|moves|items|abilities")),
    })
}

fn observe_battles<'py>(py: Python<'py>, battles: &[&Battle], sides: &[usize]) -> PyResult<Bound<'py, PyTuple>> {
    let (ni, nf, nfield, nm) = obs::sizes();
    let n = battles.len();
    let mut ints = vec![0i32; n * ni];
    let mut floats = vec![0f32; n * nf];
    let mut field = vec![0f32; n * nfield];
    let mut mask = vec![0u8; n * nm];
    let mut req = vec![0i32; n * 3];
    py.allow_threads(|| {
        ints.par_chunks_mut(ni)
            .zip(floats.par_chunks_mut(nf))
            .zip(field.par_chunks_mut(nfield))
            .zip(mask.par_chunks_mut(nm))
            .zip(req.par_chunks_mut(3))
            .enumerate()
            .for_each(|(k, ((((i, f), fi), m), r))| {
                let b = battles[k];
                let s = sides[k];
                obs::encode(b, s, i, f, fi, m);
                let rq = b.request(s);
                r[0] = req_code(rq.kind);
                r[1] = rq.slots[0] as i32;
                r[2] = rq.slots[1] as i32;
            });
    });
    let t = [
        to_np(py, ints, &[n, obs::N_MONS, obs::MON_INT])?.into_any(),
        to_np(py, floats, &[n, obs::N_MONS, obs::MON_FLOAT])?.into_any(),
        to_np(py, field, &[n, obs::FIELD_FLOAT])?.into_any(),
        to_np(py, mask, &[n, 2, N_SLOT_ACTIONS])?.into_any(),
        to_np(py, req, &[n, 3])?.into_any(),
    ];
    PyTuple::new(py, t)
}

fn req_code(k: RequestKind) -> i32 {
    match k {
        RequestKind::Wait => 0,
        RequestKind::TeamPreview => 1,
        RequestKind::Move => 2,
        RequestKind::Switch => 3,
    }
}

// ---- VecEnv -----------------------------------------------------------------------

#[pyclass(name = "VecEnv")]
pub struct PyVecEnv {
    inner: VecEnv,
    /// Pool index of each env-local team index.
    team_index: Vec<usize>,
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (n_envs, seed=0, team_indices=None, turn_limit=100, open_sheet_prob=0.5))]
    fn new(n_envs: usize, seed: u64, team_indices: Option<Vec<usize>>, turn_limit: u16, open_sheet_prob: f32) -> PyResult<Self> {
        let all = teams()?;
        let idx: Vec<usize> = match team_indices {
            Some(v) => v,
            None => (0..all.len()).filter(|&i| all[i].supported()).collect(),
        };
        if idx.is_empty() {
            return Err(PyValueError::new_err("no teams"));
        }
        let ts: Vec<TeamSpec> = idx.iter().map(|&i| all[i].clone()).collect();
        let cfg = EnvConfig { turn_limit, open_sheet_prob, ally_damage_targeting: false };
        Ok(PyVecEnv { inner: VecEnv::new(ts, n_envs, seed, cfg), team_index: idx })
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.inner.len()
    }

    /// Pool indices of the teams this env samples from.
    #[getter]
    fn team_indices(&self) -> Vec<usize> {
        self.team_index.clone()
    }

    /// (ints [n,2,12,11] i32, floats [n,2,12,F] f32, field [n,2,Ff] f32, mask [n,2,2,31] u8, req [n,2,3] i32)
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let (ni, nf, nfield, nm) = obs::sizes();
        let n = self.inner.len();
        let mut ints = vec![0i32; n * 2 * ni];
        let mut floats = vec![0f32; n * 2 * nf];
        let mut field = vec![0f32; n * 2 * nfield];
        let mut mask = vec![0u8; n * 2 * nm];
        let mut req = vec![0i32; n * 2 * 3];
        py.allow_threads(|| self.inner.observe(&mut ints, &mut floats, &mut field, &mut mask, &mut req));
        let t = [
            to_np(py, ints, &[n, 2, obs::N_MONS, obs::MON_INT])?.into_any(),
            to_np(py, floats, &[n, 2, obs::N_MONS, obs::MON_FLOAT])?.into_any(),
            to_np(py, field, &[n, 2, obs::FIELD_FLOAT])?.into_any(),
            to_np(py, mask, &[n, 2, 2, N_SLOT_ACTIONS])?.into_any(),
            to_np(py, req, &[n, 2, 3])?.into_any(),
        ];
        PyTuple::new(py, t)
    }

    /// actions: uint8 [n, 2, 3]. Returns (rewards [n,2], dones [n], n_illegal).
    fn step<'py>(&mut self, py: Python<'py>, actions: PyReadonlyArray3<'py, u8>) -> PyResult<Bound<'py, PyTuple>> {
        let n = self.inner.len();
        if actions.shape() != [n, 2, 3] {
            return Err(PyValueError::new_err(format!("actions shape {:?}, want [{n}, 2, 3]", actions.shape())));
        }
        let a: Vec<u8> = actions.as_array().iter().copied().collect();
        let mut rewards = vec![0f32; n * 2];
        let mut dones = vec![0u8; n];
        let inner = &mut self.inner;
        let errs = py.allow_threads(|| inner.step(&a, &mut rewards, &mut dones));
        let t = [
            to_np(py, rewards, &[n, 2])?.into_any(),
            to_np(py, dones, &[n])?.into_any(),
            errs.into_pyobject(py)?.into_any(),
        ];
        PyTuple::new(py, t)
    }

    /// Info for episodes that ended on the last step:
    /// (done [n] u8, winner [n] i8, turns [n] i32, team [n,2] i64 pool indices).
    fn last_infos<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let n = self.inner.len();
        let info = &self.inner.last_info;
        let done: Vec<u8> = info.iter().map(|x| x.done as u8).collect();
        let winner: Vec<i8> = info.iter().map(|x| x.winner).collect();
        let turns: Vec<i32> = info.iter().map(|x| x.turns as i32).collect();
        let team: Vec<i64> =
            info.iter().flat_map(|x| [self.team_index[x.team[0] as usize] as i64, self.team_index[x.team[1] as usize] as i64]).collect();
        let t = [
            to_np(py, done, &[n])?.into_any(),
            to_np(py, winner, &[n])?.into_any(),
            to_np(py, turns, &[n])?.into_any(),
            to_np(py, team, &[n, 2])?.into_any(),
        ];
        PyTuple::new(py, t)
    }

    /// Current teams (pool indices) of every env: [n, 2].
    fn current_teams<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArrayDyn<i64>>> {
        let n = self.inner.len();
        let v: Vec<i64> = self
            .inner
            .envs
            .iter()
            .flat_map(|e| [self.team_index[e.team[0] as usize] as i64, self.team_index[e.team[1] as usize] as i64])
            .collect();
        to_np(py, v, &[n, 2])
    }

    /// Sampling weights over this env's teams (same order as team_indices) for one side.
    fn set_team_weights(&mut self, side: usize, weights: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let w: ArrayView1<f64> = weights.as_array();
        if w.len() != self.team_index.len() || side > 1 {
            return Err(PyValueError::new_err("weights must match team_indices; side 0 or 1"));
        }
        self.inner.set_team_weights(side, &w.to_vec());
        Ok(())
    }

    /// Actions of a fixed baseline player ("random" or "greedy") for one side: uint8 [n, 3].
    #[pyo3(signature = (side, kind="random", seed=0))]
    fn baseline_actions<'py>(&self, py: Python<'py>, side: usize, kind: &str, seed: u64) -> PyResult<Bound<'py, PyArrayDyn<u8>>> {
        let greedy = match kind {
            "random" => false,
            "greedy" => true,
            _ => return Err(PyValueError::new_err("kind: random|greedy")),
        };
        let n = self.inner.len();
        let envs = &self.inner.envs;
        let out: Vec<u8> = py.allow_threads(|| {
            envs.par_iter()
                .enumerate()
                .flat_map_iter(|(i, e)| {
                    let mut rng = Rng::new(seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
                    let c = if greedy {
                        crate::baseline::greedy_choice(&e.battle, side, &mut rng)
                    } else {
                        crate::baseline::random_choice(&e.battle, side, &mut rng)
                    };
                    [c.preview, c.slots[0], c.slots[1]]
                })
                .collect()
        });
        to_np(py, out, &[n, 3])
    }

    /// Copy of env i's battle.
    fn battle(&self, i: usize) -> PyResult<PyBattle> {
        let e = self.inner.envs.get(i).ok_or_else(|| PyValueError::new_err("bad env index"))?;
        Ok(PyBattle { b: e.battle.clone() })
    }
}

// ---- single battles ------------------------------------------------------------------

#[pyclass(name = "Battle")]
#[derive(Clone)]
pub struct PyBattle {
    pub b: Battle,
}

fn choice_of(c: (u8, u8, u8)) -> Choice {
    Choice { preview: c.0, slots: [c.1, c.2] }
}

#[pymethods]
impl PyBattle {
    /// New battle between two pool teams.
    #[new]
    #[pyo3(signature = (team_a, team_b, seed=0, sheets=(false, false), turn_limit=100, log=false))]
    fn new(team_a: usize, team_b: usize, seed: u64, sheets: (bool, bool), turn_limit: u16, log: bool) -> PyResult<Self> {
        let t = teams()?;
        let (a, b) = (
            t.get(team_a).ok_or_else(|| PyValueError::new_err("bad team_a"))?,
            t.get(team_b).ok_or_else(|| PyValueError::new_err("bad team_b"))?,
        );
        let cfg = BattleConfig { turn_limit, ally_damage_targeting: false, log, sheets: [sheets.0, sheets.1] };
        Ok(PyBattle { b: Battle::new([a, b], seed, cfg) })
    }

    /// Battle from an observed-state snapshot (JSON string; see snapshot.rs).
    #[staticmethod]
    #[pyo3(signature = (snapshot, seed=0))]
    fn from_snapshot(snapshot: &str, seed: u64) -> PyResult<PyBattle> {
        let v: serde_json::Value = serde_json::from_str(snapshot).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let b = crate::snapshot::battle_from_snapshot(&v, seed).map_err(PyValueError::new_err)?;
        Ok(PyBattle { b })
    }

    /// Legal action mask per slot as a list of 31 bools, for both slots.
    fn legal_masks(&self, side: usize) -> Vec<Vec<bool>> {
        (0..2).map(|s| {
            let m = self.b.legal_mask(side, s);
            (0..N_SLOT_ACTIONS).map(|a| (m >> a) & 1 == 1).collect()
        }).collect()
    }

    fn clone_battle(&self) -> PyBattle {
        self.clone()
    }

    fn reseed(&mut self, seed: u64) {
        self.b.rng = Rng::new(seed);
    }

    /// Choices are (preview, slot0, slot1) per side.
    fn step(&mut self, c0: (u8, u8, u8), c1: (u8, u8, u8)) -> PyResult<()> {
        self.b.step([choice_of(c0), choice_of(c1)]).map_err(|e| PyValueError::new_err(e.0))
    }

    fn observe<'py>(&self, py: Python<'py>, side: usize) -> PyResult<Bound<'py, PyTuple>> {
        observe_battles(py, &[&self.b], &[side])
    }

    /// (kind: 0 wait / 1 preview / 2 move / 3 switch, slot0 acts, slot1 acts)
    fn request(&self, side: usize) -> (i32, bool, bool) {
        let r = self.b.request(side);
        (req_code(r.kind), r.slots[0], r.slots[1])
    }

    fn legal_mask(&self, side: usize, slot: usize) -> u32 {
        self.b.legal_mask(side, slot)
    }

    #[getter]
    fn ended(&self) -> bool {
        self.b.ended()
    }

    /// 0 / 1 winner, 2 draw, None while running.
    #[getter]
    fn winner(&self) -> Option<u8> {
        self.b.winner
    }

    #[getter]
    fn turn(&self) -> u16 {
        self.b.turn
    }

    #[getter]
    fn phase(&self) -> &'static str {
        match self.b.phase {
            Phase::TeamPreview => "preview",
            Phase::Move => "move",
            Phase::Switch { midturn: true } => "switch_midturn",
            Phase::Switch { midturn: false } => "switch",
            Phase::Ended => "ended",
        }
    }

    fn log(&self) -> Vec<String> {
        self.b.log().to_vec()
    }

    fn summary(&self) -> String {
        self.b.summary()
    }

    /// A copy where everything `viewer` cannot see about the opponent is resampled.
    fn determinize(&self, viewer: usize, seed: u64) -> PyResult<PyBattle> {
        let pool = worlds::set_pool().ok_or_else(|| PyRuntimeError::new_err("load() first"))?;
        let mut rng = Rng::new(seed);
        Ok(PyBattle { b: worlds::determinize(&self.b, viewer, pool, &mut rng) })
    }

    /// Species names of side's team (team order).
    fn species(&self, side: usize) -> Vec<String> {
        (0..6).map(|i| self.b.mon_name(side, i).to_string()).collect()
    }
}

// ---- batches for search -----------------------------------------------------------------

/// A batch of battles stepped / observed in parallel (search expansion).
#[pyclass(name = "BattleBatch")]
pub struct PyBattleBatch {
    pub items: Vec<Battle>,
    /// false where the expansion's choices were illegal in that world.
    pub valid: Vec<bool>,
}

#[pymethods]
impl PyBattleBatch {
    /// `n` determinized copies of `root` from `viewer`'s point of view.
    #[staticmethod]
    fn worlds(root: &PyBattle, viewer: usize, n: usize, seed: u64) -> PyResult<PyBattleBatch> {
        let pool = worlds::set_pool().ok_or_else(|| PyRuntimeError::new_err("load() first"))?;
        let mut rng = Rng::new(seed);
        let items: Vec<Battle> = (0..n).map(|_| worlds::determinize(&root.b, viewer, pool, &mut rng)).collect();
        let valid = vec![true; n];
        Ok(PyBattleBatch { items, valid })
    }

    /// `n` copies of `root` (perfect information) with different RNG seeds.
    #[staticmethod]
    fn copies(root: &PyBattle, n: usize, seed: u64) -> PyBattleBatch {
        let mut rng = Rng::new(seed);
        let items = (0..n)
            .map(|_| {
                let mut b = root.b.clone();
                b.rng = rng.fork();
                b
            })
            .collect();
        PyBattleBatch { items, valid: vec![true; n] }
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn get(&self, i: usize) -> PyResult<PyBattle> {
        self.items.get(i).map(|b| PyBattle { b: b.clone() }).ok_or_else(|| PyValueError::new_err("index"))
    }

    fn observe<'py>(&self, py: Python<'py>, side: usize) -> PyResult<Bound<'py, PyTuple>> {
        let refs: Vec<&Battle> = self.items.iter().collect();
        let sides = vec![side; refs.len()];
        observe_battles(py, &refs, &sides)
    }

    /// Children for every (world w, viewer choice i, opponent choice j):
    /// world w stepped with viewer_choices[i] and opp_choices[w, j].
    /// viewer_choices: uint8 [K, 3]; opp_choices: uint8 [W, J, 3]. Order: w, i, j.
    fn expand<'py>(
        &self,
        py: Python<'py>,
        viewer: usize,
        viewer_choices: numpy::PyReadonlyArray2<'py, u8>,
        opp_choices: PyReadonlyArray3<'py, u8>,
    ) -> PyResult<PyBattleBatch> {
        let vc = viewer_choices.as_array().to_owned();
        let oc: ArrayView3<u8> = opp_choices.as_array();
        let oc = oc.to_owned();
        let w = self.items.len();
        if oc.shape()[0] != w || vc.shape()[1] != 3 || oc.shape()[2] != 3 {
            return Err(PyValueError::new_err("viewer_choices [K,3], opp_choices [W,J,3]"));
        }
        let k = vc.shape()[0];
        let j = oc.shape()[1];
        let items = &self.items;
        let out: Vec<(Battle, bool)> = py.allow_threads(|| {
            (0..w * k * j)
                .into_par_iter()
                .map(|idx| {
                    let wi = idx / (k * j);
                    let ii = (idx / j) % k;
                    let ji = idx % j;
                    let mut b = items[wi].clone();
                    let mine = Choice { preview: vc[[ii, 0]], slots: [vc[[ii, 1]], vc[[ii, 2]]] };
                    let theirs = Choice { preview: oc[[wi, ji, 0]], slots: [oc[[wi, ji, 1]], oc[[wi, ji, 2]]] };
                    let choices = if viewer == 0 { [mine, theirs] } else { [theirs, mine] };
                    let ok = b.step(choices).is_ok();
                    (b, ok)
                })
                .collect()
        });
        let (items, valid): (Vec<Battle>, Vec<bool>) = out.into_iter().unzip();
        Ok(PyBattleBatch { items, valid })
    }

    /// Terminal value for `viewer` (+1 win, -1 loss, 0 draw), NaN if not over;
    /// NaN as well where the expansion was invalid.
    fn outcomes<'py>(&self, py: Python<'py>, viewer: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let v: Vec<f32> = self
            .items
            .iter()
            .zip(&self.valid)
            .map(|(b, &ok)| {
                if !ok {
                    return f32::NAN;
                }
                match b.winner {
                    Some(2) => 0.0,
                    Some(w) if w as usize == viewer => 1.0,
                    Some(_) => -1.0,
                    None => f32::NAN,
                }
            })
            .collect();
        Ok(v.into_pyarray(py))
    }

    fn valid_mask(&self) -> Vec<bool> {
        self.valid.clone()
    }
}

#[pymodule]
fn vgczero_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(team_names, m)?)?;
    m.add_function(wrap_pyfunction!(team_supported, m)?)?;
    m.add_function(wrap_pyfunction!(team_unsupported_reasons, m)?)?;
    m.add_function(wrap_pyfunction!(team_species, m)?)?;
    m.add_function(wrap_pyfunction!(dims, m)?)?;
    m.add_function(wrap_pyfunction!(vocab, m)?)?;
    m.add_function(wrap_pyfunction!(static_tables, m)?)?;
    m.add_function(wrap_pyfunction!(preview_options, m)?)?;
    m.add_function(wrap_pyfunction!(names, m)?)?;
    m.add_class::<PyVecEnv>()?;
    m.add_class::<PyBattle>()?;
    m.add_class::<PyBattleBatch>()?;
    Ok(())
}
