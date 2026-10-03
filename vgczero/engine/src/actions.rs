//! Action encoding shared by the engine, the observation encoder and Python.
//!
//! Per active slot, one of `N_SLOT_ACTIONS`:
//!   0           pass (nothing to do: empty slot, or waiting)
//!   1..=6       switch to team member 0..5
//!   7..=30      move m (0..3) x target t (0..2) x mega (0/1)
//!               index = 7 + (m * 3 + t) * 2 + mega
//! Targets: 0 = opposing slot 0, 1 = opposing slot 1, 2 = ally. Moves that do
//! not take a chosen target always use t = 0.
//!
//! Team preview: one of `N_PREVIEW_ACTIONS` = 90 (lead pair, back pair).

pub const N_SLOT_ACTIONS: usize = 31;
pub const PASS: u8 = 0;
pub const N_PREVIEW_ACTIONS: usize = 90;

pub const T_FOE0: u8 = 0;
pub const T_FOE1: u8 = 1;
pub const T_ALLY: u8 = 2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SlotAction {
    Pass,
    Switch(u8),
    Move { slot: u8, target: u8, mega: bool },
}

#[inline]
pub fn switch_action(mon: usize) -> u8 {
    debug_assert!(mon < 6);
    1 + mon as u8
}

#[inline]
pub fn move_action(slot: usize, target: u8, mega: bool) -> u8 {
    debug_assert!(slot < 4 && target < 3);
    7 + ((slot as u8 * 3 + target) * 2) + mega as u8
}

#[inline]
pub fn decode(a: u8) -> SlotAction {
    match a {
        0 => SlotAction::Pass,
        1..=6 => SlotAction::Switch(a - 1),
        7..=30 => {
            let k = a - 7;
            let mega = k % 2 == 1;
            let mt = k / 2;
            SlotAction::Move { slot: mt / 3, target: mt % 3, mega }
        }
        _ => SlotAction::Pass,
    }
}

/// (lead a, lead b, back c, back d) for each preview action, a<b, c<d.
pub fn preview_table() -> &'static [[u8; 4]; N_PREVIEW_ACTIONS] {
    use std::sync::OnceLock;
    static T: OnceLock<[[u8; 4]; N_PREVIEW_ACTIONS]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [[0u8; 4]; N_PREVIEW_ACTIONS];
        let mut k = 0;
        for a in 0..6u8 {
            for b in (a + 1)..6 {
                let rest: Vec<u8> = (0..6).filter(|&x| x != a && x != b).collect();
                for i in 0..4 {
                    for j in (i + 1)..4 {
                        t[k] = [a, b, rest[i], rest[j]];
                        k += 1;
                    }
                }
            }
        }
        assert_eq!(k, N_PREVIEW_ACTIONS);
        t
    })
}

pub fn preview_index(leads: [u8; 2], back: [u8; 2]) -> Option<u8> {
    let (a, b) = (leads[0].min(leads[1]), leads[0].max(leads[1]));
    let (c, d) = (back[0].min(back[1]), back[0].max(back[1]));
    preview_table().iter().position(|r| *r == [a, b, c, d]).map(|i| i as u8)
}

/// A side's choice for one decision point.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Choice {
    pub preview: u8,
    pub slots: [u8; 2],
}

impl Choice {
    pub fn preview(p: u8) -> Choice {
        Choice { preview: p, slots: [PASS, PASS] }
    }
    pub fn slots(a: u8, b: u8) -> Choice {
        Choice { preview: 0, slots: [a, b] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for m in 0..4 {
            for t in 0..3 {
                for mega in [false, true] {
                    let a = move_action(m, t, mega);
                    assert!(a < N_SLOT_ACTIONS as u8);
                    assert_eq!(decode(a), SlotAction::Move { slot: m as u8, target: t, mega });
                }
            }
        }
        for i in 0..6 {
            assert_eq!(decode(switch_action(i)), SlotAction::Switch(i as u8));
        }
        assert_eq!(preview_table().len(), 90);
        assert_eq!(preview_index([3, 1], [5, 0]).map(|i| preview_table()[i as usize]), Some([1, 3, 0, 5]));
    }
}
