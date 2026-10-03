//! Small, fast, copyable PRNG (xoshiro256**). Battles own one so a cloned
//! battle (search) continues its own independent stream after reseeding.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

#[inline]
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        Rng { s: [splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x)] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in [0, n). n must be > 0.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        // Lemire's multiply-shift; bias is negligible for our n (< 2^16).
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    /// Uniform integer in [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.below(hi - lo)
    }

    /// True with probability num/den.
    #[inline]
    pub fn chance(&mut self, num: u32, den: u32) -> bool {
        self.below(den) < num
    }

    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Derive an independent stream (for search worlds, env resets).
    pub fn fork(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }
}
