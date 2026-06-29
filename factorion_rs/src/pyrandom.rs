//! A byte-exact reimplementation of (the parts we use of) CPython's
//! `random` module — the Mersenne Twister core plus the higher-level
//! `getrandbits` / `random` / `_randbelow` / `randrange` / `randint` /
//! `choice` / `shuffle` / `sample` methods.
//!
//! This exists so the Rust port of `build_factory` can reproduce a
//! **byte-identical** factory for the same `(size, kind, seed)` as the
//! Python implementation, which (after the single-RNG refactor) draws
//! every bit of layout randomness from `random.*`. The reference is
//! CPython's `Modules/_randommodule.c` and `Lib/random.py`; the unit
//! tests below pin the output against sequences dumped from the live
//! interpreter.
//!
//! Only the surface `build_factory` needs is implemented. `getrandbits`
//! handles `k <= 64` (every `_randbelow` here uses `k <= 32`), and
//! `sample`'s big-`k` set-size table is the integer equivalent of
//! CPython's `4 ** ceil(log(k*3, 4))` — never exercised because every
//! call site uses `k <= 3`, but implemented for faithfulness.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// CPython's `random.Random`, restricted to the methods `build_factory`
/// uses. Construct via [`PyRandom::seeded`].
pub struct PyRandom {
    mt: [u32; N],
    mti: usize,
}

impl PyRandom {
    /// Create a generator seeded exactly as `random.Random(seed)` /
    /// `random.seed(seed)` would for an integer seed. Negative seeds use
    /// their absolute value, matching CPython.
    pub fn seeded(seed: u64) -> Self {
        let mut r = PyRandom {
            mt: [0; N],
            mti: N + 1,
        };
        r.seed(seed);
        r
    }

    /// `random.seed(int)`: split `seed` into little-endian 32-bit words
    /// and feed them to `init_by_array`, exactly as
    /// `random_seed`/`init_by_array` do in CPython.
    pub fn seed(&mut self, seed: u64) {
        let bits = 64 - seed.leading_zeros();
        let keymax = if bits == 0 {
            1
        } else {
            ((bits - 1) / 32 + 1) as usize
        };
        let mut key = Vec::with_capacity(keymax);
        for i in 0..keymax {
            key.push(((seed >> (32 * i)) & 0xffff_ffff) as u32);
        }
        self.init_by_array(&key);
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = (1_812_433_253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)))
                .wrapping_add(i as u32);
        }
        self.mti = N;
    }

    fn init_by_array(&mut self, init_key: &[u32]) {
        self.init_genrand(19_650_218);
        let key_length = init_key.len();
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = if N > key_length { N } else { key_length };
        while k != 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
            .wrapping_add(init_key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
            k -= 1;
        }
        for _ in 0..(N - 1) {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    fn genrand_uint32(&mut self) -> u32 {
        if self.mti >= N {
            let mag01 = [0u32, MATRIX_A];
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.random()` — 53-bit float in [0, 1) via `genrand_res53`.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_uint32() >> 5) as f64; // 27 bits
        let b = (self.genrand_uint32() >> 6) as f64; // 26 bits
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `random.getrandbits(k)` for `1 <= k <= 64`.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        debug_assert!((1..=64).contains(&k));
        if k <= 32 {
            return (self.genrand_uint32() >> (32 - k)) as u64;
        }
        let mut result: u64 = 0;
        let mut shift = 0u32;
        let mut remaining = k;
        while remaining > 0 {
            let take = remaining.min(32);
            let r = (self.genrand_uint32() >> (32 - take)) as u64;
            result |= r << shift;
            shift += 32;
            remaining -= take;
        }
        result
    }

    /// `Random._randbelow_with_getrandbits(n)` — uniform int in [0, n).
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros(); // n.bit_length()
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// `random.randrange(stop)` (single argument, `stop > 0`).
    pub fn randrange(&mut self, stop: u64) -> u64 {
        self.randbelow(stop)
    }

    /// `random.randint(a, b)` — inclusive on both ends, `a <= b`.
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        debug_assert!(a <= b);
        let width = (b - a + 1) as u64;
        a + self.randbelow(width) as i64
    }

    /// `random.choice(seq)` — returns the chosen index into a length-`len`
    /// sequence. Caller indexes its own slice (keeps this generic-free).
    pub fn choice_index(&mut self, len: usize) -> usize {
        self.randbelow(len as u64) as usize
    }

    /// `random.shuffle(x)` performed in place on a slice.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        let n = x.len();
        if n <= 1 {
            return;
        }
        for i in (1..n).rev() {
            let j = self.randbelow((i + 1) as u64) as usize;
            x.swap(i, j);
        }
    }

    /// `random.sample(population, k)` — returns `k` distinct elements in
    /// selection order, reproducing CPython's pool/set dual strategy.
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let n = population.len();
        debug_assert!(k <= n);
        let mut setsize = 21usize;
        if k > 5 {
            // Integer form of `4 ** ceil(log(k*3, 4))`: smallest power of
            // 4 that is >= k*3. Not exercised (all call sites use k <= 3).
            let mut p = 1usize;
            while p < k * 3 {
                p *= 4;
            }
            setsize += p;
        }
        let mut result: Vec<T> = Vec::with_capacity(k);
        if n <= setsize {
            let mut pool: Vec<T> = population.to_vec();
            for i in 0..k {
                let j = self.randbelow((n - i) as u64) as usize;
                result.push(pool[j].clone());
                pool[j] = pool[n - i - 1].clone();
            }
        } else {
            let mut selected: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for _ in 0..k {
                let mut j = self.randbelow(n as u64) as usize;
                while selected.contains(&j) {
                    j = self.randbelow(n as u64) as usize;
                }
                selected.insert(j);
                result.push(population[j].clone());
            }
        }
        result
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // Reference sequences dumped from CPython 3.11's `random` module.
    // See the generator in the commit that added this file.

    #[test]
    fn test_getrandbits() {
        let ks = [1u32, 2, 3, 7, 10, 16, 20, 31, 32];
        let cases: &[(u64, [u64; 9])] = &[
            (0, [1, 1, 6, 113, 430, 2653, 271493, 2073320061, 2195908194]),
            (1, [0, 2, 6, 102, 782, 4135, 267459, 253228484, 2127877499]),
            (42, [1, 0, 0, 94, 281, 16049, 234053, 299655412, 3163119785]),
            (
                12345,
                [0, 2, 0, 104, 845, 52615, 313146, 1836395613, 1582316135],
            ),
            (
                2147483648,
                [0, 1, 4, 112, 910, 58539, 170831, 1364984270, 482266895],
            ),
            (
                1099511627783,
                [1, 2, 6, 120, 967, 41911, 598097, 1515253441, 2044078442],
            ),
        ];
        for (seed, expected) in cases {
            let mut r = PyRandom::seeded(*seed);
            for (idx, &k) in ks.iter().enumerate() {
                assert_eq!(r.getrandbits(k), expected[idx], "seed={seed} k={k}");
            }
        }
    }

    #[test]
    // The reference values are CPython's 17-significant-digit dumps; each
    // rounds to the exact f64 our `random()` produces, so equality holds.
    #[allow(clippy::excessive_precision)]
    fn test_random() {
        let cases: &[(u64, [f64; 4])] = &[
            (
                0,
                [
                    0.84442185152504812,
                    0.75795440294030247,
                    0.420571580830845,
                    0.25891675029296335,
                ],
            ),
            (
                1,
                [
                    0.13436424411240122,
                    0.84743373693723267,
                    0.76377461897661403,
                    0.2550690257394217,
                ],
            ),
            (
                42,
                [
                    0.63942679845788375,
                    0.025010755222666936,
                    0.27502931836911926,
                    0.22321073814882275,
                ],
            ),
        ];
        for (seed, expected) in cases {
            let mut r = PyRandom::seeded(*seed);
            for &e in expected {
                assert_eq!(r.random(), e, "seed={seed}");
            }
        }
    }

    #[test]
    fn test_randbelow() {
        let ns = [1u64, 2, 3, 4, 5, 12, 67, 100, 1000];
        let cases: &[(u64, [u64; 9])] = &[
            (0, [0, 1, 0, 2, 4, 7, 51, 38, 991]),
            (1, [0, 0, 1, 0, 3, 7, 60, 83, 388]),
            (42, [0, 0, 2, 2, 1, 3, 17, 94, 104]),
            (12345, [0, 0, 1, 2, 1, 4, 55, 20, 382]),
        ];
        for (seed, expected) in cases {
            let mut r = PyRandom::seeded(*seed);
            for (idx, &n) in ns.iter().enumerate() {
                assert_eq!(r.randbelow(n), expected[idx], "seed={seed} n={n}");
            }
        }
    }

    #[test]
    fn test_randrange() {
        let ns = [4u64, 4, 67, 100, 121, 256];
        let cases: &[(u64, [u64; 6])] = &[
            (0, [3, 3, 5, 33, 65, 248]),
            (1, [1, 0, 32, 15, 63, 230]),
            (42, [0, 0, 35, 31, 28, 71]),
        ];
        for (seed, expected) in cases {
            let mut r = PyRandom::seeded(*seed);
            for (idx, &n) in ns.iter().enumerate() {
                assert_eq!(r.randrange(n), expected[idx], "seed={seed} n={n}");
            }
        }
    }

    #[test]
    fn test_randint() {
        let bs = [3i64, 9, 9, 11, 3];
        let cases: &[(u64, [i64; 5])] = &[
            (0, [3, 6, 0, 4, 3]),
            (1, [1, 9, 1, 4, 0]),
            (42, [0, 0, 4, 3, 1]),
        ];
        for (seed, expected) in cases {
            let mut r = PyRandom::seeded(*seed);
            for (idx, &b) in bs.iter().enumerate() {
                assert_eq!(r.randint(0, b), expected[idx], "seed={seed} b={b}");
            }
        }
    }

    #[test]
    fn test_shuffle() {
        let cases: &[(u64, [usize; 10])] = &[
            (0, [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]),
            (1, [6, 8, 9, 7, 5, 3, 0, 4, 1, 2]),
            (42, [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]),
            (12345, [8, 7, 3, 5, 1, 2, 9, 4, 0, 6]),
        ];
        for (seed, expected) in cases {
            let mut r = PyRandom::seeded(*seed);
            let mut v: Vec<usize> = (0..10).collect();
            r.shuffle(&mut v);
            assert_eq!(v, expected.to_vec(), "seed={seed}");
        }
    }

    #[test]
    fn test_sample_pool_and_set() {
        // n=12 -> pool path; n=100 -> set path.
        let pool_cases: &[(u64, [usize; 3])] = &[
            (0, [6, 11, 0]),
            (1, [2, 9, 1]),
            (42, [10, 1, 0]),
            (12345, [6, 0, 4]),
        ];
        for (seed, expected) in pool_cases {
            let mut r = PyRandom::seeded(*seed);
            let pop: Vec<usize> = (0..12).collect();
            assert_eq!(r.sample(&pop, 3), expected.to_vec(), "pool seed={seed}");
        }
        let set_cases: &[(u64, [usize; 3])] = &[
            (0, [49, 97, 53]),
            (1, [17, 72, 97]),
            (42, [81, 14, 3]),
            (12345, [53, 93, 1]),
        ];
        for (seed, expected) in set_cases {
            let mut r = PyRandom::seeded(*seed);
            let pop: Vec<usize> = (0..100).collect();
            assert_eq!(r.sample(&pop, 3), expected.to_vec(), "set seed={seed}");
        }
    }
}
