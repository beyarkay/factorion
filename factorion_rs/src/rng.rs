//! The small slice of random draws `build_factory` needs, delegated to the
//! `fastrand` crate.
//!
//! `build_factory` only needs *some* good pseudo-randomness that is
//! reproducible for a given `(size, kind, seed)` — it does **not** need to
//! match any external generator bit-for-bit. `fastrand` (a fast,
//! dependency-free PRNG) supplies the core; this is just the thin adapter
//! exposing the handful of higher-level draws the generator calls.

use fastrand::Rng as FastRng;

/// A deterministic PRNG for factory generation. Construct via
/// [`Rng::seeded`].
pub struct Rng {
    inner: FastRng,
}

impl Rng {
    /// Seed the generator. Same seed → same stream of draws.
    pub fn seeded(seed: u64) -> Self {
        Rng {
            inner: FastRng::with_seed(seed),
        }
    }

    /// Uniform integer in `[0, n)`. Returns 0 for `n == 0`.
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.inner.u64(0..n)
    }

    /// `randrange(stop)` — uniform int in `[0, stop)`.
    pub fn randrange(&mut self, stop: u64) -> u64 {
        self.below(stop)
    }

    /// `randint(a, b)` — uniform int in the inclusive range `[a, b]`.
    ///
    /// The generator deliberately calls this with `a > b` for grids too
    /// small to hold an entity, relying on the (out-of-range) result being
    /// rejected downstream, so an empty range must not panic: the width
    /// wraps to a huge value and the draw lands somewhere out of bounds.
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        let width = (b - a + 1) as u64; // wraps huge when a > b
        a.wrapping_add(self.below(width) as i64)
    }

    /// `choice(seq)` — returns the chosen index into a length-`len`
    /// sequence. Caller indexes its own slice (keeps this generic-free).
    pub fn choice_index(&mut self, len: usize) -> usize {
        self.below(len as u64) as usize
    }

    /// Fisher-Yates shuffle performed in place on a slice.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        self.inner.shuffle(x);
    }

    /// `sample(population, k)` — `k` distinct elements in selection order
    /// (partial Fisher-Yates over a copy of the population).
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        let n = population.len();
        debug_assert!(k <= n);
        let mut pool: Vec<T> = population.to_vec();
        let mut result: Vec<T> = Vec::with_capacity(k);
        for i in 0..k {
            let j = self.below((n - i) as u64) as usize;
            result.push(pool[j].clone());
            pool[j] = pool[n - i - 1].clone();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draws(seed: u64) -> Vec<u64> {
        let mut r = Rng::seeded(seed);
        (0..100).map(|_| r.randrange(1000)).collect()
    }

    #[test]
    fn same_seed_same_stream() {
        assert_eq!(draws(12345), draws(12345));
    }

    #[test]
    fn different_seeds_differ() {
        assert_ne!(draws(1), draws(2));
    }

    #[test]
    fn randrange_in_range() {
        let mut r = Rng::seeded(42);
        for n in [1u64, 2, 3, 4, 5, 12, 67, 100, 1000] {
            for _ in 0..1000 {
                assert!(r.randrange(n) < n, "n={n}");
            }
        }
        assert_eq!(r.randrange(0), 0);
    }

    #[test]
    fn randint_in_range() {
        let mut r = Rng::seeded(9);
        for _ in 0..1000 {
            let v = r.randint(-5, 5);
            assert!((-5..=5).contains(&v));
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = Rng::seeded(3);
        let mut v: Vec<usize> = (0..10).collect();
        r.shuffle(&mut v);
        v.sort();
        assert_eq!(v, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn sample_returns_distinct_elements() {
        let mut r = Rng::seeded(5);
        let pop: Vec<usize> = (0..100).collect();
        for _ in 0..100 {
            let mut s = r.sample(&pop, 3);
            assert!(s.iter().all(|&x| x < 100));
            s.sort();
            s.dedup();
            assert_eq!(s.len(), 3, "elements must be distinct");
        }
    }
}
