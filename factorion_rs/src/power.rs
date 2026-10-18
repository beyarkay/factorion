//! Electric power (wiki/electric-network.md). Every pair of poles within wire
//! reach is wired together — no manual wiring, no connection cap. Power comes
//! from the [`Outside`]: a network is powered iff one of its poles could wire
//! to a 1x1 pole on a tile just outside the grid or on an unbuildable tile
//! joined to the grid's edge. A tile is powered iff it lies in
//! the supply area of a pole on a powered network, and an entity that
//! [`Item::needs_power`] is powered iff any of its tiles is.

use crate::entities::entity_tiles;
use crate::graph::unit_anchors;
use crate::types::{Channel, Item, Pole};
use crate::world::World;

/// Which tiles of a world are powered.
pub struct PowerMap {
    powered: Vec<bool>,
    height: usize,
}

/// A pole on the grid, anchored at its min-corner tile `(x, y)`.
pub struct PlacedPole {
    pole: Pole,
    x: i64,
    y: i64,
    size: i64,
}

impl PlacedPole {
    /// `None` if `kind` isn't a pole.
    pub fn new(kind: Item, x: i64, y: i64) -> Option<Self> {
        Some(Self {
            pole: kind.pole()?,
            x,
            y,
            size: kind.size().0 as i64,
        })
    }

    /// The pole's centre in half-tiles, so a 2x2 pole's corner centre stays
    /// an integer.
    fn centre_x2(&self) -> (i64, i64) {
        (2 * self.x + self.size, 2 * self.y + self.size)
    }

    pub fn wires_to(&self, other: &PlacedPole) -> bool {
        let reach = self.pole.reach_x2.min(other.pole.reach_x2);
        let ((ax, ay), (bx, by)) = (self.centre_x2(), other.centre_x2());
        (ax - bx).pow(2) + (ay - by).pow(2) <= reach.pow(2)
    }
}

/// Where power comes from: off the map, and every unbuildable tile joined to
/// the grid's edge through unbuildable tiles (the player's base around a
/// marked-out area). An enclosed unbuildable island is not part of it.
pub struct Outside {
    joined: Vec<bool>,
    width: i64,
    height: i64,
}

impl Outside {
    pub fn of(world: &World) -> Self {
        let (w, h) = (world.width(), world.height());
        let unbuildable = |x: usize, y: usize| world.get(x, y, Channel::Footprint) == 0;
        let mut joined = vec![false; w * h];
        let mut frontier: Vec<(usize, usize)> = (0..w)
            .flat_map(|x| (0..h).map(move |y| (x, y)))
            .filter(|&(x, y)| (x == 0 || y == 0 || x == w - 1 || y == h - 1) && unbuildable(x, y))
            .collect();
        for &(x, y) in &frontier {
            joined[x * h + y] = true;
        }
        while let Some((x, y)) = frontier.pop() {
            let near = [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ];
            for (nx, ny) in near {
                if nx < w && ny < h && !joined[nx * h + ny] && unbuildable(nx, ny) {
                    joined[nx * h + ny] = true;
                    frontier.push((nx, ny));
                }
            }
        }
        Self {
            joined,
            width: w as i64,
            height: h as i64,
        }
    }

    /// Whether `pole` could wire to a 1x1 pole on an outside tile.
    pub fn reached_by(&self, pole: &PlacedPole) -> bool {
        let reach = pole.pole.reach_x2;
        // The nearest off-map tile centre lies straight out past the nearest
        // edge, half a tile aside when this pole's centre is a tile corner.
        let (cx, cy) = pole.centre_x2();
        let gap = (cx + 1)
            .min(2 * self.width + 1 - cx)
            .min(cy + 1)
            .min(2 * self.height + 1 - cy);
        let aside = (pole.size + 1) % 2;
        if gap.pow(2) + aside.pow(2) <= reach.pow(2) {
            return true;
        }
        let span = |c: i64, n: i64| ((c - reach - 1) / 2).max(0)..=((c + reach - 1) / 2).min(n - 1);
        span(cx, self.width).any(|x| {
            span(cy, self.height).any(|y| {
                self.joined[(x * self.height + y) as usize]
                    && (2 * x + 1 - cx).pow(2) + (2 * y + 1 - cy).pow(2) <= reach.pow(2)
            })
        })
    }
}

impl PowerMap {
    pub fn of(world: &World) -> Self {
        let (width, height) = (world.width() as i64, world.height() as i64);
        let anchor_of = unit_anchors(world);
        let mut poles = Vec::new();
        for x in 0..world.width() {
            for y in 0..world.height() {
                let Some(kind) = world.entity_at(x, y) else {
                    continue;
                };
                if !anchor_of.contains_key(&(x, y)) {
                    poles.extend(PlacedPole::new(kind, x as i64, y as i64));
                }
            }
        }

        let outside = Outside::of(world);
        let mut live: Vec<bool> = poles.iter().map(|p| outside.reached_by(p)).collect();
        let mut frontier: Vec<usize> = (0..poles.len()).filter(|&i| live[i]).collect();
        while let Some(i) = frontier.pop() {
            for j in 0..poles.len() {
                if !live[j] && poles[i].wires_to(&poles[j]) {
                    live[j] = true;
                    frontier.push(j);
                }
            }
        }

        let mut powered = vec![false; world.width() * world.height()];
        for p in poles
            .iter()
            .zip(&live)
            .filter(|(_, &on)| on)
            .map(|(p, _)| p)
        {
            let reach = p.pole.supply;
            for x in (p.x - reach).max(0)..(p.x + p.size + reach).min(width) {
                for y in (p.y - reach).max(0)..(p.y + p.size + reach).min(height) {
                    powered[x as usize * world.height() + y as usize] = true;
                }
            }
        }
        Self {
            powered,
            height: world.height(),
        }
    }

    pub fn tile(&self, x: usize, y: usize) -> bool {
        self.powered[x * self.height + y]
    }

    /// Whether the entity unit anchored at `(x, y)` has a powered tile.
    pub fn powers(&self, world: &World, x: usize, y: usize) -> bool {
        let kind = world.entity_at(x, y);
        let (w, h) = kind.map_or((1, 1), Item::size);
        entity_tiles(x, y, world.direction_at(x, y), w, h)
            .unwrap_or_default()
            .iter()
            .filter_map(|p| p.to_usize())
            .any(|(tx, ty)| tx < world.width() && ty < world.height() && self.tile(tx, ty))
    }

    /// Store the map in `world`'s [`Channel::Power`].
    pub fn write(&self, world: &mut World) {
        for x in 0..world.width() {
            for y in 0..world.height() {
                world.set(x, y, Channel::Power, self.tile(x, y) as i64);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::types::Direction;

    fn powered_tiles(world: &World) -> Vec<(usize, usize)> {
        let map = PowerMap::of(world);
        let mut out = Vec::new();
        for x in 0..world.width() {
            for y in 0..world.height() {
                if map.tile(x, y) {
                    out.push((x, y));
                }
            }
        }
        out
    }

    fn with_poles(size: usize, poles: &[(usize, usize, Item)]) -> World {
        let mut w = World::empty(size, size);
        for &(x, y, kind) in poles {
            let (pw, ph) = kind.size();
            assert!(w.place_multi_tile(x, y, kind, Direction::None, None, pw, ph));
        }
        w
    }

    #[test]
    fn test_supply_area_is_a_square_around_the_pole() {
        for (kind, reach) in [(Item::SmallElectricPole, 2), (Item::MediumElectricPole, 3)] {
            let tiles = powered_tiles(&with_poles(9, &[(0, 4, kind)]));
            let want: Vec<_> = (0..=reach)
                .flat_map(|x| (4 - reach..=4 + reach).map(move |y| (x, y)))
                .collect();
            assert_eq!(tiles, want, "{kind:?}");
        }
        // A substation's 18x18 reaches 8 tiles past its 2x2 body.
        let tiles = powered_tiles(&with_poles(40, &[(10, 10, Item::Substation)]));
        assert_eq!(tiles.first(), Some(&(2, 2)));
        assert_eq!(tiles.last(), Some(&(19, 19)));
        assert_eq!(tiles.len(), 18 * 18);
    }

    #[test]
    fn test_off_map_reach() {
        // Small pole: centre to the off-map tile centre must be ≤ 7.5.
        let small = |x| powered_tiles(&with_poles(30, &[(x, 15, Item::SmallElectricPole)]));
        assert!(!small(6).is_empty());
        assert!(small(7).is_empty());
        // Medium pole: ≤ 9.
        let medium = |x| powered_tiles(&with_poles(30, &[(x, 15, Item::MediumElectricPole)]));
        assert!(!medium(8).is_empty());
        assert!(medium(9).is_empty());
        // Substation: centre on a tile corner, 1.5 + 0.5 aside to the nearest
        // off-map tile when flush with the edge; ≤ 18.
        let sub = |x| powered_tiles(&with_poles(60, &[(x, 30, Item::Substation)]));
        assert!(!sub(16).is_empty());
        assert!(sub(17).is_empty());
    }

    #[test]
    fn test_unbuildable_tiles_joined_to_the_edge_feed_power() {
        // A medium pole reaches 9 tiles; unbuildable tiles 29.. on its row
        // feed it only while they run on to the grid's edge.
        let medium = |from: usize, to: usize| {
            let mut w = with_poles(40, &[(20, 20, Item::MediumElectricPole)]);
            for x in from..to {
                w.set(x, 20, Channel::Footprint, 0);
            }
            powered_tiles(&w)
        };
        assert!(!medium(29, 40).is_empty());
        assert!(medium(30, 40).is_empty());
        assert!(medium(29, 39).is_empty());
    }

    #[test]
    fn test_networks_carry_power_inward() {
        let small = Item::SmallElectricPole;
        // 7 apart wires (≤ 7.5), 8 apart doesn't.
        let chain = |gap| with_poles(40, &[(0, 20, small), (gap, 20, small)]);
        assert!(PowerMap::of(&chain(7)).tile(7 + 2, 20));
        assert!(!PowerMap::of(&chain(8)).tile(8 + 2, 20));
        // Diagonal reach is Euclidean: (5, 5) ≈ 7.07 wires, (6, 6) ≈ 8.49 doesn't.
        let diag = |d: usize| with_poles(40, &[(2, 20, small), (2 + d, 20 + d, small)]);
        assert!(PowerMap::of(&diag(5)).tile(5 + 4, 25));
        assert!(!PowerMap::of(&diag(6)).tile(6 + 4, 26));
        // A mixed pair uses the smaller reach: 8 apart is within the medium
        // pole's 9 but not the small pole's 7.5.
        let mixed = with_poles(40, &[(0, 20, Item::MediumElectricPole), (8, 20, small)]);
        assert!(!PowerMap::of(&mixed).tile(8 + 2, 20));
    }

    #[test]
    fn test_assembler_powered_by_one_corner_tile() {
        let mut w = with_poles(12, &[(0, 0, Item::SmallElectricPole)]);
        w.place_multi_tile(2, 2, Item::AssemblingMachine1, Direction::North, None, 3, 3);
        w.place_multi_tile(3, 6, Item::AssemblingMachine1, Direction::North, None, 3, 3);
        let map = PowerMap::of(&w);
        assert!(map.powers(&w, 2, 2));
        assert!(!map.powers(&w, 3, 6));
    }
}
