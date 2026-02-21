use numpy::PyUntypedArrayMethods;

use crate::types::{Channel, Direction, EntityKind, Item, Misc};

/// A wrapper around a 3D array representing a factory world.
///
/// The world has dimensions W × H × C where:
/// - W is width (x axis)
/// - H is height (y axis)
/// - C is channels (entity, direction, item, misc)
///
/// Data is stored as a flat Vec indexed as [x][y][c].
#[derive(Debug, Clone)]
pub struct World {
    data: Vec<i64>,
    width: usize,
    height: usize,
    channels: usize,
}

impl World {
    /// Create a World from a flat data vec with given dimensions.
    #[allow(dead_code)]
    pub fn new(data: Vec<i64>, width: usize, height: usize, channels: usize) -> Self {
        assert_eq!(
            data.len(),
            width * height * channels,
            "Data length {} does not match dimensions {}x{}x{}",
            data.len(),
            width,
            height,
            channels
        );
        Self {
            data,
            width,
            height,
            channels,
        }
    }

    /// Create an empty world of given dimensions, filled with zeros.
    #[allow(dead_code)]
    pub fn empty(width: usize, height: usize) -> Self {
        let channels = 4;
        Self {
            data: vec![0; width * height * channels],
            width,
            height,
            channels,
        }
    }

    /// Construct a World from a numpy array (called from PyO3).
    /// The array should have shape (W, H, C) and dtype i64.
    pub fn from_numpy(array: &numpy::PyReadonlyArray3<i64>) -> Self {
        let shape = array.shape();
        let width = shape[0];
        let height = shape[1];
        let channels = shape[2];
        let array = array.as_array();
        let mut data = vec![0i64; width * height * channels];
        for x in 0..width {
            for y in 0..height {
                for c in 0..channels {
                    data[x * height * channels + y * channels + c] = array[[x, y, c]];
                }
            }
        }
        Self {
            data,
            width,
            height,
            channels,
        }
    }

    #[inline]
    fn index(&self, x: usize, y: usize, c: usize) -> usize {
        x * self.height * self.channels + y * self.channels + c
    }

    /// Get a raw value at (x, y, channel).
    pub fn get(&self, x: usize, y: usize, channel: Channel) -> i64 {
        self.data[self.index(x, y, channel.index())]
    }

    /// Set a raw value at (x, y, channel).
    #[allow(dead_code)]
    pub fn set(&mut self, x: usize, y: usize, channel: Channel, value: i64) {
        let idx = self.index(x, y, channel.index());
        self.data[idx] = value;
    }

    /// Get the entity kind at (x, y).
    pub fn entity_at(&self, x: usize, y: usize) -> EntityKind {
        EntityKind::from_i64(self.get(x, y, Channel::Entities))
    }

    /// Get the direction at (x, y).
    pub fn direction_at(&self, x: usize, y: usize) -> Direction {
        Direction::from_i64(self.get(x, y, Channel::Direction))
    }

    /// Get the item at (x, y).
    pub fn item_at(&self, x: usize, y: usize) -> Item {
        Item::from_i64(self.get(x, y, Channel::Items))
    }

    /// Get the misc flags at (x, y).
    pub fn misc_at(&self, x: usize, y: usize) -> Misc {
        Misc::from_i64(self.get(x, y, Channel::Misc))
    }

    /// Check if coordinates are within bounds.
    /// Takes i64 to handle negative offsets from direction calculations.
    pub fn in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_3x3_world() -> World {
        // 3x3 world, 4 channels, all zeros
        World::empty(3, 3)
    }

    #[test]
    fn test_empty_world() {
        let w = make_3x3_world();
        assert_eq!(w.width(), 3);
        assert_eq!(w.height(), 3);
        for x in 0..3 {
            for y in 0..3 {
                assert_eq!(w.entity_at(x, y), EntityKind::Empty);
                assert_eq!(w.direction_at(x, y), Direction::None);
                assert_eq!(w.item_at(x, y), Item::Empty);
                assert_eq!(w.misc_at(x, y), Misc::None);
            }
        }
    }

    #[test]
    fn test_set_and_get() {
        let mut w = make_3x3_world();
        w.set(1, 2, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(1, 2, Channel::Direction, Direction::East as i64);
        w.set(1, 2, Channel::Items, Item::CopperCable as i64);

        assert_eq!(w.entity_at(1, 2), EntityKind::TransportBelt);
        assert_eq!(w.direction_at(1, 2), Direction::East);
        assert_eq!(w.item_at(1, 2), Item::CopperCable);

        // Other cells still empty
        assert_eq!(w.entity_at(0, 0), EntityKind::Empty);
    }

    #[test]
    fn test_in_bounds() {
        let w = make_3x3_world();
        assert!(w.in_bounds(0, 0));
        assert!(w.in_bounds(2, 2));
        assert!(!w.in_bounds(-1, 0));
        assert!(!w.in_bounds(0, -1));
        assert!(!w.in_bounds(3, 0));
        assert!(!w.in_bounds(0, 3));
    }

    #[test]
    fn test_set_entity_types() {
        let mut w = World::empty(5, 5);

        // Place a source at (0,0) facing east
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(0, 0, Channel::Items, Item::ElectronicCircuit as i64);

        // Place a belt at (1,0) facing east
        w.set(1, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);

        // Place a sink at (2,0) facing east
        w.set(2, 0, Channel::Entities, EntityKind::Sink as i64);
        w.set(2, 0, Channel::Direction, Direction::East as i64);
        w.set(2, 0, Channel::Items, Item::ElectronicCircuit as i64);

        assert_eq!(w.entity_at(0, 0), EntityKind::Source);
        assert_eq!(w.entity_at(1, 0), EntityKind::TransportBelt);
        assert_eq!(w.entity_at(2, 0), EntityKind::Sink);
        assert_eq!(w.item_at(0, 0), Item::ElectronicCircuit);
    }

    #[test]
    fn test_underground_belt_misc() {
        let mut w = World::empty(5, 5);
        w.set(1, 0, Channel::Entities, EntityKind::UndergroundBelt as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);
        w.set(1, 0, Channel::Misc, Misc::UndergroundDown as i64);

        w.set(3, 0, Channel::Entities, EntityKind::UndergroundBelt as i64);
        w.set(3, 0, Channel::Direction, Direction::East as i64);
        w.set(3, 0, Channel::Misc, Misc::UndergroundUp as i64);

        assert_eq!(w.misc_at(1, 0), Misc::UndergroundDown);
        assert_eq!(w.misc_at(3, 0), Misc::UndergroundUp);
        assert_eq!(w.misc_at(0, 0), Misc::None);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_mismatched_dimensions() {
        World::new(vec![0; 10], 3, 3, 4);
    }
}
