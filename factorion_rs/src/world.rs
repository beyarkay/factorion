use crate::types::{Channel, Direction, Item, Misc, NUM_CHANNELS};

/// A wrapper around a 3D array representing a factory world.
///
/// The world has dimensions W × H × C where:
/// - W is width (x axis)
/// - H is height (y axis)
/// - C is channels (entity, direction, item, misc, footprint, ores)
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

    /// Create an empty world of given dimensions with default channel values.
    /// FOOTPRINT is initialized to 1 (available); all other channels are 0.
    #[allow(dead_code)]
    pub fn empty(width: usize, height: usize) -> Self {
        let channels = NUM_CHANNELS;
        let mut data = vec![0; width * height * channels];
        for x in 0..width {
            for y in 0..height {
                data[x * height * channels + y * channels + Channel::Footprint.index()] = 1;
            }
        }
        Self {
            data,
            width,
            height,
            channels,
        }
    }

    /// Construct a World from a numpy array (called from PyO3).
    /// The array should have shape (W, H, C) and dtype i64.
    #[cfg(feature = "pyo3-bindings")]
    pub fn from_numpy(array: &numpy::PyReadonlyArray3<i64>) -> Self {
        use numpy::PyUntypedArrayMethods;

        let shape = array.shape();
        let width = shape[0];
        let height = shape[1];
        let channels = shape[2];
        let array = array.as_array();

        // Fast path: if the array is C-contiguous, copy the entire buffer at once.
        let data = if let Some(slice) = array.as_slice() {
            slice.to_vec()
        } else {
            // Fallback for non-contiguous arrays
            let mut data = vec![0i64; width * height * channels];
            for x in 0..width {
                for y in 0..height {
                    for c in 0..channels {
                        data[x * height * channels + y * channels + c] = array[[x, y, c]];
                    }
                }
            }
            data
        };

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

    /// Place an entity at (x, y) with the given direction and item.
    /// `item = None` writes 0 to the items channel (no item set).
    #[allow(dead_code)]
    pub fn place(&mut self, x: usize, y: usize, entity: Item, dir: Direction, item: Option<Item>) {
        self.set(x, y, Channel::Entities, entity as i64);
        self.set(x, y, Channel::Direction, dir as i64);
        self.set(x, y, Channel::Items, item.map_or(0, |i| i as i64));
    }

    /// Place an underground belt at (x, y) with the given direction and misc state.
    #[allow(dead_code)]
    pub fn place_underground(&mut self, x: usize, y: usize, dir: Direction, misc: Misc) {
        self.set(x, y, Channel::Entities, Item::UndergroundBelt as i64);
        self.set(x, y, Channel::Direction, dir as i64);
        self.set(x, y, Channel::Misc, misc as i64);
    }

    /// Place a multi-tile entity at (x, y), filling all tiles in its footprint.
    /// Returns false if any tile is out of bounds or already occupied.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub fn place_multi_tile(
        &mut self,
        x: usize,
        y: usize,
        entity: Item,
        dir: Direction,
        item: Option<Item>,
        width: usize,
        height: usize,
    ) -> bool {
        use crate::entities::entity_tiles;
        let tiles = match entity_tiles(x, y, dir, width, height) {
            Some(t) => t,
            None => return false,
        };
        for tile in &tiles {
            match tile.to_usize() {
                Some((tx, ty)) if self.in_bounds(tile.x, tile.y) => {
                    if self.entity_at(tx, ty).is_some() {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        for tile in tiles {
            if let Some((tx, ty)) = tile.to_usize() {
                self.place(tx, ty, entity, dir, item);
            }
        }
        true
    }

    /// Place a splitter at (x, y). Convenience wrapper around place_multi_tile.
    #[allow(dead_code)]
    pub fn place_splitter(
        &mut self,
        x: usize,
        y: usize,
        dir: Direction,
        item: Option<Item>,
    ) -> bool {
        self.place_multi_tile(x, y, Item::Splitter, dir, item, 2, 1)
    }

    /// Get the entity at (x, y). Returns `None` if the tile has no
    /// entity placed (channel value 0). The returned Item, if any,
    /// should always satisfy `is_placeable()`; non-placeable values in
    /// the entities channel are a data error.
    pub fn entity_at(&self, x: usize, y: usize) -> Option<Item> {
        Item::from_i64(self.get(x, y, Channel::Entities))
    }

    /// Get the direction at (x, y).
    pub fn direction_at(&self, x: usize, y: usize) -> Direction {
        Direction::from_i64(self.get(x, y, Channel::Direction))
    }

    /// Get the item at (x, y). Returns `None` for cells with no item set.
    pub fn item_at(&self, x: usize, y: usize) -> Option<Item> {
        Item::from_i64(self.get(x, y, Channel::Items))
    }

    /// Get the misc flags at (x, y).
    pub fn misc_at(&self, x: usize, y: usize) -> Misc {
        Misc::from_i64(self.get(x, y, Channel::Misc))
    }

    /// Get the ore terrain at (x, y). Returns `None` for cells with no ore.
    #[allow(dead_code)]
    pub fn ore_at(&self, x: usize, y: usize) -> Option<Item> {
        Item::from_i64(self.get(x, y, Channel::Ores))
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

    /// Consume the world, returning its flat `(W, H, C)` C-contiguous data
    /// buffer along with the dimensions — ready to hand to numpy. Used by the
    /// PyO3 `build_factory` binding to return the world to Python.
    pub fn into_whc(self) -> (Vec<i64>, usize, usize, usize) {
        (self.data, self.width, self.height, self.channels)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_3x3_world() -> World {
        World::empty(3, 3)
    }

    #[test]
    fn test_empty_world() {
        let w = make_3x3_world();
        assert_eq!(w.width(), 3);
        assert_eq!(w.height(), 3);
        for x in 0..3 {
            for y in 0..3 {
                assert_eq!(w.entity_at(x, y), None);
                assert_eq!(w.direction_at(x, y), Direction::None);
                assert_eq!(w.item_at(x, y), None);
                assert_eq!(w.misc_at(x, y), Misc::None);
            }
        }
    }

    #[test]
    fn test_set_and_get() {
        let mut w = make_3x3_world();
        w.place(
            1,
            2,
            Item::TransportBelt,
            Direction::East,
            Some(Item::CopperCable),
        );

        assert_eq!(w.entity_at(1, 2), Some(Item::TransportBelt));
        assert_eq!(w.direction_at(1, 2), Direction::East);
        assert_eq!(w.item_at(1, 2), Some(Item::CopperCable));

        assert_eq!(w.entity_at(0, 0), None);
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

        w.place(
            0,
            0,
            Item::Source,
            Direction::East,
            Some(Item::ElectronicCircuit),
        );

        w.place(1, 0, Item::TransportBelt, Direction::East, None);

        w.place(
            2,
            0,
            Item::Sink,
            Direction::East,
            Some(Item::ElectronicCircuit),
        );

        assert_eq!(w.entity_at(0, 0), Some(Item::Source));
        assert_eq!(w.entity_at(1, 0), Some(Item::TransportBelt));
        assert_eq!(w.entity_at(2, 0), Some(Item::Sink));
        assert_eq!(w.item_at(0, 0), Some(Item::ElectronicCircuit));
    }

    #[test]
    fn test_underground_belt_misc() {
        let mut w = World::empty(5, 5);
        w.place_underground(1, 0, Direction::East, Misc::UndergroundDown);
        w.place_underground(3, 0, Direction::East, Misc::UndergroundUp);

        assert_eq!(w.misc_at(1, 0), Misc::UndergroundDown);
        assert_eq!(w.misc_at(3, 0), Misc::UndergroundUp);
        assert_eq!(w.misc_at(0, 0), Misc::None);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_mismatched_dimensions() {
        World::new(vec![0; 10], 3, 3, NUM_CHANNELS);
    }
}
