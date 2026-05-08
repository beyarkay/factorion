use nonempty::{nonempty, NonEmpty};

/// Channels in the WHC tensor (3rd dimension).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Entities = 0,
    Direction = 1,
    Items = 2,
    Misc = 3,
    Footprint = 4,
}

pub const NUM_CHANNELS: usize = 5;

impl Channel {
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Direction an entity is facing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    None = 0,
    North = 1,
    East = 2,
    South = 3,
    West = 4,
}

impl Direction {
    pub fn from_i64(v: i64) -> Self {
        match v {
            1 => Direction::North,
            2 => Direction::East,
            3 => Direction::South,
            4 => Direction::West,
            _ => Direction::None,
        }
    }

    /// Returns (dx, dy) for the direction the entity is facing.
    /// North = (0, -1), East = (1, 0), South = (0, 1), West = (-1, 0).
    pub fn delta(self) -> (i64, i64) {
        match self {
            Direction::North => (0, -1),
            Direction::East => (1, 0),
            Direction::South => (0, 1),
            Direction::West => (-1, 0),
            Direction::None => (0, 0),
        }
    }

    /// The opposite direction.
    #[allow(dead_code)]
    pub fn opposite(self) -> Self {
        match self {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::East => Direction::West,
            Direction::West => Direction::East,
            Direction::None => Direction::None,
        }
    }
}

/// Underground belt state flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Misc {
    None = 0,
    UndergroundDown = 1,
    UndergroundUp = 2,
}

impl Misc {
    pub fn from_i64(v: i64) -> Self {
        match v {
            1 => Misc::UndergroundDown,
            2 => Misc::UndergroundUp,
            _ => Misc::None,
        }
    }
}

/// The unified entity-and-item identifier.
///
/// In Factorion's data model, **everything is an Item**. Some items are
/// *placeable* (they can sit on the grid as entities); the rest are pure
/// inventory items (raw materials and intermediates). The world tensor
/// has two channels:
///   - `ENTITIES` channel: stores the Item id of the placeable item at
///     this tile, or 0 if no entity is placed here.
///   - `ITEMS` channel: stores the Item id of the carried/recipe/filter
///     item at this tile, or 0 if none.
///
/// There is no `Empty` variant: absence of an item is encoded as
/// channel value 0 and decoded to `Option::<Item>::None`.
///
/// Integer values are arranged so placeable items occupy ids 1..=7 and
/// non-placeable items 8..=12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Item {
    // Placeable, agent-buildable (1..=5)
    TransportBelt = 1,
    Inserter = 2,
    AssemblingMachine1 = 3,
    UndergroundBelt = 4,
    Splitter = 5,
    // Placeable, env-only — Source/Sink are placed by the env, not the
    // agent. They live last among placeables so the policy's entity head
    // can be sized to exclude them.
    Sink = 6,   // bulk_inserter in Python
    Source = 7, // stack_inserter in Python
    // Non-placeable (8..=12) — recipe ingredients / products
    CopperCable = 8,
    CopperPlate = 9,
    IronPlate = 10,
    ElectronicCircuit = 11,
    IronGearWheel = 12,
}

impl Item {
    /// Decode a tensor value into an Item. Returns `None` for value 0
    /// (the "no item" sentinel) and any unknown value.
    pub fn from_i64(v: i64) -> Option<Self> {
        match v {
            1 => Some(Item::TransportBelt),
            2 => Some(Item::Inserter),
            3 => Some(Item::AssemblingMachine1),
            4 => Some(Item::UndergroundBelt),
            5 => Some(Item::Splitter),
            6 => Some(Item::Sink),
            7 => Some(Item::Source),
            8 => Some(Item::CopperCable),
            9 => Some(Item::CopperPlate),
            10 => Some(Item::IronPlate),
            11 => Some(Item::ElectronicCircuit),
            12 => Some(Item::IronGearWheel),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Item::TransportBelt => "transport_belt",
            Item::Inserter => "inserter",
            Item::AssemblingMachine1 => "assembling_machine_1",
            Item::UndergroundBelt => "underground_belt",
            Item::Sink => "bulk_inserter",
            Item::Source => "stack_inserter",
            Item::Splitter => "splitter",
            Item::CopperCable => "copper_cable",
            Item::CopperPlate => "copper_plate",
            Item::IronPlate => "iron_plate",
            Item::ElectronicCircuit => "electronic_circuit",
            Item::IronGearWheel => "iron_gear_wheel",
        }
    }

    /// Whether this item can be placed on the grid as an entity.
    /// Non-placeable items only ever live in the ITEMS channel
    /// (carried on belts, set as recipes, etc.).
    pub fn is_placeable(self) -> bool {
        matches!(
            self,
            Item::TransportBelt
                | Item::Inserter
                | Item::AssemblingMachine1
                | Item::UndergroundBelt
                | Item::Sink
                | Item::Source
                | Item::Splitter
        )
    }

    /// Maximum items/second this item can transfer when placed as an
    /// entity. Returns 0.0 for non-placeable items.
    #[allow(dead_code)]
    pub fn flow_rate(self) -> f64 {
        match self {
            Item::TransportBelt => 15.0,
            Item::Inserter => 0.86,
            Item::AssemblingMachine1 => 0.5,
            Item::UndergroundBelt => 15.0,
            Item::Sink => f64::INFINITY,
            Item::Source => f64::INFINITY,
            Item::Splitter => 30.0, // 2 lanes × 15 i/s
            // Non-placeable: cannot transfer flow on its own.
            Item::CopperCable
            | Item::CopperPlate
            | Item::IronPlate
            | Item::ElectronicCircuit
            | Item::IronGearWheel => 0.0,
        }
    }

    /// Footprint (width, height) for placeable items.
    /// Width = perpendicular to flow, height = along flow.
    /// Non-placeable items return (1, 1).
    pub fn size(self) -> (usize, usize) {
        match self {
            Item::AssemblingMachine1 => (3, 3),
            Item::Splitter => (2, 1),
            _ => (1, 1),
        }
    }
}

/// Every Item variant. The single source of truth — Python's `items`
/// dict is built from this via the PyO3 `py_items` binding.
pub fn all_items() -> &'static [Item] {
    &[
        Item::TransportBelt,
        Item::Inserter,
        Item::AssemblingMachine1,
        Item::UndergroundBelt,
        Item::Sink,
        Item::Source,
        Item::Splitter,
        Item::CopperCable,
        Item::CopperPlate,
        Item::IronPlate,
        Item::ElectronicCircuit,
        Item::IronGearWheel,
    ]
}

/// A signed grid position. Used for multi-tile entity offsets where
/// coordinates may be temporarily negative before bounds-checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pos {
    pub x: i64,
    pub y: i64,
}

impl Pos {
    pub fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }

    /// Convert to usize coordinates, returning None if either is negative.
    pub fn to_usize(self) -> Option<(usize, usize)> {
        if self.x >= 0 && self.y >= 0 {
            Some((self.x as usize, self.y as usize))
        } else {
            None
        }
    }
}

/// A crafting recipe. `consumes` and `produces` are `NonEmpty` so an
/// empty recipe (no inputs or no outputs) is unrepresentable at compile
/// time — there is no need for runtime checks of these invariants.
#[derive(Debug, Clone)]
pub struct Recipe {
    pub consumes: NonEmpty<(Item, f64)>,
    pub produces: NonEmpty<(Item, f64)>,
}

impl Recipe {
    /// Look up the consumption rate for `item`, if present.
    #[allow(dead_code)]
    pub fn consumes_rate(&self, item: Item) -> Option<f64> {
        self.consumes
            .iter()
            .find(|(i, _)| *i == item)
            .map(|(_, r)| *r)
    }

    /// Look up the production rate for `item`, if present.
    #[allow(dead_code)]
    pub fn produces_rate(&self, item: Item) -> Option<f64> {
        self.produces
            .iter()
            .find(|(i, _)| *i == item)
            .map(|(_, r)| *r)
    }
}

/// All crafting recipes in the game. The single source of truth — both
/// `get_recipe` and the Python-facing PyO3 binding read from this.
///
/// Quantities are the canonical wiki recipe values (per craft, not
/// per-second). Throughput math (`transform_flow`) is scale-invariant
/// under uniform multiplication of consumes and produces.
pub fn all_recipes() -> Vec<(Item, Recipe)> {
    vec![
        // 1 copper plate -> 2 copper cables, 0.5s
        (
            Item::CopperCable,
            Recipe {
                consumes: nonempty![(Item::CopperPlate, 1.0)],
                produces: nonempty![(Item::CopperCable, 2.0)],
            },
        ),
        // 3 copper cable + 1 iron plate -> 1 electronic circuit, 0.5s
        (
            Item::ElectronicCircuit,
            Recipe {
                consumes: nonempty![(Item::CopperCable, 3.0), (Item::IronPlate, 1.0)],
                produces: nonempty![(Item::ElectronicCircuit, 1.0)],
            },
        ),
        // 2 iron plates -> 1 iron gear wheel, 0.5s
        (
            Item::IronGearWheel,
            Recipe {
                consumes: nonempty![(Item::IronPlate, 2.0)],
                produces: nonempty![(Item::IronGearWheel, 1.0)],
            },
        ),
        // 1 iron gear wheel + 1 iron plate -> 2 transport belts, 0.5s
        (
            Item::TransportBelt,
            Recipe {
                consumes: nonempty![(Item::IronGearWheel, 1.0), (Item::IronPlate, 1.0)],
                produces: nonempty![(Item::TransportBelt, 2.0)],
            },
        ),
        // 1 EC + 1 IGW + 1 iron plate -> 1 inserter, 0.5s
        (
            Item::Inserter,
            Recipe {
                consumes: nonempty![
                    (Item::ElectronicCircuit, 1.0),
                    (Item::IronGearWheel, 1.0),
                    (Item::IronPlate, 1.0)
                ],
                produces: nonempty![(Item::Inserter, 1.0)],
            },
        ),
        // 3 EC + 5 IGW + 9 iron plates -> 1 assembling machine 1, 0.5s
        (
            Item::AssemblingMachine1,
            Recipe {
                consumes: nonempty![
                    (Item::ElectronicCircuit, 3.0),
                    (Item::IronGearWheel, 5.0),
                    (Item::IronPlate, 9.0)
                ],
                produces: nonempty![(Item::AssemblingMachine1, 1.0)],
            },
        ),
    ]
}

/// Get the recipe for a given item, if one exists.
pub fn get_recipe(item: Item) -> Option<Recipe> {
    all_recipes()
        .into_iter()
        .find(|(i, _)| *i == item)
        .map(|(_, r)| r)
}

/// Unique identifier for a node in the factory graph.
/// Matches the Python format: "entity_name\n@x,y".
///
/// `entity_kind` is an `Item` (post-unification) and should always be
/// placeable — only placeable items become graph nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId {
    pub entity_kind: Item,
    pub x: usize,
    pub y: usize,
}

impl NodeId {
    pub fn new(entity_kind: Item, x: usize, y: usize) -> Self {
        Self { entity_kind, x, y }
    }

    #[allow(dead_code)]
    pub fn label(&self) -> String {
        format!("{}\n@{},{}", self.entity_kind.name(), self.x, self.y)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_direction_from_i64() {
        assert_eq!(Direction::from_i64(0), Direction::None);
        assert_eq!(Direction::from_i64(1), Direction::North);
        assert_eq!(Direction::from_i64(2), Direction::East);
        assert_eq!(Direction::from_i64(3), Direction::South);
        assert_eq!(Direction::from_i64(4), Direction::West);
        assert_eq!(Direction::from_i64(99), Direction::None);
    }

    #[test]
    fn test_direction_delta() {
        assert_eq!(Direction::North.delta(), (0, -1));
        assert_eq!(Direction::East.delta(), (1, 0));
        assert_eq!(Direction::South.delta(), (0, 1));
        assert_eq!(Direction::West.delta(), (-1, 0));
        assert_eq!(Direction::None.delta(), (0, 0));
    }

    #[test]
    fn test_direction_opposite() {
        assert_eq!(Direction::North.opposite(), Direction::South);
        assert_eq!(Direction::South.opposite(), Direction::North);
        assert_eq!(Direction::East.opposite(), Direction::West);
        assert_eq!(Direction::West.opposite(), Direction::East);
    }

    #[test]
    fn test_item_from_i64() {
        assert_eq!(Item::from_i64(0), None);
        assert_eq!(Item::from_i64(1), Some(Item::TransportBelt));
        assert_eq!(Item::from_i64(2), Some(Item::Inserter));
        assert_eq!(Item::from_i64(3), Some(Item::AssemblingMachine1));
        assert_eq!(Item::from_i64(4), Some(Item::UndergroundBelt));
        assert_eq!(Item::from_i64(5), Some(Item::Splitter));
        assert_eq!(Item::from_i64(6), Some(Item::Sink));
        assert_eq!(Item::from_i64(7), Some(Item::Source));
        assert_eq!(Item::from_i64(8), Some(Item::CopperCable));
        assert_eq!(Item::from_i64(11), Some(Item::ElectronicCircuit));
        assert_eq!(Item::from_i64(12), Some(Item::IronGearWheel));
        assert_eq!(Item::from_i64(99), None);
        assert_eq!(Item::from_i64(-1), None);
    }

    #[test]
    fn test_item_flow_rates() {
        assert_eq!(Item::TransportBelt.flow_rate(), 15.0);
        assert_eq!(Item::Inserter.flow_rate(), 0.86);
        assert_eq!(Item::AssemblingMachine1.flow_rate(), 0.5);
        assert_eq!(Item::UndergroundBelt.flow_rate(), 15.0);
        assert!(Item::Sink.flow_rate().is_infinite());
        assert!(Item::Source.flow_rate().is_infinite());
        assert_eq!(Item::CopperCable.flow_rate(), 0.0);
        assert_eq!(Item::IronPlate.flow_rate(), 0.0);
    }

    #[test]
    fn test_item_is_placeable() {
        for placeable in [
            Item::TransportBelt,
            Item::Inserter,
            Item::AssemblingMachine1,
            Item::UndergroundBelt,
            Item::Sink,
            Item::Source,
            Item::Splitter,
        ] {
            assert!(
                placeable.is_placeable(),
                "{:?} should be placeable",
                placeable
            );
        }
        for non_placeable in [
            Item::CopperCable,
            Item::CopperPlate,
            Item::IronPlate,
            Item::ElectronicCircuit,
            Item::IronGearWheel,
        ] {
            assert!(
                !non_placeable.is_placeable(),
                "{:?} should not be placeable",
                non_placeable
            );
        }
    }

    #[test]
    fn test_recipes() {
        let ec = get_recipe(Item::ElectronicCircuit).unwrap();
        assert_eq!(ec.consumes_rate(Item::CopperCable), Some(3.0));
        assert_eq!(ec.consumes_rate(Item::IronPlate), Some(1.0));
        assert_eq!(ec.produces_rate(Item::ElectronicCircuit), Some(1.0));

        let cc = get_recipe(Item::CopperCable).unwrap();
        assert_eq!(cc.consumes_rate(Item::CopperPlate), Some(1.0));
        assert_eq!(cc.produces_rate(Item::CopperCable), Some(2.0));

        let igw = get_recipe(Item::IronGearWheel).unwrap();
        assert_eq!(igw.consumes_rate(Item::IronPlate), Some(2.0));
        assert_eq!(igw.produces_rate(Item::IronGearWheel), Some(1.0));

        let tb = get_recipe(Item::TransportBelt).unwrap();
        assert_eq!(tb.consumes_rate(Item::IronGearWheel), Some(1.0));
        assert_eq!(tb.consumes_rate(Item::IronPlate), Some(1.0));
        assert_eq!(tb.produces_rate(Item::TransportBelt), Some(2.0));

        let ins = get_recipe(Item::Inserter).unwrap();
        assert_eq!(ins.consumes_rate(Item::ElectronicCircuit), Some(1.0));
        assert_eq!(ins.consumes_rate(Item::IronGearWheel), Some(1.0));
        assert_eq!(ins.consumes_rate(Item::IronPlate), Some(1.0));
        assert_eq!(ins.produces_rate(Item::Inserter), Some(1.0));

        let am1 = get_recipe(Item::AssemblingMachine1).unwrap();
        assert_eq!(am1.consumes_rate(Item::ElectronicCircuit), Some(3.0));
        assert_eq!(am1.consumes_rate(Item::IronGearWheel), Some(5.0));
        assert_eq!(am1.consumes_rate(Item::IronPlate), Some(9.0));
        assert_eq!(am1.produces_rate(Item::AssemblingMachine1), Some(1.0));

        assert!(get_recipe(Item::IronPlate).is_none());
        assert!(get_recipe(Item::CopperPlate).is_none());
    }

    #[test]
    fn test_all_recipes_lists_known_items() {
        let items: Vec<Item> = all_recipes().into_iter().map(|(i, _)| i).collect();
        assert!(items.contains(&Item::ElectronicCircuit));
        assert!(items.contains(&Item::CopperCable));
        assert!(items.contains(&Item::IronGearWheel));
        assert!(items.contains(&Item::TransportBelt));
        assert!(items.contains(&Item::Inserter));
        assert!(items.contains(&Item::AssemblingMachine1));
    }

    #[test]
    fn test_misc_from_i64() {
        assert_eq!(Misc::from_i64(0), Misc::None);
        assert_eq!(Misc::from_i64(1), Misc::UndergroundDown);
        assert_eq!(Misc::from_i64(2), Misc::UndergroundUp);
        assert_eq!(Misc::from_i64(99), Misc::None);
    }

    #[test]
    fn test_node_id_label() {
        let id = NodeId::new(Item::TransportBelt, 3, 5);
        assert_eq!(id.label(), "transport_belt\n@3,5");
    }
}
