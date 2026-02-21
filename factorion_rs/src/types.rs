use std::collections::HashMap;

/// Channels in the WHC tensor (3rd dimension).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Entities = 0,
    Direction = 1,
    Items = 2,
    Misc = 3,
}

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

/// Item identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Item {
    Empty = 0,
    CopperCable = 1,
    CopperPlate = 2,
    IronPlate = 3,
    ElectronicCircuit = 4,
}

impl Item {
    pub fn from_i64(v: i64) -> Self {
        match v {
            1 => Item::CopperCable,
            2 => Item::CopperPlate,
            3 => Item::IronPlate,
            4 => Item::ElectronicCircuit,
            _ => Item::Empty,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Item::Empty => "empty",
            Item::CopperCable => "copper_cable",
            Item::CopperPlate => "copper_plate",
            Item::IronPlate => "iron_plate",
            Item::ElectronicCircuit => "electronic_circuit",
        }
    }
}

/// Entity type on the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Empty = 0,
    TransportBelt = 1,
    Inserter = 2,
    AssemblingMachine1 = 3,
    UndergroundBelt = 4,
    Sink = 5,   // bulk_inserter in Python
    Source = 6, // stack_inserter in Python
}

impl EntityKind {
    pub fn from_i64(v: i64) -> Self {
        match v {
            1 => EntityKind::TransportBelt,
            2 => EntityKind::Inserter,
            3 => EntityKind::AssemblingMachine1,
            4 => EntityKind::UndergroundBelt,
            5 => EntityKind::Sink,
            6 => EntityKind::Source,
            _ => EntityKind::Empty,
        }
    }

    pub fn flow_rate(self) -> f64 {
        match self {
            EntityKind::Empty => 0.0,
            EntityKind::TransportBelt => 15.0,
            EntityKind::Inserter => 0.86,
            EntityKind::AssemblingMachine1 => 0.5,
            EntityKind::UndergroundBelt => 15.0,
            EntityKind::Sink => f64::INFINITY,
            EntityKind::Source => f64::INFINITY,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            EntityKind::Empty => "empty",
            EntityKind::TransportBelt => "transport_belt",
            EntityKind::Inserter => "inserter",
            EntityKind::AssemblingMachine1 => "assembling_machine_1",
            EntityKind::UndergroundBelt => "underground_belt",
            EntityKind::Sink => "bulk_inserter",
            EntityKind::Source => "stack_inserter",
        }
    }
}

/// A crafting recipe.
#[derive(Debug, Clone)]
pub struct Recipe {
    pub consumes: HashMap<Item, f64>,
    pub produces: HashMap<Item, f64>,
}

/// Get the recipe for a given item, if one exists.
pub fn get_recipe(item: Item) -> Option<Recipe> {
    match item {
        Item::ElectronicCircuit => Some(Recipe {
            consumes: HashMap::from([
                (Item::CopperCable, 6.0),
                (Item::IronPlate, 2.0),
            ]),
            produces: HashMap::from([(Item::ElectronicCircuit, 2.0)]),
        }),
        Item::CopperCable => Some(Recipe {
            consumes: HashMap::from([(Item::CopperPlate, 2.0)]),
            produces: HashMap::from([(Item::CopperCable, 4.0)]),
        }),
        _ => None,
    }
}

/// Unique identifier for a node in the factory graph.
/// Matches the Python format: "entity_name\n@x,y"
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId {
    pub entity_kind: EntityKind,
    pub x: usize,
    pub y: usize,
}

impl NodeId {
    pub fn new(entity_kind: EntityKind, x: usize, y: usize) -> Self {
        Self { entity_kind, x, y }
    }

    pub fn label(&self) -> String {
        format!("{}\n@{},{}", self.entity_kind.name(), self.x, self.y)
    }
}

#[cfg(test)]
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
    fn test_entity_kind_from_i64() {
        assert_eq!(EntityKind::from_i64(0), EntityKind::Empty);
        assert_eq!(EntityKind::from_i64(1), EntityKind::TransportBelt);
        assert_eq!(EntityKind::from_i64(2), EntityKind::Inserter);
        assert_eq!(EntityKind::from_i64(3), EntityKind::AssemblingMachine1);
        assert_eq!(EntityKind::from_i64(4), EntityKind::UndergroundBelt);
        assert_eq!(EntityKind::from_i64(5), EntityKind::Sink);
        assert_eq!(EntityKind::from_i64(6), EntityKind::Source);
        assert_eq!(EntityKind::from_i64(-1), EntityKind::Empty);
    }

    #[test]
    fn test_entity_flow_rates() {
        assert_eq!(EntityKind::TransportBelt.flow_rate(), 15.0);
        assert_eq!(EntityKind::Inserter.flow_rate(), 0.86);
        assert_eq!(EntityKind::AssemblingMachine1.flow_rate(), 0.5);
        assert_eq!(EntityKind::UndergroundBelt.flow_rate(), 15.0);
        assert!(EntityKind::Sink.flow_rate().is_infinite());
        assert!(EntityKind::Source.flow_rate().is_infinite());
    }

    #[test]
    fn test_item_from_i64() {
        assert_eq!(Item::from_i64(0), Item::Empty);
        assert_eq!(Item::from_i64(1), Item::CopperCable);
        assert_eq!(Item::from_i64(4), Item::ElectronicCircuit);
    }

    #[test]
    fn test_recipes() {
        let ec = get_recipe(Item::ElectronicCircuit).unwrap();
        assert_eq!(ec.consumes[&Item::CopperCable], 6.0);
        assert_eq!(ec.consumes[&Item::IronPlate], 2.0);
        assert_eq!(ec.produces[&Item::ElectronicCircuit], 2.0);

        let cc = get_recipe(Item::CopperCable).unwrap();
        assert_eq!(cc.consumes[&Item::CopperPlate], 2.0);
        assert_eq!(cc.produces[&Item::CopperCable], 4.0);

        assert!(get_recipe(Item::Empty).is_none());
        assert!(get_recipe(Item::IronPlate).is_none());
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
        let id = NodeId::new(EntityKind::TransportBelt, 3, 5);
        assert_eq!(id.label(), "transport_belt\n@3,5");
    }
}
