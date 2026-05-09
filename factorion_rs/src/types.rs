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
    pub fn opposite(self) -> Self {
        match self {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::East => Direction::West,
            Direction::West => Direction::East,
            Direction::None => Direction::None,
        }
    }

    /// (dx, dy) offset to the cell on the port (left) side of a belt facing
    /// `self`. Port and starboard are *labels on lanes* inside the same 1×1
    /// belt tile — they have no offset within the tile. This helper returns
    /// the offset to the *neighbouring cell* that sits on the port side.
    ///
    /// Derived as a 90° CCW rotation of `delta()`: (dx, dy) → (dy, -dx).
    pub fn port_neighbor_offset(self) -> (i64, i64) {
        let (dx, dy) = self.delta();
        (dy, -dx)
    }

    /// (dx, dy) offset to the cell on the starboard (right) side of a belt
    /// facing `self`. 90° CW rotation of `delta()`: (dx, dy) → (-dy, dx).
    pub fn starboard_neighbor_offset(self) -> (i64, i64) {
        let (dx, dy) = self.delta();
        (-dy, dx)
    }

    /// True if `self` and `other` are perpendicular (one is N/S and the
    /// other is E/W). False if either is `Direction::None`.
    pub fn is_perpendicular(self, other: Direction) -> bool {
        match (self, other) {
            (Direction::None, _) | (_, Direction::None) => false,
            (Direction::North | Direction::South, Direction::East | Direction::West) => true,
            (Direction::East | Direction::West, Direction::North | Direction::South) => true,
            _ => false,
        }
    }

    /// Given a `(dx, dy)` offset relative to a belt facing `self`, return
    /// which lane-side that cell sits on. Returns `None` for the in-line
    /// cells (forward / backward) and any other offset.
    pub fn side_of(self, dx: i64, dy: i64) -> Option<LaneSide> {
        if (dx, dy) == self.port_neighbor_offset() {
            Some(LaneSide::Port)
        } else if (dx, dy) == self.starboard_neighbor_offset() {
            Some(LaneSide::Starboard)
        } else {
            None
        }
    }
}

/// Which side of a belt's flow direction a lane is on.
/// `Port` = left of travel direction; `Starboard` = right.
///
/// This is the geometry-side type returned by `Direction::side_of()` —
/// it answers "is this neighbour cell on the port or starboard side of a
/// belt facing me?". For *graph node identity* (which port-of-an-entity
/// a node represents), use `PortRole` below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaneSide {
    Port,
    Starboard,
}

impl LaneSide {
    pub fn opposite(self) -> Self {
        match self {
            LaneSide::Port => LaneSide::Starboard,
            LaneSide::Starboard => LaneSide::Port,
        }
    }
}

/// Identifies which I/O *port* of a multi-port entity a graph node
/// represents.
///
/// One graph node corresponds to one logical I/O port. Lane-aware
/// entities (TransportBelt, UndergroundBelt, Splitter) get separate
/// nodes per lane (`Port` and `Starboard`); lane-agnostic entities
/// (Inserter, Source, Sink, AssemblingMachine) get a single node
/// tagged `Single`.
///
/// This generalises beyond the lane axis — future entities with
/// multiple structurally distinct I/O channels (priority splitters,
/// chemical plants with two fluid inputs and one output, etc.) can add
/// additional `PortRole` variants without changing the edge model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortRole {
    /// Lane-agnostic entities expose a single node.
    Single,
    /// Lane-aware entities split into a port-side node...
    Port,
    /// ...and a starboard-side node.
    Starboard,
}

impl From<LaneSide> for PortRole {
    fn from(s: LaneSide) -> Self {
        match s {
            LaneSide::Port => PortRole::Port,
            LaneSide::Starboard => PortRole::Starboard,
        }
    }
}

/// Per-lane flow rate cap on a transport belt or splitter output. Each
/// lane carries up to 7.5 items/sec; the per-belt total of 15 i/s is
/// reached when both lanes are saturated.
pub const LANE_FLOW_RATE: f64 = 7.5;

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
/// Integer-id layout:
///   1..=5   — agent-placeable entities (TB, Inserter, AM1, UB, Splitter)
///   6..=65  — non-placeable items (recipe ingredients / products / raw
///             materials / non-modeled buildings exposed only as items)
///   66..=67 — env-spawned (Sink, Source) — MUST remain the last two
///             ids; ppo.py sizes its entity head to `len(items)-2` to
///             structurally exclude them. See `test_source_and_sink_are_last_two_ids`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Item {
    // Placeable, agent-buildable (1..=5)
    TransportBelt = 1,
    Inserter = 2,
    AssemblingMachine1 = 3,
    UndergroundBelt = 4,
    Splitter = 5,
    // Non-placeable (6..=65) — recipe ingredients / products
    CopperCable = 6,
    CopperPlate = 7,
    IronPlate = 8,
    ElectronicCircuit = 9,
    IronGearWheel = 10,
    // Raw materials (mining/chopping not modelled — no recipes)
    Wood = 11,
    Stone = 12,
    StoneBrick = 13,
    Concrete = 14,
    CopperOre = 15,
    IronOre = 16,
    // Crafted intermediates
    IronStick = 17,
    Barrel = 18,
    AdvancedCircuit = 19,
    EngineUnit = 20,
    // Smelting / refining (some have fluid recipes — added as Items, recipes skipped)
    SteelPlate = 21,
    SolidFuel = 22,
    PlasticBar = 23,
    Sulfur = 24,
    Battery = 25,
    Explosives = 26,
    // Storage
    WoodenChest = 27,
    IronChest = 28,
    StorageTank = 29,
    // Fast belt tier
    FastTransportBelt = 30,
    FastUndergroundBelt = 31,
    FastSplitter = 32,
    // Inserter variants
    BurnerInserter = 33,
    LongHandedInserter = 34,
    FastInserter = 35,
    // Power & pipes
    SmallElectricPole = 36,
    MediumElectricPole = 37,
    BigElectricPole = 38,
    Substation = 39,
    Pipe = 40,
    PipeToGround = 41,
    Pump = 42,
    // Production buildings + modules
    Accumulator = 43,
    Beacon = 44,
    Boiler = 45,
    BurnerMiningDrill = 46,
    ChemicalPlant = 47,
    EfficiencyModule = 48,
    ElectricFurnace = 49,
    HeatExchanger = 50,
    HeatPipe = 51,
    Lab = 52,
    NuclearReactor = 53,
    OffshorePump = 54,
    OilRefinery = 55,
    ProductivityModule = 56,
    Pumpjack = 57,
    QualityModule = 58,
    RepairPack = 59,
    SolarPanel = 60,
    SpeedModule = 61,
    SteamEngine = 62,
    SteamTurbine = 63,
    SteelFurnace = 64,
    StoneFurnace = 65,
    // Env-spawned, not agent-placeable — must remain the LAST two ids so
    // the policy's entity head can be sized to `len(items) - 2` and
    // structurally exclude them from sampling (see ppo.py).
    Sink = 66,   // bulk_inserter in Python
    Source = 67, // stack_inserter in Python
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
            6 => Some(Item::CopperCable),
            7 => Some(Item::CopperPlate),
            8 => Some(Item::IronPlate),
            9 => Some(Item::ElectronicCircuit),
            10 => Some(Item::IronGearWheel),
            11 => Some(Item::Wood),
            12 => Some(Item::Stone),
            13 => Some(Item::StoneBrick),
            14 => Some(Item::Concrete),
            15 => Some(Item::CopperOre),
            16 => Some(Item::IronOre),
            17 => Some(Item::IronStick),
            18 => Some(Item::Barrel),
            19 => Some(Item::AdvancedCircuit),
            20 => Some(Item::EngineUnit),
            21 => Some(Item::SteelPlate),
            22 => Some(Item::SolidFuel),
            23 => Some(Item::PlasticBar),
            24 => Some(Item::Sulfur),
            25 => Some(Item::Battery),
            26 => Some(Item::Explosives),
            27 => Some(Item::WoodenChest),
            28 => Some(Item::IronChest),
            29 => Some(Item::StorageTank),
            30 => Some(Item::FastTransportBelt),
            31 => Some(Item::FastUndergroundBelt),
            32 => Some(Item::FastSplitter),
            33 => Some(Item::BurnerInserter),
            34 => Some(Item::LongHandedInserter),
            35 => Some(Item::FastInserter),
            36 => Some(Item::SmallElectricPole),
            37 => Some(Item::MediumElectricPole),
            38 => Some(Item::BigElectricPole),
            39 => Some(Item::Substation),
            40 => Some(Item::Pipe),
            41 => Some(Item::PipeToGround),
            42 => Some(Item::Pump),
            43 => Some(Item::Accumulator),
            44 => Some(Item::Beacon),
            45 => Some(Item::Boiler),
            46 => Some(Item::BurnerMiningDrill),
            47 => Some(Item::ChemicalPlant),
            48 => Some(Item::EfficiencyModule),
            49 => Some(Item::ElectricFurnace),
            50 => Some(Item::HeatExchanger),
            51 => Some(Item::HeatPipe),
            52 => Some(Item::Lab),
            53 => Some(Item::NuclearReactor),
            54 => Some(Item::OffshorePump),
            55 => Some(Item::OilRefinery),
            56 => Some(Item::ProductivityModule),
            57 => Some(Item::Pumpjack),
            58 => Some(Item::QualityModule),
            59 => Some(Item::RepairPack),
            60 => Some(Item::SolarPanel),
            61 => Some(Item::SpeedModule),
            62 => Some(Item::SteamEngine),
            63 => Some(Item::SteamTurbine),
            64 => Some(Item::SteelFurnace),
            65 => Some(Item::StoneFurnace),
            66 => Some(Item::Sink),
            67 => Some(Item::Source),
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
            Item::Wood => "wood",
            Item::Stone => "stone",
            Item::StoneBrick => "stone_brick",
            Item::Concrete => "concrete",
            Item::CopperOre => "copper_ore",
            Item::IronOre => "iron_ore",
            Item::IronStick => "iron_stick",
            Item::Barrel => "barrel",
            Item::AdvancedCircuit => "advanced_circuit",
            Item::EngineUnit => "engine_unit",
            Item::SteelPlate => "steel_plate",
            Item::SolidFuel => "solid_fuel",
            Item::PlasticBar => "plastic_bar",
            Item::Sulfur => "sulfur",
            Item::Battery => "battery",
            Item::Explosives => "explosives",
            Item::WoodenChest => "wooden_chest",
            Item::IronChest => "iron_chest",
            Item::StorageTank => "storage_tank",
            Item::FastTransportBelt => "fast_transport_belt",
            Item::FastUndergroundBelt => "fast_underground_belt",
            Item::FastSplitter => "fast_splitter",
            Item::BurnerInserter => "burner_inserter",
            Item::LongHandedInserter => "long_handed_inserter",
            Item::FastInserter => "fast_inserter",
            Item::SmallElectricPole => "small_electric_pole",
            Item::MediumElectricPole => "medium_electric_pole",
            Item::BigElectricPole => "big_electric_pole",
            Item::Substation => "substation",
            Item::Pipe => "pipe",
            Item::PipeToGround => "pipe_to_ground",
            Item::Pump => "pump",
            Item::Accumulator => "accumulator",
            Item::Beacon => "beacon",
            Item::Boiler => "boiler",
            Item::BurnerMiningDrill => "burner_mining_drill",
            Item::ChemicalPlant => "chemical_plant",
            Item::EfficiencyModule => "efficiency_module",
            Item::ElectricFurnace => "electric_furnace",
            Item::HeatExchanger => "heat_exchanger",
            Item::HeatPipe => "heat_pipe",
            Item::Lab => "lab",
            Item::NuclearReactor => "nuclear_reactor",
            Item::OffshorePump => "offshore_pump",
            Item::OilRefinery => "oil_refinery",
            Item::ProductivityModule => "productivity_module",
            Item::Pumpjack => "pumpjack",
            Item::QualityModule => "quality_module",
            Item::RepairPack => "repair_pack",
            Item::SolarPanel => "solar_panel",
            Item::SpeedModule => "speed_module",
            Item::SteamEngine => "steam_engine",
            Item::SteamTurbine => "steam_turbine",
            Item::SteelFurnace => "steel_furnace",
            Item::StoneFurnace => "stone_furnace",
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
            | Item::IronGearWheel
            | Item::Wood
            | Item::Stone
            | Item::StoneBrick
            | Item::Concrete
            | Item::CopperOre
            | Item::IronOre
            | Item::IronStick
            | Item::Barrel
            | Item::AdvancedCircuit
            | Item::EngineUnit
            | Item::SteelPlate
            | Item::SolidFuel
            | Item::PlasticBar
            | Item::Sulfur
            | Item::Battery
            | Item::Explosives
            | Item::WoodenChest
            | Item::IronChest
            | Item::StorageTank
            | Item::FastTransportBelt
            | Item::FastUndergroundBelt
            | Item::FastSplitter
            | Item::BurnerInserter
            | Item::LongHandedInserter
            | Item::FastInserter
            | Item::SmallElectricPole
            | Item::MediumElectricPole
            | Item::BigElectricPole
            | Item::Substation
            | Item::Pipe
            | Item::PipeToGround
            | Item::Pump
            | Item::Accumulator
            | Item::Beacon
            | Item::Boiler
            | Item::BurnerMiningDrill
            | Item::ChemicalPlant
            | Item::EfficiencyModule
            | Item::ElectricFurnace
            | Item::HeatExchanger
            | Item::HeatPipe
            | Item::Lab
            | Item::NuclearReactor
            | Item::OffshorePump
            | Item::OilRefinery
            | Item::ProductivityModule
            | Item::Pumpjack
            | Item::QualityModule
            | Item::RepairPack
            | Item::SolarPanel
            | Item::SpeedModule
            | Item::SteamEngine
            | Item::SteamTurbine
            | Item::SteelFurnace
            | Item::StoneFurnace => 0.0,
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
        Item::Splitter,
        Item::CopperCable,
        Item::CopperPlate,
        Item::IronPlate,
        Item::ElectronicCircuit,
        Item::IronGearWheel,
        Item::Wood,
        Item::Stone,
        Item::StoneBrick,
        Item::Concrete,
        Item::CopperOre,
        Item::IronOre,
        Item::IronStick,
        Item::Barrel,
        Item::AdvancedCircuit,
        Item::EngineUnit,
        Item::SteelPlate,
        Item::SolidFuel,
        Item::PlasticBar,
        Item::Sulfur,
        Item::Battery,
        Item::Explosives,
        Item::WoodenChest,
        Item::IronChest,
        Item::StorageTank,
        Item::FastTransportBelt,
        Item::FastUndergroundBelt,
        Item::FastSplitter,
        Item::BurnerInserter,
        Item::LongHandedInserter,
        Item::FastInserter,
        Item::SmallElectricPole,
        Item::MediumElectricPole,
        Item::BigElectricPole,
        Item::Substation,
        Item::Pipe,
        Item::PipeToGround,
        Item::Pump,
        Item::Accumulator,
        Item::Beacon,
        Item::Boiler,
        Item::BurnerMiningDrill,
        Item::ChemicalPlant,
        Item::EfficiencyModule,
        Item::ElectricFurnace,
        Item::HeatExchanger,
        Item::HeatPipe,
        Item::Lab,
        Item::NuclearReactor,
        Item::OffshorePump,
        Item::OilRefinery,
        Item::ProductivityModule,
        Item::Pumpjack,
        Item::QualityModule,
        Item::RepairPack,
        Item::SolarPanel,
        Item::SpeedModule,
        Item::SteamEngine,
        Item::SteamTurbine,
        Item::SteelFurnace,
        Item::StoneFurnace,
        // Sink and Source MUST stay last — see test_source_and_sink_are_last_two_ids.
        Item::Sink,
        Item::Source,
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
        // ===========================================================
        // Wiki-scraped recipes (https://wiki.factorio.com), Base game.
        // Quantities are per-craft as listed on the wiki. Recipes that
        // consume or produce a fluid (water, petroleum_gas, ...) are
        // intentionally omitted — the data model has no fluid items
        // yet. The PNGs for those items are still in factorio-icons/
        // and the items themselves are present in the enum.
        // ===========================================================

        // Intermediates
        // 1 iron_plate -> 2 iron_stick, 0.5s
        (
            Item::IronStick,
            Recipe {
                consumes: nonempty![(Item::IronPlate, 1.0)],
                produces: nonempty![(Item::IronStick, 2.0)],
            },
        ),
        // 4 cable + 2 EC + 2 plastic -> 1 advanced_circuit, 6s
        (
            Item::AdvancedCircuit,
            Recipe {
                consumes: nonempty![
                    (Item::CopperCable, 4.0),
                    (Item::ElectronicCircuit, 2.0),
                    (Item::PlasticBar, 2.0)
                ],
                produces: nonempty![(Item::AdvancedCircuit, 1.0)],
            },
        ),
        // 1 IGW + 2 pipe + 1 steel_plate -> 1 engine_unit, 10s
        (
            Item::EngineUnit,
            Recipe {
                consumes: nonempty![
                    (Item::IronGearWheel, 1.0),
                    (Item::Pipe, 2.0),
                    (Item::SteelPlate, 1.0)
                ],
                produces: nonempty![(Item::EngineUnit, 1.0)],
            },
        ),
        // Smelting
        // 5 iron_plate -> 1 steel_plate, 16s
        (
            Item::SteelPlate,
            Recipe {
                consumes: nonempty![(Item::IronPlate, 5.0)],
                produces: nonempty![(Item::SteelPlate, 1.0)],
            },
        ),
        // Storage
        // 2 wood -> 1 wooden_chest, 0.5s
        (
            Item::WoodenChest,
            Recipe {
                consumes: nonempty![(Item::Wood, 2.0)],
                produces: nonempty![(Item::WoodenChest, 1.0)],
            },
        ),
        // 8 iron_plate -> 1 iron_chest, 0.5s
        (
            Item::IronChest,
            Recipe {
                consumes: nonempty![(Item::IronPlate, 8.0)],
                produces: nonempty![(Item::IronChest, 1.0)],
            },
        ),
        // 20 iron_plate + 5 steel_plate -> 1 storage_tank, 3s
        (
            Item::StorageTank,
            Recipe {
                consumes: nonempty![(Item::IronPlate, 20.0), (Item::SteelPlate, 5.0)],
                produces: nonempty![(Item::StorageTank, 1.0)],
            },
        ),
        // Fast belt tier
        // 5 IGW + 1 transport_belt -> 1 fast_transport_belt, 0.5s
        (
            Item::FastTransportBelt,
            Recipe {
                consumes: nonempty![(Item::IronGearWheel, 5.0), (Item::TransportBelt, 1.0)],
                produces: nonempty![(Item::FastTransportBelt, 1.0)],
            },
        ),
        // 40 IGW + 2 underground_belt -> 2 fast_underground_belt, 2s
        (
            Item::FastUndergroundBelt,
            Recipe {
                consumes: nonempty![(Item::IronGearWheel, 40.0), (Item::UndergroundBelt, 2.0)],
                produces: nonempty![(Item::FastUndergroundBelt, 2.0)],
            },
        ),
        // 10 EC + 10 IGW + 1 splitter -> 1 fast_splitter, 2s
        (
            Item::FastSplitter,
            Recipe {
                consumes: nonempty![
                    (Item::ElectronicCircuit, 10.0),
                    (Item::IronGearWheel, 10.0),
                    (Item::Splitter, 1.0)
                ],
                produces: nonempty![(Item::FastSplitter, 1.0)],
            },
        ),
        // Inserter variants
        // 1 IGW + 1 iron_plate -> 1 burner_inserter, 0.5s
        (
            Item::BurnerInserter,
            Recipe {
                consumes: nonempty![(Item::IronGearWheel, 1.0), (Item::IronPlate, 1.0)],
                produces: nonempty![(Item::BurnerInserter, 1.0)],
            },
        ),
        // 1 inserter + 1 IGW + 1 iron_plate -> 1 long_handed_inserter, 0.5s
        (
            Item::LongHandedInserter,
            Recipe {
                consumes: nonempty![
                    (Item::Inserter, 1.0),
                    (Item::IronGearWheel, 1.0),
                    (Item::IronPlate, 1.0)
                ],
                produces: nonempty![(Item::LongHandedInserter, 1.0)],
            },
        ),
        // 2 EC + 1 inserter + 2 iron_plate -> 1 fast_inserter, 0.5s
        (
            Item::FastInserter,
            Recipe {
                consumes: nonempty![
                    (Item::ElectronicCircuit, 2.0),
                    (Item::Inserter, 1.0),
                    (Item::IronPlate, 2.0)
                ],
                produces: nonempty![(Item::FastInserter, 1.0)],
            },
        ),
        // Power & pipes
        // 2 cable + 1 wood -> 2 small_electric_pole, 0.5s
        (
            Item::SmallElectricPole,
            Recipe {
                consumes: nonempty![(Item::CopperCable, 2.0), (Item::Wood, 1.0)],
                produces: nonempty![(Item::SmallElectricPole, 2.0)],
            },
        ),
        // 2 cable + 4 iron_stick + 2 steel_plate -> 1 medium_electric_pole, 0.5s
        (
            Item::MediumElectricPole,
            Recipe {
                consumes: nonempty![
                    (Item::CopperCable, 2.0),
                    (Item::IronStick, 4.0),
                    (Item::SteelPlate, 2.0)
                ],
                produces: nonempty![(Item::MediumElectricPole, 1.0)],
            },
        ),
        // 4 cable + 8 iron_stick + 5 steel_plate -> 1 big_electric_pole, 0.5s
        (
            Item::BigElectricPole,
            Recipe {
                consumes: nonempty![
                    (Item::CopperCable, 4.0),
                    (Item::IronStick, 8.0),
                    (Item::SteelPlate, 5.0)
                ],
                produces: nonempty![(Item::BigElectricPole, 1.0)],
            },
        ),
        // 5 advanced + 6 cable + 10 steel_plate -> 1 substation, 0.5s
        (
            Item::Substation,
            Recipe {
                consumes: nonempty![
                    (Item::AdvancedCircuit, 5.0),
                    (Item::CopperCable, 6.0),
                    (Item::SteelPlate, 10.0)
                ],
                produces: nonempty![(Item::Substation, 1.0)],
            },
        ),
        // 1 iron_plate -> 1 pipe, 0.5s
        (
            Item::Pipe,
            Recipe {
                consumes: nonempty![(Item::IronPlate, 1.0)],
                produces: nonempty![(Item::Pipe, 1.0)],
            },
        ),
        // 5 iron_plate + 10 pipe -> 2 pipe_to_ground, 0.5s
        (
            Item::PipeToGround,
            Recipe {
                consumes: nonempty![(Item::IronPlate, 5.0), (Item::Pipe, 10.0)],
                produces: nonempty![(Item::PipeToGround, 2.0)],
            },
        ),
        // 1 engine_unit + 1 pipe + 1 steel_plate -> 1 pump, 2s
        (
            Item::Pump,
            Recipe {
                consumes: nonempty![
                    (Item::EngineUnit, 1.0),
                    (Item::Pipe, 1.0),
                    (Item::SteelPlate, 1.0)
                ],
                produces: nonempty![(Item::Pump, 1.0)],
            },
        ),
        // Production buildings + modules
        // 5 battery + 2 iron_plate -> 1 accumulator, 10s
        (
            Item::Accumulator,
            Recipe {
                consumes: nonempty![(Item::Battery, 5.0), (Item::IronPlate, 2.0)],
                produces: nonempty![(Item::Accumulator, 1.0)],
            },
        ),
        // 20 advanced + 10 cable + 20 EC + 10 steel_plate -> 1 beacon, 15s
        (
            Item::Beacon,
            Recipe {
                consumes: nonempty![
                    (Item::AdvancedCircuit, 20.0),
                    (Item::CopperCable, 10.0),
                    (Item::ElectronicCircuit, 20.0),
                    (Item::SteelPlate, 10.0)
                ],
                produces: nonempty![(Item::Beacon, 1.0)],
            },
        ),
        // 4 pipe + 1 stone_furnace -> 1 boiler, 0.5s
        (
            Item::Boiler,
            Recipe {
                consumes: nonempty![(Item::Pipe, 4.0), (Item::StoneFurnace, 1.0)],
                produces: nonempty![(Item::Boiler, 1.0)],
            },
        ),
        // 3 IGW + 3 iron_plate + 1 stone_furnace -> 1 burner_mining_drill, 2s
        (
            Item::BurnerMiningDrill,
            Recipe {
                consumes: nonempty![
                    (Item::IronGearWheel, 3.0),
                    (Item::IronPlate, 3.0),
                    (Item::StoneFurnace, 1.0)
                ],
                produces: nonempty![(Item::BurnerMiningDrill, 1.0)],
            },
        ),
        // 5 EC + 5 IGW + 5 pipe + 5 steel_plate -> 1 chemical_plant, 5s
        (
            Item::ChemicalPlant,
            Recipe {
                consumes: nonempty![
                    (Item::ElectronicCircuit, 5.0),
                    (Item::IronGearWheel, 5.0),
                    (Item::Pipe, 5.0),
                    (Item::SteelPlate, 5.0)
                ],
                produces: nonempty![(Item::ChemicalPlant, 1.0)],
            },
        ),
        // 5 advanced + 5 EC -> 1 efficiency_module, 15s
        (
            Item::EfficiencyModule,
            Recipe {
                consumes: nonempty![(Item::AdvancedCircuit, 5.0), (Item::ElectronicCircuit, 5.0)],
                produces: nonempty![(Item::EfficiencyModule, 1.0)],
            },
        ),
        // 5 advanced + 10 steel_plate + 10 stone_brick -> 1 electric_furnace, 5s
        (
            Item::ElectricFurnace,
            Recipe {
                consumes: nonempty![
                    (Item::AdvancedCircuit, 5.0),
                    (Item::SteelPlate, 10.0),
                    (Item::StoneBrick, 10.0)
                ],
                produces: nonempty![(Item::ElectricFurnace, 1.0)],
            },
        ),
        // 100 copper_plate + 10 pipe + 10 steel_plate -> 1 heat_exchanger, 3s
        (
            Item::HeatExchanger,
            Recipe {
                consumes: nonempty![
                    (Item::CopperPlate, 100.0),
                    (Item::Pipe, 10.0),
                    (Item::SteelPlate, 10.0)
                ],
                produces: nonempty![(Item::HeatExchanger, 1.0)],
            },
        ),
        // 20 copper_plate + 10 steel_plate -> 1 heat_pipe, 1s
        (
            Item::HeatPipe,
            Recipe {
                consumes: nonempty![(Item::CopperPlate, 20.0), (Item::SteelPlate, 10.0)],
                produces: nonempty![(Item::HeatPipe, 1.0)],
            },
        ),
        // 10 EC + 10 IGW + 4 transport_belt -> 1 lab, 2s
        (
            Item::Lab,
            Recipe {
                consumes: nonempty![
                    (Item::ElectronicCircuit, 10.0),
                    (Item::IronGearWheel, 10.0),
                    (Item::TransportBelt, 4.0)
                ],
                produces: nonempty![(Item::Lab, 1.0)],
            },
        ),
        // 500 advanced + 500 concrete + 500 copper_plate + 500 steel_plate -> 1 nuclear_reactor, 8s
        (
            Item::NuclearReactor,
            Recipe {
                consumes: nonempty![
                    (Item::AdvancedCircuit, 500.0),
                    (Item::Concrete, 500.0),
                    (Item::CopperPlate, 500.0),
                    (Item::SteelPlate, 500.0)
                ],
                produces: nonempty![(Item::NuclearReactor, 1.0)],
            },
        ),
        // 2 IGW + 3 pipe -> 1 offshore_pump, 0.5s
        (
            Item::OffshorePump,
            Recipe {
                consumes: nonempty![(Item::IronGearWheel, 2.0), (Item::Pipe, 3.0)],
                produces: nonempty![(Item::OffshorePump, 1.0)],
            },
        ),
        // 10 EC + 10 IGW + 10 pipe + 15 steel_plate + 10 stone_brick -> 1 oil_refinery, 8s
        (
            Item::OilRefinery,
            Recipe {
                consumes: nonempty![
                    (Item::ElectronicCircuit, 10.0),
                    (Item::IronGearWheel, 10.0),
                    (Item::Pipe, 10.0),
                    (Item::SteelPlate, 15.0),
                    (Item::StoneBrick, 10.0)
                ],
                produces: nonempty![(Item::OilRefinery, 1.0)],
            },
        ),
        // 5 advanced + 5 EC -> 1 productivity_module, 15s
        (
            Item::ProductivityModule,
            Recipe {
                consumes: nonempty![(Item::AdvancedCircuit, 5.0), (Item::ElectronicCircuit, 5.0)],
                produces: nonempty![(Item::ProductivityModule, 1.0)],
            },
        ),
        // 5 EC + 10 IGW + 10 pipe + 5 steel_plate -> 1 pumpjack, 5s
        (
            Item::Pumpjack,
            Recipe {
                consumes: nonempty![
                    (Item::ElectronicCircuit, 5.0),
                    (Item::IronGearWheel, 10.0),
                    (Item::Pipe, 10.0),
                    (Item::SteelPlate, 5.0)
                ],
                produces: nonempty![(Item::Pumpjack, 1.0)],
            },
        ),
        // 5 advanced + 5 EC -> 1 quality_module, 15s
        (
            Item::QualityModule,
            Recipe {
                consumes: nonempty![(Item::AdvancedCircuit, 5.0), (Item::ElectronicCircuit, 5.0)],
                produces: nonempty![(Item::QualityModule, 1.0)],
            },
        ),
        // 2 EC + 2 IGW -> 1 repair_pack, 0.5s
        (
            Item::RepairPack,
            Recipe {
                consumes: nonempty![(Item::ElectronicCircuit, 2.0), (Item::IronGearWheel, 2.0)],
                produces: nonempty![(Item::RepairPack, 1.0)],
            },
        ),
        // 5 copper_plate + 15 EC + 5 steel_plate -> 1 solar_panel, 10s
        (
            Item::SolarPanel,
            Recipe {
                consumes: nonempty![
                    (Item::CopperPlate, 5.0),
                    (Item::ElectronicCircuit, 15.0),
                    (Item::SteelPlate, 5.0)
                ],
                produces: nonempty![(Item::SolarPanel, 1.0)],
            },
        ),
        // 5 advanced + 5 EC -> 1 speed_module, 15s
        (
            Item::SpeedModule,
            Recipe {
                consumes: nonempty![(Item::AdvancedCircuit, 5.0), (Item::ElectronicCircuit, 5.0)],
                produces: nonempty![(Item::SpeedModule, 1.0)],
            },
        ),
        // 8 IGW + 10 iron_plate + 5 pipe -> 1 steam_engine, 0.5s
        (
            Item::SteamEngine,
            Recipe {
                consumes: nonempty![
                    (Item::IronGearWheel, 8.0),
                    (Item::IronPlate, 10.0),
                    (Item::Pipe, 5.0)
                ],
                produces: nonempty![(Item::SteamEngine, 1.0)],
            },
        ),
        // 50 copper_plate + 50 IGW + 20 pipe -> 1 steam_turbine, 3s
        (
            Item::SteamTurbine,
            Recipe {
                consumes: nonempty![
                    (Item::CopperPlate, 50.0),
                    (Item::IronGearWheel, 50.0),
                    (Item::Pipe, 20.0)
                ],
                produces: nonempty![(Item::SteamTurbine, 1.0)],
            },
        ),
        // 6 steel_plate + 10 stone_brick -> 1 steel_furnace, 3s
        (
            Item::SteelFurnace,
            Recipe {
                consumes: nonempty![(Item::SteelPlate, 6.0), (Item::StoneBrick, 10.0)],
                produces: nonempty![(Item::SteelFurnace, 1.0)],
            },
        ),
        // 5 stone -> 1 stone_furnace, 0.5s
        (
            Item::StoneFurnace,
            Recipe {
                consumes: nonempty![(Item::Stone, 5.0)],
                produces: nonempty![(Item::StoneFurnace, 1.0)],
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
///
/// One graph node corresponds to one logical I/O port of an entity:
/// - Lane-aware entities (TB, UG, Splitter) place TWO nodes per anchor
///   tile, one with `port = Port` and one with `port = Starboard`.
/// - Lane-agnostic entities (Inserter, Source, Sink, AssemblingMachine)
///   place a single node with `port = Single`.
///
/// `entity_kind` should always be placeable — only placeable items
/// become graph nodes. The `(entity_kind, x, y, port)` tuple uniquely
/// identifies a node.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId {
    pub entity_kind: Item,
    pub x: usize,
    pub y: usize,
    pub port: PortRole,
}

impl NodeId {
    /// Build a NodeId for a lane-agnostic entity (Inserter, Source,
    /// Sink, AssemblingMachine). For lane-aware entities use `port` /
    /// `starboard`.
    pub fn single(entity_kind: Item, x: usize, y: usize) -> Self {
        Self {
            entity_kind,
            x,
            y,
            port: PortRole::Single,
        }
    }

    /// Build a NodeId for the port-lane node of a lane-aware entity.
    pub fn port(entity_kind: Item, x: usize, y: usize) -> Self {
        Self {
            entity_kind,
            x,
            y,
            port: PortRole::Port,
        }
    }

    /// Build a NodeId for the starboard-lane node of a lane-aware entity.
    pub fn starboard(entity_kind: Item, x: usize, y: usize) -> Self {
        Self {
            entity_kind,
            x,
            y,
            port: PortRole::Starboard,
        }
    }

    /// Build a NodeId targeting a specific lane side of a lane-aware
    /// entity. Convenience for converting from geometry to graph identity.
    pub fn lane(entity_kind: Item, x: usize, y: usize, side: LaneSide) -> Self {
        Self {
            entity_kind,
            x,
            y,
            port: side.into(),
        }
    }

    #[allow(dead_code)]
    pub fn label(&self) -> String {
        let suffix = match self.port {
            PortRole::Single => "",
            PortRole::Port => ":port",
            PortRole::Starboard => ":stbd",
        };
        format!(
            "{}{}\n@{},{}",
            self.entity_kind.name(),
            suffix,
            self.x,
            self.y
        )
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
    fn test_port_neighbor_offset() {
        // North-facer (moving up): port (left) = west
        assert_eq!(Direction::North.port_neighbor_offset(), (-1, 0));
        // East-facer (moving right): port = north
        assert_eq!(Direction::East.port_neighbor_offset(), (0, -1));
        // South-facer (moving down): port = east
        assert_eq!(Direction::South.port_neighbor_offset(), (1, 0));
        // West-facer (moving left): port = south
        assert_eq!(Direction::West.port_neighbor_offset(), (0, 1));
    }

    #[test]
    fn test_starboard_neighbor_offset() {
        // Starboard is always opposite of port and 90° CW of forward.
        assert_eq!(Direction::North.starboard_neighbor_offset(), (1, 0));
        assert_eq!(Direction::East.starboard_neighbor_offset(), (0, 1));
        assert_eq!(Direction::South.starboard_neighbor_offset(), (-1, 0));
        assert_eq!(Direction::West.starboard_neighbor_offset(), (0, -1));
    }

    #[test]
    fn test_port_starboard_are_opposite() {
        for dir in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            let (px, py) = dir.port_neighbor_offset();
            let (sx, sy) = dir.starboard_neighbor_offset();
            assert_eq!((px, py), (-sx, -sy), "port ≠ -starboard for {:?}", dir);
        }
    }

    #[test]
    fn test_is_perpendicular() {
        assert!(Direction::North.is_perpendicular(Direction::East));
        assert!(Direction::North.is_perpendicular(Direction::West));
        assert!(Direction::South.is_perpendicular(Direction::East));
        assert!(Direction::South.is_perpendicular(Direction::West));
        assert!(Direction::East.is_perpendicular(Direction::North));
        assert!(Direction::East.is_perpendicular(Direction::South));
        assert!(Direction::West.is_perpendicular(Direction::North));
        assert!(Direction::West.is_perpendicular(Direction::South));

        // Parallel cases (same or opposite) → not perpendicular
        for d in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            assert!(!d.is_perpendicular(d));
            assert!(!d.is_perpendicular(d.opposite()));
        }

        // Direction::None is never perpendicular to anything
        assert!(!Direction::None.is_perpendicular(Direction::East));
        assert!(!Direction::East.is_perpendicular(Direction::None));
        assert!(!Direction::None.is_perpendicular(Direction::None));
    }

    #[test]
    fn test_side_of() {
        // North-facing belt: cell at (-1, 0) is on its port side, (1, 0) starboard.
        assert_eq!(Direction::North.side_of(-1, 0), Some(LaneSide::Port));
        assert_eq!(Direction::North.side_of(1, 0), Some(LaneSide::Starboard));
        // In-line offsets (forward / backward) are not lane sides.
        assert_eq!(Direction::North.side_of(0, -1), None);
        assert_eq!(Direction::North.side_of(0, 1), None);
        // Non-adjacent offsets are not lane sides either.
        assert_eq!(Direction::North.side_of(2, 0), None);
        assert_eq!(Direction::North.side_of(0, 0), None);

        // South-facing belt: port = east, starboard = west.
        assert_eq!(Direction::South.side_of(1, 0), Some(LaneSide::Port));
        assert_eq!(Direction::South.side_of(-1, 0), Some(LaneSide::Starboard));

        // East-facing belt: port = north, starboard = south.
        assert_eq!(Direction::East.side_of(0, -1), Some(LaneSide::Port));
        assert_eq!(Direction::East.side_of(0, 1), Some(LaneSide::Starboard));

        // West-facing belt: port = south, starboard = north.
        assert_eq!(Direction::West.side_of(0, 1), Some(LaneSide::Port));
        assert_eq!(Direction::West.side_of(0, -1), Some(LaneSide::Starboard));
    }

    #[test]
    fn test_lane_side_opposite() {
        assert_eq!(LaneSide::Port.opposite(), LaneSide::Starboard);
        assert_eq!(LaneSide::Starboard.opposite(), LaneSide::Port);
    }

    #[test]
    fn test_port_role_from_lane_side() {
        assert_eq!(PortRole::from(LaneSide::Port), PortRole::Port);
        assert_eq!(PortRole::from(LaneSide::Starboard), PortRole::Starboard);
    }

    #[test]
    fn test_node_id_helpers() {
        let single = NodeId::single(Item::Inserter, 1, 2);
        assert_eq!(single.port, PortRole::Single);

        let p = NodeId::port(Item::TransportBelt, 3, 4);
        assert_eq!(p.port, PortRole::Port);

        let s = NodeId::starboard(Item::TransportBelt, 3, 4);
        assert_eq!(s.port, PortRole::Starboard);
        // Same entity + tile but different port → distinct NodeIds.
        assert_ne!(p, s);

        let p2 = NodeId::lane(Item::Splitter, 0, 0, LaneSide::Port);
        assert_eq!(p2.port, PortRole::Port);
    }

    #[test]
    fn test_lane_flow_rate_constant() {
        // Two lanes × 7.5 i/s = 15 i/s = belt total cap.
        assert_eq!(LANE_FLOW_RATE * 2.0, Item::TransportBelt.flow_rate());
        assert_eq!(LANE_FLOW_RATE * 2.0, Item::UndergroundBelt.flow_rate());
    }

    #[test]
    fn test_item_from_i64() {
        assert_eq!(Item::from_i64(0), None);
        assert_eq!(Item::from_i64(1), Some(Item::TransportBelt));
        assert_eq!(Item::from_i64(5), Some(Item::Splitter));
        assert_eq!(Item::from_i64(6), Some(Item::CopperCable));
        assert_eq!(Item::from_i64(10), Some(Item::IronGearWheel));
        assert_eq!(Item::from_i64(11), Some(Item::Wood));
        // Source/Sink must remain the LAST two ids — see Item enum docs.
        // Their numeric ids are derived from `all_items()` length so this
        // test stays correct as more items are added.
        let all = all_items();
        let n = all.len() as i64;
        let last = (n - 1) + 1; // ids start at 1
        assert_eq!(Item::from_i64(last - 1), Some(Item::Sink));
        assert_eq!(Item::from_i64(last), Some(Item::Source));
        assert_eq!(Item::from_i64(9999), None);
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

    /// LOAD-BEARING INVARIANT — DO NOT REMOVE.
    ///
    /// Source and Sink MUST be the last two ids in the Item enum (and the
    /// last two entries in `all_items()`). The PPO policy in ppo.py sizes
    /// its entity head to `len(items) - 2` to structurally exclude
    /// env-spawned source/sink from agent placement (`ppo.py` line 780).
    /// If anyone reorders the enum and breaks this invariant, the head
    /// will start sampling Source/Sink as agent actions, the env will
    /// reject them, and training will silently regress.
    ///
    /// Mirror tests live in tests/test_recipes.py — keep both. If you
    /// genuinely need to remove this protection, you must also rewrite
    /// `AgentCNN.__init__`'s entity-head sizing in ppo.py.
    #[test]
    fn test_source_and_sink_are_last_two_ids() {
        let all = all_items();
        let n = all.len();
        assert!(n >= 2, "all_items() must contain at least Source and Sink");
        assert_eq!(
            all[n - 2],
            Item::Sink,
            "Item::Sink must be the second-to-last entry in all_items() — \
             ppo.py's entity-head sizing depends on this"
        );
        assert_eq!(
            all[n - 1],
            Item::Source,
            "Item::Source must be the last entry in all_items() — \
             ppo.py's entity-head sizing depends on this"
        );

        // The integer values must also be the two highest. Using `as i64`
        // here is the canonical way to read the discriminant.
        let max_id = all.iter().map(|&i| i as i64).max().unwrap();
        let second_max_id = {
            let mut ids: Vec<i64> = all.iter().map(|&i| i as i64).collect();
            ids.sort();
            ids[ids.len() - 2]
        };
        assert_eq!(
            Item::Source as i64,
            max_id,
            "Item::Source must have the highest id"
        );
        assert_eq!(
            Item::Sink as i64,
            second_max_id,
            "Item::Sink must have the second-highest id"
        );
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
        // Lane-agnostic node: no suffix.
        let single = NodeId::single(Item::Inserter, 3, 5);
        assert_eq!(single.label(), "inserter\n@3,5");
        // Lane-aware nodes get a :port / :stbd suffix.
        let p = NodeId::port(Item::TransportBelt, 3, 5);
        assert_eq!(p.label(), "transport_belt:port\n@3,5");
        let s = NodeId::starboard(Item::TransportBelt, 3, 5);
        assert_eq!(s.label(), "transport_belt:stbd\n@3,5");
    }
}
