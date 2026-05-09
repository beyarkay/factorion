use crate::types::{Item, LaneTag};
use std::collections::HashMap;

/// Per-lane flow accumulator for a graph node.
///
/// Every flow has exactly two lanes — port and starboard — with no third
/// "pooled" bucket. Lane-agnostic entities (Source, Sink, Inserter,
/// AssemblingMachine) participate by writing the same value to both
/// lanes, or by routing through edges that target a specific lane.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LaneFlow {
    pub port: HashMap<Item, f64>,
    pub starboard: HashMap<Item, f64>,
}

impl LaneFlow {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `rate` of `item` to the lane identified by `tag`. Creates the
    /// item entry if absent.
    #[allow(dead_code)]
    pub fn add(&mut self, tag: LaneTag, item: Item, rate: f64) {
        *self.lane_mut(tag).entry(item).or_insert(0.0) += rate;
    }

    /// Read-only handle to the lane identified by `tag`.
    #[allow(dead_code)]
    pub fn lane(&self, tag: LaneTag) -> &HashMap<Item, f64> {
        match tag {
            LaneTag::Port => &self.port,
            LaneTag::Starboard => &self.starboard,
        }
    }

    /// Mutable handle to the lane identified by `tag`.
    #[allow(dead_code)]
    pub fn lane_mut(&mut self, tag: LaneTag) -> &mut HashMap<Item, f64> {
        match tag {
            LaneTag::Port => &mut self.port,
            LaneTag::Starboard => &mut self.starboard,
        }
    }

    /// Total rate of `item` across both lanes.
    #[allow(dead_code)]
    pub fn total(&self, item: Item) -> f64 {
        self.port.get(&item).copied().unwrap_or(0.0)
            + self.starboard.get(&item).copied().unwrap_or(0.0)
    }

    /// Cap each per-lane bucket at `cap` for every item.
    #[allow(dead_code)]
    pub fn cap_per_lane(&mut self, cap: f64) {
        for v in self.port.values_mut() {
            if *v > cap {
                *v = cap;
            }
        }
        for v in self.starboard.values_mut() {
            if *v > cap {
                *v = cap;
            }
        }
    }

    /// True if neither lane has any items.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.port.is_empty() && self.starboard.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::types::Item;

    #[test]
    fn test_default_is_empty() {
        let f = LaneFlow::default();
        assert!(f.is_empty());
        assert_eq!(f.total(Item::CopperPlate), 0.0);
    }

    #[test]
    fn test_add_to_port_and_starboard() {
        let mut f = LaneFlow::new();
        f.add(LaneTag::Port, Item::CopperPlate, 5.0);
        f.add(LaneTag::Starboard, Item::CopperPlate, 3.0);
        assert_eq!(f.lane(LaneTag::Port).get(&Item::CopperPlate), Some(&5.0));
        assert_eq!(
            f.lane(LaneTag::Starboard).get(&Item::CopperPlate),
            Some(&3.0)
        );
        assert_eq!(f.total(Item::CopperPlate), 8.0);
        assert!(!f.is_empty());
    }

    #[test]
    fn test_add_accumulates() {
        let mut f = LaneFlow::new();
        f.add(LaneTag::Port, Item::IronPlate, 2.0);
        f.add(LaneTag::Port, Item::IronPlate, 3.5);
        assert_eq!(f.lane(LaneTag::Port).get(&Item::IronPlate), Some(&5.5));
    }

    #[test]
    fn test_lane_mut_writes_through() {
        let mut f = LaneFlow::new();
        f.lane_mut(LaneTag::Starboard)
            .insert(Item::ElectronicCircuit, 9.0);
        assert_eq!(f.total(Item::ElectronicCircuit), 9.0);
    }

    #[test]
    fn test_total_is_sum_of_lanes() {
        let mut f = LaneFlow::new();
        f.add(LaneTag::Port, Item::IronGearWheel, 1.5);
        f.add(LaneTag::Starboard, Item::IronGearWheel, 2.5);
        f.add(LaneTag::Port, Item::CopperCable, 4.0);
        assert_eq!(f.total(Item::IronGearWheel), 4.0);
        assert_eq!(f.total(Item::CopperCable), 4.0);
    }

    #[test]
    fn test_cap_per_lane_caps_each_independently() {
        let mut f = LaneFlow::new();
        f.add(LaneTag::Port, Item::CopperPlate, 12.0);
        f.add(LaneTag::Starboard, Item::CopperPlate, 3.0);
        f.cap_per_lane(7.5);
        assert_eq!(f.lane(LaneTag::Port).get(&Item::CopperPlate), Some(&7.5));
        // Starboard was already under the cap and is unchanged.
        assert_eq!(
            f.lane(LaneTag::Starboard).get(&Item::CopperPlate),
            Some(&3.0)
        );
    }

    #[test]
    fn test_cap_per_lane_does_not_add_missing_items() {
        let mut f = LaneFlow::new();
        f.add(LaneTag::Port, Item::IronPlate, 100.0);
        f.cap_per_lane(7.5);
        // Starboard had nothing for IronPlate; cap doesn't introduce a 0 entry.
        assert!(!f.lane(LaneTag::Starboard).contains_key(&Item::IronPlate));
    }

    #[test]
    fn test_is_empty_after_clear() {
        let mut f = LaneFlow::new();
        f.add(LaneTag::Port, Item::IronPlate, 1.0);
        assert!(!f.is_empty());
        f.lane_mut(LaneTag::Port).clear();
        assert!(f.is_empty());
    }
}
