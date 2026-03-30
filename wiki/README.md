# Factorio Wiki Reference

Pre-parsed reference documentation from the [Factorio Wiki](https://wiki.factorio.com/)
for use by both humans and AI assistants working on the Factorion project.

## Why this exists

The Factorio wiki is large and covers many mechanics that Factorion doesn't
model (fluids, trains, circuits, Space Age, etc.). These docs contain only the
subset of game mechanics relevant to Factorion, with notes on where our
implementation simplifies or diverges from the real game.

## Entity index

| Entity | In Factorion? | Doc |
|---|---|---|
| Transport Belt | Yes | [[transport-belt]] |
| Inserter | Yes | [[inserter]] |
| Assembling Machine 1 | Yes | [[assembling-machine]] |
| Underground Belt | Yes | [[underground-belt]] |
| Splitter | Planned | [[splitter]] |
| Source / Sink | Yes (custom) | See [[inserter]] — they use inserter connection logic |

## Other docs

- [[items-and-recipes]] — items, crafting recipes, and how Factorion models them
- [[glossary]] — definitions of Factorio terms used in these docs

## What's deliberately excluded

Per the project README, Factorion ignores:

- Circuit network conditions
- Trains / railways
- Space Age (quality, new planets, new recipes)
- Fluids / pipes
- Logistics robots
- Biters / combat
- Modules / beacons
- Inserter / splitter filters
- Higher-tier entities (fast belts, assembling machine 2/3, etc.)

## How to read these docs

Each entity doc follows the same structure:

1. **Summary** — one-line description
2. **Factorio mechanics** — how it works in the real game
3. **Factorion implementation** — how we model it (enum values, simplifications)
4. **Interactions** — how it connects to other entities we care about
5. **Source** — link to the wiki page

Cross-references use `[[wiki-links]]` — e.g. `[[transport-belt]]` links to
`transport-belt.md`. These make it easy to grep for all docs that reference a
given entity.

## Adding new entries

When adding a new entity or topic to this wiki:

1. **Fetch the wiki page** using `WebFetch` from `https://wiki.factorio.com/<Entity_Name>`
2. **Extract only what's relevant** to Factorion — strip lore, history changelog
   entries, higher-tier variants, Space Age content, fluid mechanics, circuit
   network details, and anything else in the "excluded" list above
3. **Create the doc** as `wiki/<entity-name>.md` following the structure above
4. **Cross-reference** using `[[wiki-links]]` to other docs where appropriate
5. **Document simplifications** — always note where Factorion diverges from the
   real game (different ratios, missing mechanics, etc.)
6. **Check the Rust source** in `factorion_rs/src/types.rs` and
   `factorion_rs/src/entities.rs` for the actual enum values, flow rates, and
   connection logic — these are the ground truth for the "Factorion
   implementation" section
7. **Update this README** — add the entity to the index table above
8. **Update the glossary** if the new doc introduces terms not already defined
