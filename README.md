# Factorion

_(name is a placeholder, plz suggest better ones)_

The goal is to get a computer playing factorio. Or at the very least generating
reasonable factory blueprints when you prompt it. Maybe later we'll augment
this with a different bot that can actually walk around the game and build from
the blueprints suggested by the first bot.

The plan is to get a dataset of factorio blueprints (ie just JSON files) and
train a Masked Language Model (MLM) to predict missing parts of the blueprints.
A decoded blueprint looks something like:

```json
{
  "blueprint": {
    "description": "A set of 6 assembling machine that take in copper plates and iron plates, construct red science, and then move that red science into labs to do research",
    "icons": [ { "signal": { "type": "item", "name": "assembling-machine-1" }, "index": 1 } ],
    "entities": [
      {
        "entity_number": 1,
        "name": "assembling-machine-1",
        "position": { "x": -8.5, "y": 40.5 },
        "recipe": "iron-gear-wheel"
      },
      ...
      {
        "entity_number": 11,
        "name": "lab",
        "position": { "x": -4.5, "y": 45.5 }
      }
    ],
    "item": "blueprint",
    "label": "Red Science",
    "version": 281479278231552
  }
}
```

So we'll mask out the entity name `lab` and train the model to predict what
should go there, based on all the other entities surrounding it. We'll also
mask out the x and y coordinates and other factorio-specific values.
Critically, we won't mask out the JSON formatting tokens (there's no point in
teaching the model how to write JSON AFAICT) and also won't mask out the
`"description"` field, so the final model could take in just a description of a
factory and output the blueprint for that factory.

## Flaws

This technique has no measure of how "good" a factory is. It has no ability to
optimise the throughput/output of a factory. Probably we'll use this model as a
bootstrap to generate data for a reinforcement learning model that will
actually generate super-human factories.

## Task list

- [ ] **Collect data**: We need a lot of JSON blueprints with high-quality
      descriptions. Ideally we also want to classify these blueprints into "easy"
      and "everything else". Where "easy" would just have yellow belts, nothing
      else. Once the model can generate yellow belts in a plausible fashion,
      we'll move on to such exciting things as underground belts (gasp!).
- [ ] **Data augmentation**: We'll need a _lot_ of data, but really a lot of it
      is permutations. Rotate the whole thing 90 degrees. Translate the whole thing
      left/right/up/down. Upgrade certain entities. There's some basic
      augmentation in `train.py` but it could do more.
- [ ] **Get a reliable training setup**: Not sure where/how to train the model.
      I (boyd) have a basic training script setup `train.py` that also loads
      and augments the dataset, but I'm not sure about the easiest way to do
      training runs. I'm going to try get my server setup which might be
      enough.
