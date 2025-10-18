# Factorion

This project aims to create a ML model that can create high-throughput
factories in the video-game factorio. It's been through several iterations and
ideas, but the current research direction is to train an reinforcement-learning
agent to place entities in the game world, and then reward that agent based on
the throughput of the built factory.

Once an agent is successfully building factories, this agent will be integrated
into a factorio mod, so that you can just place "source" tiles and "sink"
tiles, and the agent will create a factory that'll convert whatever items are
in the source into whatever items are in the sink.

Weights & Biases report (a few months out of date, 2025-04-29): https://api.wandb.ai/links/beyarkay/wmccb7fq

## Running the code

```
uv pip install -r requirements.txt
source .venv/bin/activate
python ppo.py \
    --seed 1 \
    --env-id factorion/FactorioEnv-v0 \
    --track \
    --wandb-project-name factorion \
    --total-timesteps 50000
```

## Ideas around gradually teaching/training the agent

Environments, from easiest to hardest:

1. Given a working factory, don't break it
2. Given a working factory, try and improve it
3. Given a factory with one entity missing, add that tile
4. Given a factory with N entities missing, add those entities
5. Given a factory with only the input and output, place all the entities
   required to maximise throughput

## Ideas around the reward signal

- The throughput of the factory (items/second)
- The number of entities placed
- The number of entities which are connected to one another in the graph
- the distance that input items travel away from the source

## Ideas around the environment

Currently, the environment is a custom-built mimic of a fraction of factorio's
features. This isn't idea, but it is (probably) faster than factorio.

Really, I want to have either an actual factorio instance running, or a much
better mimic of factorio's behaviour. I'm pretty sure running a full factorio
instance is going to be very very slow, and won't parallelise without a lot of
boring effort.

Maybe, to collect training data, I could download some factorio worlds or
blueprints and use chunks of them as training data to show the RL agent?

## Open questions

- The agent struggles to learn to place even one transport belt. Something
  seems fundamentally wrong with the system.
  - Maybe the RL loop is too slow so there's not enough training? Need to
    measure time taken doing forward/backward passes vs time taken
    calculating the environment.
  - Maybe the underlying policy doesn't have the size required to actually
    figure out the problem?
- I don't feel like I have visibility about why the agent is struggling, and
  can't diagnose issues easily. Key metrics:
  - Time for environment
  - Time for forward/backward pass
  - Time for everything else
  - Number of entities on the map
  - Whether the agent's action was valid or invalid
  - Throughput of the environment
  - Number of steps taken until finish
  - Steps per second
- Also need definitions for what the various terms are, or to get better names

## Layout of ppo.py

1.  setup
2.  for iteration in range(args.num_iterations):
3.  for step in range(args.num_steps):
4.             for each environment:
5.               calculate action based on input
6.               update each env based on the action
7.               if an env is done, reset it
