# AutoAscend -- 1st place NetHack agent for [the NetHack Challenge at NeurIPS 2021](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge)


## Description
The general overview of the approach can be find [here](https://youtu.be/fVkXE330Bh0?t=4439) (1:14:00 -- 1:21:21).
For more context about the challenge and NetHack see [the entire video](https://www.youtube.com/watch?v=fVkXE330Bh0).


## Environment
We supply the repo with `Dockerfile` that contains all necessary dependencies to run the code.

`./bin/docker-build.sh` and `./bin/docker-run.sh` are convinience scripts for building and running the docker container.
Note that they should be run only from the root of the repository.
`./bin/docker-run.sh` mounts X11 socket and Xauthority within the container to enable visualization.
You may need to tune it depending on your X11 configuration.

In `Dockerfile`, besides only installing dependencies,
the [NLE](https://github.com/facebookresearch/nle) library is pulled and slightly modified
to enable game seeding, glyph to tile mapping is generated, and tileset is downloaded,
muzero is pulled and custom patch applied (needed only for experimental reinforcement learning workflows).

We encourage using docker, but if you decide that you don't want to use it, be sure to make sure that the environment is compatible,
e.g. NLE version supports seeding, tileset is downloaded and hardcoded path in the code changed,
    `autoascend/visualization/glyph2tile.py` is a proper file instead of a symlink.


## How to run
`./bin/main.py <MODE> [PARAMS]` is the main entrypoint. It has three modes:
* `simulate` -- a mode that simulates `--episodes` episodes, and saves results to `--simulation-results` json file.
    If the file exists at the beginning, it checks which episodes were already simulated not to simulate episode
    with the same seed twice. The script uses [Ray](https://www.ray.io/) to allow running episodes in parallel,
    and requires Ray instance to be running (for simpliest setup just run `ray start --head` beforehand).
    If you're planning to develop the code and add new features make sure that you understand how Ray works,
    because in some cases it may not update code properly if you don't restart the server.
* `run` -- a mode that runs a single episode with visualization.
    The visualization supports custom input to override agent action. Just type any letter to pass this input to the environment.
    If you type `backspace` key, the agent action will be executed. `delete` key works similary, but fast forward 16 frames.
    Be aware that using custom input may confuse the agent, which may result in unexpected behavior and exceptions,
    so you may consider using `--panic-on-error` flag to handle gracefully unexpected errors.
* `profile` -- a mode that profiles the code. We implemented two profilers (cProfile and pyinstrument)
    that can be set with `--profiler` flag. In pyinstrument we customly process/fake tracebacks to adjust
    the summary report to our code to be easier to read and understand (refer to the implementation for details).


## Code structure
The base strategy class with description used for defining strategies is defined in `autoascend/strategy.py`.
Strategy consists of entering condition and agent's behavior. The class contains a few methods for controling the flow
and combine strategies together using functional interface (e.g. `repeat`, `until`, `preempt`), however strategies
can be also passed into and run inside other strategies in imperative manner if needed.

The main strategy is defined in `autoascend/global_logic.py:GlobalLogic.global_strategy()`

* `autoascend/global_logic.py` -- contains definitions of the main strategy and other high-level strategies
    (e.g. altar farming, altar item identification, dipping for the Excalibur)
* `autoascend/agent.py` -- definition of the agent class. The agent class contains logic for updating the state of the game,
    wraps NLE actions into atomic actions (e.g. untrap, pray, open_door), more complex actions (e.g. go_to),
    and defines some low-level stategies.
* `autoascend/item/item.py` -- an item instance. Contains a list of possible glyphs, a list of possible objects,
    amount of items, beatitude, information about being equipped, bonuses, etc.
* `autoascend/item/inventory.py` -- item and inventory handling logic. That included atoms and strategies for
    handling items taking into account bag's items, arranging items, selecting gear to wear/wield, etc.
* `autoascend/item/inventory_items.py` -- a class representing items that are in player's inventory.
* `autoascend/item/item_manager.py` -- a class for managing general information about items in the game.
    That includes item identification helpers, known glyph to object and object to glyph mapping, content of bags, parsing items.
* `autoascend/combat` -- combat behavior and helpers.
* `autoascend/exploration_logic.py` -- exploration specific strategies, including exploration within the level and across levels.
* `autoascend/env_wrapper.py` -- an NLE environment wrapper. That includes utilities for forking the process and reloading the agent.
* `autoascend/glyph` -- hardcoded glyphs with their meaning and related helpers.
* `autoascend/object` -- hardcoded objects with their meaning and related helpers.
* `autoascend/soko_solver` -- utilities and method for solving sokoban.
* `autoascend/visualization` -- visualization tool.
