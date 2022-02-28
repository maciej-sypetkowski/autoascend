# AutoAscend -- 1st place NetHack bot for [the NetHack Challenge at NeurIPS 2021](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge)


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
to enable game seeding, glyph to tile mapping is generated, and tileset is downloaded.

We encourage using docker, but if you decide that you don't want to use it, be sure to make sure that the environment is compatible,
e.g. NLE version supports seeding, tileset is downloaded and hardcoded path in the code changed,
    `autoascend/visualization/glyph2tile.py` is a proper file instead of a symlink.


## How to run
    TODO


## Code structure
    TODO
