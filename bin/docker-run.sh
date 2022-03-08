#!/bin/bash
devices=0
docker run --gpus='"device='$devices'"' --ipc=host --net=host -it --rm \
    -e DISPLAY=$DISPLAY \
    -e PYTHONPATH=/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v `pwd`:/workspace nethack:latest "$@"
