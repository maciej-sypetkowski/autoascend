#!/bin/bash
docker run --gpus='"device=0"' --ipc=host --net=host -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v `pwd`:/workspace nethack $@

# -v /data2/checkpoints/nethack:/checkpoints nethack $@
