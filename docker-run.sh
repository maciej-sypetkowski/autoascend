#!/bin/bash
if [ $HOSTNAME == "lambda" ]; then
    devices=1,4,6
else
    devices=0
fi
docker run --gpus='"device='$devices'"' --ipc=host --net=host -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v /tmp/vis:/tmp/vis \
    -v /data2/checkpoints/nethack:/checkpoints \
    -v `pwd`:/workspace nethack:latest $@

