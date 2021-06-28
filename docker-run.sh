#!/bin/bash
docker run --gpus='"device=0"' --ipc=host --net=host -it --rm -v `pwd`:/workspace -v /data2/checkpoints/nethack:/checkpoints nethack $@
