#!/bin/bash

# Startup script for docker image
IMAGE=vframe/faceless:gpu-cu102
xhost +local:docker
DOCKER_NAME="$(echo $IMAGE | sed 's/\//-/g' | sed 's/:/-/g' | sed 's/_/-/g')"

docker run \
	-u $(whoami):$(whoami) \
	-h $(hostname)-$DOCKER_NAME \
	-it \
	--gpus all \
	-v /work:/work \
	-w /work \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	$IMAGE /bin/zsh