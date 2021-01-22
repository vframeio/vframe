#!/bin/bash
# Startup script for docker image

IMAGE=vframe/base:gpu-cu111
#docker images $IMAGE
xhost +local:docker
DOCKER_NAME="$(echo $IMAGE | sed 's/\//-/g' | sed 's/:/-/g' | sed 's/_/-/g')"

docker run \
	-u $(whoami):$(whoami) \
	-h $(hostname)-$DOCKER_NAME \
	-it \
	--gpus all \
	-v /work:/work \
	-w /work \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-p 8888:8888 \
	$IMAGE /bin/zsh
