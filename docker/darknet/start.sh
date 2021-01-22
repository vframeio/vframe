#!/bin/bash
# Startup script for docker image

IMAGE=vframe/darknet
#docker images $IMAGE
xhost +local:docker
DOCKER_NAME="$(echo $IMAGE | sed 's/\//-/g' | sed 's/:/-/g' | sed 's/_/-/g')"

docker run \
	-u $(whoami):$(whoami) \
	-h $(hostname)-$DOCKER_NAME \
	-it \
	--gpus all \
	-v /work:/work \
	-v /data_store_ssd:/data_store_ssd \
	-v /data_store_vframe:/data_store_vframe \
	-w /work \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	$IMAGE /bin/zsh