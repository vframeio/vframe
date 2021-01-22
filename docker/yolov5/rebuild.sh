#!/bin/bash

CONDA_ENV_NAME=yolov5
# --no-cache \

docker build \
	--rm  \
	--build-arg CONDA_ENV_NAME=$CONDA_ENV_NAME \
	-t vframe/yolov5 \
	-f Dockerfile .
