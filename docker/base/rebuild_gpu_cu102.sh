#!/bin/bash

docker build \
	--rm  \
	-t vframe/base:gpu-cu102 \
	--build-arg user=$(whoami) \
	-f Dockerfile_cu102.gpu .
