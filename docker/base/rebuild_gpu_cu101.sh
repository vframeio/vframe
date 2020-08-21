#!/bin/bash

docker build \
	--rm  \
	-t vframe/base:gpu-cu101 \
	--build-arg user=$(whoami) \
	-f Dockerfile_cu101.gpu .