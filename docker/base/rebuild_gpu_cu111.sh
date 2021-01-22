#!/bin/bash

docker build \
	--rm  \
	-t vframe/base:gpu-cu111 \
	--build-arg user=$(whoami) \
	-f Dockerfile_cu111.gpu .
