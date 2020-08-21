#!/bin/bash
docker build \
	--rm  \
	-t vframe/base:cpu \
	--build-arg user=$(whoami) \
	-f Dockerfile.cpu .