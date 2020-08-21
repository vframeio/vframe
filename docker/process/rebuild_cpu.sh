#!/bin/bash

# get conda env name
CONDA_ENV_NAME=$(head -1 environment.yml | cut -d" " -f2)
echo $CONDA_ENV_NAME

docker build \
  --build-arg CONDA_ENV_NAME=$CONDA_ENV_NAME \
	--rm  \
	-t vframe/faceless:cpu \
	-f Dockerfile.cpu .