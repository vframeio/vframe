#!/bin/bash

# get conda env name
CONDA_ENV_NAME=$(head -1 environment.yml | cut -d" " -f2)
echo $CONDA_ENV_NAME

# GTX 1080
CUDA_ARCH_BIN=6.1

# RTX 2080
# CUDA_ARCH_BIN=7.5

docker build \
  --build-arg CONDA_ENV_NAME=$CONDA_ENV_NAME \
  --build-arg CUDA_ARCH_BIN=$CUDA_ARCH_BIN \
	--rm  \
	-t vframe/faceless:gpu-cu101 \
	-f Dockerfile_cu101.gpu .

# Other options
# --no-cache