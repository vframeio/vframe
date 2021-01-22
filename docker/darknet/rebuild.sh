#!/bin/bash
docker build \
	--rm  \
	-t vframe/darknet \
        --no-cache \
	-f Dockerfile .
