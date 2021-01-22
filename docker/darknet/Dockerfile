#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io
# https://github.com/vframeio
#
#############################################################################

FROM vframe/base:gpu-cu102

MAINTAINER Adam Harvey / VFRAME.io

# --------------------------------------------------------


RUN sudo apt update

ENV DEBIAN_FRONTEND noninteractive
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN sudo rm -rf /var/lib/apt/lists/*
# cleanup of files from setup
RUN sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install older version of OpenCV compatible with Darknet
RUN sudo apt update
RUN sudo apt install libopencv-dev -y
RUN sudo apt install ffmpeg -y
RUN sudo apt install libcanberra-gtk-module -y

# [ Install Darknet ]

WORKDIR ${HOME}
RUN git clone https://github.com/AlexeyAB/darknet/
RUN	cd darknet && \
	sed -i 's/# ARCH= -gencode arch=compute_61/ARCH= -gencode arch=compute_61/g' Makefile && \
	sed -i 's/GPU=0/GPU=1/g' Makefile && \
	sed -i 's/CUDNN=0/CUDNN=1/g' Makefile && \
	sed -i 's/AVX=0/AVX=1/g' Makefile && \
	sed -i 's/OPENMP=0/OPENMP=1/g' Makefile && \
	sed -i 's/OPENCV=0/OPENCV=1/g' Makefile && \
	make
