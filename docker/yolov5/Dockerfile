#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io
# https://github.com/vframeio
#
#############################################################################

FROM vframe/base:gpu-cu111

MAINTAINER Adam Harvey / VFRAME.io

# --------------------------------------------------------


RUN sudo apt update

ENV DEBIAN_FRONTEND noninteractive
RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN sudo rm -rf /var/lib/apt/lists/*
# cleanup of files from setup
RUN sudo rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# [ Install YoloV5 ]

WORKDIR ${HOME}
RUN git clone https://github.com/ultralytics/yolov5


# [ conda ]

ARG CONDA_ENV_NAME
USER ${USER}
RUN sudo chown -R ${USER}:${USER} ${HOME}/.conda/
COPY requirements.txt ${DIR_TMP}/
RUN conda create -n ${CONDA_ENV_NAME} python=3.8 -y
RUN echo "conda activate ${CONDA_ENV_NAME}" >> ~/.zshrc
ENV PATH /opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH
RUN /bin/bash -c "/opt/conda/bin/activate ${CONDA_ENV_NAME}"
RUN pip install -r ${DIR_TMP}/requirements.txt


# [ update bash ]

COPY bashrc ${DIR_TMP}
RUN sudo chmod a+rwx /etc/zsh/zshrc
RUN sudo cat ${DIR_TMP}/bashrc >> /etc/zsh/zshrc
