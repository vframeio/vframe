#!/bin/bash
# source environment variables for example scripts

# base
DIR_EXAMPLES="data/examples"
DIR_IMAGES=${DIR_EXAMPLES}"/images"
DIR_IMAGES_OUT=${DIR_IMAGES}"/output"
DIR_VIDEOS=${DIR_EXAMPLES}"/videos"
DIR_VIDEO_OUT=${DIR_VIDEOS}"/output"

# face images
FACE_IMAGE=${DIR_IMAGES}"/face-snowden-x1.png"

# face videos
FACE_VIDEO=${DIR_VIDEOS}"/face-snowden-x1.mp4"

echo "Added VFRAME example filepaths to environment"