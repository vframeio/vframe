#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

import os
from os.path import join
from pathlib import Path
import logging

from dotenv import load_dotenv
import yaml
import cv2 as cv

from vframe.models.color import Color

# -----------------------------------------------------------------------------
# CV Modules
# -----------------------------------------------------------------------------

SRES_ENABLED = 'superres' in cv.getBuildInformation()
CUDA_ENABLED = 'CUDA' in cv.getBuildInformation()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG = logging.getLogger('vframe')


# -----------------------------------------------------------------------------
# Filepaths
# -----------------------------------------------------------------------------

# Project directory
SELF_CWD = os.path.dirname(os.path.realpath(__file__))  # this file
DIR_PROJECT_ROOT = str(Path(SELF_CWD).parent.parent.parent)

# source .env vars
fp_env = Path(DIR_PROJECT_ROOT) / '.env'
load_dotenv(dotenv_path=fp_env, verbose=True)

# directories
DIR_DATA_STORE = os.getenv("DATA_STORE", join(DIR_PROJECT_ROOT, 'data'))
DIR_CONFIGS = join(DIR_DATA_STORE, 'configs')
DIR_MODELZOO = join(DIR_PROJECT_ROOT, 'modelzoo')
DIR_MODELS = join(DIR_MODELZOO, 'models')
DIR_PLUGINS = join(DIR_MODELZOO, 'plugins')

DIR_CLI = join(DIR_PROJECT_ROOT, 'vframe_cli')
DIR_FONTS = join(DIR_DATA_STORE, 'fonts')

# fonts
FP_ROBOTO_300 = join(DIR_FONTS, 'roboto/Roboto_300.ttf')
FP_ROBOTO_400 = join(DIR_FONTS, 'roboto/Roboto_400.ttf')
FP_ROBOTO_500 = join(DIR_FONTS, 'roboto/Roboto_500.ttf')
FP_ROBOTO_700 = join(DIR_FONTS, 'roboto/Roboto_700.ttf')
FP_HELVETICA_NORMAL = join(DIR_FONTS, 'helvetica/Helvetica-Normal.ttf')
FP_HELVETICA_BOLD = join(DIR_FONTS, 'helvetica/Helvetica-Bold.ttf')

# filenames
FN_CSV_SUMMARY  = 'summary.csv'
FN_LABEL_COLORS = 'label_colors.json'

# VFRAME config file
FP_VFRAME_YAML = join(DIR_PROJECT_ROOT, os.getenv("FP_VFRAME_YAML", "vframe.yaml"))

# Cmake config files
FP_CMAKE_OPENCV = join(DIR_DATA_STORE, 'configs/cmake/opencv.yaml')


# -----------------------------------------------------------------------------
# Input types
# -----------------------------------------------------------------------------

VALID_PIPE_IMAGE_EXTS = ['jpg', 'jpeg', 'png']
VALID_PIPE_VIDEO_EXTS = ['mp4', 'avi', 'mov', 'webm']
VALID_PIPE_MEDIA_EXTS =  VALID_PIPE_IMAGE_EXTS + VALID_PIPE_VIDEO_EXTS
VALID_PIPE_DATA_EXTS = ['json']
VALID_PIPE_EXTS = VALID_PIPE_MEDIA_EXTS.extend(VALID_PIPE_DATA_EXTS)


# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------

RED = Color.from_rgb_int((255, 0, 0))
ORANGE = Color.from_rgb_int((255, 255, 127))
YELLOW = Color.from_rgb_int((255, 255, 0))
FUSCHIA = Color.from_rgb_int((255, 0, 127))
PINK = Color.from_rgb_int((255, 0, 255))
PURPLE = Color.from_rgb_int((127, 0, 255))
LAVENDER = Color.from_rgb_int((127, 127, 255))
CYAN = Color.from_rgb_int((0, 255, 255))
GREEN = Color.from_rgb_int((0, 255, 0))
BLUE = Color.from_rgb_int((0, 0, 255))

BLACK = Color.from_rgb_int((0, 0, 0))
WHITE = Color.from_rgb_int((255, 255, 255))
GRAY = Color.from_rgb_int((127, 127, 127))
LIGHT_GRAY = Color.from_rgb_int((170, 170, 170))
DARK_GRAY = Color.from_rgb_int((85, 85, 85))

DEFAULT_TEXT_SIZE = 14
DEFAULT_STROKE_WEIGHT = 2
DEFAULT_SIZE_LABEL = 14
DEFAULT_PADDING_PER = 0.25
DEFAULT_FONT_NAME = 'roboto'
DEFAULT_FONT_FP = FP_ROBOTO_400

class TERM_COLORS:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

# -----------------------------------------------------------------------------
# OpenCV DNN
# -----------------------------------------------------------------------------

dnn_backends = {
  'DEFAULT': cv.dnn.DNN_BACKEND_DEFAULT,
  'HALIDE': cv.dnn.DNN_BACKEND_HALIDE,  # not used
  'INFERENCE_ENGINE': cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,  # not used
  'OPENCV': cv.dnn.DNN_BACKEND_OPENCV,
  'VKCOM': cv.dnn.DNN_BACKEND_VKCOM,  # not used
}
dnn_targets = {
	'DEFAULT': cv.dnn.DNN_TARGET_CPU,  # alias to CPU
  'CPU': cv.dnn.DNN_TARGET_CPU,
  'FPGA': cv.dnn.DNN_TARGET_FPGA,  # not used
  'MYRIAD': cv.dnn.DNN_TARGET_MYRIAD,  # not used
  'OPENCL': cv.dnn.DNN_TARGET_OPENCL,  # not used
  'OPENCL_FP16': cv.dnn.DNN_TARGET_OPENCL_FP16,  # not used
  'VULKAN': cv.dnn.DNN_TARGET_VULKAN,  # not used
}

if CUDA_ENABLED:
  dnn_backends.update({'CUDA': cv.dnn.DNN_BACKEND_CUDA})
  dnn_targets.update({'CUDA': cv.dnn.DNN_TARGET_CUDA})
  dnn_targets.update({'CUDA_FP16': cv.dnn.DNN_TARGET_CUDA_FP16})


# -----------------------------------------------------------------------------
# CLI command opts
# -----------------------------------------------------------------------------
DEFAULT_DETECT_MODEL = 'coco'

# use CCW rotation where 90 means 90 CCW
ROTATE_VALS = {
  '0': None,
  '90': cv.ROTATE_90_COUNTERCLOCKWISE,
  '180': cv.ROTATE_180,
  '-90': cv.ROTATE_90_CLOCKWISE,
  '-180': cv.ROTATE_180,
  '-270': cv.ROTATE_90_COUNTERCLOCKWISE,
  '270': cv.ROTATE_90_CLOCKWISE,
}

# -----------------------------------------------------------------------------
# Haarcascades
# -----------------------------------------------------------------------------
DIR_CV2_DATA = join(Path(os.path.dirname(cv.__file__)).parent, 'data')
DEFAULT_HAARCASCADE = 'frontalface_default'

# -----------------------------------------------------------------------------
# S3 files
# -----------------------------------------------------------------------------

URL_VFRAME_S3 = 'https://download.vframe.io'
DIR_S3_VER = 'v2'
DIR_S3_MODELS = join(DIR_S3_VER, 'models')
S3_HTTP_MODELS_URL = join(URL_VFRAME_S3, DIR_S3_VER, 'models')


# -----------------------------------------------------------------------------
# Display settings
# -----------------------------------------------------------------------------

MAX_DISPLAY_SIZE = (1200, 1200)
CV_WINDOW_NAME = 'VFRAME'


# -----------------------------------------------------------------------------
# I/O settings
# -----------------------------------------------------------------------------

ZERO_PADDING = 6


# -----------------------------------------------------------------------------
# Unicode symbols for logger
# -----------------------------------------------------------------------------

UCODE_OK = u"\u2714"  # check ok
UCODE_NOK = u'\u2718'  # x no ok

LICENSE_HEADER = """#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

"""


# -----------------------------------------------------------------------------
# NVIDIA architectures for NVIDIA GPUs
# -----------------------------------------------------------------------------

GPU_ARCHS = {
  '7.0': ['v100'],
  '7.5': ['rtx 2080 ti', 'rtx 2080', 'rtx 2070', 'quadro rtx 8000', 'quadro rtx 6000', 'quadro rtx 5000', 'gtx 1650', 'tesla t4'],
  '7.2': ['xavier'],
  '6.1': ['gtx 1080 ti', 'gtx 1080', 'gtx 1060',' gtx 1050', 'gtx 1030', 'titan xp', 'tesla p40', 'tesla p4'],
  '6.0': ['GP100', 'tesla p100'],
  '5.3': ['jetson tx1', 'tegra x1', 'drive cx', 'drive px'],
  '5.2': ['jetson tx2', 'drive-px2', 'drive px'],
}


# -----------------------------------------------------------------------------
# S3 storage
# -----------------------------------------------------------------------------

try:
  S3_HTTP_BASE_URL = os.getenv("S3_HTTP_BASE_URL")
  S3_HTTP_MODELS_URL = os.getenv("S3_HTTP_MODELS_URL")
  S3_DIR_MODELS = os.getenv('S3_DIR_MODELS')
except Exception as e:
  log.error(f'S3 .env variables not set. Can not access models.')
  log.info(f'Edit {fp_env} and add your S3 access keys. Use .env-sample.')



# -----------------------------------------------------------------------------
# Synthetic
# -----------------------------------------------------------------------------

# output files
FN_METADATA = 'metadata.csv'  # filenamne
FN_ANNOTATIONS = 'annotations.csv'  # filenamne
DN_REAL = 'real'  # directory name
DN_MASK = 'mask'  # directory name
DN_COMP = 'comp'  # directory name
DN_BBOX = 'bbox'  # directory name
DN_IMAGES = 'images'  # directory name for images in concat output
OUTPUT_FILE_FORMAT = 'PNG'


# -----------------------------------------------------------------------------
# YOLO
# -----------------------------------------------------------------------------


FP_YOLO_MODELS = join(DIR_CONFIGS, 'yolo-models.yaml')
FP_DARKNET_BIN = join(DIR_PROJECT_ROOT, '3rdparty/darknet/darknet')

DN_IMAGES_LABELS = 'images_labels'
DN_BACKUP = 'backup'

FN_TRAIN_INIT = 'train_init.sh'
FN_TRAIN_RESUME = 'train_resume.sh'
FN_TRAIN_MULTI = 'train_multi.sh'
FN_TEST_INIT = 'test.sh'
FN_LOGFILE = 'training.log'
FN_META_DATA = 'meta.data'
FN_LABELS = 'labels.txt'
FN_VALID = 'valid.txt'
FN_TRAIN = 'train.txt'
LABEL_BACKGROUND = 'background'


# -----------------------------------------------------------------------------
# Yolo Training
# -----------------------------------------------------------------------------

# # load YOLO init model configs
# modelzoo_yaml = load_yaml(FP_YOLO_MODELS)

# # create dict with modelzoo name-keys and DNN values
# modelzoo = {k: dacite.from_dict(data=v, data_class=DNN) for k,v in modelzoo_yaml.items()}

# FP_YOLOV4_CFG = modelzoo.get('yolo4-init').fp_config
# FP_YOLOV4_WEIGHTS = modelzoo.get('yolo4-init').fp_model

# FP_YOLOV3_CFG = modelzoo.get('yolo3-init').fp_config
# FP_YOLOV3_WEIGHTS = modelzoo.get('yolo3-init').fp_model

# # Choose YOLO version based on .env variables
# USE_YOLO_VERSION = int(os.getenv('YOLO_VERSION', '4'))
# if USE_YOLO_VERSION == 3:
#   FP_YOLO_CFG = FP_YOLOV4_CFG
#   FP_YOLO_WEIGHTS = FP_YOLOV3_WEIGHTS
# elif USE_YOLO_VERSION == 4:
#   FP_YOLO_CFG = FP_YOLOV4_CFG
#   FP_YOLO_WEIGHTS = FP_YOLOV3_WEIGHTS
# else:
#   LOG.error(f'YOLO version {USE_YOLO_VERSION} is not a valid option. Defaulting to 4.')
#   FP_YOLO_CFG = FP_YOLOV4_CFG
#   FP_YOLO_WEIGHTS = FP_YOLOV3_WEIGHTS
