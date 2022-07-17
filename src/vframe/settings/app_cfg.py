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
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
import operator

import yaml
import cv2 as cv
from dacite import from_dict

from vframe.models.color import Color
from vframe.models.plugins import Plugins
from vframe.models.dnn import DNN


# -----------------------------------------------------------------------------
# Application information
# -----------------------------------------------------------------------------

VERSION = '0.2.0'


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG = logging.getLogger('VFRAME')


# -----------------------------------------------------------------------------
# Click processor vars
# -----------------------------------------------------------------------------

SKIP_FRAME = 'SKIP_FRAME'
SKIP_FILE = 'SKIP_FILE'
USE_PREHASH = 'USE_PREHASH'
READER = 'MEDIA_READER'
PAUSED = 'PAUSED'
USE_DRAW_FRAME = 'USE_DRAW_FRAME'
MEDIA_FILTERS = 'MEDIA_FILTERS'
SKIP_MEDIA_FILTERS = 'SKIP_MEDIA_FILTERS'
OBJECT_COLORS = 'OBJECT_COLORS'
FRAME_BUFFER_SIZE = 2048

# -----------------------------------------------------------------------------
# Caption accessors
# NB: @accessors can not include another name
# eg: use @filename:filename and @filepath:filepath
# eg: !use @filename:filename and @filename_parent:filename_parent
# -----------------------------------------------------------------------------

caption_accessors = {
  '@filename': 'filename',
  '@filestem': 'filestem',
  '@filepath': 'filepath',
  '@parentname': 'parentname',
  '@ext': 'ext',
  '@width': 'width',
  '@height': 'height',
  '@frames': 'frame_count',
  '@detections': 'n_detections'
}

compare_accessors = {
  '@width': 'width',
  '@height': 'height',
  '@frames': 'frame_count',
  '@date': 'date',
  '@year': 'year',
}

filename_accessors = {
  '@filename': 'filename',
  '@filestem': 'filestem',
  '@parentname': 'parentname',
}


# -----------------------------------------------------------------------------
# Filepaths
# -----------------------------------------------------------------------------

# Project directory
FILE = Path(__file__).resolve()
DIR_PROJECT_ROOT = FILE.parent.parent.parent.parent  # vframe root

# -----------------------------------------------------------------------------
# ENV config
# -----------------------------------------------------------------------------

# source .env vars
fp_env = Path(DIR_PROJECT_ROOT) / '.env'
load_dotenv(dotenv_path=fp_env, verbose=True)


# directories
DIR_SRC = join(DIR_PROJECT_ROOT, 'src')
DIR_3RDPARTY = join(DIR_PROJECT_ROOT, '3rdparty')
DIR_DATA_STORE = join(DIR_PROJECT_ROOT, 'data')
DIR_CONFIGS = join(DIR_DATA_STORE, 'configs')
DIR_MODELZOO = join(DIR_PROJECT_ROOT, 'modelzoo')
DIR_MODELS = join(DIR_MODELZOO, 'models')
DIR_PLUGINS = join(DIR_SRC, 'plugins')
DIR_FONTS = join(DIR_DATA_STORE, 'fonts')

# filepaths
FP_CLI = join(DIR_SRC, 'cli.py')

# display, fonts
FP_ROBOTO_300 = join(DIR_FONTS, 'roboto/Roboto_300.ttf')
FP_ROBOTO_400 = join(DIR_FONTS, 'roboto/Roboto_400.ttf')
FP_ROBOTO_500 = join(DIR_FONTS, 'roboto/Roboto_500.ttf')
FP_ROBOTO_700 = join(DIR_FONTS, 'roboto/Roboto_700.ttf')
FP_HELVETICA_NORMAL = join(DIR_FONTS, 'helvetica/Helvetica-Normal.ttf')
FP_HELVETICA_BOLD = join(DIR_FONTS, 'helvetica/Helvetica-Bold.ttf')

# model config filenames
FN_CSV_SUMMARY  = 'summary.csv'
FN_LABEL_COLORS = 'label_colors.json'
FN_CHECKSUM = 'sha256.txt'

# synthetic files
FN_METADATA = 'metadata.csv'  # filename
FN_ANNOTATIONS = 'annotations.csv'  # filename
FN_DETECTIONS = 'detections.json'  # filename
FN_ANNOTATIONS_SUMMARY = 'annotations_summary.csv'  # filename
FN_LABELMAP = 'labelmap.yaml'
DN_REAL = 'images'  # directory name
DN_MASK = 'mask'  # directory name
DN_MASK_INDEXED = 'masks_indexed'
DN_COMP = 'comp'  # directory name
DN_BBOX = 'bbox'  # directory name
DN_IMAGES = 'images'  # directory name for images in concat output

# CVAT
FN_ANNOTATIONS_CVAT = 'annotations.xml'

# VFRAME config file
FP_VFRAME_YAML = join(DIR_PROJECT_ROOT, os.getenv("FP_VFRAME_YAML", "config.yaml"))

# Cmake config files
FP_CMAKE_OPENCV = join(DIR_DATA_STORE, 'configs/cmake/opencv.yaml')



# -----------------------------------------------------------------------------
# # YAML config
# -----------------------------------------------------------------------------

# get list of active modelzoo files
with open(FP_VFRAME_YAML, 'r') as fp:
  vframe_cfg = yaml.load(fp, Loader=yaml.SafeLoader)


# -----------------------------------------------------------------------------
# ModelZoo
# -----------------------------------------------------------------------------

# HTTPS_MODELS_URL = 'https://download.vframe.io/v2/models/'
HTTPS_S3_ROOT = 'https://vframe.ams3.digitaloceanspaces.com/'
HTTPS_MODELS_URL = join(HTTPS_S3_ROOT, 'v2/models/')


modelzoo_yaml = {}

# iterate all modelzoo yamls
for m in vframe_cfg.get('modelzoo'):
  fp_cfg = m['filepath']
  fp_cfg = join(DIR_PROJECT_ROOT, fp_cfg) if not fp_cfg.startswith('/') else fp_cfg
  if Path(fp_cfg).is_file():
    with open(fp_cfg,'r') as fp:
      d = yaml.load(fp, Loader=yaml.SafeLoader).get('models', {})
      if not d:
        continue
      for k,v in d.items():
        v.update({'dp_models': DIR_MODELS})
      modelzoo_yaml.update(d)
  else:
    LOG.warn(f'{fp_cfg} does not exist. Skipping.')

# create dict with modelzoo name-keys and DNN values
modelzoo = {k: from_dict(data=v, data_class=DNN) for k,v in modelzoo_yaml.items()}

# Default models
DEFAULT_DETECT_MODEL = {'COCO': 'coco'}
DEFAULT_CLASSIFICATION_MODEL = 'places365_imagenet_vgg16'


# -----------------------------------------------------------------------------
# Plugins
# -----------------------------------------------------------------------------

with open(FP_VFRAME_YAML, 'r') as fp:
  plugins = yaml.load(fp, Loader=yaml.SafeLoader)
  plugins = from_dict(data_class=Plugins, data=plugins)


# -----------------------------------------------------------------------------
# Input types
# -----------------------------------------------------------------------------

VALID_PIPE_IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'heic']
VALID_PIPE_VIDEO_EXTS = ['mp4', 'avi', 'mov', 'mkv']  # webm, mkv frame count error
VALID_PIPE_MEDIA_EXTS =  VALID_PIPE_IMAGE_EXTS + VALID_PIPE_VIDEO_EXTS
VALID_PIPE_DATA_EXTS = ['json']
VALID_PIPE_EXTS = VALID_PIPE_MEDIA_EXTS + VALID_PIPE_DATA_EXTS


# -----------------------------------------------------------------------------
# Drawing defaults
# -----------------------------------------------------------------------------

DEFAULT_TEXT_SIZE = 14
DEFAULT_STROKE_WEIGHT = 2
DEFAULT_SIZE_LABEL = 14
DEFAULT_PADDING_PER = 0.25
DEFAULT_FONT_NAME = 'roboto'
DEFAULT_FONT_FP = FP_ROBOTO_500


# -----------------------------------------------------------------------------
# Terminal display
# -----------------------------------------------------------------------------

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
# OpenCV
# -----------------------------------------------------------------------------

try:
  SRES_ENABLED = 'superres' in cv.getBuildInformation()
except Exception as e:
  SRES_ENABLED = False

try:
  CUDA_ENABLED = 'CUDA' in cv.getBuildInformation()
except Exception as e:
  CUDA_ENABLED = False

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

backends = {
  'DEFAULT': cv.dnn.DNN_BACKEND_DEFAULT,
  'HALIDE': cv.dnn.DNN_BACKEND_HALIDE,  # not used
  'INFERENCE_ENGINE': cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,  # not used
  'OPENCV': cv.dnn.DNN_BACKEND_OPENCV,
  'VKCOM': cv.dnn.DNN_BACKEND_VKCOM,  # not used
}
targets = {
  'DEFAULT': cv.dnn.DNN_TARGET_CPU,  # alias to CPU
  'CPU': cv.dnn.DNN_TARGET_CPU,
  'FPGA': cv.dnn.DNN_TARGET_FPGA,  # not used
  'MYRIAD': cv.dnn.DNN_TARGET_MYRIAD,  # not used
  'OPENCL': cv.dnn.DNN_TARGET_OPENCL,  # not used
  'OPENCL_FP16': cv.dnn.DNN_TARGET_OPENCL_FP16,  # not used
  'VULKAN': cv.dnn.DNN_TARGET_VULKAN,  # not used
}
if CUDA_ENABLED:
  targets.update({'CUDA': cv.dnn.DNN_TARGET_CUDA})
  targets.update({'CUDA_FP16': cv.dnn.DNN_TARGET_CUDA_FP16})
  backends.update({'CUDA': cv.dnn.DNN_BACKEND_CUDA})

GPU_ARCHS = {
  '8.6': ['rtx 3080 ti', 'rtx 3090', 'rtx 3080'],
  '7.0': ['v100'],
  '7.5': ['rtx 2080 ti', 'rtx 2080', 'rtx 2070', 'quadro rtx 8000', 'quadro rtx 6000', 'quadro rtx 5000', 'gtx 1650', 'tesla t4'],
  '7.2': ['xavier'],
  '6.1': ['gtx 1080 ti', 'gtx 1080', 'gtx 1060',' gtx 1050', 'gtx 1030', 'titan xp', 'tesla p40', 'tesla p4'],
  '6.0': ['GP100', 'tesla p100'],
  '5.3': ['jetson tx1', 'tegra x1', 'drive cx', 'drive px'],
  '5.2': ['jetson tx2', 'drive-px2', 'drive px'],
}


# -----------------------------------------------------------------------------
# Haarcascades
# -----------------------------------------------------------------------------

DIR_CV2_DATA = join(DIR_DATA_STORE, 'haarcascades')
DEFAULT_HAARCASCADE = 'frontalface_default'


# -----------------------------------------------------------------------------
# Display settings
# -----------------------------------------------------------------------------

MAX_DISPLAY_SIZE = (1200, 1200)
CV_WINDOW_NAME = f'VFRAME ({VERSION})'


# -----------------------------------------------------------------------------
# YAML anchors and aliases config
# -----------------------------------------------------------------------------

INCLUDE_TOKEN = '#include:'


# -----------------------------------------------------------------------------
# I/O settings
# -----------------------------------------------------------------------------

ZERO_PADDING = 6


# -----------------------------------------------------------------------------
# Unicode symbols for logger
# -----------------------------------------------------------------------------

UCODE_OK = u"\u2714"    # check: ok
UCODE_NOK = u'\u2718'   # x: not ok
UCODE_INCREASE = '\U0001F4C8'
UCODE_DECREASE = '\U0001F4C9'

UCODE_RED_CIRCLE = "\U0001F534"
UCODE_ORANGE_CIRCLE = "\U0001F7E0"
UCODE_YELLOW_CIRCLE = "\U0001F7E1"
UCODE_GREEN_CIRCLE = "\U0001F7E2"
UCODE_BLUE_CIRCLE = "\U0001F535"
UCODE_PURPLE_CIRCLE = "\U0001F7E3"
UCODE_BROWN_CIRCLE = "\U0001F7E4"
UCODE_BLACK_CIRCLE = "\U000026AB"
UCODE_WHITE_CIRCLE = "\U000026AA"



LICENSE_HEADER = """#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) Adam Harvey and VFRAME
# https://vframe.io
# https://github.com/vframeio
#
#############################################################################

"""


# -----------------------------------------------------------------------------
# CVAT
# -----------------------------------------------------------------------------

DIR_CVAT = join(DIR_DATA_STORE, 'cvat')
DIR_CVAT_TASKS = join(DIR_CVAT, 'tasks')

FP_CVAT_STATUS = join(DIR_CVAT, 'tasks.csv')

CVAT_CMD_CLI = 'python3 utils/cli/cli.py'
CVAT_IMG_NAME = 'cvat'
CVAT_CMD_DOCKER_EXEC = f'docker exec -it {CVAT_IMG_NAME} bash -ic'
CVAT_FORMAT_VIDEOS = 'CVAT for video 1.1'
CVAT_DIR_TMP = '/tmp/'
CVAT_FP_TMP_ZIP = '/tmp/annotations.zip'
CVAT_EXPORT_FORMATS = {
  'cvat_video': 'CVAT for video 1.1',
  'cvat_images': 'CVAT for images 1.1',
}
CVAT_EXPORT_DIR_NAMES = {
  'cvat_video': 'video',
  'cvat_images': 'images',
}


# -----------------------------------------------------------------------------
# Blender config
# - set in config.yaml
# -----------------------------------------------------------------------------

# TODO change to dict to check cfg first
blender_cfg = vframe_cfg.get('blender', {})
if blender_cfg:
  FP_BLENDER = blender_cfg.get('blender')
  FP_BLENDER_PY = blender_cfg.get('python')
  FP_BLENDER_PIP = blender_cfg.get('pip')
  FP_BLENDER_ENSUREPIP = blender_cfg.get('ensurepip')

# static
FN_BLENDER_LOG = 'blender.log'

# standardized object names
anon_obj_cfg = {
  'collection_name': 'background',
  'label': 'background', 
  'label_index': 0, 
  'brightness': 0, 
  'description': 'Background',
  'trainable': False,
}
anon_obj_mappings = { 'mappings': [anon_obj_cfg] }