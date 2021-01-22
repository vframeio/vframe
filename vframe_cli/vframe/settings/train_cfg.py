############################################################################# 
#
# VFRAME Training
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import os
from os.path import join
from pathlib import Path
import collections
from dotenv import load_dotenv
import logging

import dacite

from vframe.settings import app_cfg
from vframe.models.plugins import Plugins
from vframe.utils.file_utils import load_yaml
from vframe.models.dnn import DNN


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG = logging.getLogger('vframe')


# -----------------------------------------------------------------------------
# File I/O
# -----------------------------------------------------------------------------

# Project directory
SELF_CWD = os.path.dirname(os.path.realpath(__file__))
DIR_PLUGIN_ROOT = str(Path(SELF_CWD).parent.parent)

# source .env vars
fp_env = Path(DIR_PLUGIN_ROOT) / '.env'
load_dotenv(dotenv_path=fp_env, verbose=True)

DIR_CONFIGS = join(DIR_PLUGIN_ROOT, 'data/configs')
FP_YOLO_MODELS = join(DIR_CONFIGS, 'yolo-models.yaml')
FP_DARKNET_BIN = join(DIR_PLUGIN_ROOT, '3rdparty/darknet/darknet')

DN_IMAGES_LABELS = 'images_labels'
DN_BACKUP = 'backup'

FN_TRAIN_INIT = 'train_init.sh'
FN_TRAIN_RESUME= 'train_resume.sh'
FN_TEST_INIT = 'test.sh'
FN_LOGFILE = 'training.log'
FN_META_DATA = 'meta.data'
FN_CLASSES = 'classes.txt'
FN_VALID = 'valid.txt'
FN_TRAIN = 'train.txt'

LABEL_BACKGROUND = 'background'

# standardized Blender output files
FN_METADATA = 'metadata.csv'  # filename
FN_ANNOTATIONS = 'annotations.csv'  # annotations
DN_REAL = 'real'  # directory name
DN_MASK = 'mask'  # directory name
DN_COMP = 'comp'  # directory name

# -----------------------------------------------------------------------------
# Yolo Training
# -----------------------------------------------------------------------------

# load YOLO init model configs
modelzoo_yaml = load_yaml(FP_YOLO_MODELS)

# create dict with modelzoo name-keys and DNN values
modelzoo = {k: dacite.from_dict(data=v, data_class=DNN) for k,v in modelzoo_yaml.items()}

FP_YOLOV4_CFG = modelzoo.get('yolo4-init').fp_config
FP_YOLOV4_WEIGHTS = modelzoo.get('yolo4-init').fp_model

FP_YOLOV3_CFG = modelzoo.get('yolo3-init').fp_config
FP_YOLOV3_WEIGHTS = modelzoo.get('yolo3-init').fp_model

# Choose YOLO version based on .env variables
USE_YOLO_VERSION = int(os.getenv('YOLO_VERSION', '4'))
if USE_YOLO_VERSION == 3:
  FP_YOLO_CFG = FP_YOLOV4_CFG
  FP_YOLO_WEIGHTS = FP_YOLOV3_WEIGHTS
elif USE_YOLO_VERSION == 4:
  FP_YOLO_CFG = FP_YOLOV4_CFG
  FP_YOLO_WEIGHTS = FP_YOLOV3_WEIGHTS
else:
  LOG.error(f'YOLO version {USE_YOLO_VERSION} is not a valid option. Defaulting to 4.')
  FP_YOLO_CFG = FP_YOLOV4_CFG
  FP_YOLO_WEIGHTS = FP_YOLOV3_WEIGHTS