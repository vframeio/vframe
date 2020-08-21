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

import dacite

from vframe.settings import app_cfg
from vframe.models.plugins import Plugins
from vframe.utils.file_utils import load_yaml

# -----------------------------------------------------------------------------
#
# Commands
#
# -----------------------------------------------------------------------------

# create click group commands
plugins = load_yaml(app_cfg.FP_VFRAME_CONFIG, data_class=Plugins)
