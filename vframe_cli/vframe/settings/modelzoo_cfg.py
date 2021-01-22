############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from os.path import join

from dacite import from_dict

from vframe.models.dnn import DNN
from vframe.utils.file_utils import load_yaml
from vframe.settings import app_cfg

# -----------------------------------------------------------------------------
#
# Modelzoo Config
#
# -----------------------------------------------------------------------------

log = app_cfg.LOG

# get list of active modelzoo files
vframe_cfg = load_yaml(app_cfg.FP_VFRAME_YAML)

# iterate all modelzoo yamls
modelzoo_yaml = {}
for modelzoo_cfg in vframe_cfg.get('modelzoo'):
  yaml_data = load_yaml(join(app_cfg.DIR_PROJECT_ROOT, modelzoo_cfg['filepath']))
  modelzoo_yaml.update(yaml_data)

# create dict with modelzoo name-keys and DNN values
modelzoo = {k: from_dict(data=v, data_class=DNN) for k,v in modelzoo_yaml.items()}
