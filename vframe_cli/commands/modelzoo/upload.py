############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.models import types
from vframe.utils import click_utils
from vframe.models.types import ModelZoo, ModelZooClickVar
from vframe.utils.click_utils import show_help



@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  type=ModelZooClickVar,
  multiple=True,
  help=show_help(ModelZoo))
@click.option('--all', 'opt_all', is_flag=True,
  help='Sync all models')
@click.pass_context
def cli(ctx, opt_model_enum, opt_all):
  """Upload models to S3 storage"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path

  from tqdm import tqdm
  from dacite import from_dict

  from vframe.settings import app_cfg, modelzoo_cfg
  from vframe.utils import file_utils
  from vframe.models.dnn import DNN
  from vframe.utils import s3_utils


  # ------------------------------------------------
  # start

  log = app_cfg.LOG
  modelzoo = modelzoo_cfg.modelzoo

  # error checks
  if opt_all:
    model_list = list(modelzoo.keys())
  elif opt_model_enum:
    model_list = [x.name.lower() for x in opt_model_enum]
  else:
    log.error('Model required "-m/--model"')
    log.info(list(modelzoo.keys()))
    return


  # init s3 api
  s3 = s3_utils.RemoteStorageS3()

  for model_name in tqdm(model_list):
    
    dnn_cfg = modelzoo.get(model_name)

    log.info(f'Uploading: {model_name}')
    
    # sync dir if model exists
    fpp_model = Path(dnn_cfg.fp_model)

    if fpp_model.is_file():
      dir_local_model = str(fpp_model.parent)
      
      if 'https://download.vframe.io/' in dnn_cfg.remote:
        dir_remote = dnn_cfg.remote.replace('https://download.vframe.io/', '')
        log.info(f'Sync: {dir_local_model} --> {dir_remote}')
        s3.sync_dir(dir_local_model, dir_remote)
      else:
        log.info('Model not hosted on download.vframe.io. Skipping.')
    
    else:
      log.warn(f'No file exists locally: {fpp_model}')
