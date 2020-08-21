############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

import dacite

from vframe.models import types
from vframe.utils import click_utils

actions = ['verify', 'props', 'features']

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', required=True,
  type=types.ModelZooClickVar,
  help=click_utils.show_help(types.ModelZoo))
@click.option('--gpu', 'opt_gpu', is_flag=True,
  help='GPU index, overrides config file')
@click.pass_context
def cli(ctx, opt_model_enum, opt_gpu):
  """Test ModelZoo models"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path

  import numpy as np
  import cv2 as cv

  from vframe.models.dnn import DNN
  from vframe.models.color import Color
  from vframe.settings import app_cfg, modelzoo_cfg
  from vframe.image.dnn_factory import DNNFactory
  from vframe.utils.im_utils import create_random_im


  # ------------------------------------------------
  # start

  log = app_cfg.LOG
  errors = []

  model_name = opt_model_enum.name.lower()
  log.debug(f'Basic test: {model_name}')
  dnn_cfg = modelzoo_cfg.modelzoo.get(model_name)

  # override cfg with cli vars
  if opt_gpu:
    log.info('Using GPU')
    dnn_cfg.use_gpu()

  # creates cv model
  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)

  # create random image
  im = create_random_im(640, 480)

  if dnn_cfg.features is not None:
    feat_vec = cvmodel.features(im)
    if len(feat_vec) != int(dnn_cfg.dimensions):
      errors.append(f'Feature dimensions mismatch: {len(feat_vec)} (config) != {dnn_cfg.dimensions} (computed)')
    else:
      log.info(f'Feature vector length: {len(feat_vec)}')
  
  log.info('Running inference...')
  results = cvmodel.infer(im)
  log.info(f'Ran inference ok for: {results.task_type}')

  if not errors:
    log.info(f'{app_cfg.UCODE_OK}: seems OK.')
  else:
    log.warn(f'{app_cfg.UCODE_NOK} There were {len(errors)}. Try to fix and rerun.')
    for e in errors:
      log.error(e)

        