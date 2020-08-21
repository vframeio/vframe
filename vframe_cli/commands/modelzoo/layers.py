############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click
from dacite import from_dict

from vframe.models import types
from vframe.models.dnn import DNN
from vframe.models.types import dict_to_enum
from vframe.utils import click_utils, file_utils


layer_type = ['connected', 'unconnected']

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', required=True,
  type=types.ModelZooClickVar,
  help=click_utils.show_help(types.ModelZoo))
@click.option('--gpu', 'opt_gpu', is_flag=True,
  help='GPU index, overrides config file')
@click.option('-t', '--type', 'opt_layer_type', type=click.Choice(layer_type), 
  required=True,
  help='Layer types')
@click.pass_context
def cli(ctx, opt_model_enum, opt_layer_type, opt_gpu):
  """List DNN layers"""

  # ------------------------------------------------
  # imports

  from vframe.image.dnn_factory import DNNFactory
  from vframe.settings import app_cfg, modelzoo_cfg
  from vframe.utils import im_utils

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  model_name = opt_model_enum.name.lower()
  log.debug(f'Get {opt_layer_type} layers for: {model_name}')
  dnn_cfg = modelzoo_cfg.modelzoo.get(model_name)

  
  # Omit models that are not opencv DNN compatible
  if '.params' in dnn_cfg.model:
    log.error('MXNet .params models not supported')
    return

  # override cfg with cli vars
  if opt_gpu:
    dnn_cfg.use_gpu()

  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)
  
  if opt_layer_type == 'unconnected':
    # print unconnected layers
    ucl = cvmodel.net.getUnconnectedOutLayersNames()
    log.info(ucl)
  elif opt_layer_type == 'connected':
    # print all layers
    im = im_utils.create_blank_im(320, 240)
    cvmodel._pre_process(im)

    for layer in cvmodel.net.getLayerNames():
      try:
        feat_vec = cvmodel.net.forward(layer)
        log.info(f'{layer}: {feat_vec.shape}')
      except Exception as e:
        log.info(f'{layer}: no features')