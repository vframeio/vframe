############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.utils.click_utils import processor
from vframe.utils.click_utils import show_help
from vframe.models.types import ModelZooClickVar, ModelZoo, FrameImage

rotate_opts = list(map(str, [0, -90, 90, -180, 180, -270, 270]))  #str only

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  required=True,
  type=ModelZooClickVar,
  help=show_help(ModelZoo))
@click.option('--gpu/--cpu', 'opt_gpu', is_flag=True, default=True,
  help='Use GPU or CPU for inference')
@click.option('-s', '--size', 'opt_dnn_size', default=(None, None), type=(int, int),
  help='DNN blob image size. Overrides config file')
@click.option('-t', '--threshold', 'opt_dnn_threshold', default=None, type=float,
  help='Detection threshold. Overrides config file')
@click.option('--name', '-n', 'opt_data_key', default=None,
  help='Name of data key')
@click.option('-r', '--rotate', 'opt_rotate', type=click.Choice(rotate_opts), 
  default='0',
  help='Rotate image this many degrees in counter-clockwise direction before detection')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@processor
@click.pass_context
def cli(ctx, pipe, opt_model_enum, opt_data_key, opt_gpu, 
  opt_dnn_threshold, opt_dnn_size, opt_rotate, opt_verbose):
  """Detect objects"""
  
  from os.path import join
  from pathlib import Path
  import traceback

  import dacite
  import cv2 as cv

  from vframe.settings import app_cfg
  from vframe.models.dnn import DNN
  from vframe.settings.modelzoo_cfg import modelzoo
  from vframe.image.dnn_factory import DNNFactory

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  
  model_name = opt_model_enum.name.lower()
  dnn_cfg = modelzoo.get(model_name)

  # override dnn_cfg vars with cli vars
  if opt_gpu:
    dnn_cfg.use_gpu()
  else:
    dnn_cfg.use_cpu()
  if all(opt_dnn_size):
    dnn_cfg.width = opt_dnn_size[0]
    dnn_cfg.height = opt_dnn_size[1]
  if opt_dnn_threshold is not None:
    dnn_cfg.threshold = opt_dnn_threshold
    
  # rotate
  opt_rotate = int(opt_rotate)
  np_rot_val =  opt_rotate // 90  # 90 deg rotations in counter-clockwise direction
  if opt_rotate == 90 or opt_rotate == -270:
    cv_rot_val = cv.ROTATE_90_COUNTERCLOCKWISE
  elif opt_rotate == 180 or opt_rotate == -180:
    cv_rot_val = cv.ROTATE_180
  elif opt_rotate == 270 or opt_rotate == -90:
    cv_rot_val = cv.ROTATE_90_CLOCKWISE
  else:
    cv_rot_val = None


  if not opt_data_key:
    opt_data_key = model_name

  # create dnn cvmodel
  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)

  # ---------------------------------------------------------------------------
  # process

  while True:

    # Get pipe data
    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(FrameImage.ORIGINAL)
    
    # rotate if optioned  
    if cv_rot_val is not None:
      im = cv.rotate(im, cv_rot_val)
    
    # Detect
    try:
      results = cvmodel.infer(im)

      # rotate if optioned
      if results and np_rot_val != 0:
        for detect_results in results.detections:
          detect_results.bbox = detect_results.bbox.rot90(np_rot_val)

    except Exception as e:
      results = {}
      log.error(f'Could not detect: {header.filepath}')
      tb = traceback.format_exc()
      log.error(tb)
      

    # debug
    if opt_verbose:
      log.debug(f'{cvmodel.dnn_cfg.name} detected: {len(results.detections)} objects')

    # Update data
    if results:
      pipe_data = {opt_data_key: results}
      header.add_data(pipe_data)
    
    # Continue processing
    pipe.send(pipe_item)