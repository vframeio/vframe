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
from vframe.settings import app_cfg

@click.command('')
@click.option('-t', '--threshold', 'opt_threshold', default=None, type=float,
  help='Detection threshold. Overrides config file')
@click.option('--name', '-n', 'opt_data_key', default=None,
  help='Name of data key')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_key, opt_threshold):
  """Interpolate frame gaps in detection BBoxes"""

  """
  - create a running buffer of previous N frame-bbox positions
  - for each bbox in each frame buffer, check if there is a gap
  - define gap as frame with 
  """
  
  from os.path import join
  from pathlib import Path
  import traceback

  import cv2 as cv
  import imagehash

  from vframe.models.geometry import BBox, Point

  
  # ---------------------------------------------------------------------------
  # initialize
  
  log = app_cfg.LOG

  # ---------------------------------------------------------------------------
  # process

  while True:

    # get pipe data
    pipe_item = yield
    header = ctx.obj['header']
    frame_dim = header.dim
    #im = pipe_item.get_image(FrameImage.ORIGINAL)
      
    if not opt_data_keys:
      data_keys = header.get_data_keys()
    else:
      data_keys = opt_data_keys

    for data_key in data_keys:
      
      if header.data_key_exists(data_key):
        item_data =  header.get_data(data_key)

        if item_data.detections:
          for face_idx, detect_result in enumerate(item_data.detections):
            bboxes.append(detect_result.bbox.xywh)
            confidences.append(float(detect_result.confidence))
            labels.append(detect_result.label)
            detect_results.append(detect_result)

    # remove old data keys if optioned
    if opt_remove_old:
      for data_key in data_keys:
        if data_key != opt_name:
          header.remove_data(data_key)

    # add/update merged bboxes
    detect_results = DetectResults(detect_results_nms)
    pipe_data = {opt_name: detect_results}
    header.set_data(pipe_data)
    
    
    # continue processing
    pipe.send(pipe_item)