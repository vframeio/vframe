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

@click.command('')
@click.option( '-n', '--name', 'opt_data_keys', 
  multiple=True,
  help='Data key for ROIs')
@click.option('--fac', 'opt_factor', 
  default=1.0,
  show_default=True,
  help='Strength to apply redaction filter')
@click.option('--expand', 'opt_expand', 
  default=0.25,
  show_default=True,
  help='Percentage to expand')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_keys, opt_factor, opt_expand):
  """Blurs BBoxes"""
  
  from vframe.settings import app_cfg
  from vframe.models import types
  from vframe.utils import im_utils
    
  # ---------------------------------------------------------------------------
  # TODO
  # - add oval shape blurring

  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG


  # ---------------------------------------------------------------------------
  # Example: process images as they move through pipe

  while True:

    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(types.FrameImage.DRAW)

    # get data keys
    if not opt_data_keys:
      data_keys = header.get_data_keys()
    else:
      data_keys = opt_data_keys

    # iterate data keys
    for data_key in data_keys:
      
      if data_key not in header.get_data_keys():
        log.warn(f'data_key: {data_key} not found')
      
      # get data
      item_data = header.get_data(data_key)

      # blur data
      if item_data:
        for obj_idx, detection in enumerate(item_data.detections):
          bbox_norm = detection.bbox.expand_per(opt_expand)

          # TODO: handle segmentation mask
          im = im_utils.blur_roi(im, bbox_norm)


      # resume pipe stream    
      pipe_item.set_image(types.FrameImage.DRAW, im)
      pipe.send(pipe_item)