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
  default=0.5,
  show_default=True,
  help='Strength to apply redaction filter')
@click.option('--iters', 'opt_iters', 
  default=2,
  show_default=True,
  help='Blur iterations')
@click.option('--expand', 'opt_expand', 
  default=0.25,
  show_default=True,
  help='Percentage to expand')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_keys, opt_factor, opt_iters, opt_expand):
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
    dim = im.shape[:2][::-1]
    
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
          bbox = detection.bbox.expand_per(opt_expand).redim(dim)

          # TODO: handle segmentation mask
          for i in range(opt_iters):
            im = im_utils.blur_roi(im, bbox)


      # resume pipe stream    
      pipe_item.set_image(types.FrameImage.DRAW, im)
      pipe.send(pipe_item)