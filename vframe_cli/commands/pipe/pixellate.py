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
@click.option('--cell-size', 'opt_cell_size', default=(1,1),
  help='Override factor and use pixel-based cell size')
@click.option('--expand', 'opt_expand', 
  default=0.30,
  show_default=True,
  help='Percentage to expand')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_keys, opt_expand, opt_cell_size):
  """Pixellates BBoxes"""
  
  from vframe.settings import app_cfg
  from vframe.models import types
  from vframe.utils import im_utils
  
  # ---------------------------------------------------------------------------
  # TODO
  # - sadd oval shape blurring

  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG


  # ---------------------------------------------------------------------------
  # Example: process images as they move through pipe

  while True:

    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(types.FrameImage.DRAW)

    if not opt_data_keys:
      data_keys = header.get_data_keys()
      log.debug(f'No key provided. Blurring all keys {data_keys}')
    else:
      data_keys = opt_data_keys

    # iterate data keys
    for data_key in data_keys:
      
      if data_key not in header.get_data_keys():
        log.warn(f'data_key: {data_key} not found')
        
      item_data = header.get_data(data_key)

      if item_data:

        # blur bbox
        for obj_idx, detection in enumerate(item_data.detections):
          bbox_norm = detection.bbox.expand_per(opt_expand)
          # TODO: handle segmentation mask
          im = im_utils.pixellate_roi(im, bbox_norm, cell_size=opt_cell_size)
          

    # resume pipe stream    
    pipe_item.set_image(types.FrameImage.DRAW, im)
    pipe.send(pipe_item)