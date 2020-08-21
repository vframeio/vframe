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
@click.option( '-n', '--name', 'opt_data_key', required=True,
  help='Data key for ROIs')
@click.option('--square', 'opt_square', is_flag=True,
  help='Square bbox dimensions')
@click.option('-e', '--expand', 'opt_expand', type=float,
  help='Expand percent')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_key, opt_square, opt_expand):
  """Modify BBoxes"""
  
  from vframe.settings import app_cfg
  from vframe.models import types

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  log.debug('Modify BBoxes')


  # ---------------------------------------------------------------------------
  # process

  while True:

    pipe_item = yield

    header = ctx.obj['header']
    item_data = header.get_data(opt_data_key)
    
    im = pipe_item.get_image(types.FrameImage.DRAW)
    dim = im.shape[:2][::-1]

    if item_data:
      for obj_idx, detection in enumerate(item_data.detections):
        bbox_norm = detection.bbox
        if opt_square:
          bbox_norm = bbox_norm.to_bbox_dim(dim).to_square().to_bbox_norm()
        if opt_expand:
          bbox_norm = bbox_norm.expand(opt_expand)
        detection.bbox = bbox_norm

    pipe.send(pipe_item)