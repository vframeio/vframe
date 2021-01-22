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

redact_types = ['pixellate', 'blur', 'softblur']

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
  default=0.0,
  show_default=True,
  help='Percentage to expand')
@click.option('-t', '--type', 'opt_redact_type', type=click.Choice(redact_types),
  show_default=True,
  default='softblur',
  help='Redact type')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_keys, opt_factor, opt_iters, opt_expand, opt_redact_type):
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
    data_keys = opt_data_keys if opt_data_keys else header.get_data_keys()

    # iterate data keys
    bboxes = []

    for data_key in data_keys:
      
      if data_key not in header.get_data_keys():
        log.warn(f'data_key: {data_key} not found')
      
      # get data
      item_data = header.get_data(data_key)

      # blur data
      if item_data:
        for obj_idx, detection in enumerate(item_data.detections):
          bbox = detection.bbox.expand_per(opt_expand).redim(dim)
          bboxes.append(bbox)

    # redact method
    if opt_redact_type == 'pixellate':
      im = im_utils.pixellate(im, bboxes)
    elif opt_redact_type == 'blur':
      im = im_utils.blur(im, bboxes)
    elif opt_redact_type == 'softblur':
      im = im_utils.blur_bbox_soft(im, bboxes, iters=2, expand_per=-0.15, 
        mask_k_fac=0.25, im_k_fac=0.995, multiscale=True)
    

    # resume pipe stream    
    pipe_item.set_image(types.FrameImage.DRAW, im)
    pipe.send(pipe_item)