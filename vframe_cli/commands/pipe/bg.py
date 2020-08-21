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
@click.option('-c', '--color', 'opt_color', 
  type=(int, int, int), default=(0, 0, 0),
  help='Color in RGB int (eg 0 255 0)')
@click.option('-a', '--alpha', 'opt_alpha', default=0.5,
  help='Opacity of background image. Use 1.0 for solid fill.')
@processor
@click.pass_context
def cli(ctx, pipe, opt_color, opt_alpha):
  """Add or composite background"""
  
  import cv2 as cv

  from vframe.settings import app_cfg
  from vframe.models import types
  from vframe.utils import im_utils
  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG


  # ---------------------------------------------------------------------------
  # Example: process images as they move through pipe

  while True:

    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(types.FrameImage.DRAW)

    # create colorfill image
    h,w,c = im.shape
    im_fill = im_utils.create_blank_im(w,h)
    im_fill[:,:,:] = opt_color[::-1]  # rgb to bgr
    im = cv.addWeighted(im, 1 - opt_alpha, im_fill, opt_alpha, 1.0)

    pipe_item.set_image(types.FrameImage.DRAW, im)
    pipe.send(pipe_item)

