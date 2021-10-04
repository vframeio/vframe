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
from vframe.models.types import FrameImage, FrameImageVar
from vframe.utils.click_utils import show_help

@click.command('')
@click.option('-c', '--color', 'opt_color', 
  type=(int, int, int), default=(0, 0, 0),
  help='Color in RGB int (eg 0 255 0)')
@click.option('-a', '--alpha', 'opt_alpha', default=0.5,
  help='Opacity of background image. Use 1.0 for solid fill.')
@click.option('-n', '--frame-name', 'opt_frame_type', default='original',
  type=FrameImageVar, help=show_help(FrameImage))
@processor
@click.pass_context
def cli(ctx, sink, opt_color, opt_alpha, opt_frame_type):
  """Add or composite background"""
  
  import cv2 as cv

  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY
  from vframe.models import types
  from vframe.utils import im_utils

  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME_KEY]:
      sink.send(M)
      continue

    im = M.images[opt_frame_type]

    # create colorfill image
    h,w,c = im.shape
    im_fill = im_utils.create_blank_im(w,h)
    im_fill[:,:,:] = opt_color[::-1]  # rgb to bgr
    im = cv.addWeighted(im, 1 - opt_alpha, im_fill, opt_alpha, 1.0)

    M.images[opt_frame_type] = im
    sink.send(M)
