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
@processor
@click.pass_context
def cli(ctx, sink, opt_color, opt_alpha):
  """Add or composite background"""
  
  import cv2 as cv

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME
  from vframe.models import types
  from vframe.models.types import FrameImage
  from vframe.utils import im_utils


  ctx.obj[USE_DRAW_FRAME] = True

  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    im = M.images[FrameImage.DRAW]

    # create colorfill image
    h,w,c = im.shape
    im_fill = im_utils.create_blank_im(w,h)
    im_fill[:,:,:] = opt_color[::-1]  # rgb to bgr
    im = cv.addWeighted(im, 1 - opt_alpha, im_fill, opt_alpha, 1.0)

    M.images[FrameImage.DRAW] = im
    sink.send(M)