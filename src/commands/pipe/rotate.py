############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.settings.app_cfg import ROTATE_VALS, SKIP_FRAME_KEY
from vframe.models.types import FrameImage, FrameImageVar
from vframe.utils.click_utils import processor, show_help


@click.command('')
@click.option('-r', '--rotate', 'opt_rotate', 
  type=click.Choice(ROTATE_VALS.keys()), 
  default='0',
  help='Rotate image this many degrees in counter-clockwise direction before detection')
@processor
@click.pass_context
def cli(ctx, sink, opt_rotate):
  """Rotate frame"""
  
  import cv2 as cv
  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY

  frame_types = [FrameImage.DRAW, FrameImage.ORIGINAL]
  cv_rot_val = ROTATE_VALS[opt_rotate]

  while True:

    M = yield

    if not ctx.opts[SKIP_FRAME_KEY]:
      
      # resize
      for frame_type in frame_types:
        im = M.images[frame_type]
        im = cv.rotate(im, cv_rot_val)
        M.images[frame_type] = im

    # continue
    sink.send(M)