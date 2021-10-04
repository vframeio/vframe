############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.models import types
from vframe.utils import click_utils
from vframe.utils.click_utils import processor
from vframe.models.types import FrameImage, FrameImageVar
from vframe.utils.click_utils import show_help


@click.command('')
@click.option('-k', '--kernel', 'opt_ksize', default=3, type=int,
  help="Kernel")
@click.option('-f', '--frame-name', 'opt_frame_type', default='original',
  type=FrameImageVar, help=show_help(FrameImage))
@click.option('-a', '--all', 'opt_all_frames', is_flag=True,
  help='Resize all frames')
@processor
@click.pass_context
def cli(ctx, sink, opt_ksize, opt_frame_type, opt_all_frames):
  """Blur image"""
  
  import cv2 as cv

  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY
  from vframe.settings.app_cfg import LOG
  from vframe.utils.im_utils import resize
  from vframe.utils.misc_utils import oddify

  opt_ksize = oddify(opt_ksize)

  frame_types = [FrameImage.DRAW, FrameImage.ORIGINAL] if opt_all_frames else [opt_frame_type]
  
  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME_KEY]:
      sink.send(M)
      continue

    # resize
    for frame_type in frame_types:
      im = M.images[frame_type]
      im = cv.blur(im, ksize=(opt_ksize, opt_ksize))
      # im = cv.GaussianBlur(im, (opt_ksize, opt_ksize), 0)
      # yuv = cv.cvtColor(im, cv.COLOR_BGR2YUV)
      # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
      # yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
      # # yuv[:, :, 0] = cv.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
      # im = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
      # update
      M.images[frame_type] = im
    # continue
    sink.send(M)
