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
@processor
@click.pass_context
def cli(ctx, sink):
  """Converts draw frame to transparent"""

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME
  from vframe.models.types import FrameImage
  from vframe.utils.im_utils import create_blank_im


  ctx.obj[USE_DRAW_FRAME] = True

  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    im = M.images[FrameImage.DRAW]
    h,w,c = im.shape

    # set draw frame to blank transparent background
    M.images[FrameImage.DRAW] = create_blank_im(w,h,c=4)

    sink.send(M)