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
from vframe.utils.im_utils import IMAGE_TRANSFORMS

@click.command('')
@click.option('-t', '--type', 'opt_filter', 
  type=click.Tuple([
    click.Choice(IMAGE_TRANSFORMS.keys()), 
    click.FloatRange(0, 1, clamp=True)]),
  required=True,
  help='Tuple of (filter,  factor) (eg "blur 0.5")')
@click.option('--frame', 'opt_frame_type', default='draw',
  type=FrameImageVar, help=show_help(FrameImage))
@click.option('-a', '--all/--single', 'opt_all_frames', is_flag=True, default=True,
  help='Resize all frames')
@click.option('--shuffle', 'opt_shuffle', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_filter, opt_frame_type, opt_all_frames, opt_shuffle):
  """Apply image filter/transform"""

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME


  frame_types = [FrameImage.DRAW, FrameImage.ORIGINAL] if opt_all_frames else [opt_frame_type]
  if FrameImage.DRAW in frame_types:
    ctx.obj[USE_DRAW_FRAME] = True

  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    # apply
    im = M.images[opt_frame_type]
    im = IMAGE_TRANSFORMS.get(opt_filter[0])(im, opt_filter[1])
    M.images[opt_frame_type] = im

    # update
    sink.send(M)