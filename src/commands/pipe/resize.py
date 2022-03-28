############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils import click_utils
from vframe.utils.click_utils import processor
from vframe.models.types import FrameImage, FrameImageVar, Interpolation, InterpolationVar
from vframe.utils.click_utils import show_help


@click.command('')
@click.option('-w', '--width', 'opt_width', default=None, type=int,
  help="Draw image width")
@click.option('-h', '--height', 'opt_height', default=None, type=int,
  help="Draw image height")
@click.option('-f', '--frame', 'opt_frame_type', default='original',
  type=FrameImageVar, help=show_help(FrameImage))
@click.option('-a', '--all/--single', 'opt_all_frames', is_flag=True, default=True,
  help='Resize all frames')
@click.option('--interp', 'opt_interp', default='linear',
  type=InterpolationVar, help=click_utils.show_help(Interpolation))
@processor
@click.pass_context
def cli(ctx, sink, opt_width, opt_height, opt_frame_type, opt_all_frames, opt_interp):
  """Resize image"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME
  from vframe.utils.im_utils import resize

  if not opt_width or opt_height:
    raise click.UsageError('-w/--width or -h/--height required')

  frame_types = [FrameImage.DRAW, FrameImage.ORIGINAL] if opt_all_frames else [opt_frame_type]
  if FrameImage.DRAW in frame_types:
    ctx.obj[USE_DRAW_FRAME] = True
  
  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue
  
    # resize
    for frame_type in frame_types:
      im = M.images[frame_type]
      im = resize(im, width=opt_width, height=opt_height, interp=opt_interp.value)
      # update
      M.images[frame_type] = im
  
    # continue
    sink.send(M)

"""
Some of the possible interpolation in OpenCV are:

    INTER_NEAREST – a nearest-neighbor interpolation
    INTER_LINEAR – a bilinear interpolation (used by default)
    INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
    INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood
"""