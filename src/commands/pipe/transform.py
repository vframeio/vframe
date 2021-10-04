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
    click.FloatRange(0, 1, clamp=True),
    ],
    ),
  required=True,
  help='Tuple of (filter,  factor) (eg "blur 0.5")')
@click.option('--frame', 'opt_frame_type', default='draw',
  type=FrameImageVar, help=show_help(FrameImage))
@click.option('--shuffle', 'opt_shuffle', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_filter, opt_frame_type, opt_shuffle):
  """Apply image filter/transform"""

  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY
  

  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME_KEY]:
      sink.send(M)
      continue

    # apply
    im = M.images[opt_frame_type]
    im = IMAGE_TRANSFORMS.get(filter_name)(im, fac)
    M.images[opt_frame_type] = im

    # update
    sink.send(M)