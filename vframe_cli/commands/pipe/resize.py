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

@click.command('')
@click.option('-w', '--width', 'opt_width', default=None, type=int,
  help="Draw image width")
@click.option('-h', '--height', 'opt_height', default=None, type=int,
  help="Draw image height")
@click.option('-f', '--frame', 'opt_frame_type', default='draw',
  type=types.FrameImageVar,
  help=click_utils.show_help(types.FrameImage))
@click.option('--interp', 'opt_interp', default='cubic',
  type=types.InterpolationVar,
  help=click_utils.show_help(types.Interpolation))
@processor
@click.pass_context
def cli(ctx, pipe, opt_frame_type, opt_width, opt_height, opt_interp):
  """Resize input images"""
  

  from vframe.settings import app_cfg
  from vframe.utils import im_utils
  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG


  # ---------------------------------------------------------------------------
  # process

  while True:

    pipe_item = yield
    im = pipe_item.get_image(opt_frame_type)
    header = ctx.obj.get('header')

    if opt_width or opt_height:
      im = im_utils.resize(im, width=opt_width, height=opt_height, interp=opt_interp.value)

    pipe_item.set_image(opt_frame_type, im)
    pipe.send(pipe_item)