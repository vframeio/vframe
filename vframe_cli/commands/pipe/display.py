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
from vframe.models import types
from vframe.utils.click_utils import processor

@click.command('')
@click.option('--delay', 'opt_delay', default=150,
  type=click.IntRange(1, 1000), show_default=True,
  help='Delay between images in milliseconds')
@click.option('--pause/--no-pause', 'opt_pause', is_flag=True, 
  default=True,
  help="Pause between frames")
@click.option('--autoplay', 'opt_autoplay', is_flag=True,
  help='Autoplay video')
@click.option('--frame', 'opt_frame_type', default='draw',
  type=types.FrameImageVar,
  help=click_utils.show_help(types.FrameImage))
@processor
@click.pass_context
def cli(ctx, pipe, opt_delay, opt_pause, opt_autoplay, opt_frame_type):
  """Display images to screen"""
  
  from vframe.settings import app_cfg
  from vframe.utils.display_utils import DisplayUtils

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  display_utils = DisplayUtils()
  if opt_autoplay:
    opt_pause = False
    try:
      opt_delay =  int(header.mspf)  # milliseconds per frame
    except Exception as e:
      pass

  # ---------------------------------------------------------------------------
  # process 

  while True:

    pipe_item = yield
    header = ctx.obj['header']

    im = pipe_item.get_image(opt_frame_type)
    display_utils.show_ctx(ctx, im, pause=opt_pause, delay=opt_delay)

    pipe.send(pipe_item)