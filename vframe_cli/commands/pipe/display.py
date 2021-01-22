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
@click.option('--auto', 'opt_autoplay', is_flag=True,
  help='Autoplay video')
@click.option('--frame', 'opt_frame_type', default='draw',
  type=types.FrameImageVar,
  help=click_utils.show_help(types.FrameImage))
@processor
@click.pass_context
def cli(ctx, pipe, opt_delay, opt_pause, opt_autoplay, opt_frame_type):
  """Display images to screen"""
  
  import time

  from vframe.settings import app_cfg
  from vframe.utils.display_utils import DisplayUtils

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  display_utils = DisplayUtils()  # TODO change to class method
  if opt_autoplay:
    opt_pause = False
    frame_delay = 0
  else:
    frame_delay = opt_delay

  st = time.time()

  # ---------------------------------------------------------------------------
  # process 

  while True:

    pipe_item = yield
    header = ctx.obj['header']

    # if first video or new video, set frame delay based on video fps
    if (opt_autoplay and not frame_delay) or (header.frame_index == header.last_frame_index):
      try:
        mspf =  int(header.mspf)  # milliseconds per frame
      except Exception as e:
        mspf = opt_delay

    # dynamically adjust framerate
    if opt_autoplay:
      frame_delay = int(max(1, mspf - (time.time() - st)/1000))
      st = time.time()
    
    # get and display image
    im = pipe_item.get_image(opt_frame_type)
    display_utils.show_ctx(ctx, im, pause=opt_pause, delay=frame_delay)

    pipe.send(pipe_item)