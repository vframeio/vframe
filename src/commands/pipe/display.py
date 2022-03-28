############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.models.types import MediaType
from vframe.utils.click_utils import processor
from vframe.models.types import FrameImage, FrameImageVar
from vframe.utils.click_utils import show_help

@click.command('')
@click.option('--fps', 'opt_fps', default=25, show_default=True,
  type=click.IntRange(1,1000),
 help='Frames per second. Use 0 to pause.')
@click.option('--pause/--play', 'opt_pause', is_flag=True,
  help='Autoplay video')
@click.option('--frame', 'opt_frame_type', default='draw',
  type=FrameImageVar, help=show_help(FrameImage))
@click.option('--filter', 'opt_filter', type=click.Choice(['detections', 'no-detections']),
  help='Only display frames with detections')
@processor
@click.pass_context
def cli(ctx, sink, opt_fps, opt_pause, opt_frame_type, opt_filter):
  """Display images to screen"""
  
  import time

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME
  from vframe.settings.app_cfg import PAUSED, READER
  from vframe.utils.display_utils import DisplayUtils

  
  # ---------------------------------------------------------------------------
  # initialize

  ctx.obj[USE_DRAW_FRAME] = True

  display_utils = DisplayUtils()
  target_mspf = 1000 / opt_fps
  ctx.obj[PAUSED] = opt_pause
  st = time.time()

  # ---------------------------------------------------------------------------
  # process 

  while True:
    
    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue
    
    # override pause if single image
    if ctx.obj[READER].n_files == 1 and M.type == MediaType.IMAGE:
      ctx.obj[PAUSED] = True
    
    # dynamically adjust framerate
    actual_mspf = (time.time() - st) / 1000
    frame_delay_ms = int(max(1, target_mspf - actual_mspf))
    
    # get and display image
    fde = M.frame_detections_exist

    if not opt_filter or ((fde and opt_filter == 'detections') or (fde == False and opt_filter == 'no-detections')):
      im = M.images.get(opt_frame_type)
      display_utils.show_ctx(ctx, im, delay=frame_delay_ms)
      
    # continue
    sink.send(M)
    st = time.time()