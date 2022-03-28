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
@click.option('-i', '--interval', 'opt_frame_interval', default=2,
  help='Number of frames to decimate/skip for every frame read')
@click.option('--override', 'opt_override', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_frame_interval, opt_override):
  """Skip frames at regular interval"""
  
  from vframe.models.types import MediaType
  from vframe.settings.app_cfg import LOG, SKIP_FRAME, READER
    

  while True:

    M = yield
    R = ctx.obj[READER]

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    idx = R.index if M.type == MediaType.IMAGE else M.index
    skip = idx % opt_frame_interval  # valid frame = 0

    if opt_override:
      ctx.obj[SKIP_FRAME] = skip
    else:
      ctx.obj[SKIP_FRAME] = (ctx.obj[SKIP_FRAME] or skip)
    
    sink.send(M)