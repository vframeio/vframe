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
@click.option('--interval', 'opt_frame_interval', default=2,
  help='Number of frames to decimate/skip for every frame read')
@processor
@click.pass_context
def cli(ctx, sink, opt_frame_interval):
  """Skip frames at regular interval"""
  
  from vframe.models.types import MediaType
  from vframe.settings.app_cfg import LOG, SKIP_FRAME
    

  while True:

    M = yield
    R = ctx.obj['reader']

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME]:
      sink.send(M)
      continue

    idx = R.index if M.type == MediaType.IMAGE else M.index
    skip_frame = idx % opt_frame_interval  # valid frame = 0
    ctx.opts[SKIP_FRAME] = skip_frame
    
    sink.send(M)