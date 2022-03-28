############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils.click_utils import processor, show_help

@click.command('')
@processor
@click.pass_context
def cli(ctx, sink, ):
  """Reset metadata if frame skipped"""
  

  from vframe.settings.app_cfg import LOG, SKIP_FRAME

  while True:

    M = yield

    if ctx.obj.get(SKIP_FRAME):
      M.metadata[M.index] = {}
    
    sink.send(M)