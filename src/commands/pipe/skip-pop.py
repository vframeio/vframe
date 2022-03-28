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
@click.option('--include', 'opt_includes', multiple=True,
  help='Include this label')
@click.option('--exclude', 'opt_excludes', multiple=True,
  help='Include only this label')
@processor
@click.pass_context
def cli(ctx, sink, opt_includes, opt_excludes):
  """Pop (remove) metadata on skip frames"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME
    
  
  while True:

    M = yield

    if ctx.obj[SKIP_FRAME]:
      M.metadata[M.index] = {}  # clear
    
    sink.send(M)