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
  """Skip frames if include/exclude labels"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME
    
  
  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    # if exc/inc classes
    valid_inc = M.includes_labels(opt_includes) if opt_includes else True
    valid_exc = M.excludes_labels(opt_excludes) if opt_excludes else True

    skip = not (valid_inc or valid_exc)
    ctx.obj[SKIP_FRAME] = skip
    
    sink.send(M)