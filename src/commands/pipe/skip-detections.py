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
@click.option('-t', '--threshold', 'opt_threshold', 
  type=click.FloatRange(0,1), default=0.5,
  help='Threshold below which frames are skipped (1.0 = all skipped, 0.0 = none skipped)')
@click.option('--label', 'opt_labels', multiple=True,
  help='Detection label')
@click.option('--skip/--keep', 'opt_skip', is_flag=True, default=True,
  help='Inverts threshold to keep frames below threshold')
@processor
@click.pass_context
def cli(ctx, sink, opt_threshold, opt_labels, opt_skip):
  """Skip frames based on presence/absence of detections"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY
    

  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME_KEY]:
      sink.send(M)
      continue

    # returns True if dets exist > threshold
    exist = M.frame_detections_exist(threshold=opt_threshold, labels=opt_labels)
    skip = not exist if opt_skip else exist
    ctx.opts[SKIP_FRAME_KEY] = skip
    
    sink.send(M)