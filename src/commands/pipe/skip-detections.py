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
@click.option('--keep', 'opt_keep', is_flag=True,
  help='Inverts threshold to keep frames below threshold')
@click.option('--override', 'opt_override', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_threshold, opt_labels, opt_keep, opt_override):
  """Skip frames based on presence/absence of detections"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME
    

  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts.get(SKIP_FRAME):
      sink.send(M)
      continue

    # skip if no detections exist above the threshold
    skip = not M.frame_detections_exist(threshold=opt_threshold, labels=opt_labels)
    skip = not skip if opt_keep else skip
    if opt_override:
      ctx.opts[SKIP_FRAME] = skip
    else:
      ctx.opts[SKIP_FRAME] = (ctx.opts.get(SKIP_FRAME) or skip)
    
    sink.send(M)