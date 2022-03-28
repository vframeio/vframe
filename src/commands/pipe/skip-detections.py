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
@click.option('--min', 'opt_threshold_lt', 
  type=click.FloatRange(0,1), default=0.0,
  help='Skip detections less than this threshold')
@click.option('--max', 'opt_threshold_gt', 
  type=click.FloatRange(0,1), default=1.0,
  help='Skip detections greater than this threshold')
@click.option('--label', 'opt_labels', multiple=True,
  help='Detection label')
@click.option('--keep', 'opt_keep', is_flag=True,
  help='Inverts threshold to keep frames below threshold')
@click.option('--override', 'opt_override', is_flag=True)
@click.option('--pop', 'opt_pop', is_flag=True, help='Remove detection if skipped')
@processor
@click.pass_context
def cli(ctx, sink, opt_threshold_lt, opt_threshold_gt, 
  opt_labels, opt_keep, opt_override, opt_pop):
  """Skip frames based on presence/absence of detections"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME
    
  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj.get(SKIP_FRAME):
      sink.send(M)
      continue

    # skip if no detections exist above the threshold
    t = (opt_threshold_lt, opt_threshold_gt)
    exist = M.frame_detections_exist(thresholds=t, labels=opt_labels)
    skip = not exist if opt_keep else exist
    if opt_override:
      ctx.obj[SKIP_FRAME] = skip
    else:
      ctx.obj[SKIP_FRAME] = (ctx.obj.get(SKIP_FRAME) or skip)
    
    sink.send(M)