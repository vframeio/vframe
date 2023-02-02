############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

import operator
from datetime import date

from vframe.settings.app_cfg import LOG, READER
from vframe.utils.click_utils import processor, operator_validator_multi
from vframe.models.types import MediaType


@click.command('')
@click.option('--if', 'opt_if_evals', multiple=True,
  type=str, callback=operator_validator_multi)
@click.option('--skip/--keep', 'opt_skip', is_flag=True, default=True)
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.option('--skip-frame', 'opt_skip_frame', is_flag=True, 
  help='Skip file if current skip frame is True')
@processor
@click.pass_context
def cli(ctx, sink, opt_if_evals, opt_skip, opt_skip_frame, opt_verbose):
  """Skip file by filtering attributes"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FILE
  from vframe.settings.app_cfg import MEDIA_FILTERS, SKIP_MEDIA_FILTERS
  
  
  # init media filter
  if not (opt_if_evals or opt_skip_frame):
    return
    
  for opt_if_eval in opt_if_evals:
    opt_if_eval.is_skip = opt_skip

  ctx.obj[MEDIA_FILTERS] = opt_if_evals


  while True:

    M = yield

    # ignore filters if already applied in media file reader
    if ctx.obj.get(SKIP_MEDIA_FILTERS):
      sink.send(M)
      continue


    if (M.type == MediaType.VIDEO and M.is_first_item) or M.type == MediaType.IMAGE:
    
      skip_results = []

      for opt_if_eval in opt_if_evals:
      
        val = getattr(M, opt_if_eval.attribute)
        skip = opt_if_eval.evaulate(val)
        skip_results.append(skip)

      ctx.obj[SKIP_FILE] = any(skip_results) or opt_skip_frame

      if opt_verbose and opt_if_evals:
        LOG.info(f'\nSkipping: {skip_results}. because: {opt_if_eval.attribute}: {val} or opt_skip_frame: {opt_skip_frame}\n')
      elif opt_verbose:
        LOG.info(f'\nSkipping: {skip_results}. because: opt_skip_frame: {opt_skip_frame}\n')

    
    sink.send(M)