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
@click.option('-o', '--output', 'opt_fp_out', required=True,
  help='Path to output file')
@click.option('--minify/--no-minify', 'opt_minify', is_flag=True, default=True,
  help='Minify JSON output')
@click.option('-v', '--verbose', 'opt_verbose', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_fp_out, opt_minify, opt_verbose):
  """Save frame data as JSON"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY
  from vframe.utils.file_utils import get_ext, write_json

  
  # ---------------------------------------------------------------------------
  # initialize

  # error check
  if not get_ext(opt_fp_out).lower() == 'json':
    LOG.error('Only JSON export supported')
    return

  # ---------------------------------------------------------------------------
  # process 

  metadata = []
  
  # accumulate all pipe items
  while True:

    M = yield
    R = ctx.obj['reader']

    if M.is_last_item:
      # append after processing each file
      metadata.append(M.to_dict())
    if R.is_last_item and M.is_last_item:
      # save after processing all files
      write_json(opt_fp_out, metadata, minify=opt_minify, verbose=opt_verbose)
    
    sink.send(M)