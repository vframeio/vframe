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
@click.option('--suffix-slice/--no-suffix-slice', 'opt_append_slice', 
  is_flag=True, default=True,
  help='Append open command slice indices to filename')
@click.option('-v', '--verbose', 'opt_verbose', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_fp_out, opt_minify, opt_append_slice, opt_verbose):
  """Save frame data as JSON"""
  
  from pathlib import Path

  from vframe.settings.app_cfg import LOG, READER, SKIP_FILE, SKIP_FRAME
  from vframe.utils.file_utils import get_ext, write_json, add_suffix

  
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
    R = ctx.obj[READER]

    if M.is_last_item and not ctx.obj[SKIP_FILE]:
      # append after processing each file
      metadata.append(M.to_dict())

    if R.is_last_item and (M.is_last_item or ctx.obj[SKIP_FILE]):
      if opt_append_slice and all([x > -1 for x in R.slice_idxs]):
        suffix = f'_{R.slice_idxs[0]}_{R.slice_idxs[1]}'
        fp_out = add_suffix(opt_fp_out, suffix)
      else:
        fp_out = opt_fp_out
      # save after processing all files
      write_json(fp_out, metadata, minify=opt_minify, verbose=opt_verbose)
    
    sink.send(M)