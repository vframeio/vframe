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
@click.option('-o', '--output', 'opt_output', required=True,
  help='Path to output JSON file')
@click.option('--minify/--no-minify', 'opt_minify', is_flag=True, default=True,
  help='Minify JSON output')
@click.option('--suffix-slice/--no-suffix-slice', 'opt_append_slice', 
  is_flag=True, default=True,
  help='Append open command slice indices to filename')
@click.option('--subdirs', 'opt_subdirs', is_flag=True)
@click.option('-v', '--verbose', 'opt_verbose', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_output, opt_minify, opt_append_slice, opt_subdirs, opt_verbose):
  """Save file and frame metadata to JSON"""
  
  from pathlib import Path
  from os.path import join

  from vframe.settings.app_cfg import LOG, READER, SKIP_FILE, FN_DETECTIONS
  from vframe.utils.file_utils import get_ext, write_json, add_suffix

  
  # ---------------------------------------------------------------------------
  # initialize

  # error check
  if Path(opt_output).is_dir() or not Path(opt_output).is_file():
    opt_output = join(opt_output, FN_DETECTIONS)
  elif not get_ext(opt_output).lower() == 'json':
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

      if opt_subdirs:
        fp_out = join(Path(opt_output).parent, M.filestem, FN_DETECTIONS)
        write_json(fp_out, metadata, minify=opt_minify, verbose=opt_verbose)
        metadata = []  

    if R.is_last_item and (M.is_last_item or ctx.obj[SKIP_FILE]) and not opt_subdirs:
      if opt_append_slice and all([x > -1 for x in R.slice_idxs]):
        suffix = f'_{R.slice_idxs[0]}_{R.slice_idxs[1]}'
        fp_out = add_suffix(opt_output, suffix)
      else:
        fp_out = opt_output
      # save after processing all files
      write_json(fp_out, metadata, minify=opt_minify, verbose=opt_verbose)
    
    sink.send(M)