############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

@click.command('')
@click.option('-i', '--input', 'opt_inputs', required=True, multiple=True,
    help="Path to input")
@click.option('-o', '--output', 'opt_output', required=True,
    help="Path to output")
@click.pass_context
def cli(ctx, opt_inputs, opt_output):
  """Merge detection JSON"""

  from pathlib import Path
  from tqdm import tqdm

  from vframe.utils.file_utils import glob_multi, get_ext, load_json, write_json
  from vframe.settings.app_cfg import LOG


  fp_files = []

  for opt_input in opt_inputs:
    if Path(opt_input).is_dir():
      fp_files += glob_multi(opt_input, exts=['json'])
    elif get_ext(opt_input) == 'json':
      fp_files.append(opt_input)

  if not len(fp_files) > 1:
    LOG.error('More than 1 file required.')
    LOG.debug(fp_files)
    return

  results = []
  for fp_file in tqdm(fp_files):
    results += load_json(fp_file)

  LOG.debug(f'Writing data to {opt_output}...')
  write_json(opt_output, results)

  