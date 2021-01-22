############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.settings import app_cfg

ext_choices = ['jpg', 'png']

@click.command()
@click.option('-i', '--input', 'opt_input', required=True,
  help='Path to directory with images already removed')
@click.option('-o', '--output', 'opt_output', required=True,
  help='Path to full directory where images will be removed')
@click.option('--dry-run/--confirm', 'opt_dry_run', is_flag=True, default=True,
  help='Dry run, do not delete any files')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_dry_run, opt_verbose):
  """Removes images from input if exist in output"""

  """
  Example:
  - two folders (A, B) of same images with same name where A == B
  - remove X images from A
  - remove !X from B (remove images from B if they are in A)
  """

  import os
  from os.path import join
  from pathlib import Path
  from glob import glob

  from tqdm import tqdm

  log = app_cfg.LOG

  fps_in = sorted([im for im in glob(str(Path(opt_input) / '*.png'))])
  fps_out = sorted([im for im in glob(str(Path(opt_output) /  '*.png'))])
  # delete images in output if they exist in input
  fns_input = [Path(fp).name for fp in fps_in]
  fps_output_delete = [fp for fp in fps_out if Path(fp).name in fns_input]
  n_delete = len(fps_output_delete)
  log.info(f'Found: {n_delete} images in output directory that are in input directory')

  if not n_delete > 0:
    log.info('Same number of images.')
    return


  if opt_dry_run:
    log.info(f'Add "--confirm" to delete {n_delete:,} images in {opt_output}')
  else:
    log.info(f'Deleting {n_delete:,} images...')

  # delete images
  for fp in tqdm(fps_output_delete):
    if not opt_dry_run:
      if opt_verbose:
        log.info(f'Deleting: {fp}')
      os.remove(fp)
    else:
      if opt_verbose:
        log.info(f'Dry run. Did not delete: {fp}')




