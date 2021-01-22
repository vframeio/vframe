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
@click.option('-i', '--input', 'opt_dir_ims', required=True,
  help='Path to balanced directory images')
@click.option('--dry-run/--confirm', 'opt_dry_run', is_flag=True, default=True,
  help='Dry run, do not delete any files')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.pass_context
def cli(ctx, opt_dir_ims, opt_dry_run, opt_verbose):
  """Remove orphaned mask images """

  import os
  from os.path import join
  from pathlib import Path
  from glob import glob

  from tqdm import tqdm

  log = app_cfg.LOG

  fp_ims_real = sorted([im for im in glob(str(Path(opt_dir_ims) / app_cfg.DN_REAL / '*.png'))])
  fp_ims_mask = sorted([im for im in glob(str(Path(opt_dir_ims) / app_cfg.DN_MASK / '*.png'))])
  n_delete = len(fp_ims_mask) - len(fp_ims_real)
  log.info(f'Real: {len(fp_ims_real):,}. Masks: {len(fp_ims_mask):,}. Deleting: {n_delete} masks')

  if not n_delete > 0:
    log.info('Same number of images.')
    return

  # create list of files to delete
  # list of real image names to check
  fns_real = [Path(fp).name for fp in fp_ims_real]
  fps_delete = [fp for fp in fp_ims_mask if not Path(fp).name in fns_real]

  if opt_dry_run:
    log.info(f'Add "--confirm" to delete {n_delete:,} mask images')
  else:
    log.info(f'Deleting {n_delete:,} images...')

  # delete images
  for fp in tqdm(fps_delete):
    if not opt_dry_run:
      if opt_verbose:
        log.info(f'Deleting: {fp}')
      os.remove(fp)
    else:
      if opt_verbose:
        log.info(f'Dry run. Did not delete: {fp}')




