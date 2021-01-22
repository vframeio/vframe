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
@click.option('-i', '--input', 'opt_fp_in', required=True,
  help='Path to directory of edited images')
@click.option('-e', '--ext', 'opt_ext', default='png',
  type=click.Choice(ext_choices),
  help='Image types to glob for')
@click.option('--prefix', 'opt_prefix',
  help='Filename prefix')
@click.option('--suffix', 'opt_suffix',
  help='Filename suffix')
@click.option('--replace', 'opt_replace', type=str, nargs=2,
  help='String replace (from, to)')
@click.option('--dry-run/--confirm', 'opt_dry_run', is_flag=True, default=True,
  help='Dry run, do not delete any files')
@click.option('--number', 'opt_number', is_flag=True,
  help='Number filenames')
@click.option('--n-zeros', 'opt_n_zeros', default=6,
  help='Number of zeros if using numbered filename')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.pass_context
def cli(ctx, opt_fp_in, opt_ext, opt_replace, opt_prefix, opt_suffix, 
  opt_number, opt_n_zeros, opt_dry_run, opt_verbose):
  """Rename files in render subdirectories"""

  import os
  from os.path import join
  from pathlib import Path
  from glob import glob

  from tqdm import tqdm
  from vframe.utils.file_utils import get_ext, zpad

  log = app_cfg.LOG
  fp_dirs = [join(opt_fp_in, d) for d in os.listdir(opt_fp_in) if os.path.isdir(join(opt_fp_in, d))]

  log.info(f'Found {len(fp_dirs)} subdirectories')
  
  if not opt_prefix and not opt_suffix and not opt_replace:
    log.error('No prefix, suffix, or replacement provided. Exiting')
    return
  
  if opt_dry_run:
    log.debug('Dry run. Not renaming any images')

  # Assume all subdirs have balanced names
  for fp_dir in fp_dirs:
    
    log.info(f'Renaming files in: {fp_dir}')
    fp_ims = sorted(glob(join(fp_dir, f'*.{opt_ext}')))
    
    for i, fp_src in enumerate(fp_ims):
      
      ext = get_ext(fp_src)

      if opt_number:
        fn_stem = f'{zpad(i, zeros=opt_n_zeros)}'
      else:
        fn_stem = Path(fp_src).stem
      if opt_prefix:
        fn_stem = f'{opt_prefix}{fn_stem}'
      if opt_suffix:
        fn_stem = f'{fn_stem}{opt_suffix}'
      if opt_replace:
        fn_stem = fn_stem.replace(*opt_replace)

      fn = f'{fn_stem}.{ext}'
      fp_dst = join(fp_dir, fn)
      
      if fp_dst is not fp_src:
        if opt_verbose:
          log.info(f'Rename {Path(fp_src).name} --> {fn}')
        if not opt_dry_run:
         os.rename(fp_src, fp_dst)
        else:
          log.info(f'Dry not. Did not rename: {fp_src} --> {fp_dst}')


