############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

ext_choices = ['jpg', 'png']

@click.command()
@click.option('-i', '--input', 'opt_dirs_in', required=True, multiple=True,
  help='Path to directory of images')
@click.option('--subdirs', 'opt_subdirs', is_flag=True,
  help='Glob all sub directories for each input directory, but not recursively')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output dir')
@click.option('-e', '--ext', 'opt_ext', default='png',
  type=click.Choice(ext_choices),
  help='Path to output dir')
@click.option('--symlink/--copy', 'opt_symlink', is_flag=True,
  default=True,
  help='Symlink or copy images to new directory')
@click.option('--masks/--no-masks', 'opt_concat_masks', is_flag=True, 
  default=False,
  help='Concat masks')
@click.pass_context
def cli(ctx, opt_dirs_in, opt_subdirs, opt_dir_out, opt_ext, opt_symlink, opt_concat_masks):
  """Concatenate multiple render directories"""

  from os.path import join
  from pathlib import Path
  from glob import glob

  import pandas as pd
  from tqdm import tqdm

  from vframe.settings import app_cfg
  from vframe.utils import file_utils

  log = app_cfg.LOG

  # output
  fp_dir_out_real = join(opt_dir_out, app_cfg.DN_REAL)
  file_utils.ensure_dir(fp_dir_out_real)

  log.debug(f'opt_concat_masks: {opt_concat_masks}')
  if opt_concat_masks:
    log.info('Copying masks enabled')
    fp_dir_out_mask = join(opt_dir_out, app_cfg.DN_MASK)
    file_utils.ensure_dir(fp_dir_out_mask)


  # group input directories
  if opt_subdirs:
    dirs_input = []
    for d in opt_dirs_in:
      dirs_input += glob(join(d, '*'))
  else:
    dirs_input = opt_dirs_in

  log.info(f'Concatenating {len(dirs_input)} directories')
  dfs_annos = []

  for dir_in in dirs_input:
    log.debug(dir_in)
    # concat dataframe
    fp_annos = join(dir_in, app_cfg.FN_ANNOTATIONS)
    if not Path(fp_annos).is_file():
      log.warn(f'{fp_annos} does not exist. Skipping')
      continue
    df_annos = pd.read_csv(fp_annos)
    dfs_annos.append(df_annos)

    # symlink/copy real images
    render_dir_names = [app_cfg.DN_REAL]
    concat_dir_names = [fp_dir_out_real]
    if opt_concat_masks:
      render_dir_names.append(app_cfg.DN_MASK)
      concat_dir_names.append(fp_dir_out_mask)

    for sf in df_annos.itertuples():
      # real
      for concat_dir_name, render_dir_name in zip(concat_dir_names, render_dir_names):
        fp_src = join(dir_in, render_dir_name, sf.filename)
        fpp_dst = Path(join(concat_dir_name, sf.filename))
        if fpp_dst.is_symlink():
          fpp_dst.unlink()
        fpp_dst.symlink_to(fp_src)


  df_annos = pd.concat(dfs_annos)

  fp_out = join(opt_dir_out, app_cfg.FN_ANNOTATIONS)
  df_annos.to_csv(fp_out, index=False)
  log.info(f'Wrote new annotations file with {len(df_annos):,} items')