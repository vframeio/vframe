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
  help='Path to directory with real and masks')
@click.option('-f', '--force', 'opt_force', is_flag=True)
@click.option('-e', '--ext', 'opt_ext', default='png')
@click.pass_context
def cli(ctx, opt_dir_ims, opt_ext, opt_force):
  """Create blank mask image for negative real images"""

  import os
  from os.path import join
  from pathlib import Path
  from glob import glob

  import cv2 as cv
  import numpy as np
  from tqdm import tqdm

  from vframe.settings import app_cfg

  log = app_cfg.LOG
  dir_masks = join(opt_dir_ims, app_cfg.DN_MASK)
  dir_reals = join(opt_dir_ims, app_cfg.DN_REAL)
  fps_ims_real = sorted([im for im in glob(join(dir_reals,  f'*.{opt_ext}'))])
  fps_ims_mask = sorted([im for im in glob(join(dir_masks, f'*.{opt_ext}'))])

  log.info(f'found {len(fps_ims_mask)} mask images')
  log.info(f'found {len(fps_ims_real)} real images')

  for fp_im_real in tqdm(fps_ims_real):
    fn = Path(fp_im_real).name
    fp_mask = join(dir_masks, fn)
    if Path(fp_mask).is_file() and not opt_force:
      log.warn(f'File exists: {fn} Use "-f/--force" to overwrite')
    else:
      im = cv.imread(fp_im_real)
      im_blank = np.zeros_like(im)
      cv.imwrite(fp_mask, im_blank)

    