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


@click.command()
@click.option('-i', '--input', 'opt_input', required=True,
  help='Path to input directory of images')
@click.option('-o', '--output', 'opt_output', required=True,
  help='Path to output directory to save degraded images')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('-t', '--threads', 'opt_threads', 
  help='Number threads')
@click.option('-e', '--ext', 'opt_exts', default=['jpg', 'png'],
  multiple=True,
  help='Glob extension')
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_slice, opt_threads, opt_exts):
  """Degrades images, save to another directory"""

  
  from os.path import join
  import random

  from PIL import Image
  import pandas as pd
  from glob import glob
  from pathlib import Path

  import cv2 as cv
  import numpy as np
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count

  from vframe.utils import log_utils, file_utils, im_utils
  from vframe.utils.degrade_utils import quality, zoom_shift, scale
  from vframe.utils.degrade_utils import motion_blur_v, motion_blur_h
  from vframe.utils.degrade_utils import enhance, auto_adjust
  from vframe.utils.degrade_utils import chromatic_aberration

  log = app_cfg.LOG
  log.info('Degrade data to match target domain')

  if opt_input == opt_output:
    log.error('Input can not equal output directory. Change input or output.')
    return

  opt_threads = opt_threads if opt_threads else cpu_count()
  file_utils.ensure_dir(opt_output)

  # glob images
  fps_ims = file_utils.glob_multi(opt_input, exts=opt_exts, sort=True)
  if any(opt_slice):
    fps_ims = fps_ims[opt_slice[0]:opt_slice[1]]
  log.info(f'found {len(fps_ims)} images in {opt_input}')

  # multiproc pool
  def pool_worker(fp_im):
    im = cv.imread(fp_im)
    try:
      w, h = im.shape[:2][::-1]
    except Exception as e:
      log.error(f'Could not process: {fp_im}. {e}')
      return

    # randomly degrade image
    im = quality(im, value_range=(30, 90), alpha_range=(0.5, 1.0), rate=1.0)
    # im = motion_blur_v(im, value_range=(0.01, 0.1), alpha_range=(0.25, 0.75), rate=0.15)
    # im = motion_blur_h(im, value_range=(0.01, 0.1), alpha_range=(0.25, 0.75), rate=0.15)
    # im = scale(im, value_range=(0.05, 0.1), rate=0.15)
    im = zoom_shift(im, value_range=(1, 6), alpha_range=(0.1, 0.6), rate=0.1)
    im = enhance(im, 'sharpness', value_range=(0.5, 6.0), rate=0.15)
    im = enhance(im, 'brightness', value_range=(0.75, 1.25), rate=0.15)
    im = enhance(im, 'contrast', value_range=(0.75, 1.25), rate=0.15)
    im = enhance(im, 'color', value_range=(0.75,1.25), rate=0.15)
    im = auto_adjust(im, 'equalize', alpha_range=(0.05, 0.1), rate=0.15)  # caution
    im = auto_adjust(im, 'autocontrast', alpha_range=(0.1, 0.5), rate=0.15)
    im = chromatic_aberration(im, 0, value_range=(1, 1), rate=0.1)
    im_pil = im_utils.np2pil(im)

    fp_out = join(opt_output, Path(fp_im).name)
    cv.imwrite(fp_out, im)

  # fill pool
  with Pool(opt_threads) as p:
    d = f'Degrading x{opt_threads}'
    pool_results = list(tqdm(p.imap(pool_worker, fps_ims), total=len(fps_ims), desc=d))

