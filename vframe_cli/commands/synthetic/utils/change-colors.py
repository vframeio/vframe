############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click


@click.command()
@click.option('-i', '--input', 'opt_input', required=True)
@click.option('-o', '--output', 'opt_output',
  help='Path to video output file')
@click.option('-t', '--threads', 'opt_threads', type=int,
  help='Number threads')
@click.option('--from', 'opt_colors_from', type=int, nargs=3,
  multiple=True,
  required=True,
  help='RGB source color')
@click.option('--to', 'opt_color_to', type=int, nargs=3,
  required=True,
  help='RGB destination color')
@click.option('--dry-run', 'opt_dry_run', is_flag=True,
  help='Overwrite existing files')
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_colors_from, opt_color_to, opt_threads, opt_dry_run):
  """Swaps colors in mask images"""
  
  import pandas as pd
  from os.path import join
  from glob import glob
  from pathlib import Path

  import cv2 as cv
  import numpy as np
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count

  from vframe.settings import app_cfg
  from vframe.utils import file_utils, im_utils
  from vframe.models.color import Color

  log = app_cfg.LOG
  log.info(f'Swapping colors in: {Path(opt_input).name}')

  # default output is input
  opt_output = opt_output if opt_output else opt_input
  colors_from = [Color.from_rgb_int(c) for c in opt_colors_from]
  color_to = Color.from_rgb_int(opt_color_to)

  # set N threads
  opt_threads = opt_threads if opt_threads else cpu_count()
  log.info(f'Using {opt_threads} threads')

  # ensure output dir
  file_utils.ensure_dir(opt_output)

  # glob images
  fp_masks = sorted([im for im in glob(str(Path(opt_input) / '*.png'))])
  log.info(f'found {len(fp_masks)} mask images')

  if opt_dry_run:
    log.info('Dry run. Use --confirm to overwrite images')
    return

  def pool_worker(item):
    """Swaps image colors and writes image
    """
    im = cv.imread(item['fp_in'])
    for color_from in item['colors_from']:
      im = im_utils.swap_color(im, color_from, item['color_to'])
    cv.imwrite(item['fp_out'], im)
    return True


  # ----------------------------------------------------------------------------------
  # Process images

  pool_items = []
  for fp in fp_masks:
    fp_out = join(opt_output, Path(fp).name)
    o = {
      'fp_in':fp, 
      'fp_out':fp_out,
      'colors_from': colors_from,
      'color_to': color_to,
      'opt_dry_run': opt_dry_run
    }
    pool_items.append(o)

  with Pool(opt_threads) as p:
    d = f'Compositing x{opt_threads}'
    pool_results = list(tqdm(p.imap(pool_worker, pool_items), total=len(pool_items), desc=d))