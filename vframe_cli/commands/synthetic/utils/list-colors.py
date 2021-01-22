############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import os

import click

from vframe.models import types
from vframe.models.color import Color
from vframe.settings import app_cfg
from vframe.utils import click_utils


@click.command()
@click.option('-i', '--input', 'opt_input', required=True,
  help='Path to project folder (metadata.csv, mask, real)')
@click.option('-t', '--threads', 'opt_threads', default=12, show_default=True,
  help='Number threads')
@click.pass_context
def cli(ctx, opt_input, opt_threads):
  """List unique colors in masks"""
  
  import os
  from os.path import join
  from glob import glob
  from pathlib import Path
  from dataclasses import asdict
  import logging

  import dacite
  from pathos.multiprocessing import ProcessingPool as Pool
  #from multiprocessing.pool import ThreadPool as Pool
  #from functools import partial
  from numba import jit, njit
  from PIL import Image
  import pandas as pd
  import cv2 as cv
  import numpy as np
  from tqdm import tqdm

  from vframe.utils import file_utils, im_utils
  from vframe.models.geometry import BBox
  from vframe.models.annotation import Annotation
  from vframe.utils import anno_utils


  # init log
  log = app_cfg.LOG

  # post-process input
  opt_threads = opt_threads if opt_threads else pathos.multiprocessing.cpu_count()

  # glob mask
  fps_masks = glob(join(opt_input, '*.png'))


  @jit(nopython=True, parallel=True)
  def fast_np_trim(im):
    '''Trims ndarray of blackspace/zeros
    :param im: np.ndarray image in BGR or RGB
    :returns np.ndarray image in BGR or RGB
    '''
    # Warning: does not throw error when used in @jit mode
    # this can cause early termination of pool processes
    # comment out @jit if early termination
    npza = np.array([0,0,0], dtype=np.uint8)
    w, h = im.shape[:2][::-1]
    im = im.reshape((w * h, 3))
    idxs = np.where(im > npza)
    if len(idxs[0]):
      return im[min(idxs[0]):max(idxs[0])]
    else:
      return im


  def pool_worker(fp_mask):
    fn_mask = Path(fp_mask).name
    im_mask = cv.imread(fp_mask)
    w, h = im_mask.shape[:2][::-1]

    # flatten image and find unique colors
    im_mask_rgb = cv.cvtColor(im_mask, cv.COLOR_BGR2RGB)

    # Opt 1: use Numpy unique
    #im_flat_rgb = im_mask_rgb.reshape((w * h, 3))
    #rgb_unique = np.unique(im_flat_rgb, axis=0)

    # Opt 2: use numba then set list (faster)
    n_colors_found = 0
    results = []

    try:
      im_flat_rgb_trim = fast_np_trim(im_mask_rgb)
    except Exception as e:
      log.error(fp_mask)
      log.error(e)
    im_flat_rgb_trim = im_flat_rgb_trim.tolist()
    rgb_unique = list(set(tuple(map(tuple, im_flat_rgb_trim))))
    colors = [Color.from_rgb_int(c) for c in rgb_unique]
    return colors

  with Pool(opt_threads) as p:
    d = f'Get Colors x{opt_threads}'
    t = len(fps_masks)
    pool_results = list(tqdm(p.imap(pool_worker, fps_masks), total=t, desc=d))

  colors = []
  for pool_result in pool_results:
    for c in pool_result:
      colors.append(c.to_rgb_int())

  colors = list(set(colors))
  for c in colors:
    log.info(c)

