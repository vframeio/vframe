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
  help='Path to masks')
@click.option('-t', '--threads', 'opt_threads', default=12, show_default=True,
  help='Number threads')
@click.pass_context
def cli(ctx, opt_input, opt_threads):
  """Convert mask PNGs from RGBA to RGB"""

  import os
  from os.path import join
  from glob import glob
  from pathlib import Path

  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count
  from PIL import Image
  from tqdm import tqdm

  # init log
  log = app_cfg.LOG

  # post-process input
  opt_threads = opt_threads if opt_threads else cpu_count()

  # glob mask
  fps_masks = glob(join(opt_input, '*.png'))

  def pool_worker(fp):
    Image.open(fp).convert('RGB').save(fp)
    return

  with Pool(opt_threads) as p:
    d = f'Convert to RGB x{opt_threads}'
    t = len(fps_masks)
    pool_results = list(tqdm(p.imap(pool_worker, fps_masks), total=t, desc=d))
