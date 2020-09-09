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
@click.option('-i', '--input', 'opt_dir_in', required=True)
@click.option('-r', '--recursive', 'opt_recursive', is_flag=True)
@click.option('-e', '--ext', 'opt_exts', default=['jpg', 'png'], multiple=True, 
  help='Glob extension')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('-t', '--threads', 'opt_threads', default=None)
@click.pass_context
def cli(ctx, opt_dir_in, opt_recursive, opt_exts, opt_slice, opt_threads):
  """Multiprocessor image template"""


  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  from dataclasses import asdict

  import numpy as np
  import cv2 as cv
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count

  from vframe.settings import app_cfg
  from vframe.settings.modelzoo_cfg import modelzoo
  from vframe.models.dnn import DNN
  from vframe.image.dnn_factory import DNNFactory
  from vframe.utils import file_utils
  from vframe.utils.video_utils import FileVideoStream, mediainfo


  log = app_cfg.LOG

  # set N threads
  if not opt_threads:
    opt_threads = cpu_count()  # maximum

  # glob items
  fp_items = file_utils.glob_multi(opt_dir_in, opt_exts, recursive=opt_recursive)
  if any(opt_slice):
    fp_items = fp_items[opt_slice[0]:opt_slice[1]]
  log.info(f'Processing: {len(fp_items):,} files')


  # -----------------------------------------------------------
  # start pool worker

  def pool_worker(pool_item):

    # init threaded video reader
    fp = pool_item['fp']
    result = {'fp': fp}

    # add media metadata
    im = cv.imread(fp)
    for i in range(20):
      im = cv.blur(im, (35,35))

    return result

  # end pool worker
  # -----------------------------------------------------------


  # convert file list into object with 
  pool_items = [{'fp': fp} for fp in fp_items]

  # init processing pool iterator
  # use imap instead of map via @hkyi Stack Overflow 41920124
  desc = f'image-mp x{opt_threads}'
  with Pool(opt_threads) as p:
    pool_results = list(tqdm(p.imap(pool_worker, pool_items), total=len(fp_items), desc=desc))