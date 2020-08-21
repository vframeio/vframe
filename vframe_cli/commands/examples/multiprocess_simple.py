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
@click.option('-o', '--output', 'opt_fp_out', required=True)
@click.option('-r', '--recursive', 'opt_recursive', is_flag=True)
@click.option('-e', '--ext', 'opt_exts', default=['mp4'], multiple=True, 
  help='Glob extension')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('-t', '--threads', 'opt_threads', default=2)
@click.pass_context
def cli(ctx, opt_dir_in, opt_fp_out, opt_recursive, opt_exts, opt_slice, opt_threads):
  """Multiprocess simple boilerplate. Convert directory of media to SHA256 JSON"""

  """
  Example using the multiprocess pool to batch process videos or images.
  This example computes the SHA256, then saves it to a JSON file.
  Adapt or modify this file to create your custom script.

  """


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
  log.debug(f'Using {opt_threads} threads')

  # glob items
  fp_items = file_utils.glob_multi(opt_dir_in, opt_exts, recursive=opt_recursive)
  if any(opt_slice):
    fp_items = fp_items[opt_slice[0]:opt_slice[1]]
  log.info(f'Processing: {len(fp_items):,} videos')


  # -----------------------------------------------------------
  # pool worker

  def pool_worker(pool_item):

    # init threaded video reader
    #log = app_cfg.LOG
    fp = pool_item['fp']
    result = {'fp': fp}

    # add sha256
    sha256 = file_utils.sha256(fp)
    result.update({'sha256': sha256})

    # add media metadata
    media_meta = asdict(mediainfo(fp))
    result.update(media_meta)

    return result

  # end pool worker
  # -----------------------------------------------------------


  # convert file list into object with 
  pool_items = [{'fp': fp} for fp in fp_items]

  # init processing pool iterator
  # use imap instead of map via @hkyi Stack Overflow 41920124
  desc = f'Generating SHA256'
  with Pool(opt_threads) as p:
    pool_results = list(tqdm(p.imap(pool_worker, pool_items), total=len(fp_items), desc=desc))

  # print status report
  n_ok = len([x for x in pool_results if x['sha256'] is not None])
  n_nok = len(pool_results) - n_ok
  log.info(f'Success: {n_ok:,}')
  if n_nok > 0:
    log.error(f'Errors: {n_nok:,}')

  # write metadata to JSON
  file_utils.ensure_dir(opt_fp_out)
  file_utils.write_json(pool_results, opt_fp_out, minify=False)