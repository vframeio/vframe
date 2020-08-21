############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.models import types
from vframe.utils import click_utils

@click.command('')
@click.option('-i', '--input', 'opt_dir_in', required=True)
@click.option('-o', '--output', 'opt_dir_out', required=True)
@click.option('-r', '--recursive', 'opt_recursive', is_flag=True)
@click.option('-e', '--ext', 'opt_exts', default=['mp4'], multiple=True, 
  help='Glob extension')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('-t', '--threads', 'opt_threads', default=2)
@click.option('-m', '--model', 'opt_model_enum',
  type=types.ModelZooClickVar,
  default='caffe_places365_googlenet',
  help=click_utils.show_help(types.ModelZoo))
@click.option('--min-width', 'opt_width_min', default=224,
  help='Filter out media below this width')
@click.option('--min-height', 'opt_height_min', default=224,
  help='Filter out media below this height')
@click.option('--size', 'opt_size_process', default=256,
  help='Process media at this size')
@click.pass_context
def cli(ctx, opt_dir_in, opt_dir_out, opt_recursive, opt_exts, opt_slice, opt_threads,
  opt_model_enum, opt_width_min, opt_height_min, opt_size_process):
  """Multiprocess video processor boilerplate"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path

  import numpy as np
  import cv2 as cv
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count
  
  from vframe.settings import app_cfg
  from vframe.settings.modelzoo_cfg import modelzoo
  from vframe.models.dnn import DNN
  from vframe.image.dnn_factory import DNNFactory
  from vframe.utils.file_utils import glob_multi
  from vframe.utils.video_utils import FileVideoStream, mediainfo


  log = app_cfg.LOG

  # set N threads
  if not opt_threads:
    opt_threads = cpu_count()  # maximum
  log.debug(f'Using {opt_threads} threads')

  # glob items
  fp_items = glob_multi(opt_dir_in, opt_exts, recursive=opt_recursive)
  if any(opt_slice):
    fp_items = fp_items[opt_slice[0]:opt_slice[1]]
  log.info(f'Processing: {len(fp_items):,} videos')


  # multithreaded worker
  def pool_worker(pool_item):

    # init threaded video reader
    log = app_cfg.LOG
    fp_item = pool_item['fp']
    result = True

    # ensure file still exists after long globs
    if not Path(fp_item).is_file():
      log.error(f'File disappeared: {fp_item}')
      return False

    # Read (faster) metadata first to filter out by sizes
    meta = mediainfo(fp_item)
    w, h = (meta.width, meta.height)
    if not (w > pool_item['width_min'] and h > pool_item['height_min']):
      log.error(f'Size criteria fail ({w}, {h})')
      return False

    # init
    frame_idx = 0

    # dnn cfg
    model_cfg = modelzoo.get(pool_item['opt_model_name'])
    if pool_item['opt_gpu']:
      model_cfg.use_gpu()
    else:
      model_cfg.use_cpu()

    # init dnn
    cvmodel = DNNFactory.from_dnn_cfg(model_cfg)

    # read video into opencv and create focal mask
    video = cv.VideoCapture(fp_item)
    
    # read attributes
    w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
  
    while video.isOpened():

      frame_ok, frame = video.read()
      if not frame_ok:
        break

      # process video frames
      results = cvmodel.infer(frame)

    return {'success': result, 'fp_item': fp_item}



  # -----------------------------------------------------------
  # convert file list into objects

  pool_items = []

  for fp_item in fp_items:
    o = {
      'fp': fp_item,
      'opt_dir_out': opt_dir_out,
      'opt_model_name': opt_model_enum.name.lower(),
      'opt_gpu': True,
      'opt_size_process': opt_size_process,
      'height_min': opt_height_min,
      'width_min': opt_width_min,
    }
    pool_items.append(o)

  # Multiprocess/threading use imap instead of map via @hkyi Stack Overflow 41920124
  desc = f'Model: {opt_model_enum.name.lower()}'
  with Pool(opt_threads) as p:
    pool_results = list(tqdm(p.imap(pool_worker, pool_items), total=len(fp_items), desc=desc))

  # print status report
  # n_ok = sum([x fo xpool_results)
  # n_nok = len(pool_results) - n_ok
  # log.info(f'Success: {n_ok:,}')
  # if n_nok > 0:
  #   log.error(f'Errors: {n_nok:,}')