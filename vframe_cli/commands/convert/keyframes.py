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
@click.option('-o', '--output', 'opt_dir_out', required=False)
@click.option('-r', '--recursive/--no-recursive', 'opt_recursive', is_flag=True, default=True)
@click.option('-e', '--ext', 'opt_exts', default=['mp4'], multiple=True, 
  help='Glob extension')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('-t', '--threads', 'opt_threads', default=2)
@click.option('-m', '--model', 'opt_model_enum',
  type=types.ModelZooClickVar,
  default = 'imagenet_alexnet',
  help=click_utils.show_help(types.ModelZoo))
@click.option('--min-width', 'opt_width_min', default=224,
  help='Filter out media below this width')
@click.option('--min-height', 'opt_height_min', default=224,
  help='Filter out media below this height')
@click.option('--size', 'opt_size_process', default=256,
  help='Process media at this size')
@click.option('--focal-mask/--no-focal-mask', 'opt_use_focal_mask', 
  is_flag=True,  default=True,
  help='Use focal mask to deprioritize edge pixels')
@click.option('--remove-existing/--keep-existing', 'opt_remove_existing', is_flag=True,
  default=False,
  help='Removes existing keyframes')
@click.pass_context
def cli(ctx, opt_dir_in, opt_dir_out, opt_recursive, opt_exts, opt_slice, opt_threads,
  opt_model_enum, opt_width_min, opt_height_min, opt_size_process,
  opt_use_focal_mask, opt_remove_existing):
  """Detect and save keyframes"""

  # ------------------------------------------------
  # imports

  import os
  from os.path import join
  from glob import glob
  from pathlib import Path
  from dataclasses import asdict
  import shutil
  
  import dacite

  import imagehash
  import numpy as np
  import cv2 as cv
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count
  from sklearn.metrics.pairwise import cosine_similarity

  from vframe.settings import app_cfg
  from vframe.utils import file_utils, im_utils, draw_utils, model_utils, keyframe_utils
  from vframe.utils import video_utils
  from vframe.models.dnn import DNN
  from vframe.models.mediameta import KeyframeMediaMeta
  from vframe.image.dnn_factory import DNNFactory
  from vframe.settings.modelzoo_cfg import modelzoo
  from vframe.utils.video_utils import FileVideoStream


  log = app_cfg.LOG

  if not opt_threads:
    opt_threads = cpu_count()

  log.debug(f'Using {opt_threads} threads')

  fp_items = file_utils.glob_multi(opt_dir_in, opt_exts, recursive=opt_recursive)
  if any(opt_slice):
    fp_items = fp_items[opt_slice[0]:opt_slice[1]]

  log.info(f'Processing: {len(fp_items):,} videos')

  # multithreaded worker
  def pool_worker(pool_item):

    # init threaded video reader
    log = app_cfg.LOG
    fp_item = pool_item['fp']

    # ensure file still exists, globbing duration may be long
    if not Path(fp_item).is_file():
      log.error(f'File disappeared: {fp_item}')
      return False

    # if image, copy and return
    if file_utils.get_ext(fp_item) == 'jpg':
      sha256 = file_utils.sha256(fp_item)
      fp_dir_out = join(pool_item['opt_dir_out'], sha256)
      frame_idx = 1
      fn = f'{file_utils.zpad(frame_idx)}.jpg'
      fp_out = join(fp_dir_out, Path(fp_item).name)
      file_utils.ensure_dir(fp_out)
      shutil.copy(fp_item, fp_out)
      return True

    # Read (faster) metadata first to filter out by sizes

    sha256 = file_utils.sha256(fp_item)
    fp_dir_out = join(pool_item['opt_dir_out'], sha256)
    if Path(fp_dir_out).is_dir():
      if pool_item['opt_remove_existing']:
        shutil.rmtree(fp_dir_out)
      else:
        log.debug(f'Skipping {sha256}')
        return False
    
    # mk output dir
    file_utils.ensure_dir(fp_dir_out)

    log.debug(f'Process: {fp_item}')
    meta = video_utils.mediainfo(fp_item)
    w, h = (meta.width, meta.height)
    if not (w > pool_item['width_min'] and h > pool_item['height_min']):
      log.error(f'Size criteria fail ({w}, {h})')
      return False

    # opts
    opt_width = pool_item['opt_size_process']
    max_phash_delta = 64
    opt_phash_thresh = 32 / max_phash_delta  # 30/64
    opt_phash_dnn_thresh = 14 / max_phash_delta
    opt_phash_thresh_multi = 28 / max_phash_delta
    opt_vec_thresh_solo = 0.225
    opt_vec_thresh_multi = 0.15
    opt_ext = 'jpg'

    # init
    phash_deltas = []
    dnn_deltas = []
    scene_frame_idxs = []
    scene_frame_dnn_deltas = []
    scene_frame_phash_deltas = []
    vec_prev = None
    im_blank = im_utils.create_blank_im(opt_size_process, opt_size_process)
    im_blank_pil = im_utils.np2pil(im_blank)
    phash_prev = imagehash.phash(im_blank_pil)
    frame_idx = 0

    # write keyframes

    # dnn cfg
    model_cfg = modelzoo.get(pool_item['opt_model_name'])
    if pool_item['opt_gpu']:
      model_cfg.use_gpu()
    else:
      model_cfg.use_cpu()

    # init dnn
    cvmodel = DNNFactory.from_dnn_cfg(model_cfg)
    vec_prev = cvmodel.features(im_blank)

    # read video into opencv and create focal mask
    video = cv.VideoCapture(fp_item)
    #w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))  # unreliable attribute
    #h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))  # unreliable attribute
    frame_ok, frame = video.read()
    if not frame_ok:
      log.error(f'Could not read video: {fp_item}')
      return False

    video.set(cv.CAP_PROP_POS_FRAMES, 0)  # rewind

    frame_sm = im_utils.resize(frame, width=opt_width)
    h,w,c = frame_sm.shape
    im_focal_mask = keyframe_utils.create_focal_mask((w,h))
  
    
    # find keyframes
    while video.isOpened():

      frame_ok, frame = video.read()
      
      if not frame_ok:
        break

      # resize
      frame_sm = im_utils.resize(frame, width=opt_width)

      # focal mask
      if pool_item['use_focal_mask']:
        frame_sm = keyframe_utils.apply_focal_mask(frame_sm, im_focal_mask)

      # phash: every frame
      frame_sm_pil = im_utils.np2pil(frame_sm)
      phash_cur = imagehash.phash(frame_sm_pil)
      phash_per = min((phash_cur - phash_prev) / max_phash_delta, 1.0)
      
      # feature delta
      if phash_per > opt_vec_thresh_solo:
        vec_cur = cvmodel.features(frame_sm)
        cos_result = cosine_similarity([vec_cur, vec_prev])
        vec_per = 1 - cos_result[0][1]
      else:
        vec_per = 0

      # update
      phash_deltas.append(phash_per)
      dnn_deltas.append(vec_per)
          
      # detect change
      if phash_per > opt_phash_thresh or vec_per > opt_vec_thresh_solo \
        or (phash_per > opt_phash_thresh_multi and vec_per > opt_vec_thresh_multi):

        # append frame 
        scene_frame_idxs.append(frame_idx)
        scene_frame_phash_deltas.append(phash_cur)
        scene_frame_dnn_deltas.append(vec_cur)
        
        # reset reference frame metrics
        phash_prev = phash_cur
        vec_prev = vec_cur

      # advance frame index
      frame_idx += 1


    log.debug(f'Found {len(scene_frame_idxs)} keyframes')

    # TODO: deduplicate across dnn metrics
    drop_idxs = []

    # TODO: deduplicate across phash metrics
    drop_idxs = []

    # write metadata.json
    keyframe_meta = asdict(meta)
    keyframe_meta.update({'sha256': sha256})
    keyframe_meta = asdict(dacite.from_dict(data=keyframe_meta, data_class=KeyframeMediaMeta))
    fp_out_json = fp_out = join(fp_dir_out, 'metadata.json')
    file_utils.write_json(keyframe_meta, fp_out_json)
    # seek to frames
    frame_writes_ok = True
    for frame_idx in scene_frame_idxs:
      
      # read frame
      video.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
      frame_ok, frame = video.read()
      
      # save frame
      fn = f'{file_utils.zpad(frame_idx)}.jpg'
      fp_out = join(fp_dir_out, fn)
      try:
        frame_write_ok = cv.imwrite(fp_out, frame)
        if not frame_write_ok:
          frame_writes_ok = False
      except Exception as e:
        log.error(f'Could not write frame {frame_idx} in video {fp_item}')
        frame_writes_ok = False

    return frame_writes_ok



  # -----------------------------------------------------------
  # convert file list into objects

  pool_items = []

  # assume opt_dir_out is our dataset, then ensure keyframes go into a keyframes folder
  if not opt_dir_out:
    opt_dir_out = opt_dir_in
  opt_dir_out = join(opt_dir_out, 'keyframes')

  for fp_item in fp_items:
    o = {
      'fp': fp_item,
      'opt_dir_out': opt_dir_out,
      'opt_model_name': opt_model_enum.name.lower(),
      'opt_gpu': True,
      'opt_size_process': opt_size_process,
      'width_min': opt_width_min,
      'height_min': opt_height_min,
      'use_focal_mask': opt_use_focal_mask,
      'opt_remove_existing': opt_remove_existing,
    }
    pool_items.append(o)

  # Multiprocess/threading use imap instead of map via @hkyi Stack Overflow 41920124
  if opt_threads > 1:
    with Pool(opt_threads) as p:
      pool_results = list(tqdm(p.imap(pool_worker, pool_items), total=len(fp_items)))
  else:
    # If multithreading is disabled, don't use the pool at all
    pool_results = [ pool_worker(item) for item in tqdm(pool_items, total=len(fp_items)) ]

  log.info(f'{len(fp_items)} items processed')
  n_ok = sum(pool_results)
  log.info(f'Wrote keyframes for {n_ok:,}')
  n_nok = len(pool_results) - n_ok
  log.info(f'Did not write keyframes for: {n_nok:,}')