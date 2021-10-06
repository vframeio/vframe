############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils.click_utils import processor, show_help

@click.command('')
@click.option('--type', 'opt_type', 
  default='phash', type=click.Choice(['phash', 'ahash', 'dhash']),
  help='Hashing function')
@click.option('-t', '--threshold', 'opt_thresh', 
  default=0.035, type=click.FloatRange(0,1),
  help='Hash threshold')
@click.option('--all-frames/--last-frame', 'opt_all_frames', is_flag=True)
@click.option('--prehash', 'opt_prehash', is_flag=True,)
@processor
@click.pass_context
def cli(ctx, sink, opt_thresh, opt_type, opt_all_frames, opt_prehash):
  """Skip similar frames using perceptual hash"""
  
  from pathlib import Path
  from PIL import Image
  import cv2 as cv
  import numpy as np
  import imagehash

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_PHASH
  from vframe.settings.app_cfg import USE_DRAW_FRAME
  from vframe.utils.im_utils import resize, np2pil
  from vframe.models.types import FrameImage, MediaType

  cur_file = None
  cur_subdir = None

  # init perceptual hash
  hash_functions = {
    'ahash': imagehash.average_hash,
    'phash': imagehash.phash,
    'dhash': imagehash.dhash
  }
  hasher = hash_functions.get(opt_type)
  # blank image to init
  im_blank = Image.new('RGB', (100,100), (0,0,0))
  hash_pre = imagehash.phash(im_blank)
  hashes = [hash_pre]
  hash_size = 8
  highfreq_factor = 4
  hash_im_size = hash_size * highfreq_factor
  hash_thresh_int = opt_thresh * 64

  ctx.obj[USE_PHASH] = opt_prehash


  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME]:
      sink.send(M)
      continue

    # -------------------------------------------------------------------------
    # init

    if (M.type == MediaType.VIDEO and cur_file != M.filepath) or \
      (M.type == MediaType.IMAGE and cur_subdir != Path(M.filepath).parent):
      # reset hashes
      im_blank = Image.new('RGB', (100,100), (0,0,0))
      hash_pre = imagehash.phash(im_blank)
      del hashes
      hashes = [hash_pre]
      cur_file = M.filepath
      cur_subdir = Path(M.filepath).parent

    hash_changed = False

    # -------------------------------------------------------------------------
    # perceptual hash thresholding

    if opt_prehash:
      hash_cur = M.phash
    else:
      im = M.images.get(FrameImage.ORIGINAL)
      im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
      im = resize(im, width=hash_im_size, height=hash_im_size, force_fit=True)
      hash_cur = hasher(np2pil(im))

    hash_changed = (hash_cur - hash_pre) > hash_thresh_int
    
    if hash_changed and opt_all_frames:
      hash_changed = all([abs(hash_cur - x) > hash_thresh_int for x in hashes])
      if hash_changed:
        hashes.append(hash_cur)
    if hash_changed:
      hash_pre = hash_cur


    # -------------------------------------------------------------------------
    # set flag

    ctx.opts[SKIP_FRAME] = not hash_changed
    
    sink.send(M)