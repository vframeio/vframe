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
@click.option('-t', '--threshold', 'opt_thresh', 
  default=0.05, type=click.FloatRange(0,1),
  help='Skip frames if similarity below threshold. Higher skips more frames.')
@click.option('--all-frames/--last-frame', 'opt_all_frames', is_flag=True)
@click.option('--prehash', 'opt_prehash', is_flag=True,)
@click.option('--override', 'opt_override', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_thresh, opt_all_frames, opt_prehash, opt_override):
  """Skip similar frames using perceptual hash"""
  
  from pathlib import Path
  from PIL import Image
  import cv2 as cv
  import numpy as np
  import imagehash

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_PREHASH, SKIP_FILE
  from vframe.settings.app_cfg import USE_DRAW_FRAME
  from vframe.utils.im_utils import resize, np2pil, phash, create_blank_im
  from vframe.models.types import FrameImage, MediaType

  cur_file = None
  cur_subdir = None

  # init perceptual hash
  hash_thresh_int = opt_thresh * 64  # length of imagehash

  # blank image to init
  hash_wh = 32  # check im_utils
  im_blank = create_blank_im(hash_wh, hash_wh)
  hash_pre = phash(im_blank)
  hashes = [hash_pre]

  ctx.obj[USE_PREHASH] = opt_prehash


  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts.get(SKIP_FRAME) or ctx.opts.get(SKIP_FILE):
      sink.send(M)
      continue

    # -------------------------------------------------------------------------
    # check for new media

    if (M.type == MediaType.VIDEO and cur_file != M.filepath) or \
      (M.type == MediaType.IMAGE and cur_subdir != Path(M.filepath).parent):
      # new file, reset hashes
      im_blank = create_blank_im(hash_wh, hash_wh)
      hash_pre = phash(im_blank)
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
      hash_cur = phash(im)

    hash_changed = (hash_cur - hash_pre) > hash_thresh_int
    
    if hash_changed and opt_all_frames:
      hash_changed = all([abs(hash_cur - x) > hash_thresh_int for x in hashes])
      if hash_changed:
        hashes.append(hash_cur)
    if hash_changed:
      hash_pre = hash_cur


    # -------------------------------------------------------------------------
    # skip frame if hash has not changed (ie same looking frame)

    skip = not hash_changed
    if opt_override:
      ctx.opts[SKIP_FRAME] = skip
    else:
      ctx.opts[SKIP_FRAME] = (ctx.opts[SKIP_FRAME] or skip)
    
    sink.send(M)