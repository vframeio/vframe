
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
  default=0.75, type=click.FloatRange(0,1),
  help='Skip frames above this perceptual similar. Higher number means skip fewer frames. Lower number skip more frames. 0.0: completely different, 1.0: exactly same')
@click.option('--all-frames/--last-frame', 'opt_all_frames', is_flag=True,
  help='Compare with all previous frames or only last/previous frame.')
@click.option('--prehash', 'opt_prehash', is_flag=True,)
@click.option('--override', 'opt_override', is_flag=True)
@click.option('--roi', 'opt_use_roi', is_flag=True,
  help='Use BBox ROI as phash source')
@click.option('--roi-label', 'opt_roi_labels', multiple=True, default=[],
  help='Labels to use for ROI phash')
@processor
@click.pass_context
def cli(ctx, sink, opt_thresh, opt_all_frames, opt_prehash, opt_override,
  opt_use_roi, opt_roi_labels):
  """Skip similar frames using perceptual hash"""
  
  from pathlib import Path
  from PIL import Image
  import cv2 as cv
  import numpy as np
  import imagehash

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_PREHASH, SKIP_FILE
  from vframe.settings.app_cfg import USE_DRAW_FRAME
  from vframe.utils.im_utils import resize, np2pil, phash, create_blank_im, crop_roi
  from vframe.utils.file_utils import ensure_dir
  from vframe.models.types import FrameImage, MediaType
  from vframe.models.geometry import BBox

  cur_file = None
  cur_subdir = None

  # init perceptual hash
  opt_thresh = 1.0 - opt_thresh  # invert
  hash_thresh_int = opt_thresh * 32  # length of imagehash

  # blank image to init
  hash_wh = 32  # check im_utils
  im_blank = create_blank_im(hash_wh, hash_wh)
  hash_pre = phash(im_blank)
  hashes = [hash_pre]

  ctx.obj[USE_PREHASH] = opt_prehash


  def isolate_roi(im, meta, dim):
    bboxes = []
    data_keys = list(meta.keys())
    for data_key in data_keys:
      # get detection metadata
      item_data = meta.get(data_key)
      if item_data:
        for detection in item_data.detections:
          bboxes.append(detection.bbox.redim(dim))
    if not bboxes:
      return im
    
    # create blank im
    im_roi = create_blank_im(*dim)
    
    # paste regions
    for bbox in bboxes:
      x1,y1,x2,y2 = bbox.xyxy_int
      im_roi[y1:y2, x1:x2] = im[y1:y2, x1:x2]

    # merge bboxes and crop
    x1 = min([bbox.x1_int for bbox in bboxes])
    y1 = min([bbox.y1_int for bbox in bboxes])
    x2 = max([bbox.x2_int for bbox in bboxes])
    y2 = max([bbox.y2_int for bbox in bboxes])
    roi = BBox(x1,y1,x2,y2, *dim)
    return crop_roi(im_roi, roi)


  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj.get(SKIP_FRAME) or ctx.obj.get(SKIP_FILE):
      sink.send(M)
      continue

    # -------------------------------------------------------------------------
    # check for new media

    if (M.type == MediaType.VIDEO and cur_file != M.filepath) \
    or (M.type == MediaType.IMAGE and cur_subdir != Path(M.filepath).parent):
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
      dim = im.shape[:2][::-1]
      if opt_use_roi:
        im_roi = isolate_roi(im, M.metadata[M.index],dim)
        hash_cur = phash(im_roi)
      else:
        hash_cur = phash(im)

    hash_changed = (hash_cur - hash_pre) > hash_thresh_int
    # LOG.debug(f'hash diff: {hash_cur - hash_pre}, thresh: {hash_thresh_int}')
    
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
      ctx.obj[SKIP_FRAME] = skip
    else:
      ctx.obj[SKIP_FRAME] = (ctx.obj[SKIP_FRAME] or skip)

    if skip and M.type == MediaType.IMAGE:
      ctx.obj[SKIP_FILE] = True
    
    sink.send(M)