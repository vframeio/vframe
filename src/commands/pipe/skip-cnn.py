############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.models.types import ModelZooClickVar, ModelZoo
from vframe.utils.click_utils import processor, show_help

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  default=None,
  type=ModelZooClickVar,
  help=show_help(ModelZoo))
@click.option('-t', '--threshold', 'opt_threshold', default=0.25, type=click.FloatRange(0,1),
  help='Number of frames to decimate/skip for every frame read (0.05)')
@click.option('--all-frames/--last-frame', 'opt_all_frames', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_model_enum, opt_threshold, opt_all_frames):
  """Skip similar frames using CNN features"""
  
  from pathlib import Path
  from PIL import Image
  import cv2 as cv
  import numpy as np
  from sklearn.metrics.pairwise import cosine_similarity

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, modelzoo
  from vframe.utils.im_utils import resize, np2pil
  from vframe.models.types import FrameImage, MediaType
  from vframe.image.dnn_factory import DNNFactory

  cur_file = None
  cur_subdir = None    

  # init cnn
  if opt_model_enum:
    dnn_cfg = modelzoo.get(opt_model_enum.name.lower())
    # dnn_cfg.override(device=opt_device, size=opt_dnn_size, threshold=opt_dnn_threshold)
    cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)
    feat_pre = np.zeros(dnn_cfg.dimensions)
    features = [feat_pre]
    feat_thresh = 1.0 - opt_threshold


  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    # -------------------------------------------------------------------------
    # init

    if (M.type == MediaType.VIDEO and cur_file != M.filepath) or \
      (M.type == MediaType.IMAGE and cur_subdir != Path(M.filepath).parent):
      # rest features
      feat_pre = np.zeros(dnn_cfg.dimensions)
      del features
      features = [feat_pre]
      cur_file = M.filepath
      cur_subdir = Path(M.filepath).parent

    feat_changed = False
    
    # -------------------------------------------------------------------------
    # cnn embeddings

    im = M.images.get(FrameImage.ORIGINAL)
    feat_cur = cvmodel.features(im)
    # comppare one-to-one feature vectors
    feat_sim = cosine_similarity([feat_cur], [feat_pre])[0][0]  # 1.0 = same frame
    feat_changed = feat_sim < feat_thresh
    if feat_changed and opt_all_frames:
      # compare current to all previous feature vectors
      scores = cosine_similarity([feat_cur], features)[0]
      feat_changed = all(scores < np.array([feat_thresh]))
      if feat_changed:
        features.append(feat_cur)
    if feat_changed:
      feat_pre = feat_cur

    # -------------------------------------------------------------------------
    # set flag

    ctx.obj[SKIP_FRAME] = not feat_changed    
    sink.send(M)