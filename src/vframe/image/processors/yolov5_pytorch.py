#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################


import os
from os.path import join
import math
import time
from pathlib import Path
import logging
import random

import torch

from vframe.settings.app_cfg import LOG, DIR_3RDPARTY
from vframe.models.geometry import BBox
from vframe.utils.im_utils import np2pil, create_random_im
from vframe.image.processors.base import Detection
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.utils.file_utils import load_txt


class YOLOV5PyTorch(Detection):

  def __init__(self, cfg):
    """Instantiate an DNN network model
    """
    self.cfg = cfg
    dp_yolo = join(DIR_3RDPARTY, 'yolov5')
    if Path(dp_yolo).is_dir():
      source = 'local'
    else:
      dp_yolo = 'ultralytics/yolov5'
      source = 'github'
    device = 'cpu' if cfg.device == -1 else f'cuda:{cfg.device}'

    try:
      # verbose var is not used if source is local
      # but can override the yolov5 default logger by creating it first
      logging.basicConfig(format="%(message)s",level=logging.CRITICAL)
    except Exception as e:
      pass

    self.model = torch.hub.load(dp_yolo, 'custom', 
        path=cfg.fp_model, 
        device=device,
        verbose=False,
        source=source)
    try:
      logging.getLogger("").handlers.clear()
    except Exception as e:
      pass
    self.model.half = False  # to FP16
    self.model.conf = cfg.threshold
    self.model.iou = cfg.iou

    # load labels
    self.labels = load_txt(cfg.fp_labels) if cfg.labels_exist else []
    if not self.labels:
      LOG.debug(f'No labels or missing file: {cfg.fp_labels}')


  def fps_batch(self, n_iters=10, dim=(640,480), batch_size=12):
    """Benchmark model FPS on image
    :param im: (np.ndarray) image
    :pram n_iters: (int) iterations
    """
    ims = [create_random_im(*dim) for i in range(batch_size)]
    _ = self.infer(random.choice(ims))  # warmup
    st = time.perf_counter()
    for i in range(n_iters):
      _ = self.infer(ims)
    fps = batch_size * n_iters / (time.perf_counter() - st)
    return fps
    

  def infer(self, ims):
    """Runs pre-processor, inference, and post-processor
    This processing uses batch inference
    """
    if not isinstance(ims, list):
      ims = [ims]
    if not ims:
      return []
    self.dim = ims[0].shape[:2][::-1]
    
    ims_batch = [np2pil(im) for im in ims]
    dims = [im.size for im in ims_batch]
    batch_outputs = self.model(ims_batch, size=self.cfg.width)
    return self._post_process(batch_outputs, dims)


  def _post_process(self, outputs, dims):
    """Convert to list DetectResults
    """
    batch_dets = []
    for batch_idx, idx_outputs in enumerate(outputs.pred):
      idx_dets = []
      idx_outputs = idx_outputs.tolist()
      if not idx_outputs:
        batch_dets.append(DetectResults(idx_dets))  # empty results
        continue
      for idx_output in idx_outputs:
        if idx_output:
          bbox = BBox(*idx_output[:4], *dims[batch_idx])
          conf, class_idx = (float(idx_output[4]), int(idx_output[5]))
          det = DetectResult(class_idx, conf, bbox, self.labels[class_idx])
          idx_dets.append(det)

      batch_dets.append(DetectResults(idx_dets))

    return batch_dets