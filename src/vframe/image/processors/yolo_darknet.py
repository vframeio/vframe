############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import numpy as np
import cv2 as cv

from vframe.settings.app_cfg import LOG
from vframe.models.geometry import BBox
from vframe.image.processors.base import Detection
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.utils import im_utils


class YOLODarknet(Detection):

  def _pre_process(self, im):
    """Pre-process image
    """

    cfg = self.dnn_cfg

    if cfg.width % 32:
      wh = int(round(cfg.width / 32)) * 32
    elif cfg.height % 32:
      wh = int(round(cfg.height / 32)) * 32
    else:
      wh = None
    if wh:
      cfg.override(size=(wh, wh))
      self.log.warn(f'YOLO width and height must be multiple of 32. Using width scale to: {wh}')

    self.frame_dim_orig = im.shape[:2][::-1]
    im = im_utils.resize(im, width=cfg.width, height=cfg.height, force_fit=cfg.fit)
    self.frame_dim_resized = im.shape[:2][::-1]
    dim = self.frame_dim_resized if cfg.fit else cfg.size
    blob = cv.dnn.blobFromImage(im, cfg.scale, dim, cfg.mean, crop=cfg.crop, swapRB=cfg.rgb)
    self.net.setInput(blob)

  def _post_process(self, outs):
    """Post process net output for YOLO object detection
    Network produces output blob with a shape NxC where N is a number of
    detected objects and C is a number of classes + 4 where the first 4
    numbers are [center_x, center_y, width, height]
    """

    detect_results = []

    for out in outs:
      out_filtered_idxs = np.where(out[:, 5:] > self.dnn_cfg.threshold)
      out = [out[x] for x in out_filtered_idxs[0]]
      for detection in out:
        scores = detection[5:]
        class_idx = np.argmax(scores)
        confidence = scores[class_idx]
        LOG.debug(confidence)
        if confidence > self.dnn_cfg.threshold:
          cx, cy, w, h = detection[0:4]
          bbox = BBox.from_cxcywh_norm(cx, cy, w, h, *self.frame_dim_orig)
          label = self.labels[class_idx] if self.labels else ''
          LOG.debug(label)
          detect_result = DetectResult(class_idx, confidence, bbox, label)
          detect_results.append(detect_result)

    if self.dnn_cfg.nms:
      detect_results = self._nms(detect_results)

    return DetectResults(detect_results)
