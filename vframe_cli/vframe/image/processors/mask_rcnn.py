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

from vframe.utils import im_utils
from vframe.models.color import Color
from vframe.settings import app_cfg
from vframe.models.geometry import BBox, Point
from vframe.image.processors.base import DetectionProc, NetProc
#from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.models.cvmodels import SegmentResult, SegmentResults


class MaskRCNNProc(NetProc):

  def __post_init__(self):
    self.colors = np.array([Color.random().to_rgb_int() for x in range(len(self.labels))])


  def _pre_process(self, im):
    """Pre-process image
    """
    cfg = self.dnn_cfg
    im = im_utils.resize(im, width=cfg.width, height=cfg.height, force_fit=cfg.fit)
    self.frame_dim = im.shape[:2][::-1]
    dim = self.frame_dim if cfg.fit else cfg.size
    blob = cv.dnn.blobFromImage(im, crop=cfg.crop, swapRB=cfg.rgb)
    self.net.setInput(blob)


  def _post_process(self, outs):
    (boxes, masks) = outs
    self.log.debug(f'boxes: {boxes.shape}, masks: {masks.shape}')
    self.log.debug(f'dim: {self.frame_dim}')
    w, h = self.frame_dim
    detect_results = []

    for i in range(0, boxes.shape[2]):
      class_idx = int(boxes[0, 0, i, 1])
      confidence = boxes[0, 0, i, 2]
      if confidence > self.dnn_cfg.threshold:
        box = boxes[0, 0, i, 3:7] * np.array([w, h, w, h])
        xyxy = box.astype("int")
        bbox_norm = BBox(*xyxy, *self.frame_dim)
        label = self.labels[class_idx] if self.labels else ''
        mask = masks[i, class_idx]
        detect_result = SegmentResult(class_idx, confidence, bbox_norm, mask, label)
        detect_results.append(detect_result)

    if self.dnn_cfg.nms:
      detect_results = self._nms(detect_results, boxes, conf)

    return SegmentResults(detect_results, self._perf_ms())