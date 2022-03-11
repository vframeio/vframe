#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

from vframe.settings import app_cfg
from vframe.models.geometry import BBox, Point
from vframe.image.processors.base import Detection
from vframe.models.cvmodels import DetectResult, DetectResults

class SSD(Detection):

  def _post_process(self, outs):
    """Post process net output for object detection
    """

    detect_results = []
    
    for detection in outs[0, 0]:
      confidence = float(detection[2])
      if confidence > self.dnn_cfg.threshold:
        class_idx = int(detection[1])  # skip background ?
        xyxy = detection[3:7]
        bbox = BBox.from_xyxy_norm(*xyxy, *self.frame_dim_orig)
        label = self.labels[class_idx] if self.labels else ''
        detect_result = DetectResult(class_idx, confidence, bbox, label)
        detect_results.append(detect_result)

    if self.dnn_cfg.nms:
      detect_results = self._nms(detect_results)

    return DetectResults(detect_results)
