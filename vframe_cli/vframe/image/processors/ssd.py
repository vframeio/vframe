############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from vframe.settings import app_cfg
from vframe.models.bbox import BBoxNorm, BBoxDim
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import DetectResult, DetectResults

class SSDProc(DetectionProc):

  def _post_process(self, outs):
    """Post process net output for object detection
    """

    detect_results = []

    for detection in outs[0, 0]:
      confidence = float(detection[2])
      if confidence > self.dnn_cfg.threshold:
        class_idx = int(detection[1])  # skip background ?
        (x1,y1,x2,y2) = detection[3:7]
        bbox_norm = BBoxNorm(x1,y1,x2,y2)
        label = self.labels[class_idx] if self.labels else ''
        detect_result = DetectResult(class_idx, confidence, bbox_norm, label)
        detect_results.append(detect_result)

    if self.dnn_cfg.nms:
      detect_results = self._nms(detect_results, boxes, conf)

    return DetectResults(detect_results, self._perf_ms())
