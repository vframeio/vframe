#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

# -----------------------------------------------------------------------------
#
# MXNET RetinaFace Detector
#
# pip install mxnet-cu102 for CUDA 10.2
# pip install mxnet-cu100 for CUDA 10.0
# pip install mxnet for CPU
# -----------------------------------------------------------------------------

from os.path import join
import time

from vframe.settings import app_cfg
from vframe.settings.app_cfg import LOG
from vframe.models.geometry import BBox, Point
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import DetectResult, DetectResults


class RetinaFaceMXNetProc(DetectionProc):

  class_idx = 0
  label = 'face'
  net_param = 'net3'

  def __init__(self, dnn_cfg):
    self.dnn_cfg = dnn_cfg
    # set filepaths

    #fp_config = join(dnn_cfg.fp_model, dnn_cfg.config)
    #fp_labels = join(dnn_cfg.fp_model, dnn_cfg.labels)
    # keep imports inside init to avoid loading mxnet library externally
    # temp fix: set "cudnn_tune" layers to "off" in params JSON
    #os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    import mxnet
    from vframe.image.processors.utils import insightface_utils
    # instantiate detector model
    self.net = insightface_utils.FaceDetector(dnn_cfg.fp_model, self.net_param, dnn_cfg)
    LOG.debug(f'initializing InsightFace with GPU: {dnn_cfg.device}')
    self.net.prepare(dnn_cfg.device)


  def _pre_process(self, im):
    self.frame_dim = im.shape[:2][::-1]

  def infer(self, im):
    # run detection inference
    self._pre_process(im)
    h,w,c = im.shape
    scale = min(self.dnn_cfg.width / w, self.dnn_cfg.height / h)
    outs = self.net.detect(im, threshold=self.dnn_cfg.threshold, scale=scale)
    results = self._post_process(outs)
    return results


  def _post_process(self, outs):
    """InsightFace RetinaFace mxnet detector
    """

    detections, _landmarks = outs
    detect_results = []

    # convert to DetectionResult
    for detection in detections:
      xyxy = detection[:4]
      conf = detection[4]
      bbox = BBox(*xyxy, *self.frame_dim)
      detect_result = DetectResult(self.class_idx, conf, bbox, self.label)
      detect_results.append(detect_result)

    return DetectResults(detect_results)
