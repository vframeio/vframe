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
from vframe.models.bbox import BBoxNorm, BBoxDim
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import DetectResult, DetectResults


class MobileNetMXNetProc(DetectionProc):

  class_idx = 0
  label = 'face'
  net_param = 'net3l'

  def __init__(self, dnn_cfg):
    self.log = app_cfg.LOG
    self.dnn_cfg = dnn_cfg
    # set filepaths
    fp_model = join(app_cfg.DIR_MODELS, dnn_cfg.model)
    fp_config = join(app_cfg.DIR_MODELS, dnn_cfg.config)
    fp_labels = join(app_cfg.DIR_MODELS, dnn_cfg.labels)
    # keep imports inside init to avoid loading mxnet library externally
    # temp fix: set "cudnn_tune" layers to "off" in params JSON
    #os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    import mxnet
    from vframe.image.processors.utils import insightface_mobilenet
    # instantiate detector model
    self.net = insightface_utils.FaceDetector(fp_model, self.net_param, dnn_cfg)
    self.log.debug(f'initializing InsightFace with GPU index: {dnn_cfg.gpu}')
    self.net.prepare(dnn_cfg.gpu)


  def _pre_process(self, im):
    self.frame_dim = im.shape[:2][::-1]

  def infer(self, im):
    # run detection inference
    self._pre_process(im)
    start_time = time.time()
    outs = self.net.detect(im, threshold=self.dnn_cfg.threshold)
    perf_ms = time.time() - start_time
    results = self._post_process(outs, perf_ms)
    return results
    

  def _post_process(self, outs, perf_ms):
    """InsightFace RetinaFace mxnet detector
    """

    detections, _landmarks = outs
    detect_results = []

    # convert to DetectionResult
    for detection in detections:
      xyxy = detection[:4]
      conf = detection[4]
      bbox_norm = BBoxDim.from_xyxy_dim(xyxy, self.frame_dim).to_bbox_norm()
      detect_result = DetectResult(self.class_idx, conf, bbox_norm, self.label)
      detect_results.append(detect_result)
    
    return DetectResults(detect_results, perf_ms)

