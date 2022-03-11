#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

from os.path import join
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from vframe.models.cvmodels import ClassifyResult, ClassifyResults
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.models.cvmodels import BenchmarkResult
from vframe.utils import file_utils, im_utils
from vframe.settings import app_cfg


class Processor:
  """Basic Net Processor for OpenCV DNN models
  """

  def __init__(self, dnn_cfg):
    """Instantiate an DNN network model
    """
    self.log = app_cfg.LOG
    self.dnn_cfg = dnn_cfg

    # build dnn net
    if self.dnn_cfg.fp_config is None:
      self.net = cv.dnn.readNet(self.dnn_cfg.fp_model)
    else:
      self.net = cv.dnn.readNet(self.dnn_cfg.fp_model, self.dnn_cfg.fp_config)
    if self.dnn_cfg.device > -1:
      # TODO: add options for other opencv cuda backends
      self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
      self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
      # TODO: add options for other opencv backends
      self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
      self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # load class labels
    if self.dnn_cfg.labels_exist:
      self.labels = file_utils.load_txt(dnn_cfg.fp_labels)  # line-delimited list of class labels
    else:
      self.log.debug(f'labels file does not exist: {dnn_cfg.fp_labels}. Using empty label set.')
      self.labels = []

    self.__post_init__()


  def __post_init__(self):
    """Override me
    """
    pass


  def _pre_process(self, im):
    """Pre-process image
    """
    cfg = self.dnn_cfg

    if cfg.width == cfg.height and not cfg.width == cfg.height:
      cfg.width = min(cfg.width, cfg.height)
      cfg.height = cfg.width
      self.log.warning(f'Width and height must be equal. Forcing to lowest size: {cfg.width}')

    self.frame_dim_orig = im.shape[:2][::-1]
    im = im_utils.resize(im, width=cfg.width, height=cfg.height, force_fit=cfg.fit)
    self.frame_dim_resized = im.shape[:2][::-1]
    dim = self.frame_dim_resized if cfg.fit else cfg.size
    blob = cv.dnn.blobFromImage(im, cfg.scale, dim, cfg.mean, crop=cfg.crop, swapRB=cfg.rgb)
    self.net.setInput(blob)


  def _post_process(self, outs):
    """Post process net output and return ProcessorResult
    """
    self.log.error('Override this')
    return ProcessorResult(0, 0.0)


  def features(self):
    """Probably only used for classification networks. Consider removing.
    """
    self.log.error('Override this')
    return np.array()


  def fps(self, n_iters=10, dim=(640,480)):
    """Benchmark model FPS on image
    :param im: (np.ndarray) image
    :pram n_iters: (int) iterations
    """
    im = im_utils.create_random_im(640, 480)
    _ = self.infer(im)  # warmup
    start_time = time.time()
    for i in range(n_iters):
      _ = self.infer(im)
    fps = n_iters / (time.time() - start_time)
    return fps

  def infer(self, im):
    """Runs pre-processor, inference, and post-processor
    """
    self._pre_process(im)
    if self.dnn_cfg.layers:
      outs = self.net.forward(self.dnn_cfg.layers)
    else:
      outs = self.net.forward()
    results = self._post_process(outs)
    return results


  def _perf_ms(self):
    """Returns network forward pass performance time in milliseconds
    """
    t, _ = self.net.getPerfProfile()
    return t * 1000.0 / cv.getTickFrequency()



# -----------------------------------------------------------------------------
#
# Classification networks
#
# -----------------------------------------------------------------------------


class Classification(Processor):
  """Classification inference processor
  """
  limit = 1

  def _post_process(self, outs):
    preds = outs.flatten()
    classify_results = []
    if self.limit > 1:
      idxs = np.argsort(preds)[::-1][:self.limit]  # top N indices
    else:
      idxs = [np.argmax(preds)]
    for idx in idxs:
      if preds[idx] > self.dnn_cfg.threshold:
        classify_result = ClassifyResult(idx, preds[idx], self.labels[idx])
        classify_results.append(classify_result)

    return ClassifyResults(classify_results)


  def features(self, im):
    cfg = self.dnn_cfg
    self._pre_process(im)
    feat_vec = np.squeeze(self.net.forward(cfg.features).flatten())
    return feat_vec / np.linalg.norm(feat_vec)



# -----------------------------------------------------------------------------
#
# Detection base
#
# -----------------------------------------------------------------------------


class Detection(Processor):


  def _nms(self, detect_results):
    """Apply non-maximum suppression and filter detection results
    :param detect_results: List[DetectResult]
    :returns List[DetectResult]
    """
    confidences = [float(d.conf) for d in detect_results]
    boxes = [d.bbox.xywh_int for d in detect_results]
    idxs = cv.dnn.NMSBoxes(boxes, confidences, self.dnn_cfg.threshold, self.dnn_cfg.nms_threshold)
    detect_results_nms = [detect_results[i] for i in idxs]
    return detect_results_nms
