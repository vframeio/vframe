#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

import onnxruntime
import torchvision  # explicit import to avoid nms error
import torch
import numpy as np
import cv2 as cv

from vframe.image.processors.utils.yolov5_utils import letterbox, non_max_suppression
from vframe.settings.app_cfg import LOG
from vframe.models.geometry import BBox
from vframe.image.processors.base import Detection
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.utils.file_utils import load_txt
from vframe.utils.im_utils import resize


class YOLOV5ONNX(Detection):

  def __init__(self, dnn_cfg):
    """Instantiate an ONNX DNN network model
    """
    self.dnn_cfg = dnn_cfg
    cfg = self.dnn_cfg

    # model
    providers = ['CUDAExecutionProvider'] if cfg.device > -1 else ['CPUExecutionProvider']
    self.model = onnxruntime.InferenceSession(cfg.fp_model, providers=providers)
    self.input_name = self.model.get_inputs()[0].name
    self.dim_model = self.model.get_inputs()[0].shape[2:4][::-1]
    self.batch_size = self.model.get_inputs()[0].shape[0]

    # load class labels
    if cfg.labels_exist:
      self.labels = load_txt(cfg.fp_labels)  # line-delimited class labels
    else:
      LOG.debug(f'Labels file missing: {cfg.fp_labels}')
      self.labels = []


  def _pre_process(self, im):
    """Pre-process image
    :param im: (numpy.ndarray) in BGR
    """
    # resize image using letterbox
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)  # ONNX is using RGB
    if im.shape[2] < self.dnn_cfg.width:
      im = resize(im, width=self.dnn_cfg.width, interp=cv.INTER_AREA)
    self.im_preproc, ratio, self.pad = letterbox(im, new_shape=self.dim_model, auto=False, scaleup=False)
    self.im_preproc = np.transpose(self.im_preproc, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    self.im_preproc = np.expand_dims(self.im_preproc, axis=0)
    self.im_preproc /= 255.0
    # image dims
    self.dim_orig = im.shape[:2][::-1]
    self.dim_resized = self.dim_model
    self.dim_letterbox = (self.dim_resized[0], self.dim_resized[1] - self.pad[1]*2)


  def infer(self, im):
    """Runs pre-processor, inference, and post-processor
    :param im: (numpy.ndarry) image in BGR
    :returns (ProcessorResult)
    """
    self._pre_process(im)
    outputs = self.model.run(None, {self.input_name: self.im_preproc})
    results = self._post_process(outputs)
    return results


  def _post_process(self, outputs):
    """Post process net output for YOLOV5 object detection.
    Post-process outputs of ONNX model inference into DetectResults
    Code based on YOLOV5 Ultralytics by Glenn Jocher
    :param outputs: ONNX model inference output
    :returns (DetectResult): of object detection resultss
    """

    detect_results = []

    detections = torch.from_numpy(np.array(outputs[0]))
    detections = non_max_suppression(detections,
        conf_thres=self.dnn_cfg.nms_threshold,
        iou_thres=self.dnn_cfg.iou)

    # benchmark
    if detections[0] is not None:
      outputs = np.array([(d.tolist()) for d in detections[0]])

      for output in outputs:
        output = output.tolist()
        conf = output[4]
        if conf >= self.dnn_cfg.threshold:
            xyxy = list(map(int, output[:4]))
            bbox = BBox(*xyxy, *self.dim_resized).translate(0, -self.pad[1]).to_dim(self.dim_letterbox).redim(self.dim_orig)
            class_idx = int(output[5])
            label = self.labels[class_idx]
            detect_results.append(DetectResult(class_idx, conf, bbox, label))

    return DetectResults(detect_results)
