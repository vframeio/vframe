############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import time

import cv2 as cv

from vframe.settings import app_cfg
from vframe.models.geometry import BBox, Point
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.image.processors.utils.pose_utils import mk_valid_pairs, mk_personwise_keypoints, pose_to_bbox
from vframe.image.processors.utils.pose_utils import mk_detected_keypoints, keypoints_to_persons, persons_to_bboxes

class COCOPoseFaceProc(DetectionProc):

  opt_blur_kernel = 7
  #min_wh = (24, 24)
  min_wh = (30, 30)

  def _pre_process(self, im):
    """Pre-process image
    """
    
    cfg = self.dnn_cfg
    self.frame_dim_orig = im.shape[:2][::-1]
    self.frame_dim_resized = im.shape[:2][::-1]
    dim = self.frame_dim_orig
    w = int((cfg.height / dim[1])*dim[0])
    blob = cv.dnn.blobFromImage(im, cfg.scale, (w, cfg.height), cfg.mean, crop=cfg.crop, swapRB=cfg.rgb)
    self.net.setInput(blob)


  def infer(self, im):
    # run detection inference
    self._pre_process(im)
    start_time = time.time()
    outs = self.net.forward()
    results = self._post_process(outs)
    return results


  def _post_process(self, output):
    """Post process net output for object detection
    """

    detect_results = []
    class_idx = 0
    label = self.labels[class_idx] if self.labels else ''
    dim = self.frame_dim_orig

    detected_keypoints, keypoints_list = mk_detected_keypoints(output, dim, 
      threshold=self.dnn_cfg.threshold, k=self.opt_blur_kernel)
    valid_pairs, invalid_pairs = mk_valid_pairs(output, dim, detected_keypoints)
    personwise_keypoints = mk_personwise_keypoints(valid_pairs, invalid_pairs, keypoints_list)
    persons = keypoints_to_persons(detected_keypoints, personwise_keypoints)
    bboxes = persons_to_bboxes(persons, dim)
    n_persons = len(personwise_keypoints)

    # TODO: average confidence for each keypoint
    confidence = 1.0

    for bbox in bboxes:
      if bbox.width > self.min_wh[0] or bbox.height > self.min_wh[1]:
        detect_result = DetectResult(class_idx, confidence, bbox, label)
        detect_results.append(detect_result)

    return DetectResults(detect_results)
