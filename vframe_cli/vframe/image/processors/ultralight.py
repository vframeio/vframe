############################################################################# 
#
# https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
#
#############################################################################

"""
MIT License

Copyright (c) 2019 linzai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import numpy as np
import cv2 as cv

from vframe.settings import app_cfg
from vframe.models.bbox import BBoxNorm, BBoxDim
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import DetectResult, DetectResults


class UltralightRetinaFaceProc(DetectionProc):

  image_std = 128.0
  iou_threshold = 0.3
  center_variance = 0.1
  size_variance = 0.2
  min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
  strides = [8.0, 16.0, 32.0, 64.0]


  def _define_img_size(self, image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
      feature_map = [int(math.ceil(size / stride)) for stride in self.strides]
      feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(self.strides)
    priors = self._generate_priors(feature_map_w_h_list, shrinkage_list, image_size, self.min_boxes)
    return priors


  def _generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for index in range(0, len(feature_map_list[0])):
      scale_w = image_size[0] / shrinkage_list[0][index]
      scale_h = image_size[1] / shrinkage_list[1][index]
      for j in range(0, feature_map_list[1][index]):
        for i in range(0, feature_map_list[0][index]):
          x_center = (i + 0.5) / scale_w
          y_center = (j + 0.5) / scale_h

          for min_box in min_boxes[index]:
            w = min_box / image_size[0]
            h = min_box / image_size[1]
            priors.append([x_center, y_center, w,h])
    print("priors nums:{}".format(len(priors)))
    return np.clip(priors, 0.0, 1.0)


  def _hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = self._iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


  def _area_of(self, left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


  def _iou_of(self, boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = self._area_of(overlap_left_top, overlap_right_bottom)
    area0 = self._area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = self._area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


  def _predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
      probs = confidences[:, class_index]
      mask = probs > prob_threshold
      probs = probs[mask]
      if probs.shape[0] == 0:
        continue
      subset_boxes = boxes[mask, :]
      box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
      box_probs = self._hard_nms(box_probs,
                           iou_threshold=iou_threshold,
                           top_k=top_k,
                           )
      picked_box_probs.append(box_probs)
      picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
      return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    labels = [self.labels[x - 1] for x in picked_labels]
    return picked_box_probs[:, :4].astype(np.int32), labels, picked_box_probs[:, 4]


  def _convert_locations_to_boxes(self, locations, priors, center_variance, size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]], axis=len(locations.shape) - 1)


  def _center_form_to_corner_form(self, locations):
    return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                           locations[..., :2] + locations[..., 2:] / 2], 
                           len(locations.shape) - 1)


  def __post_init__(self):
    self.log.debug(f'post init: {self.dnn_cfg.size}')
    self.priors = self._define_img_size(list(self.dnn_cfg.size))


  def _post_process(self, outs):
    class_idx = 0
    boxes, scores = outs
    boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
    scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
    boxes = self._convert_locations_to_boxes(boxes, self.priors, self.center_variance, self.size_variance)
    boxes = self._center_form_to_corner_form(boxes)
    w,h = self.frame_dim
    boxes, labels, probs = self._predict(w, h, scores, boxes, self.dnn_cfg.threshold)

    detect_results = []
    for box, label, prob in zip(boxes, labels, probs):
      bbox_norm = BBoxDim(*box, self.frame_dim).to_bbox_norm()
      detect_results.append(DetectResult(class_idx, prob, bbox_norm, label))

    return DetectResults(detect_results, self._perf_ms())










