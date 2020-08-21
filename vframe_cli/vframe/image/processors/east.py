############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import math

import cv2 as cv


from vframe.settings import app_cfg
from vframe.utils import im_utils
from vframe.models.bbox import BBoxNorm, BBoxDim, PointNorm, PointDim
from vframe.models.bbox import RotatedBBoxNorm, RotatedBBoxDim
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import RotatedDetectResult, RotatedDetectResults


class EASTProc(DetectionProc):


  def _decode(self, scores, geometry, threshold=0.5):

    # From Satya Mallick @spmallick
    # From https://github.com/spmallick/learnopencv/tree/master/TextDetectionEAST

    detections = []
    confidences = []

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    
    height = scores.shape[2]
    width = scores.shape[3]

    for y in range(0, height):

      # Extract data from scores
      scoresData = scores[0][0][y]
      x0_data = geometry[0][0][y]
      x1_data = geometry[0][1][y]
      x2_data = geometry[0][2][y]
      x3_data = geometry[0][3][y]
      anglesData = geometry[0][4][y]

      for x in range(0, width):
        score = scoresData[x]

        # If score is lower than threshold score, move to next x
        if(score < threshold):
          continue

        # Calculate offset
        offsetX = x * 4.0
        offsetY = y * 4.0
        angle = anglesData[x]

        # Calculate cos and sin of angle
        cosA = math.cos(angle)
        sinA = math.sin(angle)
        h = x0_data[x] + x2_data[x]
        w = x1_data[x] + x3_data[x]

        # Calculate offset
        offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

        # Find points for rectangle
        p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
        p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
        center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
        detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
        confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


  def _post_process(self, outs):
    """Post process net output and return ProcessorResult
    """

    nms_threshold = self.dnn_cfg.nms_threshold
    conf_threshold = self.dnn_cfg.threshold
    results = []
    geometry = outs[0]
    scores = outs[1]
    
    [boxes, confidences] = self._decode(scores, geometry, conf_threshold)
    
    indices = cv.dnn.NMSBoxesRotated(boxes, confidences, conf_threshold, nms_threshold)
    w,h = self.frame_dim

    for i in indices:
      # get 4 corners of the rotated rect
      vertices = cv.boxPoints(boxes[i[0]])
      # output format
      #((163.83588152587978, 66.38432557600203), (132.39764, 20.20161), -0.32462181577778104)
      points = [PointDim(int(x), int(y)) for x,y in vertices]
      rbbox_norm = RotatedBBoxDim(*points, self.frame_dim).to_rbbox_norm()
      bbox_norm = rbbox_norm.to_bbox_norm()
      box_orig = boxes[i[0]][:2]
      x1, y1 = box_orig[0]
      x2, y2 = box_orig[1]
      bbox_norm_orig = BBoxNorm(x1/w, y1/h, x2/w, y2/h)
      angle = boxes[i[0]][0]
      rdr = RotatedDetectResult(0, confidences[i[0]], bbox_norm, rbbox_norm, bbox_norm_orig, angle)
      results.append(rdr)
    
    return RotatedDetectResults(results, 0.0)
