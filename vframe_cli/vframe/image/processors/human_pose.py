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
from vframe.models.cvmodels import HumanPoseDetectResult, HumanPoseDetectResults


class HumanPoseProc(DetectionProc):



  def _post_process(self, outs):
    """Post process net output and return ProcessorResult
    """
    results = []
    pose_result = HumanPoseDetectResult(0, 1.0, bbox_norm, pose_keypoints)
    results.append(rdr)
    
    return HumanPoseDetectResults(results, 0.0)
