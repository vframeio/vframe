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
from vframe.models.geometry import BBox, Point
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import HumanPoseDetectResult, HumanPoseDetectResults


class HumanPoseProc(DetectionProc):


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


  def _post_process(self, outs):
    """Post process net output and return ProcessorResult
    """
    results = []
    pose_result = HumanPoseDetectResult(0, 1.0, bbox_norm, pose_keypoints)
    results.append(rdr)
    
    return HumanPoseDetectResults(results)
