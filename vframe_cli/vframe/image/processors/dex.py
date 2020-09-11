############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import numpy as np
import cv2 as cv

from vframe.settings import app_cfg
from vframe.image.processors.base import ClassificationProc
from vframe.models.cvmodels import ClassifyResult
from vframe.utils import im_utils

class DEXProc(ClassificationProc):


  def _pre_process(self, im):
    """Pre-process image
    """
    
    cfg = self.dnn_cfg
    self.frame_dim_orig = im.shape[:2][::-1]
    im = im_utils.resize(im, width=cfg.width, height=cfg.height, force_fit=cfg.fit)
    self.frame_dim_resized = im.shape[:2][::-1]
    dim = self.frame_dim_resized if cfg.fit else cfg.size
    blob = cv.dnn.blobFromImage(im, cfg.scale, dim, cfg.mean, crop=cfg.crop, swapRB=cfg.rgb)
    self.net.setInput(blob)


  def _post_process(self, outs):
    """Post process net output for DEX object detection
    Network produces output blob with a shape NxC where N is a number of
    detected objects and C is a number of classes + 4 where the first 4
    numbers are [center_x, center_y, width, height]
    """
    preds = outs.flatten()
    age_range = 100
    ages = np.arange(0, age_range + 1).reshape(age_range + 1, 1)
    age = preds.dot(ages).flatten()[0]
    results = [ClassifyResult(int(age), age, 'age')]
    return results