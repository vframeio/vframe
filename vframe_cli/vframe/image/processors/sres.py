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
from cv2 import dnn_superres

from vframe.models.cvmodels import BenchmarkResult
from vframe.utils import file_utils, im_utils
from vframe.settings import app_cfg


class SuperResolution:
  """Basic SuperResolution processor for OpenCV DNN models
  """

  def __init__(self, dnn_cfg):
    """Instantiate an OpenCV DNN Super Resolution
    """
    self.log = app_cfg.LOG
    self.dnn_cfg = dnn_cfg

    # build dnn net
    self.net = dnn_superres.DnnSuperResImpl_create()
    self.net.readModel(dnn_cfg.model_local)
    self.net.setModel(dnn_cfg.algorithm, dnn_cfg.scale_factor)
    
    # set backend
    self.net.setPreferableBackend(dnn_cfg.dnn_backend)
    self.net.setPreferableTarget(dnn_cfg.dnn_target)
    
    self.__post_init__()


  def __post_init__(self):
    """Override in subclass
    """
    pass
    

  def upsample(self, im):
    """Run upsampling algorithm
    """
    return self.net.upsample(im)


  def _perf_ms(self):
    """FIXME: remove? Returns network forward pass performance time in milliseconds
    """
    t, _ = self.net.getPerfProfile()
    return t * 1000.0 / cv.getTickFrequency()


  def fps(self, im=None, n_iters=10):
    """Benchmark model FPS on image
    :param im: (np.ndarray) image
    :pram n_iters: (int) iterations
    """
    if im is None:
      im = im_utils.create_blank_im(640, 480)
    _ = self.upsample(im)  # warmup
    start_time = time.time()
    for i in range(n_iters):
      _ = self.upsample(im)
    fps = n_iters / (time.time() - start_time)
    return fps
