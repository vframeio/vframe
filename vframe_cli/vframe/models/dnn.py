############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from os.path import join
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import logging
from pathlib import Path

from vframe.settings import app_cfg


log = app_cfg.LOG


@dataclass
class DNN:
  name: str  # descriptive name of model
  processor: str  # type of processor
  output: str  # type of output
  local: str  # root directory for model files
  model: str  # model filename
  remote: str=None  # root directory for model files
  width: int=None  # width of image tensor/blob
  height: int=None  # height of image tensor/blob
  fit: bool=True  # force fit image to exact width and height\
  allow_resize: bool=False
  # model file locations
  config: str=''  # filename prototxt, pbtxt, .cfg, etc
  labels: str = ''  # filename path to labels.txt line-delimeted
  # preprocessing
  mean: List[float] = field(default_factory=lambda: [])
  scale: float=0.0
  rgb: bool=True
  crop: bool=False
  # processing
  features: str=None  # name of layer to extract embeddings from
  layers: List[str] = field(default_factory=lambda: [])
  backend: str='DEFAULT'  # cv.dnn backend name
  target: str='DEFAULT'  # cv.dnn target name
  gpu: bool=False  # use gpu or cpu
  algorithm: str=None  # additional param for non cv.dnn nets
  scale_factor: int=None   # super resolution scale factor
  dimensions: int=None
  # post-processing
  threshold: float=0.8  # detection confidence threshold
  nms: bool = False  # use non-maximum suppression
  nms_threshold: float=0.4  # nms threshold
  # metadata
  credit: str=''  # how credit should be displayed
  repo: str=''  # author/repo URL
  license: str=''  # filepath to license
  license_tag: str=''  # eg "mit", see https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository#choosing-the-right-license

  model_exists: bool=False
  config_exists: bool=False
  labels_exist: bool=False

  def __post_init__(self):
    self.log = app_cfg.LOG
    if not self.local[0] == '/':
      self.local = join(app_cfg.DIR_PROJECT_ROOT, self.local)

    # Check if files exist locally
    self.model_exists = Path(join(self.local, self.model)).is_file()
    self.config_exists = Path(join(self.local, self.config)).is_file()
    self.labels_exist = Path(join(self.local, self.labels)).is_file()
    self.license_exists = Path(join(self.local, self.license)).is_file()

    # model
    self.fp_model = join(self.local, self.model)
    self.url_model = join(self.remote, self.model)

    # config
    if self.config:
      self.fp_config = join(self.local, self.config)
      self.url_config = join(self.remote, self.config)
    else:
      self.fp_config = None

    # labels
    if self.labels:
      self.fp_labels = join(self.local, self.labels)
      self.url_labels = join(self.remote, self.labels)
    else:
      self.fp_labels = None

    # labels
    if self.license:
      self.fp_license = join(self.local, self.license)
      self.url_license = join(self.remote, self.license)
    else:
      self.fp_license = None


  def override(self, gpu=None, size=(None, None), threshold=None):
    if gpu is not None:
      if gpu:
        self.use_gpu()
      else:
        self.use_cpu()
    if all(size):
      if not self.allow_resize:
        log.warn(f'Resizing DNN input size not permitted for this model')
      else:
        self.width, self.height = size
    if threshold is not None:
      self.threshold = threshold


  def use_gpu(self):
    self.backend = 'CUDA'
    self.target = 'CUDA'


  def use_cpu(self):
    self.backend = 'DEFAULT'
    self.target = 'DEFAULT'

      
  @property
  def size(self):
    return (self.width, self.height)
  
  @property
  def dnn_backend(self):
    if 'CUDA' in self.backend and not app_cfg.CUDA_ENABLED:
      self.log.error('GPU was selected but CUDA not enabled. Using CPU. Try Docker GPU?')
    backend = app_cfg.dnn_backends.get(self.backend.upper(), app_cfg.dnn_backends.get('DEFAULT'))
    #app_cfg.LOG.debug(f'backend: {backend}')
    return backend

  @property
  def dnn_target(self):
    if 'CUDA' in self.backend and not app_cfg.CUDA_ENABLED:
      self.log.error('GPU was selected but CUDA not enabled. Using CPU. Try Docker GPU?')
    target = app_cfg.dnn_targets.get(self.target.upper(), app_cfg.dnn_targets.get('DEFAULT'))
    #app_cfg.LOG.debug(f'target: {target}')
    return target
