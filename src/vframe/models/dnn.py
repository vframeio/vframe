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

from vframe.models.color import Color

LOG = logging.getLogger('VFRAME')

@dataclass
class DNN:
  name: str  # descriptive name of model
  processor: str  # type of processor
  output: str  # type of output
  local: str  # root directory for model files
  model: str  # model filename
  dp_models: str  # path to modelzoo/
  remote: str=None  # root directory for model files
  width: int=None  # width of image tensor/blob
  height: int=None  # height of image tensor/blob
  fit: bool=True  # force fit image to exact width and height\
  resize_enabled: bool=False
  # model file locations
  config: str=''  # filename prototxt, pbtxt, .cfg, etc
  labels: str = 'labels.txt'  # filename path to labels.txt line-delimeted
  colors: str='colors.txt'  # line-delimeted hex colors #FF0000
  # preprocessing
  mean: List[float] = field(default_factory=lambda: [])
  scale: float=0.0
  rgb: bool=True
  crop: bool=False
  # processing
  features: str=None  # name of layer to extract embeddings from
  layers: List[str] = field(default_factory=lambda: [])
  device: bool=False  # use gpu or cpu
  algorithm: str=None  # additional param for non cv.dnn nets
  scale_factor: int=None   # super resolution scale factor
  dimensions: int=None
  # post-processing
  threshold: float=0.8  # detection confidence threshold
  iou: float=0.45  # intersection over union
  nms: bool = False  # use non-maximum suppression
  nms_threshold: float=0.4  # nms threshold
  # metadata
  credit: str=''  # how credit should be displayed
  repo: str=''  # author/repo URL
  license: str='LICENSE.txt'  # filepath to license
  license_tag: str=''  # eg "mit", see https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository#choosing-the-right-license
  batch_enabled: bool=False

  model_exists: bool=False
  config_exists: bool=False
  labels_exist: bool=False

  def __post_init__(self):
    if not self.local[0] == '/':
      self.local = join(self.dp_models, self.local)

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

    # colors
    fp = join(self.local, self.colors)
    if self.colors and Path(fp).is_file():
      with open(fp, 'rt') as fp:
        lines = fp.read().rstrip('\n').split('\n')
      self.colorlist = [Color.from_rgb_hex_str(x) for x in lines]
    else:
      self.colorlist = None

    # # update device
    # if not self.device or any(self.device) == -1:
    #   self.device = -1  # CPU
    # else:
    #   devices_available = os.getenv('CUDA_VISIBLE_DEVICES')
    # device: 0  # gpu FIXME conflicts with gpu property


  def override(self, device=0, dnn_size=(None, None), threshold=None, **kwargs):
    self.device = device
    if all(dnn_size):
      if not self.resize_enabled:
        LOG.warn(f'Resizing DNN input size not permitted for this model')
      else:
        self.width, self.height = dnn_size
    if threshold is not None:
      self.threshold = threshold


  @property
  def size(self):
    return (self.width, self.height)
