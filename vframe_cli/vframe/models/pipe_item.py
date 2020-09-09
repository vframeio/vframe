############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import cv2 as cv
from PIL import Image
from dataclasses import asdict
import dacite

from vframe.settings import app_cfg
from vframe.models import types
from vframe.models.cvmodels import ClassifyResults, DetectResults
from vframe.utils import im_utils, file_utils
from vframe.models.geometry import BBox, Point


# -----------------------------------------------------------------------------
#
# Context header meta for pipe items
#
# -----------------------------------------------------------------------------

@dataclass
class PipeContextHeader:

  filepath: str

  def __post_init__(self):
    self.log = app_cfg.LOG
    self.filename = Path(self.filepath).name
    self.ext = file_utils.get_ext(self.filename)
    self._frame_idx = 0
    self._frames_data = {}  # meta data from processors
    self._frames_data[self._frame_idx]  = {} # init frames data dict
    self.video = None

    if self.ext == 'jpg':
      im = Image.open(self.filepath)
      self.dim = im.size
      self.dim_draw = self.dim
      self.width, self.height = self.dim
      self.frame_count = 1
      self._frame_idx_start = 0
      self._frame_idx_end = 0

    elif self.ext == 'mp4':
      try:
        self.video = cv.VideoCapture(self.filepath)
        self.video.get(cv.CAP_PROP_FOURCC)
        self.frame_count = int(self.video.get(cv.CAP_PROP_FRAME_COUNT))
        self.height = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        self.dim = (self.width, self.height)
        self.dim_draw = (self.width, self.height)
        self.bitrate = (self.video.get(cv.CAP_PROP_BITRATE))
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        self.spf = 1 / self.fps
        self.mspf = 1 / self.fps * 1000  # milliseconds per frame
        self._frame_idx_start = 0  # opencv frame index starts at 0 (not 1)
        self._frame_idx_end = self.frame_count - 1  # 0-indexed frames
      except Exception as e:
        self.log.error(f'Could not load file: {self.filename}')

  def increment_frame(self):
    """Increments frame count and adds empty frame data dict
    """
    n = self._frame_idx + 1
    self.set_frame_index(n)


  def set_frame_index(self, frame_num):
    self._frame_idx = frame_num
    if not self._frames_data[self._frame_idx]:
      self._frames_data[self._frame_idx] = {}  # init frames data dict


  def set_frame_min_max(self, frame_start, frame_end, opt_decimate):
    self._frame_idx_start = self._frame_idx_start if frame_start is None else frame_start
    self._frame_idx_end = self._frame_idx_end if frame_end is None else frame_end
    if opt_decimate:
      decimate_intervals = (self._frame_idx_end // opt_decimate)
      self._frame_idx_end = decimate_intervals * opt_decimate

  

  @property
  def frame_start(self):
    return self._frame_idx_start
  
  
  @property
  def frame_end(self):
    return self._frame_idx_end

  def is_last_frame(self):
    return self._frame_idx > self._frame_idx_end or self._frame_idx >= self.frame_count
  

  @property
  def frame_index(self):
    return self._frame_idx
  
  @property
  def _frame_data(self):
    """Returns the current frame's data
    """
    return self._frames_data[self._frame_idx]

  
  def set_frame_index(self, n):
    self._frame_idx = n
    if not n in self._frames_data.keys():
      self._frames_data[self._frame_idx] = {}


  def data_key_exists(self, k):
    return k in self._frame_data.keys()

    
  def add_data(self, data):
    self._frame_data.update(data)


  def set_data(self, data):
    self._frame_data.update(data)

  
  def get_data_keys(self):
    return list(self._frame_data.keys())

    
  def remove_data(self, data_key):
    self._frame_data.pop(data_key)


  def get_data(self, data_key):
    return self._frame_data.get(data_key, None)


  def to_dict(self):
    """Serializes PipeContextHeader for export to JSON
    """
    frames_data = {}
    
    for frame_num, frame_data in self._frames_data.items():
      frames_data[frame_num] = {}
      for data_key, data in frame_data.items():
        frames_data[frame_num][data_key] = data.to_dict()

    pipe_data = {
      'filepath': self.filepath,
      'frames_data': frames_data,
    }

    return pipe_data


  @classmethod
  def from_dict(cls, data_dict):
    """Create pipe image item from dict
    """

    # Init new item
    pipe_item = cls(data_dict['filepath'])

    # add data
    for frame_num, frame_data in data_dict['frames_data'].items():
      
      pipe_item.set_frame_index(int(frame_num))
      
      for data_key, outputs in frame_data.items():
        task_type = outputs['task_type']
        if task_type == 'detection':
          outputs['task_type'] = types.Processor.DETECTION  # FIXME
          d = dacite.from_dict(data=outputs, data_class=DetectResults)
          pipe_item.add_data({data_key: d})
          
        elif task_type == 'classification':
          outputs['task_type'] = types.Processor.CLASSIFICATION  # FIXME
          d = dacite.from_dict(data=outputs, data_class=ClassifyResults)
          pipe_item.add_data({data_key: d})

    return pipe_item



  

# -----------------------------------------------------------------------------
#
# Pipe Image item
#
# -----------------------------------------------------------------------------


class PipeFrame:
  """Container for pipe frame
  """

  def __init__(self, im):
    self.im_original = im
    self.im_draw = self.im_original.copy()
    self.height, self.width = self.im_original.shape[:2]


  def get_image(self, frame_type):
    if frame_type == types.FrameImage.ORIGINAL:
      return self.im_original
    elif frame_type == types.FrameImage.DRAW:
      return self.im_draw

  def set_image(self, frame_type, im):
    if frame_type == types.FrameImage.ORIGINAL:
      self.im_original = im
    elif frame_type == types.FrameImage.DRAW:
      self.im_draw = im

    self.height, self.width = im.shape[:2]
