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

    if self.ext.lower() in ['jpg', 'png']:
      im = Image.open(self.filepath)
      self.dim = im.size
      self.dim_draw = self.dim
      self.width, self.height = self.dim
      self.frame_count = 1
      self._frame_idx_start = 0
      self._frame_idx_end = 0

    elif self.ext.lower() in ['mp4', 'mov']:
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

  
  # ---------------------------------------------------------------------------
  # Properties
  # ---------------------------------------------------------------------------

  @property
  def first_frame_index(self):
    """Returns the 0-based index of first frame
    :returns (int) first frame number index
    """
    return self._frame_idx_start
  
  @property
  def last_frame_index(self):
    """Returns the 0-based index of last frame
    :returns (int) last frame number index
    """
    return self._frame_idx_end

  @property
  def frame_index(self):
    """Returns the 0-based current frame index
    :returns (int) current frame number index
    """
    return self._frame_idx
  
  @property
  def _frame_data(self):
    """Current frame-indexed data
    :returns (dict) {'frame_index': {frame_data}}
    """
    return self._frames_data[self._frame_idx]

  def is_last_frame(self):
    """Returns True if current frame index is the last frame of a video
    """
    return self._frame_idx > self._frame_idx_end or self._frame_idx >= self.frame_count
  

  # ---------------------------------------------------------------------------
  # Frame data
  # ---------------------------------------------------------------------------

  def set_frame_index(self, n):
    self._frame_idx = n
    if not n in self._frames_data.keys():
      self._frames_data[self._frame_idx] = {}


  def data_key_exists(self, data_key, frame_idx=None):
    frame_idx = frame_idx if frame_idx else self._frame_idx
    return data_key in self._frames_data[frame_idx].keys()


  def set_data(self, data, frame_idx=None):
    """Sets data on current frame
    :param data: (dict) {data_key: frame_data}
    """
    frame_idx = frame_idx if frame_idx else self._frame_idx
    self._frame_data.update(data)

  
  def get_data_keys(self, frame_idx=None):
    """Get all data keys on current frame
    :returns (list) of keys (str)
    """
    frame_idx = frame_idx if frame_idx else self._frame_idx
    return list(self._frame_data.keys())

    
  def remove_data(self, data_key, frame_idx=None):
    """Removes data on current frame
    :param data_key: (str) dict key
    """
    frame_idx = frame_idx if frame_idx else self._frame_idx
    self._frame_data.pop(data_key)


  def get_data(self, data_key, frame_idx=None):
    """Gets frame data on current frame index
    :param data_key: (str) dict key
    :returns (dict) of DetectResults
    """
    frame_idx = frame_idx if frame_idx else self._frame_idx
    return self._frames_data[frame_idx].get(data_key, None)

  """
  "0": {
    "retinaface_r0": {
      "detections": [
        {
          "bbox": {
            "dh": 720,
            "dw": 1280,
            "x1": 886.5418090820312,
            "x2": 946.8053588867188,
            "y1": 489.9473876953125,
            "y2": 558.052978515625
          },
          "confidence": 0.9975469708442688,
          "index": 0,
          "label": "face"
        },
      "task_type": "detection"
    },
    "coco": {
      "detections": [
        {
          "bbox": {
            "dh": 720,
            "dw": 1280,
            "x1": 886.5418090820312,
            "x2": 946.8053588867188,
            "y1": 489.9473876953125,
            "y2": 558.052978515625
          },
          "confidence": 0.9975469708442688,
          "index": 0,
          "label": "face"
        },
      "task_type": "detection"
    }
  },
  """




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
          pipe_item.set_data({data_key: d})
          
        elif task_type == 'classification':
          outputs['task_type'] = types.Processor.CLASSIFICATION  # FIXME
          d = dacite.from_dict(data=outputs, data_class=ClassifyResults)
          pipe_item.set_data({data_key: d})

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
      self.im_draw = self.im_original.copy()
      self.height, self.width = self.im_original.shape[:2]
    elif frame_type == types.FrameImage.DRAW:
      self.im_draw = im

    self.height, self.width = im.shape[:2]
