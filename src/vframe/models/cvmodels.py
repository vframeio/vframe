#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

import sys
from os.path import join
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from enum import Enum
import datetime

import dacite
import cv2 as cv
import numpy as np

from vframe.models.dnn import DNN
from vframe.models.types import Processor
from vframe.models.geometry import BBox, Point, RotatedBBox
from vframe.settings.app_cfg import LOG, DN_REAL
from vframe.utils.file_utils import date_modified


# -----------------------------------------------------------------------------
#
# Benchmark result
#
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
  model: str
  fps: float
  iterations: int
  image_width: int
  image_height: int
  dnn_width: int
  dnn_height: int
  user_width: int=None
  user_height: int=None
  processor: str=''
  opencv_version: str=''
  python_version: str=''
  batch_size: int=1

  def __post_init__(self):
    self.opencv_version = cv.__version__
    self.python_version = sys.version.split(' ')[0]
    if not self.user_width:
      self.user_width = self.dnn_width
    if not self.user_height:
      self.user_height = self.dnn_height



# -----------------------------------------------------------------------------
#
# Processor result
#
# -----------------------------------------------------------------------------

@dataclass
class ProcessorResult:
  """Base class for processor result
  """

  index: int
  conf: float



@dataclass
class ClassifyResult(ProcessorResult):
  """Extends base processor class to hold results from classification
  """
  label: str

  def to_dict(self):
    return {
      'index': int(self.index),
      'label': self.label,
      'conf': float(self.conf),
    }



@dataclass
class DetectResult(ProcessorResult):
  """Store results of object detection processor
  """
  bbox: BBox
  label: str = ''
  track: int=0

  
  @classmethod
  def from_anno(cls, anno):
    return cls(anno.label_index, 1.0, anno.bbox, anno.label_enum)


  def to_dict(self):
    return {
      'index': int(self.index),
      'label': self.label,
      'conf': float(self.conf),
      'bbox': self.bbox.to_dict()
    }
    # 'track': self.track,




@dataclass
class SegmentResult(ProcessorResult):
  """Store results of segmentation detection processor
  """
  bbox: BBox
  conf: float
  mask: np.ndarray
  label: str = ''

  def to_dict(self):
    return {
      'bbox': self.bbox.to_dict(),
      'mask': self.mask.to_dict(),
      'label': self.label,
      'conf': self.conf,
    }



@dataclass
class RotatedDetectResult(ProcessorResult):
  """Store results of rotated detection processor
  """
  #index: int
  #conf: float
  bbox: BBox
  rbbox: RotatedBBox
  bbox_unrotated: BBox
  angle: float
  label: str = ''

  def to_dict(self):
    return {
      'bbox': self.bbox.to_dict(),
      'rbbox': self.rbbox.to_dict(),
      'label': self.label,
      'conf': self.conf,
    }



@dataclass
class PoseKeypoints:
  nose: Point
  neck: Point
  right_shoulder: Point
  right_elbow: Point
  right_wrist: Point
  left_shoulder: Point
  left_elbow: Point
  left_wrist: Point
  right_hip: Point
  right_knee: Point
  right_ankle: Point
  left_hip: Point
  left_knee: Point
  left_ankle: Point
  right_eye: Point
  left_eye: Point
  right_ear: Point
  left_ear: Point

#   def __post_init__(self):
#   POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
#               [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
#               [1,0], [0,14], [14,16], [0,15], [15,17],
#               [2,17], [5,16] ]

# # index of pafs correspoding to the POSE_PAIRS
# # e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
# mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
#           [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
#           [47,48], [49,50], [53,54], [51,52], [55,56],
#           [37,38], [45,46]]

# colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
#          [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
#          [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]



@dataclass
class HumanPoseDetectResult(ProcessorResult):
  """Store results of human pose detection
  """
  bbox: BBox
  keypoints: List[PoseKeypoints]
  label: str = ''


  def to_dict(self):
    return {
      'bbox': self.bbox.to_dict(),
      'label': self.label,
      'conf': self.conf,
    }



# -----------------------------------------------------------------------------
#
# Processor result containers
#
# -----------------------------------------------------------------------------

@dataclass
class ClassifyResults:
  """Container for ClassifyResult items
  """
  classifications: List[ClassifyResult]
  task_type: Enum = Processor.CLASSIFICATION


  def to_dict(self):
    return {
      'classifications': [ d.to_dict() for d in self.classifications ],
    }




@dataclass
class DetectResults:
  """Container for DetectResult items
  """
  detections: List[DetectResult]
  task_type: Enum = Processor.DETECTION

  @classmethod
  def from_priors(cls, dets):
    return cls( [dacite.from_dict(data=d, data_class=DetectResult) for d in dets] )

  def to_dict(self):
    return {
      'detections': [ d.to_dict() for d in self.detections ],
    }




  def rot90(self, k=1):
    """Rotates BBox 90 degrees this many times
    """
    detections_copy = detections.copy()



@dataclass
class SegmentResults:
  """Container for SegmentResult items
  """
  detections: List[SegmentResult]
  task_type: Enum = Processor.SEGMENTATION

  def to_dict(self):
    return {
      'detections': [ d.to_dict() for d in self.detections ],
    }


@dataclass
class RotatedDetectResults:
  """Container for RotatatedDetectReusult with
  """
  detections: List[RotatedDetectResult]
  task_type: Enum = Processor.DETECTION_ROTATED

  def to_dict(self):
    return {
      'detections': [ d.to_dict() for d in self.detections ],
    }




@dataclass
class HumanPoseDetectResults:
  """Container for HumanPoseDetectResult with
  """
  detections: List[HumanPoseDetectResult]
  task_type: Enum = Processor.DETECTION_POSE

  def to_dict(self):
    return {
      'detections': [ d.to_dict() for d in self.detections ],
    }


@dataclass
class FileMeta:
  filepath: str
  height: int
  width: int
  frame_count: int
  datetime: str
  # sha256: str=''

  def __post_init__(self):
    self.datetime = datetime.datetime.fromisoformat(self.datetime)

  @property
  def n_frames(self):
    return self.frame_count

  @property
  def date(self):
    return self.datetime.date()

  @property
  def year(self):
    return self.datetime.year


@dataclass
class ProcessedFile:
  # represents a processed media file and its metadata
  file_meta: FileMeta
  frames_meta: dict=field(default_factory=lambda: {}, repr=False)
  detections: dict=field(default_factory=lambda: {})

  def __post_init__(self):
    # TODO: too slow for frame meta with high number of detections
    for i, frame_meta in self.frames_meta.items():
      frame_dets = {}
      for data_key, detections in frame_meta.items():
        frame_dets[data_key] = dacite.from_dict(data=detections, data_class=DetectResults)
      self.detections[int(i)] = frame_dets


  @classmethod
  def from_df_annotation(cls, fn, df, fp_csv):
    """Converts DataFrame annotation to ProcessedFile
    Images only. Not compatible with videos.
    """
    LOG.debug('TODO')
    fp_im = join(Path(fp_csv).parent, DN_REAL, fn)
    date = date_modified(fp_im)  # using modified to get created
    file_meta = FileMeta(fp_im, df[0].dw, df[0].dh, 1, date)
    
    data_key = 'annotation'
    frame_dets = {}

    dets = DetectResults([DetectResult.from_annos(Annotation.from_row(row)) \
        for i, row in df.iterrows()])
  
    frames_meta[data_key] = dets
    return cls(file_meta, [frames_meta])



