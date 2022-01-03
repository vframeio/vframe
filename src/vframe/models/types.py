############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

"""Enumerated application data types
"""

import os
from enum import Enum

import cv2 as cv

from vframe.settings import app_cfg
from vframe.settings.app_cfg import modelzoo
from vframe.utils.click_utils import ParamVar

SELF_CWD = os.path.dirname(os.path.realpath(__file__))


# ---------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------

def create_interp_enum():
  """Creates interpolation enum
  """
  interps = {
    'area': cv.INTER_AREA,
    'cubic': cv.INTER_CUBIC,
    'linear': cv.INTER_LINEAR,
    'lanczos4': cv.INTER_LANCZOS4,
    'linear_exact': cv.INTER_LINEAR_EXACT,
    'max': cv.INTER_MAX,
    'nearest': cv.INTER_NEAREST,
  }
  return {k.upper(): v for k, v in interps.items()}

Interpolation = Enum('Interpolation', create_interp_enum())
InterpolationVar = ParamVar(Interpolation)


# ---------------------------------------------------------------------
# Logger, monitoring
# --------------------------------------------------------------------

class LogLevel(Enum):
  """Loger vebosity
  """
  DEBUG, INFO, WARN, ERROR, CRITICAL = range(5)


# ---------------------------------------------------------------------
# Processors
# --------------------------------------------------------------------

class Processor(Enum):
  """Loger vebosity
  """
  DETECTION, CLASSIFICATION, SEGMENTATION, DETECTION_ROTATED, DETECTION_POSE = range(5)

  def __repr__(self):
    return self.name.lower()


class ImageFileExt(Enum):
  """Image file extensions
  """
  JPG, PNG = range(2)


class VideoFileExt(Enum):
  """Video file extensions
  """
  MP4, AVI, MOV = range(3)


class FrameImage(Enum):
  """Frame image type
  """
  ORIGINAL, DRAW = range(2)


class MediaType(Enum):
  """Type of pipe item
  """
  IMAGE, VIDEO, METADATA, UNKNOWN = range(4)


class DrawAction(Enum):
  """Draw type
  """
  LABEL_BOX, BOX, BLUR, COLORFILL = range(4)


class AnnoyMetric(Enum):
  """Annoy distance metrics
  """
  ANGULAR, EUCLIDEAN, MANHATTAN, HAMMING, DOT = range(5)


class Haarcascade(Enum):
  """Haarcascade filenames
  """
  FRONTALFACE_DEFAULT, FRONTALFACE_ALT, FRONTALFACE_ALT2, FRONTALFACE_ALT_TREE, PROFILEFACE = range(5)


class CVATFormats(Enum):
  """CVAT export type
  """
  CVAT_IMAGES, CVAT_VIDEO = range(2)


# ---------------------------------------------------------------------
# Dynamic enums
# --------------------------------------------------------------------

def dict_to_enum(name, cfg):
  """Dynamically create ModelZoo enum type using model zoo YAML
  :param name: (str) name of enum
  :param cfg: (dict)
  :returns (Enum)
  """
  enum_data = {kv[0].upper(): i for i, kv in enumerate(cfg.items())}
  return Enum(name, enum_data)


ModelZoo = dict_to_enum('ModelZoo', modelzoo)



# ---------------------------------------------------------------------
# Click option vars
# --------------------------------------------------------------------

LogLevelVar = ParamVar(LogLevel)
DrawActionVar = ParamVar(DrawAction)
FrameImageVar = ParamVar(FrameImage)
ImageFileExtVar = ParamVar(ImageFileExt)
VideoFileExtVar = ParamVar(VideoFileExt)
ProcessorVar = ParamVar(Processor)
AnnoyMetricVar = ParamVar(AnnoyMetric)
HaarcascadeVar = ParamVar(Haarcascade)
ModelZooClickVar = ParamVar(ModelZoo)


# ---------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def find_type(name, enum_type):
  for enum_opt in enum_type:
    if name == enum_opt.name.lower():
      return enum_opt
  return None


# ---------------------------------------------------------------------
# Custom data types
# --------------------------------------------------------------------

class HexInt(int):
  """Represent HexInt for use in writing YAML
  """
  pass