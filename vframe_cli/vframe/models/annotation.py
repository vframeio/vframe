#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

import logging
import random
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import numpy as np

from vframe.models.color import Color
from vframe.models.geometry import BBox

log = logging.getLogger('vframe')


# ---------------------------------------------------------------------------
#
# CVAT
#
# ---------------------------------------------------------------------------

@dataclass
class CVATLabel:
  name: str
  color: str=''
  attributes: list = field(default_factory=list)

# ---------------------------------------------------------------------------
#
# Label Map
#
# ---------------------------------------------------------------------------

@dataclass
class Label:
  """Label map data
  """
  index: int
  enum: str
  display: str

  def to_cvat_label(self):
    return CVATLabel(self.enum)

@dataclass
class LabelMaps:
  """List of label maps
  """
  labels: List[Label]



# ---------------------------------------------------------------------------
#
# Annotation
#
# ---------------------------------------------------------------------------

@dataclass
class Annotation:
  """Annotation data object
  """
  filename: str
  label_index: int
  label_enum: str
  label_display: str
  bbox: BBox
  color: Color

  def to_dict(self):
    d = self.bbox.to_dict()
    r,g,b = self.color.to_rgb_int()
    d.update(
      {
      'label_display': self.label_display,
      'label_enum': self.label_enum,
      'label_index': self.label_index,
      'color_hex': self.color.to_rgb_hex(),
      'r': r,
      'g': g,
      'b': b,
      'filename': self.filename
      }
    )
    return d

  def to_darknet_str(self):
    return f'{self.label_index} {self.bbox.cx_norm} {self.bbox.cy_norm} {self.bbox.w_norm} {self.bbox.h_norm}'

  @classmethod
  def from_anno_series_row(cls, row):
    bbox = BBox(row.x1, row.y1, row.x2, row.y2, row.dw, row.dh)
    return cls(row.filename, row.label_index, row.label_enum, row.label_display, bbox, Color.from_rgb_hex(row.color_hex))
