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
from dataclasses import dataclass

import numpy as np

from vframe.models.color import Color
from vframe.models.geometry import BBox

log = logging.getLogger('vframe')

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
  label: str
  label_index: int
  bbox: BBox
  color: Color

  def to_dict(self):
    d = self.bbox.to_dict()
    d.update(
      {
      'label': self.label,
      'label_index': self.label_index,
      'color': self.color.to_rgb_hex(),
      'filename': self.filename
      }
    )
    return d

  def to_darknet_str(self):
    return f'{self.label_index} {self.bbox.cx_norm} {self.bbox.cy_norm} {self.bbox.w_norm} {self.bbox.h_norm}'

  @classmethod
  def from_anno_series_row(cls, row):
    bbox = BBox(row.x1, row.y1, row.x2, row.y2, row.dw, row.dh)
    return cls(row.filename, row.label, row.label_index, bbox, Color.from_rgb_hex(row.color))
