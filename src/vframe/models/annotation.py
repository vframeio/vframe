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

from vframe.models.color import Color, BLACK
from vframe.models.geometry import BBox, Point, Polygon

LOG = logging.getLogger('VFRAME')


# ---------------------------------------------------------------------------
#
# CVAT
#
# ---------------------------------------------------------------------------

@dataclass
class CVATLabel:
  name: str
  color: str=''
  attributes: List = field(default_factory=List)

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
  color_hex: int=0x000000

  @property
  def color(self):
    return Color.from_rgb_hex_int(self.color_hex)
  
  def to_cvat_label(self):
    return CVATLabel(self.enum)

  def as_dict(self):
    return {
      'index': self.index,
      'enum': self.enum,
      'display': self.display,
      'color_hex': self.color.to_rgb_hex_str(),
    }


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
  """Annotation data object. Stored using CSV.
  """
  filename: str
  label_index: int
  label_enum: str
  label_display: str
  bbox: BBox
  color: Color=BLACK
  anno_index: int=0
  polygon_str: str=''
  polygon: Polygon=None

  def __post_init__(self):
    if self.polygon_str and not self.polygon:
      try:
        self.polygon = Polygon.from_str(self.polygon_str, self.bbox.dw, self.bbox.dh)
      except Exception as e:
        pass
        # LOG.warn(f'Could not create polygon_str: {e}, {self.polygon_str}')

  def as_dict(self):
    """Deprecated"""
    LOG.warning('Deprecated. Use "to_dict()')
    return self.to_dict()

  def to_dict(self):
    d = self.bbox.to_dict()
    r,g,b = self.color.to_rgb_int()
    polygon_str = '' if not self.polygon else self.polygon.to_str()
    d.update(
      {
      'label_display': self.label_display,
      'label_enum': self.label_enum,
      'label_index': self.label_index,
      'color_hex': self.color.to_rgb_hex_str(),
      'r': r,
      'g': g,
      'b': b,
      'filename': self.filename,
      'anno_index': self.anno_index,
      'polygon_str': polygon_str,
      }
    )
    return d


  def to_yolo_str(self, use_polygon=False):
    s = [str(self.label_index)]
    if use_polygon and self.polygon is not None:
      # normalized list of xy points defining polygon
      # coords = [p.xy_norm for p in self.polygon.points]
      coords = [' '.join([str(p.x_norm), str(p.y_norm)]) for p in self.polygon.points]
    else:
      # normalized list of centerX, centerY, width, height
      coords = [self.bbox.cx_norm, self.bbox.cy_norm, self.bbox.w_norm, self.bbox.h_norm]
    s += list(map(str, coords))
    return ' '.join(s)


  @classmethod
  def from_row(cls, row):
    bbox = BBox(row.x1, row.y1, row.x2, row.y2, row.dw, row.dh)

    if not 'polygon_str' in row.keys():
      LOG.warning('"polygon_str" column missing.')
      row['polygon_str'] = ''

    return cls(row.filename, 
      row.label_index, 
      row.label_enum, 
      row.label_display, 
      bbox, 
      Color.from_rgb_hex_str(row.color_hex),
      row.anno_index, 
      row.polygon_str)
