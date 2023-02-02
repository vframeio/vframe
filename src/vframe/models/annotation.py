#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

from pathlib import Path
import logging
import random
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from dacite import from_dict
import numpy as np
from PIL import Image

from vframe.settings.app_cfg import CRYPTO_ASSET_TYPES
from vframe.models.color import Color, BLACK, GREEN
from vframe.settings.app_cfg import LABEL_INDEX_NEG, LABEL_ENUM_NEG, LABEL_DISPLAY_NEG
from vframe.models.geometry import BBox, Point, Polygon

LOG = logging.getLogger('VFRAME')


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
  r: int=0
  g: int=0
  b: int=0

  @property
  def color(self):
    return Color.from_rgb_int((self.r, self.g, self.b))

  def as_dict(self):
    return {
      'index': self.index,
      'enum': self.enum,
      'display': self.display,
      'r': self.r,
      'g': self.g,
      'b': self.b
    }


@dataclass
class LabelMaps:
  """List of label maps
  """
  labels: List[Label]


@dataclass
class RenderedFilepath:
  filepath: str
  filename: str
  filename_root: str
  stem: str
  frame_num_str: str
  frame_num: int

  def as_image_name(self):
    """Return mask filename formatted as image filename for lookup tables
    """
    return f'{self.fn_root}_'

  @classmethod
  def from_filepath(cls, fp):
    fn = Path(fp).name
    stem = Path(fn).stem
    ridx = stem.rindex('_')
    fn_root = stem[:ridx+1]
    frame_num_str = stem[ridx+1:]
    frame_num = int(frame_num_str)
    return cls(fp, fn, fn_root, stem, frame_num_str, frame_num)



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
  filename_mask: str
  label_index: int
  label_enum: str
  label_display: str
  bbox: BBox
  color: Color=BLACK
  anno_index: int=0


  def to_dict(self):
    r,g,b = self.color.to_rgb_int()
    d = {
      'filename': self.filename,
      'filename_mask': self.filename_mask,
      'label_display': self.label_display,
      'label_enum': self.label_enum,
      'label_index': self.label_index,
      'r': r,
      'g': g,
      'b': b,
      'anno_index': self.anno_index,
      }
    d.update(self.bbox.to_dict())
    return d


  def to_yolo_str(self):
    """Returns normalized space-delimited string of centerX, centerY, width, height
    """
    s = [str(self.label_index)]
    coords = [self.bbox.cx_norm, self.bbox.cy_norm, self.bbox.w_norm, self.bbox.h_norm]
    s += list(map(str, coords))
    return ' '.join(s)


  @classmethod
  def from_row(cls, row):
    """Create from annotation DataFrame row
    """
    return cls(row.filename, 
      row.get('filename_mask', ''),
      row.label_index, 
      row.label_enum, 
      row.label_display, 
      BBox(row.x1, row.y1, row.x2, row.y2, row.dw, row.dh), 
      Color.from_rgb_int((row.r, row.g, row.b)),
      row.anno_index)

  @classmethod
  def from_negative(cls, fp, wh=None):
    wh = wh if wh else Image.open(fp).size
    return cls(
      Path(fp).name,      # filename
      '',                 # filename_mask
      LABEL_INDEX_NEG,    # label_index: int
      LABEL_ENUM_NEG,     # label_enum
      LABEL_DISPLAY_NEG,  # label_display: str
      BBox(0,0,0,0,wh[0],wh[1]),  # bbox
      BLACK,              # color 
      -1,                  # anno_index
    )


# ---------------------------------------------------------------------------
#
# Synthetic Annotations
#
# ---------------------------------------------------------------------------

@dataclass
class Cryptomatte:
  object_name: str
  label_enum: str
  label_display: str
  label_index: int
  filename: str=''  # defaults to object name

  def __post_init__(self):
    self.filename = self.object_name if not self.filename else self.filename

  
@dataclass
class Cryptomattes:
  save_image: bool=True
  save_cryptomatte: bool=True
  save_masks: bool=True
  asset_type: str='CryptoAsset'
  mattes: List[Cryptomatte]=field(default_factory=lambda:[])

  def __post_init__(self):
    if not self.asset_type in CRYPTO_ASSET_TYPES:
      raise ValueError(f'Invalid type: {self.asset_type}. Use {CRYPTO_ASSET_TYPES}.')