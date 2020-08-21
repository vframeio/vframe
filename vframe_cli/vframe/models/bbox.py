############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

"""
VFRAME BBox classes. Used in all VFRAME repos
"""

import logging
import random
import math
from dataclasses import dataclass

import numpy as np

from vframe.models.color import Color


# ---------------------------------------------------------------------------
#
# Point classes
#
# ---------------------------------------------------------------------------

@dataclass
class Point:
  x: float
  y: float

  def distance(self, p):
    dx = self.x - p.x
    dy = self.y - p.y
    return int(math.sqrt(math.pow(dx, 2) + math.pow(dy, 2)))

  @property
  def xy(self):
    return (self.x, self.y)

  def scale(self, scale):
    self.x *= scale
    self.y *= scale



@dataclass
class PointNorm(Point):
  x: float
  y: float

  def to_point_dim(self, dim):
    return PointDim(int(self.x * dim[0]), int(self.y * dim[1]))


@dataclass
class PointDim(Point):
  x: int
  y: int

  def to_point_norm(self, dim):
    return PointNorm(self.x / dim[0], self.y / dim[1])


# ---------------------------------------------------------------------------
#
# Bounding Box base class
#
# ---------------------------------------------------------------------------

@dataclass
class BBoxBase:
  x1: float
  y1: float
  x2: float
  y2: float

  def __post_init__(self):
    # clamp values
    self.x1 = float(min(max(self.x1, 0.0), 1.0))
    self.y1 = float(min(max(self.y1, 0.0), 1.0))
    self.x2 = float(min(max(self.x2, 0.0), 1.0))
    self.y2 = float(min(max(self.y2, 0.0), 1.0))

  def to_dict(self):
    """Override me"""
    pass


  @property
  def w(self):
    return (self.x2 - self.x1)
  
  @property
  def width(self):
    return self.w

  @property
  def h(self):
    return (self.y2 - self.y1)

  @property
  def wh(self):
    return (self.w, self.h)

  @property
  def cx(self):
    return self.x1 + (self.width / 2)

  @property
  def cy(self):
    return self.y1 + (self.height / 2)

  @property
  def cxcy(self):
    return (self.cx, self.cy)
  
  @property
  def height(self):
    return self.h

  @property
  def area(self):
    return self.w * self.h
  
  @property
  def p1(self):
    return PointNorm(self.x1, self.y1)

  @property
  def p2(self):
    return PointNorm(self.x2, self.y2)

  @property
  def xyxy(self):
    return (self.x1, self.y1, self.x2, self.y2)

  @property
  def xy(self):
    return (self.x1, self.y1)
  
  @property
  def xywh(self):
    return (self.x1, self.y1, self.w, self.h)

  def to_scale_wh(self, sw, sh):
    return self.__class__(self.x1, self.y1, self.w * sw + self.x1, self.h*sh + self.y1)

  def scale(self, per):
    """Scale by percentage value
    """
    return self.__class__(self.x1 * per, self.y1 * per, self.x2 * per, self.y2 * s)

  def scale(self, sw, sh):
    """Scale by width and height values
    """
    return self.__class__(self.x1*sw, self.y1*sh, self.x2*sw, self.y2*sh)

  def to_bbox_labeled(self, label, label_index, color, fn):
    return BBoxLabeled(self.x1, self.y1, self.x2, self.y2, label, label_index, color, fn)
  
  def jitter(self, per):
    '''Jitters the center xy and the wh of BBox
    :returns BBox[SubClass] jittered
    '''
    amtw = per * self.w
    amth = per *self.h
    w = self.w + (self.w * random.uniform(-amtw, amtw))
    h = self.h + (self.h * random.uniform(-amth, amth))
    cx = self.cx + (self.cx * random.uniform(-amtw, amtw))
    cy = self.cy + (self.cy * random.uniform(-amth, amth))
    orig_type = type(self.x1)
    xyxy_mapped = list(map(orig_type, [cx - w/2, cx - w/2, cx + w/2, cx + w/2]))
    self.__class__(*xyxy_mapped)

  def contains_point(self, p):
    '''Checks if this BBox contains the normalized point
    :param p: (Point)
    :returns (bool)
    '''
    return (p.x >= self.x1 and p.x <= self.x2 and p.y >= self.y1 and p.y <= self.y2)

  def contains_bbox(self, b):
    '''Checks if this BBox fully contains another BBox
    :param b: (BBox)
    :returns (bool)
    '''
    return (b.x1 >= self.x1 and b.x2 <= self.x2 and b.y1 >= self.y1 and b.y2 <= self.y2)

  @classmethod
  def from_xywh(cls, xywh):
    x, y, w, h = xywh
    x1, y1, x2, y2 = (x, y, x + w, y + h)
    return cls(x1, y1, x2, y2)

  @classmethod
  def from_xyxy(cls, xyxy):
    return cls(*xyxy)

  @classmethod
  def from_cxcywh(cls, cxcywh):
    cx, cy, w, h = cxcywh
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return cls(x1, y1, x2, y2)

  def rot90(self, k=1):
    """Rotates BBox by 90 degrees N times
    :param k: number of 90 rotations
    """
    w,h = (1.0, 1.0)
    if k == 1:
      # 90 degrees
      x1,y1 = (h - self.y2, self.x1)
      x2,y2 = (x1 + self.h, y1 + self.w)
      return self.__class__(x1, y1, x2, y2)
    elif k == 2:
      # 180 degrees
      x1,y1 = (w - self.x2, h - self.y2)
      x2, y2 = (x1 + self.w, y1 + self.h)
      return self.__class__(x1, y1, x2, y2)
    elif k == 3 or k == -1:
      # 270 degrees
      x1,y1 = (self.y1, w - self.x2)
      x2, y2 = (x1 + self.h, y1 + self.w)
      return self.__class__(x1, y1, x2, y2)
    else:
      return self


  def to_expanded(self):
    pass


  @classmethod
  def translate(self, xyxy):
    pass


  
# ---------------------------------------------------------------------------
#
# Bounding Box normalized coords
#
# ---------------------------------------------------------------------------


@dataclass
class BBoxNorm(BBoxBase):
  
  x1: float
  y1: float
  x2: float
  y2: float

  @property
  def cx(self):
    return self.x1 + (self.width / 2)

  @property
  def cy(self):
    return self.y1 + (self.height / 2)

  def to_expanded(self):
    pass

  def to_dict(self):
    return {
        'x1': float(self.x1),
        'y1': float(self.y1),
        'x2': float(self.x2),
        'y2': float(self.y2),
      }

  def translate(self, xyxy):
    x1, y1, x2, y2 = xyxy
    xyxy = (self.x1 + x1, self.y1 + y1, self.x2 + x2, self.y2 + y2)
    xyxy = [min(1.0, max(0.0, a)) for a in xyxy]
    return self.__class__(*xyxy)

  
  def expand(self, per):
    """Expands BBox by percentage
    :param per: (float) percentage to expand 0.0 - 1.0
    :returns (BBoxNorm) expanded
    """
    dw, dh = [(self.w * per), (self.h * per)]
    x1, y1, x2, y2 = list(np.array(self.xyxy) + np.array([-dw, -dh, dw, dh]))
    # threshold expanded rectangle
    x1 = max(x1, 0.0)
    y1 = max(y1, 0.0)
    x2 = min(x2, 1.0)
    y2 = min(y2, 1.0)
    return BBoxNorm(x1, y1, x2, y2)


  def expand_per(self, per):
    '''Deprecated
    '''
    return self.expand(per)


  def to_bbox_dim(self, dim):
    w,h = dim
    x1,y1,x2,y2 = [int(a) for a in [self.x1 * w, self.y1 * h, self.x2 * w, self.y2 * h]]
    return BBoxDim(x1, y1, x2, y2, dim)

  def to_labeled(self, label, label_index, fn):
    return BBoxNormLabel(*self.xyxy, label, label_index, fn)

  def to_labeled_colored(self, label, label_index, fn, color):
    return BBoxNormLabelColor(*self.xyxy, label, label_index, fn, color)


  @classmethod
  def from_bbox_dim(cls, rd):
    w,h = rd.dim
    x1,y1,x2,y2 = list(map(int, (rd.x1 / w, rd.y1 / h, rd.x2 / w, rd.y2 / h)))
    return cls(x1, y1, x2, y2)

  # convert image to new size centered at bbox
  def to_ratio(self, dim, ratio, expand=0.5):
    
    # expand/padd bbox
    w,h = dim
    bbox_norm_exp = self.expand_per(expand)
    # diemsion
    bbox_dim = self.to_bbox_dim(dim)
    bbox_exp_dim = bbox_norm_exp.to_bbox_dim(dim)
    # determine ratios
    rwh_new =  ratio[0]/ratio[1]
    rwh_bbox = bbox_exp_dim.w / bbox_exp_dim.h
    rhw_new =  1/rwh_new
    rhw_bbox = 1/rwh_bbox

    x1,y1,x2,y2 = bbox_norm_exp.xyxy

    # real width:height ratio smaller than target
    if rwh_new > rwh_bbox:
      # resize width of bbox
      r = rwh_new / rwh_bbox
      new_w = bbox_norm_exp.w * r
      new_wd = new_w - bbox_norm_exp.w
      x1 = x1 - new_wd/2
      x2 = x2 + new_wd/2
      if x1 < 0 and x2 < 1.0:
        # try to allocate to right side
        x2 += 0 - x1
      elif x1 > 0 and x2 > 1.0:
        # try to allocate to left side
        x1 -= x2 - 1.0
      x1, x2 = (max(0, x1), min(1.0, x2))

      new_w = x2 - x1
      new_h = (new_w * w) / rwh_new / h
      new_hd = (y2 - y1) - new_h
      y1 = y1 + new_hd/2
      y2 = y2 - new_hd/2

    elif rwh_new < rwh_bbox:
      # resize width of bbox
      r = rhw_new / rhw_bbox
      new_h = bbox_norm_exp.h * r
      new_hd = new_h - bbox_norm_exp.h
      x1 = x1 - new_hd/2
      x2 = x2 + new_hd/2
      if y1 < 0 and y2 < 1.0:
        y2 += 0 - y1
      elif y1 > 0 and y2 > 1.0: 
        y1 -= y2 - 1.0
      y1, y2 = (max(0, y1), min(1.0, y2))

      new_h = y2 - y1
      new_w = (new_h * h) / rhw_new / w
      new_wd = (x2 - x1) - new_w
      x1 = x1 + new_wd/2
      x2 = x2 - new_wd/2

      #xyxy = (x1, y1, x2, y2)
      #xyxy = (min())
    x1, x2 = (max(0, x1), min(1.0, x2))
    y1, y2 = (max(0, y1), min(1.0, y2))
      
    return self.__class__(x1,y1,x2,y2)



# ---------------------------------------------------------------------------
#
# Rotated BBoxNorm
#
# ---------------------------------------------------------------------------

@dataclass
class RotatedBBox:
  
  p1: Point
  p2: Point
  p3: Point
  p4: Point

  @property
  def vertices(self):
    return [self.p1, self.p2, self.p3, self.p4]



@dataclass
class RotatedBBoxNorm(RotatedBBox):

  p1: PointNorm
  p2: PointNorm
  p3: PointNorm
  p4: PointNorm

  @classmethod
  def from_rbbox_dim(cls, rbbox_dim):
    verts_norm = [p.to_point_norm(rbbox_dim.dim) for p in rbbox_dim.vertices]
    return cls(verts_norm)

  def to_rbbox_dim(self, dim):
    verts_dim = [p.to_point_dim(dim) for p in self.vertices]
    return RotatedBBoxDim(*verts_dim, dim)

  def to_bbox_norm(self):
    """Converts RotatedBBoxDim boundaries to a normal BBoxNorm
    """
    x1 = min([p.x for p in self.vertices])
    y1 = max([p.y for p in self.vertices])
    x2 = max([p.y for p in self.vertices])
    y2 = min([p.y for p in self.vertices])
    return BBoxNorm(x1, y1, x2, y2)




@dataclass
class RotatedBBoxDim(RotatedBBox):

  p1: PointDim
  p2: PointDim
  p3: PointDim
  p4: PointDim
  dim: (int, int)

  @classmethod
  def from_rbbox_norm(self, rbbox_norm, dim):
    verts = [p.to_point_dim(dim) for p in rbbox_norm.vertices]
    return RotatedBBoxDim(*verts, dim)

  def to_rbbox_norm(self):
    verts = [p.to_point_norm(self.dim) for p in self.vertices]
    return RotatedBBoxNorm(*verts)

  def to_bbox_dim(self):
    """Converts RotatedBBoxDim boundaries to a normal BBoxDim
    """
    x1 = min([p.x for p in self.vertices])
    y1 = min([p.y for p in self.vertices])
    x2 = max([p.y for p in self.vertices])
    y2 = max([p.y for p in self.vertices])
    return BBoxDim(x1, y1, x2, y2, self.dim)


# ---------------------------------------------------------------------------
#
# Bounding Box with dimension
#
# ---------------------------------------------------------------------------

@dataclass
class BBoxDim(BBoxNorm):
  
  x1: int
  y1: int
  x2: int
  y2: int  
  dim: (int, int)

  def __post_init__(self):
    # clamp values
    self.x1 = int(min(max(self.x1, 0), self.dim[0]))
    self.y1 = int(min(max(self.y1, 0), self.dim[1]))
    self.x2 = int(min(max(self.x2, 0), self.dim[0]))
    self.y2 = int(min(max(self.y2, 0), self.dim[1]))

  @property
  def cx(self):
    return int(self.x1 + (self.width // 2))

  @property
  def cy(self):
    return int(self.y1 + (self.height // 2))

  def translate(self, xyxy):
    w,h = self.dim
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(w, self.x1 + x1))
    y1 = max(0, min(h, self.y1 + y1))
    x2 = max(0, min(w, self.x2 + x2))
    y2 = max(0, min(h, self.y2 + y2))
    return self.__class__(x1, y1, x2, y2, self.dim)


  def to_square(self):
    if self.w == self.h:
      return self
    x1, y1, x2, y2 = self.xyxy
    w, h = self.wh
    # expand outward
    if w > h:
      # landscape: expand height
      delta = (w - h) / 2
      y1 = max(y1 - delta, 0)
      y2 = min(y2 + delta, self.dim[1])
    elif h > w:
      # portrait: expand width
      delta = (h - w) / 2
      x1 = max(x1 - delta, 0)
      x2 = min(x2 + delta,  self.dim[0])
    # try again
    w, h = (x2 - x1, y2 - y1)
    # if still not square, contract
    if w > h:
      # landscape: contract width
      delta = (w - h) / 2
      x1 = max(x1 + delta, 0)
      x2 = min(x2 - delta, self.dim[0])
    elif h > w:
      # portrait: contract height
      delta = (h - w) / 2
      y1 = max(y1 + delta, 0)
      y2 = min(y2 - delta, self.dim[0])
    return BBoxDim(x1, y1, x2, y2, self.dim)


  def rot90(self, k=1):
    """Rotates BBox by 90 degrees N times
    :param n
    """
    w,h = self.dim
    if k == 1:
      # 90 degrees
      x1,y1 = (h - self.y2, self.x1)
      x2,y2 = (x1 + self.h, y1 + self.w)
      return self.__class__(x1, y1, x2, y2, (h,w))
    elif k == 2:
      # 180 degrees
      x1,y1 = (w - self.x2, h - self.y2)
      x2, y2 = (x1 + self.w, y1 + self.h)
      return self.__class__(x1, y1, x2, y2, (w,h))
    elif k == 3:
      # 270 degrees
      x1,y1 = (self.y1, w - self.x2)
      x2, y2 = (x1 + self.h, y1 + self.w)
      return self.__class__(x1, y1, x2, y2, (h,w))
    else:
      return self


  @classmethod
  def from_xyxy_dim(cls, xyxy, dim):
    x1, y1, x2, y2 = xyxy
    return cls(x1, y1, x2, y2, dim)  # **xyxy?

  def to_bbox_norm(self):
    w,h = self.dim
    x1,y1,x2,y2 = (self.x1 / w, self.y1 / h, self.x2 / w, self.y2 / h)
    return BBoxNorm(x1, y1, x2, y2)
  
  def to_expanded(self, ltrb, redistribute=True):
    """Expands BBox by number of pixels
    :param box: (tuple) left, top, right, bottom
    :returns (BBoxDim) in pixel dimensions
    """
    print('Not yet implemented')
    

  def to_labeled(self, label, label_index, fn):
    return BBoxDimLabel(*self.xyxy, self.dim, label, label_index, fn)

  @classmethod
  def from_xywh_dim(cls, xywh, dim):
    x,y,w,h = xywh
    return cls(x, y, x + w, y + h, dim)


@dataclass
class BBoxNormLabel(BBoxNorm):
  '''Represent general BBox'''
  label: str
  label_index: int
  filename: str

  def to_colored(self, color):
    """Converts BBoxNorm to BBoxLabeled
    :param color: (Color)
    """
    return BBoxNormLabelColor(*self.xyxy, self.label, self.label_index, self.filename, color)



@dataclass
class BBoxDimLabel(BBoxDim):
  '''Represent general BBox info
  '''
  label: str
  label_index: int
  filename: str

  def to_colored(self, color):
    return BBoxDimLabelColor(self.xyxy, self.dim, self.label, self.label_index, self.filename, color)


@dataclass
class BBoxNormLabelColor(BBoxNormLabel):
  '''Represent BBox info from pixel masks as norm floats. 
  Used for Blender mask annotations
  '''
  color: Color


@dataclass
class BBoxDimLabelColor(BBoxDimLabel):
  '''Represent BBox info from pixel masks as int. 
  Used for Blender mask annotations
  '''
  color: Color

