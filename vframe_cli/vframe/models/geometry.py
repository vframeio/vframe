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

log = logging.getLogger('vframe')

# ---------------------------------------------------------------------------
#
# Point classes
#
# ---------------------------------------------------------------------------

@dataclass
class Point:
  """XY point in coordinate plane
  """
  x: float
  y: float
  dw: int=None  # dimension width
  dh: int=None  # dimension height

  def __post_init__(self):
    # if no bounds, use x,y as bounding plane
    self.dw = self.x if self.dw is None else self.dw
    self.dh = self.x if self.dh is None else self.dh
    self.dim = (self.dw, self.dh)
    # clamp values. only useful if real dimensions provided
    self.x = min(max(self.x, 0), self.dw)
    self.y = min(max(self.y, 0), self.dh)


  # ---------------------------------------------------------------------------
  # transformations

  def move(self, x, y):
    """Moves point by x,y
    """
    return self.__class__(self.x1 + x, self.y1 + y, self.x2, self.y2, *self.dim)


  def scale(self, scale):
    """Scales point
    """
    return self.__class__((self.x * scale, self.y * scale))

  
  def distance(self, p2):
    """Calculate distance between this point and another
    :param p2: Point
    :returns float of distance
    """
    dx = self.x - p2.x
    dy = self.y - p2.y
    return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))


  # ---------------------------------------------------------------------------
  # classmethods

  @classmethod
  def from_xy_int(cls, x, y, dim):
    return cls(x / dw, y / dim[1], dim)

  @classmethod
  def from_xy_norm(cls, x, y, dim):
    return cls(x, y, dim)

  @classmethod
  def from_bbox_tl(cls, bbox):
    return cls(bbox.x1, bbox.y1, *bbox.dim)

  @classmethod
  def from_bbox(cls, bbox):
    return cls.from_bbox_tl(bbox)

  # ---------------------------------------------------------------------------
  # properties

  @property
  def x_int(self):
    return int(self.x)

  @property
  def y_int(self):
    return int(self.y)
  
  @property
  def xy_int(self):
    return (self.x_int, self.y_int)

  @property
  def x_norm(self):
    return (self.x / self.dw)

  @property
  def y_norm(self):
    return (self.y / self.dh)
  
  @property
  def xy_norm(self):
    return (self.x_norm, self.y_norm)



# ---------------------------------------------------------------------------
#
# Bounding Box with dimension
#
# ---------------------------------------------------------------------------

@dataclass
class BBox:
  """A box defined by x1, y1, x2, y2 points and a bounding frame dimension
  """
  x1: float
  y1: float
  x2: float
  y2: float
  dw: int=None  # dimension width
  dh: int=None  # dimension height


  def __post_init__(self):
    # clamp values
    self.dw = self.width if self.dw is None else self.dw
    self.dh = self.height if self.dh is None else self.dh
    self.dim = (self.dw, self.dh)
    self.x1 = min(max(self.x1, 0), self.dw)
    self.y1 = min(max(self.y1, 0), self.dh)
    self.x2 = min(max(self.x2, 0), self.dw)
    self.y2 = min(max(self.y2, 0), self.dh)


  def redim(self, dim):
    w,h = dim
    x1, y1, x2, y2 = list(np.array(self.xyxy_norm) * np.array([w, h, w, h]))
    return self.__class__(x1, y1, x2, y2, w, h)

  # ---------------------------------------------------------------------------
  # Expanding
  # ---------------------------------------------------------------------------

  def _expand_wh(self, w, h):
    x1, y1, x2, y2 = list(np.array(self.xyxy) + np.array([-w, -h, w, h]))
    return self.__class__(x1, y1, x2, y2, *self.dim)


  def expand(self, k):
    """Expands by pixels in all directions
    :param k: (int) pixels
    :returns BBox
    """
    return self._expand_wh(k, k)


  def expand_w(self, k):
    return self._expand_wh(k, 0)


  def expand_h(self, k):
    return self._expand_wh(0, k)


  def expand_per(self, k):
    """Expands BBox by percentage of current width and height
    :param per: (float) percentage to expand 0.0 - 1.0
    :returns BBox expanded
    """
    w, h = (int(k * self.w), int(k * self.h))
    return self._expand_wh(w, h)

  def expand_per_w(self, k):
    """Expands BBox by percentage of current width
    :param per: (float) percentage to expand 0.0 - 1.0
    :returns BBox expanded
    """
    w, h = (int(k * self.w), int(k * self.h))
    return self._expand_wh(w, 0)

  def expand_per_h(self, k):
    """Expands BBox by percentage of current height
    :param per: (float) percentage to expand 0.0 - 1.0
    :returns BBox expanded
    """
    h = int(k * self.h)
    return self._expand_wh(0, h)


  # ---------------------------------------------------------------------------
  # Scaling
  # ---------------------------------------------------------------------------

  def _scale(self, w, h):
    return self.__class__(self.x1 * s, self.y1 * h, self.x2 * s, self.y2 * h, *self.dim)


  def scale(self, k):
    """Scale by width and height values
    """
    self._scale(k, k)


  def scale_w(self, k):
    return self._scale(k, 1)


  def scale_h(self, k):
    return self._scale(1, k)
    

  def scale_wh(self, w, h):
    """Scales width and height independently
    """
    self._scale(w, h)


  # ---------------------------------------------------------------------------
  # Transformations
  # ---------------------------------------------------------------------------

  def translate(self, x, y):
    """Translates BBox points
    """
    return self.__class__(self.x1 + x, self.y1 + y, self.x2 + x, self.y2 + y, *self.dim) 
  

  def move_to(self, x, y):
    """Moves to new XY location
    """
    return self.__class__(x, y, x + self.w, y + self.h, *self.dim)


  def shift(self, x1, y1, x2, y2):
    """Moves corner locations
    """
    return self.__class__(self.x1 + x1, self.y1 + y1, self.x2 + x2, self.y2 + y2, *self.dim)


  def jitter(self, k):
    '''Jitters the center xy and the wh of BBox
    :returns BBox
    '''
    amtw = k * self.w
    amth = k *self.h
    w = self.w + (self.w * random.uniform(-amtw, amtw))
    h = self.h + (self.h * random.uniform(-amth, amth))
    cx = self.cx + (self.cx * random.uniform(-amtw, amtw))
    cy = self.cy + (self.cy * random.uniform(-amth, amth))
    orig_type = type(self.x1)
    xyxy_mapped = list(map(orig_type, [cx - w/2, cx - w/2, cx + w/2, cx + w/2]))
    self.__class__(*xyxy_mapped, self.dim)


  def rot90(self, k=1):
    """Rotates BBox by 90 degrees N times
    :param k: int number of 90 degree rotations
    """
    k %= 4
    w, h = self.dim
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


  # convert image to new size centered at bbox
  def ratio(self, dim, ratio, expand=0.5):
    
    # expand/padd bbox
    w,h = dim
    bbox_norm_exp = self.expand(expand)
    # dimension
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
  

  def square(self):
    """Forces to square ratio
    """
    if self.w == self.h:
      return self
    x1, y1, x2, y2 = self.xyxy
    w, h = self.wh
    # expand outward
    if w > h:
      # landscape: expand height
      delta = (w - h) / 2
      y1 = max(y1 - delta, 0)
      y2 = min(y2 + delta, self.dh)
    elif h > w:
      # portrait: expand width
      delta = (h - w) / 2
      x1 = max(x1 - delta, 0)
      x2 = min(x2 + delta,  self.dw)
    # try again
    w, h = (x2 - x1, y2 - y1)
    # if still not square, contract
    if w > h:
      # landscape: contract width
      delta = (w - h) / 2
      x1 = max(x1 + delta, 0)
      x2 = min(x2 - delta, self.dw)
    elif h > w:
      # portrait: contract height
      delta = (h - w) / 2
      y1 = max(y1 + delta, 0)
      y2 = min(y2 - delta, self.dw)
    return self.__class__(x1, y1, x2, y2, self.dim)
  

  # ---------------------------------------------------------------------------
  # Comparisons
  # ---------------------------------------------------------------------------

  def contains_point(self, p2):
    '''Checks if this BBox contains the normalized point
    :param p: (Point)
    :returns (bool)
    '''
    return (p2.x >= self.x1 and p2.x <= self.x2 and p2.y >= self.y1 and p2.y <= self.y2)


  def contains_bbox(self, b2):
    '''Checks if this BBox fully contains another BBox
    :param b: (BBox)
    :returns (bool)
    '''
    return (b2.x1 >= self.x1 and b2.x2 <= self.x2 and b2.y1 >= self.y1 and b2.y2 <= self.y2)


  def overlaps(self, b2):
    """Calculates overlap percentage between another BBox
    :param b2: BBox
    :returns percentage as float
    """
    log.warn('Not yet implemented')
    return 0.0


  # ---------------------------------------------------------------------------
  # Represent
  # ---------------------------------------------------------------------------

  def to_dict(self):
    """Converts BBox to dict of xyxy and dim
    """
    return {
      'x1': self.x1,
      'y1': self.y1,
      'x2': self.x2,
      'y2': self.y2,
      'dw': self.dw,
      'dh': self.dh
      }


  # ---------------------------------------------------------------------------
  # Classmethods
  # ---------------------------------------------------------------------------

  @classmethod
  def from_xywh(cls, x, y, w, h, dw, dh):
    return cls(x, y, x + w , y + h, dw, dh)

  @classmethod
  def from_xywh_norm(cls, x, y, w, h, dw, dh):
    xyxy = tuple(np.array((x, y, x + w, y + h)) * np.array([dw, dh, dw, dh]))
    return cls(x, y, x + w, y + h, dw, dh)

  @classmethod
  def from_xyxy_norm(cls, x1, y1, x2, y2, dw, dh):
    xyxy = tuple(np.array((x1, y1, x2, y2)) * np.array([dw, dh, dw, dh]))
    return cls(*xyxy, dw, dh)

  @classmethod
  def from_cxcywh(cls, cx, cy, w, h, dw, dh):
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return cls(x1, y1, x2, y2, dw, dh)

  @classmethod
  def from_cxcywh_norm(cls, cx, cy, w, h, dw, dh):
    x1 = dw * (cx - w/2)
    y1 = dh * (cy - h/2)
    x2 = dw * (cx + w/2)
    y2 = dh * (cy + h/2)
    return cls(x1, y1, x2, y2, dw, dh)


  # ---------------------------------------------------------------------------
  # Properties
  # ---------------------------------------------------------------------------

  # --------------------------------------------------------------------------- 
  # width

  @property
  def w(self):
    return (self.x2 - self.x1)

  @property
  def w_int(self):
    return int(self.x2 - self.x1)

  @property
  def width(self):
    return self.w

  @property
  def width_int(self):
    return int(self.w)  

  @property
  def w_norm(self):
    n = 0 if self.dw is 0 else self.w / self.dw
    return self.w / self.dw

  @property
  def width_norm(self):
    return self.w_norm

  # --------------------------------------------------------------------------- 
  # height
  
  @property
  def h(self):
    return (self.y2 - self.y1)

  @property
  def h_int(self):
    return int(self.y2 - self.y1)

  @property
  def height(self):
    return self.h

  @property
  def height_int(self):
    return int(self.h)  

  @property
  def h_norm(self):
    n = 0 if self.dh is 0 else self.h / self.dh
    return n

  @property
  def height_norm(self):
    return self.h_norm


  # --------------------------------------------------------------------------- 
  # center

  @property
  def cx(self):
    return int(self.x1 + (self.width / 2))

  @property
  def cx_norm(self):
    return (self.x1 + (self.width / 2)) / self.dw

  @property
  def cy(self):
    return int(self.y1 + (self.height / 2))

  @property
  def cy_norm(self):
    return (self.y1 + (self.height / 2)) / self.dh
  
  @property
  def cxcy(self):
    return (*self.cx, *self.cy)

  @property
  def cxcy_norm(self):
    return (*self.cx_norm, *self.cy_norm)


  # --------------------------------------------------------------------------- 
  # x
  
  @property
  def x1_int(self):
    return int(self.x1)

  @property
  def x2_int(self):
    return int(self.x2)

  @property
  def x1_norm(self):
    return self.x1 / self.dw

  @property
  def x2_norm(self):
    return self.x2 / self.dw

  # --------------------------------------------------------------------------- 
  # y
  
  @property
  def y1_int(self):
    return int(self.y1)

  @property
  def y2_int(self):
    return int(self.y2)

  @property
  def y1_norm(self):
    return self.y1 / self.dh

  @property
  def y2_norm(self):
    return self.y2 / self.dh


  # --------------------------------------------------------------------------- 
  # xy
  
  @property
  def xy(self):
    return (self.x1, self.y1)

  @property
  def xy_int(self):
    return tuple(map(int, self.xy))

  @property
  def xy_norm(self):
    return (self.x1_norm, self.y1_norm)


  # --------------------------------------------------------------------------- 
  # xywy

  @property
  def xyxy(self):
    return (self.x1, self.y1, self.x2, self.y2)
  
  @property
  def xyxy_int(self):
    return tuple(map(int, self.xyxy))

  @property
  def xyxy_norm(self):
    return (self.x1_norm, self.y1_norm, self.x2_norm, self.y2_norm)


  # --------------------------------------------------------------------------- 
  # wh

  @property
  def wh(self):
    return (self.w, self.h)

  @property
  def wh_int(self):
    return tuple(map, int, (self.w, self.h))

  @property
  def wh_norm(self):
    return (self.w_norm, self.h_norm)

  
  # --------------------------------------------------------------------------- 
  # xywh

  @property
  def xywh(self):
    return (self.x1, self.y1, self.w, self.h)

  @property
  def xywh_int(self):
    return tuple(map(int, self.xywh))

  @property
  def xywh_norm(self):
    return (self.x1_norm, self.y1_norm, self.w_norm, self.h_norm)


  # --------------------------------------------------------------------------- 
  # area
  
  @property
  def area(self):
    return self.w * self.h

  @property
  def area_int(self):
    return int(self.area)


  # --------------------------------------------------------------------------- 
  # points

  @property
  def p1(self):
    return Point(self.x1, self.y1, self.dim)

  @property
  def p2(self):
    return Point(self.x2, self.y2, self.dim)



# ---------------------------------------------------------------------------
#
# Rotated BBoxNorm: 
# FIXME: Not yet implemented
#
# ---------------------------------------------------------------------------

@dataclass
class RotatedBBox:
  """TODO: not yet implemented
  """
  p1: Point
  p2: Point
  p3: Point
  p4: Point
  
  @property
  def vertices(self):
    return [self.p1, self.p2, self.p3, self.p4]

  @property
  def points_norm(self):
    return (self.p1.xy_norm, self.p2.xy_norm, self.p3.xy_norm, self.p4.xy_norm)

  @property
  def points_int(self):
    return (self.p1.xy, self.p2.xy, self.p3.xy, self.p4.xy)



# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 
# Deprecated, phasing out
#
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

@dataclass
class BBoxNormLabel:
  """TODO: not yet implemented
  """
  # label: str
  # label_index: int
  # filename: str

  def to_colored(self, color):
    pass


@dataclass
class BBoxLabel:
  """TODO: not yet implemented
  """
  # label: str
  # label_index: int
  # filename: str

  def to_colored(self, color):
    pass


@dataclass
class BBoxNormLabelColor(BBoxNormLabel):
  """TODO: not yet implemented
  """
  color: Color


@dataclass
class BBoxLabelColor:
  """TODO: not yet implemented
  """
  color: Color
