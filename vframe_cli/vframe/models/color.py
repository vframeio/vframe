############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from dataclasses import dataclass
import colorsys
import random
import logging

@dataclass
class Color:
  r: float
  g: float
  b: float
  a: float=1.0

  def __post_init__(self):
    self.log = logging.getLogger('vframe')


  @classmethod
  def random(cls, r_range=None, g_range=None, b_range=None):
    r = random.uniform(*r_range) if r_range else random.uniform(0.0, 1.0)
    g = random.uniform(*g_range) if g_range else random.uniform(0.0, 1.0)
    b = random.uniform(*b_range) if b_range else random.uniform(0.0, 1.0)
    return cls(r, g, b)

  @classmethod
  def from_rgb_int(cls, rgb_int):
    r, g, b = rgb_int
    rgb_norm = (r / 255, g / 255, b / 255)
    return cls(*rgb_norm)

  @classmethod
  def from_rgb_norm(cls, rgba_norm):
    '''From RGBA using (1.0, 1.0, 1.0, 1.0) scale
    '''
    return cls(*rgba_norm)

  @classmethod
  def from_rgba_int(cls, rgba_int):
    '''From RGBA using (255,255,255,100) scale
    '''
    r, g, b, a = rgba_int
    rgba_norm = (r / 255, g / 255, b / 255, a / 100)
    return cls(*rgba_norm)

  @classmethod
  def from_rgba_norm(cls, rgba_norm):
    '''From RGBA using (1.0, 1.0, 1.0, 1.0) scale
    '''
    return cls(*rgba_norm)

  @classmethod
  def from_hsv_int(cls, hsv_int):
    h, s, v = hsv_int
    rgb_norm = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100)
    return cls(*rgb_norm)

  @classmethod
  def from_hsv_norm(cls, hsv_norm):
    rgb_norm = colorsys.hsv_to_rgb(*hsv_norm)
    return cls(*rgb_norm)

  @classmethod
  def from_hsva_int(cls, hsva_int):
    '''From HSVA using (360, 100, 100, 100) scale
    '''
    h, s, v, a = hsva_int
    hsv_norm = (h / 360, s / 100, v / 100, a / 100)
    rgb_norm = colorsys.hsv_to_rgb(*hsv_norm)
    return cls(*rgb_norm)

  @classmethod
  def from_hsva_norm(cls, hsva_norm):
    '''From HSVA using (1.0, 1.0, 1.0, 1.0) scale
    '''
    rgb_norm = colorsys.hsv_to_rgb(*hsva_norm)
    return cls(*rgb_norm)

  @classmethod
  def from_rgb_hex(cls, rgb_hex):
    '''From hexidecimal RGB str using (eg 0xFFFFFF)
    '''
    rgb_hex = rgb_hex.replace('0x', '').replace('#', '')
    r,g,b = tuple(int(rgb_hex[i:i+2], 16) for i in (0, 2, 4))
    return cls.from_rgb_int((r, g, b))

  def swap_rb(self):
    """Swaps color channels for OpenCV BGR format
    """
    r = self.r
    b = self.b
    self.b = r
    self.r = b

  def to_rgb_int(self):
    rgb_int = (int(255 * self.r), int(255 * self.g), int(255 * self.b))
    return rgb_int

  def to_rgb_norm(self):
    rgb_norm = (self.r, self.g, self.b)
    return rgb_norm

  def to_rgba_int(self):
    rgb_int = (int(255 * self.r), int(255 * self.g), int(255 * self.b), int(255 * self.a))
    return rgb_int
  
  def to_rgba_norm(self):
    rgba_norm = (self.r, self.g, self.b, self.a)
    return rgba_norm

  def to_hsv_int(self):
    h, s, v = colorsys.rgb_to_hsv(self.r, self.g, self.b)
    hsv_int = (int(360 * h), int(100 * s), int(100 * v))
    return hsv_int

  def to_hsv_norm(self):
    hsv_norm = colorsys.rgb_to_hsv(self.r, self.g, self.b)
    return hsv_norm
  
  def to_hsva_int(self):
    h, s, v, a = self.to_hsva_norm()
    hsva_int = (int(360 * h), int(100 * s), int(100 * v), int(100 * a))
    return hsva_int

  def to_hsva_norm(self):
    h, s, v = colorsys.rgb_to_hsv(self.r, self.g, self.b)
    a = self.a
    hsva_norm = (h, s, v, a)
    return hsva_norm

  def to_rgb_hex(self, separator="0x"):
    r, g, b = self.to_rgb_int()
    rgb_hex_str = f"{separator}{r:02x}{g:02x}{b:02x}"
    return rgb_hex_str
  
  def to_rgba_hex(self, separator="0x"):
    r, g, b, a = self.to_rgba_int()
    rgb_hex_str = f"{separator}{r:02x}{g:02x}{b:02x}{a:02x}"
    return rgba_hex_str

  def to_int(self):
    self.log.warn('Not yet implemented')

  def get_fg_color(self):
    val = 1 - (0.299 * self.r + 0.587 * self.g + 0.114 * self.b)
    if (val < 0.5):
      b = 0  # bright colors - black font
    else:
      b = 255  # dark colors - white font    

    return self.__class__.from_rgb_int((b,b,b))
    