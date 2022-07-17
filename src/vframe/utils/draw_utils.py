############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import sys
from math import sqrt
from os.path import join

import numpy as np
import cv2 as cv
import PIL
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cycler as mpl_cycler

from vframe.models import types
from vframe.models.geometry import BBox, Point
from vframe.models.color import Color, BLACK, GREEN, BLUE, RED
from vframe.utils import im_utils
from vframe.settings import app_cfg
from vframe.settings.app_cfg import LOG


# -----------------------------------------------------------------------------
#
# Font Manager
#
# -----------------------------------------------------------------------------

class FontManager:
  
  fonts = {}
    
  def __init__(self):
    # build/cache a dict of common font sizes
    name = app_cfg.DEFAULT_FONT_NAME
    fp_font = app_cfg.DEFAULT_FONT_FP
    self.fonts[name] = {}
    self.fonts[name]['fp'] = fp_font
    for i in range(10, 60, 2):
      self.fonts[name][i] = ImageFont.truetype(join(fp_font), i)

  def add_font(self, name, fp_font, font_size=16):
    self.fonts[name] = {}
    self.fonts[name]['fp'] = fp_font
    self.fonts[name][font_size] = ImageFont.truetype(fp_font, font_size)

  def get_font(self, font_name, font_size):
    """Returns font and creates/caches if doesn't exist
    """
    font = self.fonts[font_name]
    if not font_size in font.keys():
      font[font_size] = ImageFont.truetype(font['fp'], font_size)
    return font[font_size]


# -----------------------------------------------------------------------------
#
# init instances
#
# -----------------------------------------------------------------------------

font_mngr = FontManager()



# -----------------------------------------------------------------------------
#
# Matplotlib utils
#
# -----------------------------------------------------------------------------

def pixels_to_figsize(opt_dim, opt_dpi):
  """Converts pixel dimension to inches figsize
  """
  w, h = opt_dim
  return (w / opt_dpi, h / opt_dpi)


# Plot style
def set_matplotlib_style(plt, style_name='ggplot', figsize=(12,6)):
  """Sets matplotlib stylesheet with custom colors
  """

  plt.style.use(style_name)

  plt.rcParams['font.family'] = 'sans-serif'
  plt.rcParams['font.serif'] = 'Helvetica'
  plt.rcParams['font.monospace'] = 'Andale Mono'
  plt.rcParams['font.size'] = 14
  plt.rcParams['axes.labelsize'] = 14
  plt.rcParams['axes.labelweight'] = 'bold'
  plt.rcParams['axes.titlepad'] = 20
  plt.rcParams['axes.labelpad'] = 14
  plt.rcParams['axes.titlesize'] = 18
  plt.rcParams['xtick.labelsize'] = 12
  plt.rcParams['ytick.labelsize'] = 12
  plt.rcParams['legend.fontsize'] = 12
  plt.rcParams['figure.titlesize'] = 16

  cycler = mpl_cycler('color', ['#0A1EFF', '#1EBAA8', '#CABD84', '#BC8D49', '#C04D3C', '#8EBA42', '#FFB5B8'])
  plt.rcParams['axes.prop_cycle'] = cycler
  


# -----------------------------------------------------------------------------
#
# Drawing utils
#
# -----------------------------------------------------------------------------

fonts = {}



# -----------------------------------------------------------------------------
#
# Rotated BBox
#
# -----------------------------------------------------------------------------

def draw_rotated_bbox_cv(im, rbbox_norm, stroke=2, color=GREEN):
  """Draw rotated bbox using opencv
  """
  if im_utils.is_pil(im):
    im = im_utils.pil2np()
    was_pil = True
  else:
    color.swap_rb()
    was_pil = False

  dim = im.shape[:2][::-1]
  rbbox_dim = rbbox_norm.to_rbbox_dim(dim)
  vertices = rbbox_dim.vertices
  color_rgb = color.rgb_int
  for i, p in enumerate(vertices):
    p1 = vertices[i]
    p2 = vertices[(i + 1) % 4]
    if stroke == -1:
      im = cv.line(im, p1.xy, p2.xy, color_rgb, 0, cv.LINE_AA, -1)
    else:
      im = cv.line(im, p1.xy, p2.xy, color_rgb, stroke, cv.LINE_AA)

  if was_pil:
    im = im_utils.pil2np(im)

  return im



def draw_rotated_bbox_pil(im, rbbox_norm, stroke=2, color=GREEN, expand=0.0):
  """Draw rotated bbox using PIL
  """
  if im_utils.is_np(im):
    im = im_utils.np2pil(im)
    was_np = True
  else:
    was_np = False

  # TODO implement expand on rbbox
  rbbox_dim = rbbox_norm.to_rbbox_dim(im.size)
  points = rbbox_dim.vertices
  vertices = [p.xy for p in points]
  color_rgb = color.rgb_int
  canvas = ImageDraw.Draw(im)

  if stroke == -1:
    canvas.polygon(vertices, fill=color_rgb)
  else:
    canvas.polygon(vertices, outline=color_rgb)

  del canvas

  if was_np:
    im = im_utils.pil2np(im)

  return im


# -----------------------------------------------------------------------------
#
# Segmentation masks
#
# -----------------------------------------------------------------------------

def draw_mask(im, bbox_norm, mask, threshold=0.3,  mask_blur_amt=21, color=GREEN, blur_amt=None, color_alpha=0.6):
  """Draw image mask overlay
  """
  dim = im.shape[:2][::-1]
  bbox_dim = bbox_norm.to_bbox_dim(dim)
  x1, y1, x2, y2 = bbox_dim.xyxy
  mask = cv.resize(mask, bbox_dim.wh, interpolation=cv.INTER_NEAREST)
  mask = cv.blur(mask, (mask_blur_amt, mask_blur_amt))
  mask = (mask > threshold)
  # extract the ROI of the image
  roi = im[y1:y2,x1:x2]
  if blur_amt is not None:
    roi = cv.blur(roi, (blur_amt, blur_amt))
  roi = roi[mask]
  if color is not None:
    color_rgb = color.rgb_int[::-1]  # rgb to bgr
    roi = ((color_alpha * np.array(color_rgb)) + ((1 - color_alpha) * roi))
  # store the blended ROI in the original image
  im[y1:y2,x1:x2][mask] = roi.astype("uint8")
  return im


def _draw_bbox_pil(canvas, bbox, color, stroke):
  """Draws BBox onto PIL.ImageDraw
  :param bbox: BBoxDiim
  :param color: Color
  :param stroke: int
  :returns PIL.ImageDraw
  """
  xyxy = bbox.xyxy_int
  if stroke == -1:
    canvas.rectangle(xyxy, fill=color.rgb_int)
  else:
    canvas.rectangle(xyxy, outline=color.rgb_int, width=stroke)
  return canvas
  

def draw_bbox(im, bboxes, color=GREEN, stroke=4, expand=None,
  label=None, color_label=None, font_size=None, padding=None, font_name=None):
  """Draws bboxes on image
  :param im: PIL.Image or numpy.ndarray
  :param bboxes: list(BBox) or (BBox)
  :param color: Color
  :param stroke: int
  :param expand: float percentage
  :param label: String
  :param font_size: int
  :param padding: int
  :returns numpy.ndarray image
  """

  if not bboxes:
    return im

  # ensure pil format
  if im_utils.is_np(im):
    im = im_utils.np2pil(im)
    was_np = True
  else:
    was_np = False

  # init kwargs
  bboxes = bboxes if isinstance(bboxes, list) else [bboxes]
  bboxes = [b.expand_per(expand) for b in bboxes] if expand is not None else bboxes
  color = color if color else GREEN
  stroke = stroke if stroke else app_cfg.DEFAULT_STROKE_WEIGHT
  font_name = font_name if font_name else app_cfg.DEFAULT_FONT_NAME
  canvas = ImageDraw.ImageDraw(im)
  W,H = im.size
  
  pxd = 1  # pixel offset to adjust text to account for aliasing?

  for bbox in bboxes:

    # redimension to current image size
    bbox = bbox.redim((W,H))

    # draw bbox
    _draw_bbox_pil(canvas, bbox, color, stroke)

    # draw label-background
    if label:
      label = label.upper()
      
      # init font styles
      color_label = color_label if color_label else color.get_fg_color()
      font_size = font_size if font_size else app_cfg.DEFAULT_font_size
      padding = padding if padding is not None else int(app_cfg.DEFAULT_PADDING_PER * font_size)
      padding += stroke
      font = font_mngr.get_font(font_name, font_size)
      
      # bbox of label background
      bbox_bg = _bbox_from_text(bbox, label, font, padding)
      
      # adjust box size if font height is not font size
      th = font.getsize('0')[1]
      if th != font_size:
        label_adj = font_size - th
      else:
        label_adj = 0

      # check if space permits outer label
      if bbox_bg.h < bbox.y1:
        # move outside
        bbox_bg = bbox_bg.translate(0, 0 - bbox_bg.h + stroke - pxd)
      
      _draw_bbox_pil(canvas, bbox_bg,  color, -1)
      
      # point of label origin
      bbox_label = bbox_bg.shift(padding, padding-label_adj-pxd, 0, 0)
      _draw_text_pil(canvas, label, Point.from_bbox(bbox_label), color_label, font)

  # cleanup
  del canvas

  # ensure original format
  if was_np:
    im = im_utils.pil2np(im)

  return im



def draw_polygon(im, polygon, color=GREEN, stroke=1, radius=1):
  
  # ensure numpy format
  was_pil = False
  if im_utils.is_pil(im):
    im = im_utils.pil2np(im)
    was_pil = True

  h,w = im.shape[:2]
  rgb = color.rgb_int[::-1]
  polygon = polygon.redim((w,h))
  pt_pre = None
  for pt in polygon.points:
    if pt_pre:
      im = cv.line(im, pt_pre.xy, pt.xy, rgb, stroke)
    pt_pre = pt
    im = cv.circle(im, pt.xy, radius*2, rgb, -1)
  # connect last to first
  im = cv.line(im, pt_pre.xy, polygon.points[0].xy, rgb, stroke)

  if was_pil:
    im = im_utils.np2pil(im)

  return im


# -----------------------------------------------------------------------------
#
# Text
#
# -----------------------------------------------------------------------------

def mk_font_chip(txt, font_name, font_size=16, font_color=None, bg_color=None):
  """Draw font chip for perceptual similarity scoring with synthetic font-graphics
  """
  # get text bounds
  font = font_mngr.get_font(font_name, font_size)
  tw, th = font.getsize(txt)

  # create temp image
  font_color = font_color if font_color else WHITE
  bg_color = bg_color if bg_color else BLACK
  im = Image.new('RGB', (tw,th), bg_color.rgb_int)
  p = Point(0,0, *im.size)
  im = draw_text(im, txt, p, font_name=font_name, font_color=font_color, font_size=font_size)
  return im


def _bbox_from_text(obj_geo, text, font, padding):
  """Creates BBox based on text, font size, and padding
  :returns BBox
  """
  x,y = obj_geo.xy_int
  tw, th = font.getsize(text)
  return BBox(x, y, x + tw + padding * 2, y + th + padding * 2, *obj_geo.dim)


def _draw_text_pil(canvas, text, pt, color, font):
  """Draws bbox and annotation
  """
  canvas.text(pt.xy_int, text, color.rgb_int, font)


def draw_text(im, text, pt, font_name=None, font_size=None, font_color=None, 
  bg=False, padding_text=None, color_bg=None, upper=False):
  """Draws text with background
  :param im: PIL.Image or numpy.ndarray
  :param pt: Point of upper left corner
  :param text: String
  :param color: Color
  :param stroke: int
  :param font_size: int
  :param expand: float
  """
  if im_utils.is_np(im):
    im = im_utils.np2pil(im)
    dim = im.size
    was_np = True
  else:
    was_np = False

  # init font styles and canvas
  if upper:
    text = text.upper()
  font_name = font_name if font_name else app_cfg.DEFAULT_FONT_NAME
  font_color = font_color if font_color else GREEN
  font_size = font_size if font_size else app_cfg.DEFAULT_TEXT_SIZE
  font = font_mngr.get_font(font_name, font_size)
  canvas = ImageDraw.ImageDraw(im)

  if bg:
    padding_text = int(app_cfg.DEFAULT_PADDING_PER * font_size) if padding_text is None else padding_text
    # bbox of text background
    bbox_bg = _bbox_from_text(pt, text, font, padding_text)
    _draw_bbox_pil(canvas, bbox_bg, color_bg, -1)
    # point of text origin
    bbox_text = bbox_bg.shift(padding_text, padding_text, -padding_text, -padding_text)
  else:
    bbox_text = _bbox_from_text(pt, text, font, 0)

  pt = Point.from_bbox(bbox_text)
  _draw_text_pil(canvas, text, pt, font_color, font)

  del canvas

  # ensure original format
  if was_np:
    im = im_utils.pil2np(im)
  return im




# -----------------------------------------------------------------------------
#
# BBoxes
#
# -----------------------------------------------------------------------------

def draw_bbox_cv(im, bboxes, stroke=4, color=(0,255,0)):
  """Draws BBox onto Numpy image using np broadcasting
  :param bbox: BBoxDiim
  :param color: Color
  :param stroke: int
  """
  bboxes = bboxes if isinstance(bboxes, list) else [bboxes]
  for bbox in bboxes:
    x1, y1, x2, y2 = bbox.xyxy_int
    im = cv.rectangle(im, (x1,y1), (x2,y2), color, stroke)
  return im


# def _draw_bbox_np(im, bboxes, color, stroke):
#   """Draws BBox onto cv image using np broadcasting
#   :param bbox: BBoxDiim
#   :param color: Color
#   :param stroke: int
#   """
#   for bbox in bboxes:
#     im[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = color.bgr_int
#   return im
