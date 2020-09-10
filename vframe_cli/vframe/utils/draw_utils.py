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

import logging
import numpy as np
import cv2 as cv
import PIL
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cycler as mpl_cycler

from vframe.models import types
from vframe.models.geometry import BBox, Point
from vframe.models.color import Color
from vframe.utils import im_utils
from vframe.settings import app_cfg

# -----------------------------------------------------------------------------
#
# Font Manager
#
# -----------------------------------------------------------------------------

class FontManager:
  
  fonts = {}
  log = logging.getLogger('vframe')
  
  def __init__(self):
    # build/cache a dict of common font sizes
    for i in range(10, 60, 2):
      self.fonts[i] = ImageFont.truetype(join(app_cfg.FP_ROBOTO_400), i)

  def get_font(self, pt):
    """Returns font and creates/caches if doesn't exist
    """
    if not pt in self.fonts.keys():
      self.fonts[pt] = ImageFont.truetype(join(app_cfg.FP_ROBOTO_400), pt)
    return self.fonts[pt]



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
log = logging.getLogger('vframe')
color_red = Color.from_rgb_int((255, 0, 0))
color_green = Color.from_rgb_int((0, 255, 0))
color_blue = Color.from_rgb_int((0, 0, 255))
color_white = Color.from_rgb_int((255, 255, 255))
color_black = Color.from_rgb_int((0, 0, 0))



# -----------------------------------------------------------------------------
#
# Rotated BBox
#
# -----------------------------------------------------------------------------

def draw_rotated_bbox_cv(im, rbbox_norm, stroke=2, color=color_green):
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



def draw_rotated_bbox_pil(im, rbbox_norm, stroke=2, color=color_green, expand=0.0):
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

def draw_mask(im, bbox_norm, mask, threshold=0.3,  mask_blur_amt=21, color=color_green, blur_amt=None, color_alpha=0.6):
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
  

def draw_bbox(im, bbox, color=None, stroke=None, expand=None,
  label=None, color_label=None, size_label=None, padding_label=None):
  """Draws BBox on image
  :param im: PIL.Image or numpy.ndarray
  :param bbox: list(BBox)
  :param color: Color
  :param stroke: int
  :param expand: float percentage
  :param label: String
  :param size_label: int
  :param padding_label: int
  """

  # ensure pil format
  if im_utils.is_np(im):
    im = im_utils.np2pil(im)
    was_np = True
  else:
    was_np = False

  bbox = bbox if expand is None else bbox.expand(expand)

  # init font styles and canvas
  stroke = app_cfg.DEFAULT_STROKE_WEIGHT if stroke is None else stroke
  canvas = ImageDraw.ImageDraw(im)

  # draw bbox
  _draw_bbox_pil(canvas, bbox, color, stroke)

  # draw label-background if optioned
  if label:
    label = label.upper()
    # init font styles
    color_label = color.get_fg_color() if color_label is None else color_label
    size_label = app_cfg.DEFAULT_size_label if size_label is None else size_label
    font = font_mngr.get_font(size_label)
    padding_label = int(app_cfg.DEFAULT_PADDING_PER * size_label) if padding_label is None else padding_label
    # bbox of label background
    bbox_bg = _bbox_from_text(bbox, label, font, padding_label)
    # check if space permits outer label
    if bbox_bg.h > bbox.y1:
      # move inside
      bbox_bg = bbox_bg.translate(stroke, bbox_bg.h + stroke)
    else:
      bbox_bg = bbox_bg.translate(0, 0 - bbox_bg.h)
    _draw_bbox_pil(canvas, bbox_bg,  color, -1)
    # point of label origin
    bbox_label = bbox_bg.shift(padding_label, padding_label, -padding_label, -padding_label)
    _draw_text_pil(canvas, label, Point.from_bbox(bbox_label), color_label, font)


  # cleanup
  del canvas

  # ensure original format
  if was_np:
    im = im_utils.pil2np(im)
  return im




# -----------------------------------------------------------------------------
#
# Text
#
# -----------------------------------------------------------------------------


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


def draw_text(im, text, pt, color=None, size_text=None, color_text=None, 
  bg=False, padding_text=None, color_bg=None, upper=False):
  """Draws text with background
  :param im: PIL.Image or numpy.ndarray
  :param bboxes: list(BBox)
  :param color: Color
  :param text: String
  :param stroke: int
  :param size_text: int
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
  color_text = app_cfg.GREEN if color is None else color
  size_text = app_cfg.DEFAULT_size_text if size_text is None else size_text
  font = font_mngr.get_font(size_text)
  canvas = ImageDraw.ImageDraw(im)

  if bg:
    padding_text = int(app_cfg.DEFAULT_PADDING_PER * size_text) if padding_text is None else padding_text
    # bbox of text background
    bbox_bg = _bbox_from_text(pt, text, font, padding_text)
    _draw_bbox_pil(canvas, bbox_bg, color_bg, -1)
    # point of text origin
    bbox_text = bbox_bg.shift(padding_text, padding_text, -padding_text, -padding_text)
  else:
    bbox_text = _bbox_from_text(pt, text, font, 0)
  _draw_text_pil(canvas, text, Point.from_bbox(bbox_text), color_text, font)

  del canvas

  # ensure original format
  if was_np:
    im = im_utils.pil2np(im)
  return im



# -----------------------------------------------------------------------------
#
# init instances
#
# -----------------------------------------------------------------------------

font_mngr = FontManager()



# -----------------------------------------------------------------------------
#
# junkyard
#
# -----------------------------------------------------------------------------

# def draw_text_cv(im, text, pt, size=1.0, color=None):
#   """Draws degrees as text over image
#   """
#   if im_utils.is_pil(im):
#     im = im_utils.pil2np(im)
#     was_pil = True
#   else:
#     was_pil = False

#   dim = im.shape[:2][::-1]
#   pt_dim = pt.to_point_dim(dim)
#   color = app_cfg.GREEN if not color else color
#   rgb = color.rgb_int
#   cv.putText(im, text, pt_dim.xy, cv.text_HERSHEY_SIMPLEX, size, rgb, thickness=1, lineType=cv.LINE_AA)

#   if was_pil:
#     im = im_utils.pil2np(im)

#   return im


# -----------------------------------------------------------------------------
#
# BBoxes
#
# -----------------------------------------------------------------------------

# def draw_bbox_cv(im, bboxes, color, stroke):
#   """Draws BBox onto Numpy image using np broadcasting
#   :param bbox: BBoxDiim
#   :param color: Color
#   :param stroke: int
#   """
#   for bbox in bboxes:
#     im = cv.rectangle(im, bbox_dim.p1.xy, bbox_dim.p2.xy, color, stroke)
#   return im


# def _draw_bbox_np(im, bboxes, color, stroke):
#   """Draws BBox onto cv image using np broadcasting
#   :param bbox: BBoxDiim
#   :param color: Color
#   :param stroke: int
#   """
#   for bbox in bboxes:
#     im[bbox.y1:bbox.y2, bbox.x1:bbox.x2] = color.bgr_int
#   return im
