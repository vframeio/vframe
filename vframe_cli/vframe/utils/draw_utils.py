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
from vframe.utils import im_utils
from vframe.settings import app_cfg


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

class DrawUtils:

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

  def draw_rotated_bbox_cv(self, im, rbbox_norm, stroke_weight=2, color=None):
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
    color_rgb = color.to_rgb_int()
    for i, p in enumerate(vertices):
      p1 = vertices[i]
      p2 = vertices[(i + 1) % 4]
      if stroke_weight == -1:
        im = cv.line(im, p1.xy, p2.xy, color_rgb, 0, cv.LINE_AA, -1)
      else:
        im = cv.line(im, p1.xy, p2.xy, color_rgb, stroke_weight, cv.LINE_AA)

    if was_pil:
      im = im_utils.pil2np(im)

    return im


  def draw_rotated_bbox_pil(self, im, rbbox_norm, stroke_weight=2, color=None, expand=0.0):
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
    color_rgb = color.to_rgb_int()
    canvas = ImageDraw.Draw(im)

    if stroke_weight == -1:
      canvas.polygon(vertices, fill=color_rgb)
    else:
      canvas.polygon(vertices, outline=color_rgb)

    del canvas

    if was_np:
      im = im_utils.pil2np(im)

    return im


  def draw_mask(self, im, bbox_norm, mask, threshold=0.3,  mask_blur_amt=21, color=None, blur_amt=None, color_alpha=0.6):
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
      color_rgb = color.to_rgb_int()[::-1]  # rgb to bgr
      roi = ((color_alpha * np.array(color_rgb)) + ((1 - color_alpha) * roi))
    # store the blended ROI in the original image
    im[y1:y2,x1:x2][mask] = roi.astype("uint8")
    return im


  def draw_bbox_cv(self, im, bboxes_norm, color=(0,255,0), stroke_weight=2, expand=0.0):
    '''Draws BBox onto cv image
    '''
    if im_utils.is_pil(im):
      im = im_utils.pil2np(im)
      was_pil = True
    else:
      was_pil = False

    if not type(bboxes_norm) == list:
      bboxes_norm = [bboxes_norm]

    for bbox_norm in bboxes_norm:
      bbox_dim = bbox_norm.expand(expand).to_bbox_dim(im.shape[:2][::-1])
      #rgb_int = color.to_rgb_int()
      im = cv.rectangle(im, bbox_dim.p1.xy, bbox_dim.p2.xy, color, stroke_weight)

    if was_pil:
      im = im_utils.pil2np(im)

    return im


  def draw_bbox_np(self, im, bboxes_norm, color=(0,255,0), expand=0.0):
    '''Draws BBox onto cv image using np broadcasting
    '''
    if im_utils.is_pil(im):
      im = im_utils.pil2np(im)
      was_pil = True
    else:
      was_pil = False

    rgb_int = color.to_rgb_int()
    bgr_int = rgb_int[::-1]
    
    if not type(bboxes_norm) == list:
      bboxes_norm = [bboxes_norm]

    for bbox_norm in bboxes_norm:
      bbox_dim = bbox_norm.expand(expand).to_bbox_dim(im.shape[:2][::-1])
      im[bbox_dim.y1:bbox_dim.y2, bbox_dim.x1:bbox_dim.x2] = bgr_int

    if was_pil:
      im = im_utils.pil2np(im)

    return im
    

  def draw_bbox_pil(self, im, bboxes_norm, color, stroke_weight=2, fill=True, expand=0.0):
    '''Draws BBox onto cv image
    :param color: RGB value
    '''
    if im_utils.is_np(im):
      im = im_utils.np2pil(im)
      was_np = True
    else:
      was_np = False

    if not type(bboxes_norm) == list:
      bboxes_norm = [bboxes_norm]
    
    im_draw = ImageDraw.ImageDraw(im)
    rgb_int = color.to_rgb_int()
    for bbox_norm in bboxes_norm:
      bbox_dim = bbox_norm.expand(expand).to_bbox_dim(im.size)
      xyxy = (bbox_dim.p1.xy, bbox_dim.p2.xy)  
      if stroke_weight == -1:
        im_draw.rectangle(xyxy, fill=rgb_int)
      else:
        im_draw.rectangle(xyxy, outline=rgb_int, width=stroke_weight)
    del im_draw

    if was_np:
      im = im_utils.pil2np(im)

    return im


  def draw_bbox_labeled_pil(self, im, bbox_nlc, stroke_weight=2, font_size=14, expand=0.0):
    '''Draws bbox and annotation
    :param im: PIL image RGB
    :param nlc: BBoxNormLabeledColor
    '''
    if im_utils.is_np(im):
      im = im_utils.np2pil(im)
      was_np = True
    else:
      was_np = False

    fill_color = bbox_nlc.color.to_rgb_int()
    font_color = bbox_nlc.color.get_fg_color().to_rgb_int()
    canvas = ImageDraw.ImageDraw(im)
    bbox_dim = bbox_nlc.expand(expand).to_bbox_dim(im.size)
    xyxy = bbox_dim.xyxy
    if stroke_weight == -1:
      canvas.rectangle(xyxy, fill=fill_color)
    else:
      canvas.rectangle(xyxy, outline=fill_color, width=stroke_weight)

    # draw label
    t = bbox_nlc.label.upper()
    font = self.get_font(font_size)
    tw, th = font.getsize(t)
    xfac, yfac = (1.2, 1.4)
    pad_left, pad_top = (0.055, 0.1)

    if th * yfac > bbox_dim.y1:
      # draw inside
      xyxy = (bbox_dim.x1, bbox_dim.y1, tw * xfac + bbox_dim.x1, bbox_dim.y1 + (th * yfac))
      x1, y1, x2, y2 = xyxy
      canvas.rectangle(xyxy, fill=fill_color)
      canvas.text((x1 + (pad_left * tw), y1 + (pad_top * th)), t, font_color, font)
    else:
      # draw on top
      xyxy = (0 + bbox_dim.x1, bbox_dim.y1 - (th * yfac), tw * xfac + bbox_dim.x1, bbox_dim.y1)
      x1,y1,x2,y2 = xyxy
      canvas.rectangle(xyxy, fill=fill_color)
      canvas.text((x1 + (pad_left * tw), y1 + (pad_top * th)), t, font_color, font)
    del canvas

    if was_np:
      im = im_utils.pil2np(im)

    return im



  def draw_text_pil(self, im, pt, text, font_size=14, color=(255,255,255), knockout=(0,0), knockout_color=(0,0,0)):
    """Untested
    """
    if im_utils.is_np(im):
      im = im_utils.np2pil(im)
      was_np = True
    else:
      was_np = False

    dim = im.size
    pt_dim = pt.to_point_dim(dim)

    font = self.get_font(font_size)
    tw, th = font.getsize(text)
    canvas = ImageDraw.ImageDraw(im)
    
    if any(knockout) > 0:
      canvas.text((pt_dim.x + knockout[0], pt_dim.y - th + knockout[1]), text, knockout_color, font)

    canvas.text((pt_dim.x, pt_dim.y - th), text, color, font)
    
    del canvas

    if was_np:
      im = im_utils.pil2np(im)

    return im


  def draw_text_cv(self, im, pt, text, size=1.0, color=(255,255,255)):
    '''Draws degrees as text over image
    '''
    if im_utils.is_pil():
      im = im_utils.pil2np()
      was_pil = True
    else:
      was_pil = False

    dim = im.shape[:2][::-1]
    pt_dim = pt.to_point_dim(dim)
    cv.putText(im, text, pt.xy, cv.FONT_HERSHEY_SIMPLEX, size, color, thickness=1, lineType=cv.LINE_AA)

    if was_pil:
      im = im_utils.pil2np(im)

    return im


