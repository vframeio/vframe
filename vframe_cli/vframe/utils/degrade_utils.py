############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import random
from io import BytesIO

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2 as cv

from vframe.settings import app_cfg
from vframe.utils import im_utils
from vframe.models.geometry import BBox

"""
TODO
- glitch regions https://github.com/Kareeeeem/jpglitchhttps://github.com/Kareeeeem/jpglitch
- create standalone version without ranges
"""


log = app_cfg.LOG

def quality(im, value_range=(30,90), rate=0.5, alpha_range=(0.25, 0.75)):
  """Degrade image using JPEG compression
  """
  if random.random() > rate:
    return im
  im_pil = im_utils.np2pil(im)
  buffer = BytesIO()
  quality = random.randint(*value_range)
  im_pil.save(buffer, 'JPEG', quality=quality)
  buffer.seek(0)
  im_pil = Image.open(buffer)
  im_adj = im_utils.pil2np(im_pil)
  # blend
  alpha = random.uniform(*alpha_range)
  im_adj = cv.addWeighted(im, 1.0 - alpha, im_adj, alpha, 1.0)
  return im_adj
  

def motion_blur_v(im, value_range=(0.25, 0.75), rate=0.5,  alpha_range=(0.25, 0.75)):
  """Degrade image using vertical motion blur
  """
  if random.random() > rate:
    return im
  w,h = im.shape[:2][::-1]
  
  k = max(1, int((random.uniform(*value_range) * 0.03) * max(w,h)))  # 0.01, 0.016
  k = k + 1 if k % 2 else k
  kernel_v = np.zeros((k, k))
  kernel_v[:, int((k - 1)/2)] = np.ones(k)  # Fill middle row with ones
  kernel_v /= k  # Normalize
  im_adj = cv.filter2D(im, -1, kernel_v)
  # blend
  alpha = random.uniform(*alpha_range)
  im_adj = cv.addWeighted(im, 1.0 - alpha, im_adj, alpha, 1.0)
  return im_adj


def motion_blur_h(im, value_range=(0.25, 0.75), rate=0.5, alpha_range=(0.25, 0.75)):
  """Degrade image using horizontal motion blur
  """
  if random.random() > rate:
    return im
  w,h = im.shape[:2][::-1]
  k = max(1, int((random.uniform(*value_range) * 0.03) * max(w,h)))  # 0.01, 0.016
  k = k + 1 if k % 2 else k
  kernel_h = np.zeros((k, k)) 
  kernel_h[int((k - 1)/2), :] = np.ones(k)  # Fill middle row with ones
  kernel_h /= k  # Normalize
  im_adj = cv.filter2D(im, -1, kernel_h)
  #im_adj = cv.GaussianBlur(im, (kernel_h,kernel_h), 0, 0, kernel_h)
  # blend
  alpha = random.uniform(*alpha_range)
  im_adj = cv.addWeighted(im, 1.0 - alpha, im_adj, alpha, 1.0)
  return im_adj


def bilateral_blur(im, fac=1.0, k=None, rate=0.5):
  """Degrade image using bilateral blurring. This reduces texture and noise.
  """
  if random.random() > rate:
    return im
  #value_range = list(np.clip(value_range, 0.0, 1.0))
  if k is None:
    # randomly choose kernel based on image size
    w,h = im.shape[:2][::-1]
    k = max(1, int(w * random.uniform(0.0, 0.016)))
  
  im_adj = cv.bilateralFilter(im, (k, k))
  
  if fac < 1.0 and fac > 0.0:
    im_adj = cv.addWeighted(im, 1.0 - fac, im_adj, fac, 1.0)
  return im_adj


def scale(im, value_range=(0.25,0.75), rate=0.5, alpha_range=(0.25, 0.75)):
  """Degrades image by reducing scale then rescaling to original size
  """
  if random.random() > rate:
    return im
  value_range = list(np.clip(value_range, 0.1, 1.0))
  scale = random.uniform(*value_range)
  w,h = im.shape[:2][::-1]
  nw,nh = (int(scale * w), int(scale * h))
  im_adj = im_utils.resize(im, width=nw, height=nh)
  im_adj = im_utils.resize(im_adj, width=w, height=h, force_fit=True)
  # blend
  alpha = random.uniform(*alpha_range)
  im_adj = cv.addWeighted(im, 1.0 - alpha, im_adj, alpha, 1.0)
  return im_adj


def enhance(im, enhancement, value_range=(0.5, 10), rate=0.5, alpha_range=(0.25, 0.75)):
  if random.random() > rate:
    return im
  im_adj = im_utils.np2pil(im)
  value_range = list(np.clip(value_range, 0.0, 1.0))
  if enhancement == 'sharpness':
    enhancer = ImageEnhance.Sharpness(im_adj)
  elif enhancement == 'brightness':
    enhancer = ImageEnhance.Brightness(im_adj)
  elif enhancement == 'contrast':
    enhancer = ImageEnhance.Contrast(im_adj)
  elif enhancement == 'color':
    enhancer = ImageEnhance.Color(im_adj)
  else:
    log.error(f'{enhancement} not valid option')
    return im_utils.pil2np(im_adj)
  amt = random.uniform(*value_range)
  im_adj = enhancer.enhance(amt)
  del enhancer
  im_adj = im_utils.pil2np(im_adj)
  # blend
  alpha = random.uniform(*alpha_range)
  im_adj = cv.addWeighted(im, 1.0 - alpha, im_adj, alpha, 1.0)
  return im_adj


def auto_adjust(im, opt, alpha_range=(0.25, 0.75), rate=0.5):
  """TODO
  """
  if random.random() > rate:
    return im
  value_range = list(np.clip(alpha_range, 0.0, 1.0))
  alpha = random.uniform(*alpha_range)
  # convert to pil
  im = im_utils.np2pil(im)
  # image op
  if opt == 'equalize':
    im_adj = ImageOps.equalize(im)
    im_adj = Image.blend(im, im_adj, alpha)
  elif opt == 'autocontrast':
    im_adj = ImageOps.autocontrast(im, cutoff=0)
    im_adj = Image.blend(im, im_adj, alpha)
  else:
    log.error(f'{opt} not valid option')
  # return np
  im_adj = im_utils.pil2np(im_adj)
  return im_adj


def zoom_shift(im, value_range=(1, 10), rate=0.5, alpha_range=(0.25, 0.75)):
  """Degrades image by superimposing image with offset xy
  """
  if random.random() > rate:
    return im
  w,h = im.shape[:2][::-1]
  dx,dy = value_range
  # crop
  xyxy = list(np.array([0,0,w,h]) + np.array([dx, dy, -dx, -dy]))
  value_range_xy2 = (1.0 - value_range[0], 1.0 - value_range[1])
  bbox = BBox(*xyxy, w, h)
  im_adj = im_utils.crop_roi(im, bbox)
  # scale
  im_adj = im_utils.resize(im_adj, width=w, height=h, force_fit=True)
  # blend
  alpha = random.uniform(*alpha_range)
  im_adj = cv.addWeighted(im, 1.0 - alpha, im_adj, alpha, 1.0)
  return im_adj


def chromatic_aberration(im, channel=None, value_range=(1,10), rate=0.5, alpha_range=(0.25, 0.75)):
  """Scale-shift color channel and then superimposes it back into image
  :param channel: int for BGR channel 0 = B
  :param value_range: int pixel shift
  """
  if not channel:
    channel = random.randint(0,2)
  w,h = im.shape[:2][::-1]
  im_c = im.copy()
  dx,dy = value_range
  # crop
  xyxy = list(np.array([0,0,w,h]) + np.array([dx, dy, -dx, -dy]))
  value_range_xy2 = (1.0 - value_range[0], 1.0 - value_range[1])
  bbox = BBox(*xyxy, w, h)
  im_c = im_utils.crop_roi(im_c, bbox)
  # resize
  im_c = im_utils.resize(im_c, width=w, height=h, force_fit=True)
  # add color channel
  im_adj = im.copy()
  im_adj[:,:,channel] = im_c[:,:,channel]
  # blend
  alpha = random.uniform(*alpha_range)
  im_adj = cv.addWeighted(im, 1.0 - alpha, im_adj, alpha, 1.0)
  return im_adj