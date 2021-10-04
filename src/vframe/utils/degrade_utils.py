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
import math

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2 as cv

from vframe.settings import app_cfg
from vframe.settings.app_cfg import LOG
from vframe.utils import im_utils
from vframe.utils.im_utils import ensure_pil, ensure_np
from vframe.models.geometry import BBox

"""
TODO
- glitch regions https://github.com/Kareeeeem/jpglitchhttps://github.com/Kareeeeem/jpglitch
- video codecs
- gif compression simulation
- logo/font overlay
"""


class Degrade:


  def __init__(self):
    pass

  @classmethod
  def sometimes(cls, rate):
    return random.random() <= rate


  @classmethod
  def map_range(cls, val, range_to):
    return np.interp(val, [0.0, 1.0], range_to)


  @classmethod
  def blend(cls, im_orig, im_new, alpha):
    """Blend the new image over the original image
    :param im_orig: numpy.ndarray original image
    :param im_new: numpy.ndarray new image
    :param alpha: (float) 0.0 - 1.0 of the new image
    :returns numpy.ndarray blended composite image
    """
    return cv.addWeighted(im_orig, 1.0 - alpha, im_new, alpha, 1.0)    


  @classmethod
  def equalize(cls, im, fac=1.0):
    """Equalize histograms using CLAHE
    :param im: numpy.ndarray bgr image
    :param alpha_range: alpha range for blending
    :returns numpy.ndarray bgr image
    """
    im = ensure_np(im)
    yuv = cv.cvtColor(im, cv.COLOR_BGR2YUV)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    # yuv[:, :, 0] = cv.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    im_eq = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
    return cls.blend(im, im_eq, fac)

  @classmethod
   def compress(cls, im, fac, im_type='jpg'):
    """Degrade image using JPEG or WEBP compression
    :param im: numpy.ndarray BGR image
    :param fac: image compression where 1.0 maps to quality=0, 0.0 maps to quality=100
    """
    q_flag = cv.IMWRITE_JPEG_QUALITY if im_type == 'webp' else cv.IMWRITE_JPEG_QUALITY
    q = int(np.interp(val, [0.0, 1.0], (100, 0)))
    # en/decode
    _, im_enc = cv.imencode(f'.{im_type}', im, (int(q_flag), quality))
    im = cv.imdecode(im_enc, cv.IMREAD_UNCHANGED)
    return im


  @classmethod
  def motion_blur_v(cls, im, fac):
    """Degrade image using vertical motion blur
    """
    w,h = im.shape[:2][::-1]
    k = max(1, int((fac * 0.01125) * max(w,h)))  # 0.01, 0.016
    k = k + 1 if k % 2 else k
    kernel_v = np.zeros((k, k))
    kernel_v[:, int((k - 1)/2)] = np.ones(k)  # Fill middle row with ones
    kernel_v /= k  # Normalize
    im_deg = cv.filter2D(im, -1, kernel_v)
    return im_deg


  @classmethod
  def motion_blur_h(cls, im, fac):
    """Degrade image using horizontal motion blur
    """
    w,h = im.shape[:2][::-1]
    k = max(1, int((fac * 0.01125) * max(w,h)))  # 0.01, 0.016
    k = k + 1 if k % 2 else k
    kernel_h = np.zeros((k, k)) 
    kernel_h[int((k - 1)/2), :] = np.ones(k)  # Fill middle row with ones
    kernel_h /= k  # Normalize
    im_deg = cv.filter2D(im, -1, kernel_h)
    return im_deg


  @classmethod
  def bilateral_blur(cls, im, fac):
    """Degrade image using bilateral blurring. This reduces texture and noise.
    """
    fac = np.interp(fac, [0.0, 1.0], [0.0, 0.1])
    dim_max = max(im.shape[:2])
    k = max(1, int(fac * dim_max))
    k = k if k % 2 else k + 1
    radius = k//5
    # blur = cv2.bilateralFilter(img,9,75,75)
    return cv.bilateralFilter(im, radius, k, k)


  @classmethod
  def gaussian_blur(cls, im, fac):
    """Degrade image using bilateral blurring. This reduces texture and noise.
    """
    fac = np.interp(fac, [0.0, 1.0], [0.0, 0.1])
    dim_max = max(im.shape[:2])
    k = max(1, int(fac * dim_max))
    k = k if k % 2 else k + 1
    # dst = cv.blur(src, (i, i))
    return cv.GaussianBlur(im, (k, k), 0)


  # @classmethod
  # def warp_ripple(cls, im, fac):

  #   A = im.shape[0] / 3.0
  #   w = 2.0 / im.shape[1]

  #   shift = lambda x: A * np.sin(2.0*np.pi*x * w)

  #   for i in range(im.shape[1]):
  #       im[:,i] = np.roll(im[:,i], int(shift(i)))

  #   return im


  @classmethod
  def destructive_scale(cls, im, fac):
    """Degrades image by reducing scale then rescaling to original size
    """
    amt = np.interp(val, [0.0, 1.0], (1.0, 0.25))
    w,h = im.shape[:2][::-1]
    nw,nh = (int(amt * w), int(amt * h))
    im_deg = im_utils.resize(im, width=nw, height=nh, interp=cv.INTER_CUBIC)
    im_deg = im_utils.resize(im_deg, width=w, height=h, force_fit=True, interp=cv.INTER_CUBIC)
    return im_deg


  @classmethod
  def _enhance(cls, im, enhancement, amt):
    im_pil = enhancement(ensure_pil(im)).enhance(amt)
    return im_utils.ensure_np(im_pil)


  @classmethod
  def sharpness(cls, im, fac):
    amt = np.interp(val, [0.0, 1.0], (0, 20))
    return cls._enhance(im, ImageEnhance.Sharpness, amt)


  @classmethod
  def brightness(cls, im, fac):
    amt = np.interp(val, [0.0, 1.0], (0, 1.5))
    return cls._enhance(im, ImageEnhance.Brightness, amt)


  @classmethod
  def contrast(cls, im, fac):
    amt = np.interp(val, [0.0, 1.0], (0,3))
    return cls._enhance(im, ImageEnhance.Contrast, amt)


  @classmethod
  def shift(cls, im, fac):
    """Degrades image by superimposing image with offset xy
    """
    w,h = im.shape[:2][::-1]
    max_px = int(max(im.shape[:2]) * 0.02)
    D = max(1, int(np.interp(val, [0.0, 1.0], (0, max_px))))
    rad = random.uniform(0, 2 * math.pi)
    dx = int(math.cos(rad) / D)
    dy = int(math.sin(rad) / D)
    # pad
    im_deg = cv.copyMakeBorder(im, D, D, D, D, cv.BORDER_CONSTANT, value=[0,0,0])
    # paste
    x1,y1,x2,y2 = list(np.array([0,0,w,h]) + np.array([dx, dy, -dx, -dy]))
    # crop
    xyxy = list(np.array([0,0,w,h]) + np.array([D, D, -D, -D]))
    bbox = BBox(*xyxy, w + D, h + D)
    im_deg = im_utils.crop_roi(im_deg, bbox)
    # scale
    im_deg = im_utils.resize(im_deg, width=w, height=h, force_fit=True)
    # blend
    alpha = random.uniform(0.1, 0.35)
    return cls.blend(im, im_deg, alpha)


  @classmethod
  def chromatic_aberration(cls, im, fac, channel=0):
    """Scale-shift color channel and then superimposes it back into image
    :param channel: int for BGR channel 0 = B
    """
    # TODO: use shift method to overlay channels
    channel = channel if channel else random.randint(0,2)
    w,h = im.shape[:2][::-1]
    im_c = im.copy()
    # dx,dy = value_range
    dx = np.interp(val, [0.0, 1.0], (0, 5))
    dy = np.interp(val, [0.0, 1.0], (0, 5))
    # inner crop
    xyxy = list(np.array([0,0,w,h]) + np.array([dx, dy, -dx, -dy]))
    bbox = BBox(*xyxy, w, h)
    im_c = im_utils.crop_roi(im_c, bbox)
    # resize back to original dims
    im_c = im_utils.resize(im_c, width=w, height=h, force_fit=True)
    # add color channel
    im_deg = im.copy()
    im_deg[:,:,channel] = im_c[:,:,channel]
    return im_deg


#   @classmethod
#   def compress_deprecated(cls, im, fac):
#     """Degrade image using JPEG compression
#     :param im: numpy.ndarray bgr image
#     :param alpha_range: alpha range for blending
#     """
#     im_pil = ensure_pil(im)
#     buffer = BytesIO()
#     q = int(np.interp(val, [0.0, 1.0], (90, 12)))  # 0 = 100 = no compression)
#     im_pil.save(buffer, 'JPEG', quality=q)
#     buffer.seek(0)
#     im_pil = Image.open(buffer)
#     return im_utils.pil2np(im_pil)