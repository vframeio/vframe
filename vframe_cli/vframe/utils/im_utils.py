############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import logging

import cv2 as cv
from PIL import Image
import numpy as np


log = logging.getLogger('vframe')

def np2pil(im, swap=True):
  """Ensure image is Pillow format
    :param im: image in numpy or PIL.Image format
    :returns image in Pillow RGB format
  """
  try:
    im.verify()
    log.warn('Expected Numpy received PIL')
    return im
  except:
    if swap:
      if im.shape[2] == 4:
        im = bgra2rgba(im)
        color_mode = 'RGBA'
      elif im.shape[2] == 3:
        im = bgr2rgb(im)
        color_mode = 'RGB'
    return Image.fromarray(im.astype('uint8'), color_mode)


def pil2np(im, swap=True):
  """Ensure image is Numpy.ndarry format
    :param im: image in numpy or PIL.Image format
    :returns image in Numpy uint8 format
  """
  if type(im) == np.ndarray:
    log.warn('Expected PIL received Numpy')
    return im
  im = np.asarray(im, np.uint8)
  if swap:
    if im.shape[2] == 4:
      im = bgra2rgba(im)
    elif im.shape[2] == 3:
      im = bgr2rgb(im)
  return im


def is_pil(im):
  '''Ensures image is Pillow format
  :param im: PIL.Image image
  :returns bool if is PIL.Image
  '''
  try:
    im.verify()
    return True
  except:
    return False


def is_np(im):
  '''Checks if image if numpy
  '''
  return type(im) == np.ndarray
  

def num_channels(im):
  '''Number of channels in numpy.ndarray image
  '''
  if len(im.shape) > 2:
    return im.shape[2]
  else:
    return 1


def is_grayscale(im, threshold=5):
  """Returns True if image is grayscale
  :param im: (numpy.array) image
  :return (bool) of if image is grayscale"""
  b = im[:,:,0]
  g = im[:,:,1]
  mean = np.mean(np.abs(g - b))
  return mean < threshold


def crop_roi(im, bbox):
  """Crops ROI
  :param im: (np.ndarray) image BGR
  :param bbox: (BBox)
  :returns (np.ndarray) BGR image ROi
  """
  dim = im.shape[:2][::-1]
  x1, y1, x2, y2 = bbox.xyxy_int
  im_roi = im[y1:y2, x1:x2]
  return im_roi


def blur_roi(im, bbox, per=0.33, iters=1):
  """Blur ROI
  :param im: (np.ndarray) image BGR
  :param bbox: (BBox)
  :param cell_size: (int, int) pixellated cell size
  :returns (np.ndarray) BGR image
  """
  dim = im.shape[:2][::-1]
  x1, y1, x2, y2 = bbox.xyxy_int
  im_roi = im[y1:y2, x1:x2]
  h,w,c = im_roi.shape
  ksize = int(max(per * w, per * h))
  ksize = ksize if ksize % 2 else ksize + 1
  for n in range(iters):
    im_roi = cv.blur(im_roi, ksize=(ksize,ksize))
    #im_roi = cv.GaussianBlur(im_roi, (ksize,ksize), 0)
  im[y1:y2, x1:x2] = im_roi
  return im


def pixellate_roi(im, bbox, cell_size=(1,1)):
  """Pixellates ROI
  :param im: (np.ndarray) image BGR
  :param bbox: (BBox)
  :param cell_size: (int, int) pixellated cell size
  :returns (np.ndarray) BGR image
  """
  dim = im.shape[:2][::-1]
  x1, y1, x2, y2 = bbox.xyxy_int
  im_roi = im[y1:y2, x1:x2]
  h,w,c = im_roi.shape
  fw,fh = cell_size

  #im_roi = resize(im_roi, width=fw, height=fh, interp=cv.INTER_NEAREST, force_fit=True)
  im_roi = cv.resize(im_roi, cell_size, interpolation=cv.INTER_NEAREST)
  #im_roi = resize(im_roi, width=w, height=h, interp=cv.INTER_NEAREST, force_fit=True)
  im_roi = cv.resize(im_roi, (w,h), interpolation=cv.INTER_NEAREST)
  im[y1:y2, x1:x2] = im_roi
  return im


# -----------------------------------------------------------------------------
#
# Placeholder images
#
# -----------------------------------------------------------------------------


def create_blank_im(w, h, c=3, dtype=np.uint8):
  """Creates blank np image
  :param w: width
  :param h: height
  :param c: channels
  :param dtype: data type
  :returns (np.ndarray)
  """
  if c == 1:
    im = np.zeros([h, w], dtype=np.uint8)
  elif c == 3:
    im = np.zeros([h, w, c], dtype=np.uint8)
  else:
    im = None  # TODO handle error
  return im


def create_random_im(w, h, c=3, low=0, high=255, dtype=np.uint8):
  """Creates blank np image
  :param w: width
  :param h: height
  :param c: channels
  :param dtype: data type
  :returns (np.ndarray)
  """
  if c == 1:
    im = np.random.randint(low, high, (h * w)).reshape((h, w)).astype(dtype)
  elif c == 3:
    im = np.random.randint(low, high, (h * w * c)).reshape((h, w, c)).astype(dtype)
  else:
    im = None  # TODO handle error
  return im


# -----------------------------------------------------------------------------
#
# Resizing
#
# -----------------------------------------------------------------------------

def resize(im, width=None, height=None, force_fit=False, interp=cv.INTER_LINEAR):
  """Resizes image
  :param im: (nump.ndarray) image
  :param width: int
  :param height: int
  :returns (nump.ndarray) image
  """
  if not width and not height:
    return im
  else:
    im_width, im_height = im.shape[:2][::-1]
    if width and height:
      if force_fit:
        scale_x = width / im_width
        scale_y = height / im_height
      else:
        scale_x = min(width / im_width, height / im_height)
        scale_y = scale_x
    elif width and not height:
      scale_x = width / im_width
      scale_y = scale_x
    elif height and not width:
      scale_y = height / im_height
      scale_x = scale_y
    w, h = int(scale_x * im_width), int(scale_y * im_height)
    im = cv.resize(im, (w ,h), interpolation=interp)
    return im



# -----------------------------------------------------------------------------
#
# OpenCV aliases
#
# -----------------------------------------------------------------------------

def bgr2gray(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGR)
    :returns Numpy.ndarray (Gray)
  """
  return cv.cvtColor(im, cv.COLOR_BGR2GRAY)

def gray2bgr(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (Gray)
    :returns Numpy.ndarray (BGR)
  """
  return cv.cvtColor(im, cv.COLOR_GRAY2BGR)

def bgr2rgb(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGR)
    :returns Numpy.ndarray (RGB)
  """
  return cv.cvtColor(im, cv.COLOR_BGR2RGB)

def rgb2bgr(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (RGB)
    :returns Numpy.ndarray (RGB)
  """
  return cv.cvtColor(im, cv.COLOR_RGB2BGR)

def bgra2rgba(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (BGRA)
    :returns Numpy.ndarray (RGBA)
  """
  return cv.cvtColor(im, cv.COLOR_BGRA2RGBA)

def rgba2bgra(im):
  """Wrapper for cv2.cvtColor transform
    :param im: Numpy.ndarray (RGB)
    :returns Numpy.ndarray (RGB)
  """
  return cv.cvtColor(im, cv.COLOR_RGBA2BGRA)

def bgr2luma(im):
  """Converts BGR image to grayscale Luma
  :param im: np.ndarray BGR uint8
  :returns nd.array GRAY uint8
  """
  im_y, _, _ = cv.split(cv.cvtColor(im, cv.COLOR_BGR2YUV))
  return im_y