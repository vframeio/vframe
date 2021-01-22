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
from vframe.utils.misc_utils import odd, even
from vframe.models.geometry import BBox

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
      if len(im.shape) == 2:
        color_mode = 'L'
      elif im.shape[2] == 4:
        im = bgra2rgba(im)
        color_mode = 'RGBA'
      elif im.shape[2] == 3:
        im = bgr2rgb(im)
        color_mode = 'RGB'
    else:
      print('wt')
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
    if len(im.shape) == 2:
      # grayscale, ignore swap and return current image
      return im
    elif len(im.shape) > 2 and im.shape[2] == 4:
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


def blur_bbox(im, bboxes, per=0.33, iters=1):
  """Blur ROI
  :param im: (np.ndarray) image BGR
  :param bbox: (BBox)
  :param cell_size: (int, int) pixellated cell size
  :returns (np.ndarray) BGR image
  """
  if not bboxes:
    return im
  elif not type(bboxes) == list:
    bboxes = list(bboxes)

  for bbox in bboxes:
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


def pixellate_bbox(im, bboxes, cell_size=(5,6), expand_per=0.0):
  """Pixellates ROI using Nearest Neighbor inerpolation
  :param im: (numpy.ndarray) image BGR
  :param bbox: (BBox)
  :param cell_size: (int, int) pixellated cell size
  :returns (numpy.ndarray) BGR image
  """
  if not bboxes:
    return im
  elif not type(bboxes) == list:
    bboxes = list(bboxes)

  for bbox in bboxes:
    if expand_per > 0:
      bbox = bbox.expand_per(expand_per)
    x1,y1,x2,y2 = bbox.xyxy_int
    im_roi = im[y1:y2, x1:x2]
    h,w,c = im_roi.shape
    # pixellate
    im_roi = cv.resize(im_roi, cell_size, interpolation=cv.INTER_NEAREST)
    im_roi = cv.resize(im_roi, (w,h), interpolation=cv.INTER_NEAREST)
    im[y1:y2, x1:x2] = im_roi

  return im



def mk_mask(bbox, shape='ellipse', blur_kernel_size=None, blur_iters=1):
  bboxes = bbox if isinstance(bbox, list) else [bbox]
  # mk empty mask
  im_mask = create_blank_im(*bboxes[0].dim, 1)
  # draw mask shapes
  color = (255,255,255)
  for bbox in bboxes:
    if shape == 'rectangle':
      im_mask = cv.rectangle(im_mask, bbox.p1.xy_int, bbox.p2.xy_int, color, -1)
    elif shape == 'circle':
      im_mask = cv.circle(im_mask, bbox.cxcy_int, bbox.w, color, -1)
    elif shape == 'ellipse':
      im_mask = cv.ellipse(im_mask, bbox.cxcy_int, bbox.wh_int, 0, 0, 360, color, -1)
  # blur if k
  k = blur_kernel_size
  if k:
    k = odd(k)
    for i in range(blur_iters):
      im_mask = cv.GaussianBlur(im_mask, (k, k), k, k)
  return im_mask

def mask_composite(im, im_masked, im_mask):
  """Masks two images together using grayscale mask
  :param im: the base image
  :param im_masked: the image that will be masked on top of the base image
  :param im_mask: the grayscale image used to mask
  :returns (numpy.ndarray): masked composite image
  """
  im_mask_alpha = im_mask / 255.
  im = im_mask_alpha[:, :, None] * im_masked + (1 - im_mask_alpha)[:, :, None] * im
  return (im).astype(np.uint8)


def blur_bbox_soft(im, bbox, iters=1, expand_per=-0.1, multiscale=True,
                              mask_k_fac=0.125, im_k_fac=0.33, shape='ellipse'):
  """Blurs objects using multiple blur scale per bbox
  """
  if not bbox:
    return im
  bboxes = bbox if isinstance(bbox, list) else [bbox]
  bboxes_mask = [b.expand_per(expand_per, keep_edges=True) for b in bboxes]
  if multiscale:
    # use separate kernel size for each bboxes (slower but more accurate)
    im_blur = im.copy()
    dim = im.shape[:2][::-1]
    im_mask = create_blank_im(*dim, 1)
    for bbox in bboxes_mask:
      # create a temp mask, draw shape, blur, and add to cummulative mask
      k = min(bbox.w_int, bbox.h_int)
      k_mask = odd(int(k * mask_k_fac))  # scale min bbox dim for desired blur intensity
      im_mask_next = mk_mask(bbox, shape=shape, blur_kernel_size=k_mask, blur_iters=iters)
      bounding_rect = cv.boundingRect(im_mask_next)
      bbox_blur = BBox.from_xywh(*bounding_rect, *bbox.dim)
      im_mask = cv.add(im_mask, im_mask_next)
      # blur the masked area bbox in the original image

    k = max([min(b.w_int, b.h_int) for b in bboxes])
    k_im = odd(int(k * im_k_fac)) # scaled image blur kernel
    im_blur = cv.GaussianBlur(im, (k_im,k_im), k_im/4, 0)
  else:
    # use one kernel size for all bboxes (faster but less accurate)
    k = max([min(b.w_int, b.h_int) for b in bboxes])
    k_im = odd(int(k * im_k_fac)) # scaled image blur kernel
    k_mask = odd(int(k * mask_k_fac))  # scale min bbox dim for desired blur intensity
    im_mask = mk_mask(bboxes, shape=shape, blur_kernel_size=k_mask, blur_iters=iters)
    im_blur = cv.GaussianBlur(im, (k_im,k_im), k_im, 0, k_im)

  # iteratively blend image
  k = max([min(b.w_int, b.h_int) for b in bboxes])
  k_im = odd(int(k * im_k_fac)) # scaled image blur kernel
  im_dst = im.copy()
  for i in range(iters):
    im_alpha = im_mask / 255.
    im_dst = im_alpha[:, :, None] * im_blur + (1 - im_alpha)[:, :, None] * im_dst
    im_mask = cv.GaussianBlur(im_mask, (k_im, k_im), k_im, 0)

  #im_dst = mask_composite(im, im_blur, im_mask)

  return (im_dst).astype(np.uint8)



def swap_color(im, color_src, color_dst):
  """Swaps colors in image
  :param im: (numpy.ndarray) in BGR
  :param color_src: (Color) source color
  :param color_dst: (Color) destination color
  :returns im: (numpy.ndarray) in BGR
  """
  bgr = list(color_src.to_bgr_int())
  idxs = np.all(im == bgr, axis=2)
  im[idxs] = list(color_dst.to_bgr_int())
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
  """FIXME: Resizes image, scaling issues with floating point numbers
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
        #scale_x = width / im_width
        #scale_y = height / im_height
        w, h = width, height
      else:
        scale_x = min(width / im_width, height / im_height)
        scale_y = scale_x
        w, h = int(scale_x * im_width), int(scale_y * im_height)
    elif width and not height:
      scale_x = width / im_width
      scale_y = scale_x
      w, h = int(scale_x * im_width), int(scale_y * im_height)
    elif height and not width:
      scale_y = height / im_height
      scale_x = scale_y
      w, h = int(scale_x * im_width), int(scale_y * im_height)
    w,h = int(w), int(h)
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






# -----------------------------------------------------------------------------
#
# Deprecated. For reference.
#
# -----------------------------------------------------------------------------

def _deprecated_circle_blur_soft_edges(im, bboxes, im_ksize=51, mask_ksize=51,
  sigma_x=51, sigma_y=None, iters=2):
  """Blurs ROI using soft edges
  """
  if not bboxes:
    return im
  elif not type(bboxes) == list:
    bboxes = list(bboxes)

  # force kernels odd
  im_ksize = im_ksize if im_ksize % 2 else im_ksize + 1
  mask_ksize = mask_ksize if mask_ksize % 2 else mask_ksize + 1
  sigma_x = sigma_x if sigma_x % 2 else sigma_x + 1
  sigma_y = sigma_y if sigma_y else sigma_x

  # mk empty mask
  h,w,c = im.shape
  im_mask = np.zeros((h,w))

  # draw mask shapes
  for bbox in bboxes:
    #im_mask = cv.rectangle(im_mask, bbox.p1.xy_int, bbox.p2.xy_int, (255,255,255), -1)
    #im_mask = cv.circle(im_mask, bbox.cxcy_int, bbox.w,(255,255,255), -1)
    im_mask = cv.ellipse(im_mask, bbox.cxcy_int, bbox.wh_int, 0, 0, 360, (255,255,255), -1)

  # use sigma 1/4 size of blur kernel
  im_blur = cv.GaussianBlur(im, (im_ksize,im_ksize), im_ksize//4, 0, im_ksize//4)
  im_dst = im.copy()

  for i in range(iters):
    im_mask = cv.blur(im_mask, ksize=(mask_ksize, mask_ksize))
    im_mask = cv.GaussianBlur(im_mask, (mask_ksize, mask_ksize), mask_ksize, mask_ksize)
    im_alpha = im_mask / 255.
    im_dst = im_alpha[:, :, None] * im_blur + (1 - im_alpha)[:, :, None] * im_dst

  return (im_dst).astype(np.uint8)


def _deprecated_compound_blur_bboxes(im, bboxes, iters=2, expand_per=-0.15, opt_pixellate=True):
  """Pixellates and blurs object
  """
  if not bboxes:
    return im
  elif not type(bboxes) == list:
    bboxes = list(bboxes)

  k = max([min(b.w_int, b.h_int) for b in bboxes])
  im_ksize = k // 2  # image blur kernel
  mask_ksize = k // 10  # mask blur kernel

  print('mask_ksize', mask_ksize, 'im k', im_ksize)
  bboxes_inner = [b.expand_per(expand_per, keep_edges=True) for b in bboxes]
  im = circle_blur_soft_edges(im, bboxes_inner, im_ksize=im_ksize, mask_ksize=mask_ksize, iters=iters)
  return im
