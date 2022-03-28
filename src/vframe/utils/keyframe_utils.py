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
import numpy as np

from vframe.utils import im_utils

log = logging.getLogger('VFRAME')



def create_focal_mask(wh, cross_dim=(0.5, 0.5), center_dim=(0.75, 0.75), margin=(0.1, 0.15)):
    """Create a binary mask to center focal weight
    param: wh: width, height tuple int
    param: cross_dim: the vertical and horizontal width of the focal area
    param: center_dim: the ratio of the center focal area
    param: margin: space between focal mask and edges
    returns: Numpy.ndaray bool of the binary focal mask
    """

    w,h = wh
    # cross zone
    x1 = int(( (0.5 - (cross_dim[0] / 2)) * w ))
    x2 = int(( (0.5 + (cross_dim[0] / 2)) * w ))
    y1 = int(( (0.5 - (cross_dim[1] / 2)) * h ))
    y2 = int(( (0.5 + (cross_dim[1] / 2)) * h ))
    focal_zone_col = (x1, 0, x2, h)
    focal_zone_row = (0, y1, w, y2)

    # middle zone
    x1 = int(( (0.5 - (center_dim[0] / 2)) * w ))
    x2 = int(( (0.5 + (center_dim[0] / 2)) * w ))
    y1 = int(( (0.5 - (center_dim[1] / 2)) * h ))
    y2 = int(( (0.5 + (center_dim[1] / 2)) * h ))
    focal_zone_center = (x1,y1,x2,y2)
    
    # stack
    focal_regions = [focal_zone_col, focal_zone_row,focal_zone_center]
    
    # init blank
    im_fm = np.zeros((h,w)).astype(np.bool)
    
    # iterate mask stack
    for x1,y1,x2,y2 in focal_regions:
      im_fm[y1:y2,x1:x2] = 1
    
    # apply margin
    im_margin = np.zeros_like(im_fm)
    mx, my = (int(margin[0] * w)//2, int(margin[1] * h)//2)
    im_margin[my:-my, mx:-mx] = 1
    im_fm = np.logical_and(im_fm, im_margin)
  
    return im_fm


def apply_focal_mask(im, im_mask):
  im_mask = im_mask.astype(np.uint8) * 255
  im = cv.bitwise_and(im, im, mask = im_mask)
  return im


def luma_delta(im_a,  im_b, opt_thresh=127/2, k=3, width=100, height=None):
  """Converts 2 BGR images to grayscale luma difference image
  :param im_a: np.ndarray LUMA uint8
  :param im_b: np.ndarray LUMA uint8
  :param opt_thresh: threshold for converting image to black/white
  :param k: kernel size for blurring
  :param width: image width
  :param height: image height
  :returns np.ndarray GRAY uint8 thresholded image
  """
  
  # apply blur, or not
  if k > 0:
    k = (k + 1) if not k % 2 else k
    im_a = cv.blur(im_utils.resize(im_a, width=width), ksize=(k, k))
    im_b = cv.blur(im_utils.resize(im_b, width=width), ksize=(k, k))
  
  # convert to luma
  im_a = im_utils.bgr2luma(im_a)
  im_b = im_utils.bgr2luma(im_b)
  im_delta = cv.absdiff(im_a, im_b)
  
  # find max
  #max_val = im_a.reshape((im_a.shape[0] * im_a.shape[1], 3)).max(axis=0)
  
  # threshold delta image to grayscale uint8 255
  thresh, im_delta = cv.threshold(im_delta, opt_thresh, 255, cv.THRESH_BINARY)
  per = cv.countNonZero(im_delta) / (im_delta.shape[0] * im_delta.shape[1])
  return per