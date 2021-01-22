############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import numpy as np
import cv2 as cv

from vframe.models.geometry import BBox


# FIXME: Check Blender color classes or modify settings
# Pixel color classes are sometimes off by +/-1 in in each RGB color

def get_color_boundaries(color):
  '''Creates +/-1 padded pixel in each rgb direction
  This is because of Blender issue with rendering exact pixel colors
  '''
  c = list(color).copy()
  colors = [c]
  # minus
  colors.append([max(0, c[0] - 1), c[1], c[2]])  # R-1
  colors.append([c[0], max(0, c[1] - 1), c[2]])  # G-1
  colors.append([c[0], c[1], max(0, c[2] - 1)])  # B-1
  colors.append([max(0, x - 1) if x is not 0 else 0 for x in c])  # RGB-1
  # add
  colors.append([min(255, c[0] + 1), c[1], c[2]])  # R+1
  colors.append([c[0], min(255, c[1] + 1), c[2]])  # R+1
  colors.append([c[0], c[1], min(255, c[2] + 1)])  # R+1
  colors.append([min(255, x + 1) if x is not 0 else 0 for x in c])  # R+1
  return colors


def color_mask_to_rect(im, color, threshold=40):
  '''Converts color masks areas to BBoxes for RGB image
  :param im: (numpy) image in RGB
  :param color: RGB uint8 color tuple
  :param threshold: minimum number of non-zero pixels
  :returns (BBox) or None
  '''
  im_mask = np.zeros_like(im, dtype = "uint8")
  dim = im.shape[:2][::-1]

  colors = get_color_boundaries(color)
  
  # set all matching pixels to white  
  for color in colors:
    # color = color[::-1]  # RGB --> BGR
    indices = np.all(im == color, axis=-1)
    im_mask[indices] = [255, 255, 255]

  im_gray = cv.cvtColor(im_mask, cv.COLOR_RGB2GRAY)
  if cv.countNonZero(im_gray) > threshold:
    xywh = cv.boundingRect(im_gray)
    bbox = BBox.from_xywh(*xywh, *dim)
    return bbox
  else:
    return None