############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import cv2 as cv
import numpy as np
from lapsolver import solve_dense
from tqdm import tqdm
from dataclasses import dataclass, field
import dlib

from vframe.models.pipe_item import PipeContextHeader
from vframe.settings import app_cfg
from vframe.utils.video_utils import FileVideoStream
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.utils.im_utils import bgr2rgb

"""
- DLIB tracker wanders off into nonsense
- KCF fairly stable, medium speed, does not seem to wander
- MOSSE: fairly stable, fast, does not seem to wander
- MEDIANFLOW: 
"""

log = app_cfg.LOG


class ObjectTracker:
  """Track multiple objects within frame span
  """
  
  track_counter = 0
  detect_results = []  # DetectResult
  phash_buffer = []
  frames_since_last_bbox = 0
  expired = False

  def __init__(self, frame_buffer_size, tracker_type='KCF'):
    self.frame_buffer_size = frame_buffer_size
    if opt_tracker_type == 'KCF':
      self.tracker = cv.TrackerKCF_create()
    elif opt_tracker_type == 'MOSSE':
      self.tracker = cv.TrackerMOSSE_create()
    elif opt_tracker_type == 'MEDIANFLOW':
      self.tracker = cv.TrackerMedianFlow_create()
    elif opt_tracker_type == 'BOOSTING':
      self.tracker = cv.TrackerBoosting_create()
    elif opt_tracker_type == 'MIL':
      self.tracker = cv.TrackerMIL_create()
    elif opt_tracker_type == 'TLD':
      self.tracker = cv.TrackerTLD_create()
    elif opt_tracker_type == 'GOTURN':
      self.tracker = cv.TrackerGOTURN_create()
    elif opt_tracker_type == 'CSRT':
      self.tracker = cv.TrackerCSRT_create()

  def init(self, im, bbox):
    #im_rgb = cv.cvtColor(im, cv.BGR2RGB)
    #dlib_rect = dlib.rectangle(*bbox.xyxy_int)
    #self.tracker_model.start_track(im_rgb, dlib_rec)  # convert to generic method
    self.tracker.init(im, bbox.xywh_int)  # opencv trackers use x,y,w,h
    im_roi = im_utils.crop_roi(im, bbox)
    im_roi_pil = im_utils.np2pil(im_roi)
    self.phash_origin = imagehash.phash(im_roi_pil)


  def update(self, frame_idx, frame_data, frames_buffer, frames_data_buffer):
    """Seek back/ahead to find BBoxes 
    - frame_idx is current frame index
    - frames buffer is array of all frames within frame track range
    - frame_data_buffer is all frame data
    """

    frame_cur = frames_buffer[frame_idx]
    frame_data_cur = frames_data_buffer[frame_idx]

    n_frames = list(range(self.frame_buffer_size + 1))  # number of frames in each head and tail
    mid_idx = n_frames // 2 + 1
    #tail_idxs = reverse([(1+midpoint):]) tail
    tail_idx = list(range(n_buffer))
    head_idx = reverse(list(range(n_buffer)))
    print(a[:midpoint])
    
    # head
    #for frame, frame_data in zip(frames_buffer, frames_data_buffer):

    # tail
    track_ok, result_tracker = tracker.update(frame)
    

    if is_found:
      frames_since_last_bbox = 0
    else:
      frames_since_last_bbox +=1
    if frames_since_last_bbox > self.max_unseen_frames:
      self.expired = True



class MultiObjectTracker:

  object_trackers = {}  # object tracker for each model?
  object_tracker_ids = {}
  frame_buffer = []  # shared across trackers

  def __init__(self, frame_buffer_size=10):
    self.frame_buffer_size = frame_buffer_size


  def preprocess(self, im, data_keys):
    """Set current frame and update frame buffer
    """
    self.frame_buffer.append(im)
    if len(self.frame_buffer) > self.frame_buffer_size:
      self.frame_buffer.pop(0)
    for data_key in data_keys:
      if data_key not in self.object_trackers.keys():
        self.object_trackers[data_keys] = ObjectTracker(self.frame_buffer_size)

  
  def process(self, frame_idx, frame_data):
    for model_name, tracker in self.object_trackers.items():
      if tracker.expired:
        continue
      else:
        tracker.update(frame_idx, frames_data)

    # somehow convert this to results
    tracker_results['retinaface'] = 
    return tracker_results


  def postprocess(self):
    pass


  def set_frame_data(self, frame_idx, frame_data):
    """Adds frame_data {"model_name": DetectionResult}
    """
    self.detections.append(detection)
  

"""
  "0": {
    "face": {
      "detections": [
        {
          "bbox": {
            "dh": 720,
            "dw": 1280,
            "x1": 251,
            "x2": 276,
            "y1": 498,
            "y2": 527
          },
          "confidence": 0.9494602084159851,
          "index": 0,
          "label": "face"
        },
"""





def associate(bboxes_detector, bboxes_tracker, sigma=0.5):
  """Associates indices from a list of detected BBoxes with a list of tracker BBoxes
  If no association was made, the index of the BBox is not returned.
  :param bboxes_detector: list(BBox) of BBoxes created using object detector
  :param bboxes_tracker: list(BBox) of proposed BBoxes created using object tracker
  :param sigma: (float) minimum IOU overlap required for a valid association
  :returns (tuple): detector ids and tracker ids in corresponding order. 
  """
  costs = np.empty(shape=(len(bboxes_detector), len(bboxes_tracker)), dtype=np.float32)
  
  for row, bbox_detector in enumerate(bboxes_detector):
    for col, bbox_tracker in enumerate(bboxes_tracker):
      costs[row, col] = 1 - bbox_detector.iou(bbox_tracker)

  np.nan_to_num(costs)
  costs[costs > 1 - sigma] = np.nan
  ids_detector, ids_tracker = solve_dense(costs)
  return ids_detector, ids_tracker