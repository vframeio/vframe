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

from vframe.models.pipe_item import PipeContextHeader
from vframe.settings import app_cfg
from vframe.utils.video_utils import FileVideoStream

"""
- install KCF from https://github.com/smatsumt/KCFcpp-py-wrapper/
- if using custom OpenCV build may need to change pkgconfig.exists('opencv4') to 'opencv'
"""

log = app_cfg.LOG

try:
  import KCF
  KCF2_available = True
except ImportError:
  KCF = None
  KCF2_available = False


class VisTracker:
  # ---------------------------------------------------------
  # IOU Tracker
  # Copyright (c) 2019 TU Berlin, Communication Systems Group
  # Licensed under The MIT License [see LICENSE for details]
  # Written by Erik Bochinski
  # ---------------------------------------------------------
  kcf2_warning_printed = False

  def __init__(self, tracker_type, bbox, img, keep_height_ratio=1.):
    """ Wrapper class for various visual trackers."
    Args:
      tracker_type (str): name of the tracker. either the ones provided by opencv-contrib or KCF2 for a different
                implementation for KCF (requires https://github.com/uoip/KCFcpp-py-wrapper)
      bbox (tuple): box to initialize the tracker (x1, y1, x2, y2)
      img (numpy.ndarray): image to intialize the tracker
      keep_height_ratio (float, optional): float between 0.0 and 1.0 that determines the ratio of height of the
                         object to track to the total height of the object for visual tracking.
    """
    if tracker_type == 'KCF2' and not KCF:
      tracker_type = 'KCF'
      if not VisTracker.kcf2_warning_printed:
        log.debug("[warning] KCF2 not available, falling back to KCF. please see README.md for further details")
        VisTracker.kcf2_warning_printed = True

    self.tracker_type = tracker_type
    self.keep_height_ratio = keep_height_ratio

    if tracker_type == 'BOOSTING':
      self.vis_tracker = cv.TrackerBoosting_create()
    elif tracker_type == 'MIL':
      self.vis_tracker = cv.TrackerMIL_create()
    elif tracker_type == 'KCF':
      self.vis_tracker = cv.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
      self.vis_tracker = cv.TrackerMOSSE_create()
    elif tracker_type == 'KCF2':
      self.vis_tracker = KCF.kcftracker(False, True, False, False)  # hog, fixed_window, multiscale, lab
    elif tracker_type == 'TLD':
      self.vis_tracker = cv.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
      self.vis_tracker = cv.TrackerMedianFlow_create()
    elif tracker_type == 'CSRT':
      self.vis_tracker = cv.TrackerCSRT_create()
    elif tracker_type == 'GOTURN':
      self.vis_tracker = cv.TrackerGOTURN_create()
    elif tracker_type == 'NONE':  # dummy tracker that does nothing but fail
      self.vis_tracker = None
      self.ok = False
      return
    else:
      raise ValueError("Unknown tracker type '{}".format(tracker_type))

    y_max = img.shape[0] - 1
    x_max = img.shape[1] - 1
    #
    bbox = list(bbox)
    bbox[0] = max(0, min(bbox[0], x_max))
    bbox[2] = max(0, min(bbox[2], x_max))
    bbox[1] = max(0, min(bbox[1], y_max))
    bbox[3] = max(0, min(bbox[3], y_max))

    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # x1, y1, x2, y2 -> x1, y1, w, h
    bbox[3] *= self.keep_height_ratio

    if self.tracker_type == 'KCF2':
      self.vis_tracker.init(bbox, img)
      self.ok = True
    else:
      self.ok = self.vis_tracker.init(img, tuple(bbox))
      pass

  def update(self, img):
    """
    Args:
      img (numpy.ndarray): image for update

    Returns:
    bool: True if the update was successful, False otherwise
    tuple: updated bounding box in (x1, y1, x2, y2) format
    """
    if not self.ok:
      return False, [0, 0, 0, 0]

    if self.tracker_type == 'KCF2':
      bbox = self.vis_tracker.update(img)
      ok = True
    else:
      ok, bbox = self.vis_tracker.update(img)
      bbox = list(bbox)

    bbox[3] /= self.keep_height_ratio
    # x1, y1, w, h -> x1, y1, x2, y2
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]

    return ok, tuple(bbox)



def track_viou(fp_video, detections, sigma_l=0, sigma_h=0.5, sigma_iou=0.5, t_min=2, ttl=5, 
  tracker_type='KCF2', keep_upper_height_ratio=1.0):
  """ V-IOU Tracker.
  See "Extending IOU Based Multi-Object Tracking by Visual Information by E. Bochinski, T. Senst, T. Sikora" for
  more information.
  %sigma_l = 0;
  %sigma_h = 0.95;
  %sigma_iou = 0.6;
  %t_min = 7;
  Args:
     video_stream (FileVideoStream): streaming video buffer object
     detections (list): list of detections per frame
     sigma_l (float): low detection threshold.
     sigma_h (float): high detection threshold.
     sigma_iou (float): IOU threshold.
     t_min (float): minimum track length in frames.
     ttl (float): maximum number of frames to perform visual tracking.
            this can fill 'gaps' of up to 2*ttl frames (ttl times forward and backward).
     tracker_type (str): name of the visual tracker to use. see VisTracker for more details.
     keep_upper_height_ratio (float): float between 0.0 and 1.0 that determines the ratio of height of the object
                      to track to the total height of the object used for visual tracking.
  Returns:
    list: list of tracks.
  """
  if tracker_type == 'NONE':
    assert ttl == 1, "ttl should not be larger than 1 if no visual tracker is selected"

  tracks_active = []
  tracks_extendable = []
  tracks_finished = []
  frame_buffer = []

  header = PipeContextHeader(fp_video)
  n_frames = header.frame_end - (header.frame_start)
  video = FileVideoStream(fp_video)
  video.start()
  pbar = tqdm(total=n_frames, desc=header.filename, initial=header.frame_start, leave=False)


  #for frame_num, detections_frame in enumerate(tqdm(detections), start=1):
  frame_num = 0

  while video.running():
    frame_num += 1
    frame = video.read()
    detections_frame = detections.get(str(frame_num - 1), [])
    if not detections_frame:
      log.debug(f'frame: {frame_num}, {detections_frame}')
    pbar.update()

    frame_buffer.append(frame)
    if len(frame_buffer) > ttl + 1:
      frame_buffer.pop(0)

    # apply low threshold to detections
    dets = [det for det in detections_frame if det and det['score'] >= sigma_l]

    track_ids, det_ids = associate(tracks_active, dets, sigma_iou)
    updated_tracks = []

    for track_id, det_id in zip(track_ids, det_ids):
      tracks_active[track_id]['bboxes'].append(dets[det_id]['bbox'])
      tracks_active[track_id]['max_score'] = max(tracks_active[track_id]['max_score'], dets[det_id]['score'])
      tracks_active[track_id]['classes'].append(dets[det_id]['class'])
      tracks_active[track_id]['scores'].append(dets[det_id]['score'])
      tracks_active[track_id]['det_counter'] += 1

      if tracks_active[track_id]['ttl'] != ttl:
        # reset visual tracker if active
        tracks_active[track_id]['ttl'] = ttl
        tracks_active[track_id]['visual_tracker'] = None

      updated_tracks.append(tracks_active[track_id])

    tracks_not_updated = [tracks_active[idx] for idx in set(range(len(tracks_active))).difference(set(track_ids))]

    for track in tracks_not_updated:
      if track['ttl'] > 0:
        if track['ttl'] == ttl:
          # init visual tracker
          track['visual_tracker'] = VisTracker(tracker_type, track['bboxes'][-1], frame_buffer[-2],
                             keep_upper_height_ratio)
        # viou forward update
        ok, bbox = track['visual_tracker'].update(frame)

        if not ok:
          # visual update failed, track can still be extended
          tracks_extendable.append(track)
          continue

        track['ttl'] -= 1
        track['bboxes'].append(bbox)
        updated_tracks.append(track)
      else:
        tracks_extendable.append(track)

    # update the list of extendable tracks. tracks that are too old are moved to the finished_tracks. this should
    # not be necessary but may improve the performance for large numbers of tracks (eg. for mot19)
    tracks_extendable_updated = []
    for track in tracks_extendable:
      if track['start_frame'] + len(track['bboxes']) + ttl - track['ttl'] >= frame_num:
        tracks_extendable_updated.append(track)
      elif track['max_score'] >= sigma_h and track['det_counter'] >= t_min:
        tracks_finished.append(track)
    tracks_extendable = tracks_extendable_updated

    new_dets = [dets[idx] for idx in set(range(len(dets))).difference(set(det_ids))]
    dets_for_new = []

    for det in new_dets:
      finished = False
      # go backwards and track visually
      boxes = []
      vis_tracker = VisTracker(tracker_type, det['bbox'], frame, keep_upper_height_ratio)

      for f in reversed(frame_buffer[:-1]):
        ok, bbox = vis_tracker.update(f)
        if not ok:
          # can not go further back as the visual tracker failed
          break
        boxes.append(bbox)

        # sorting is not really necessary but helps to avoid different behaviour for different orderings
        # preferring longer tracks for extension seems intuitive, LAP solving might be better
        for track in sorted(tracks_extendable, key=lambda x: len(x['bboxes']), reverse=True):

          offset = track['start_frame'] + len(track['bboxes']) + len(boxes) - frame_num
          # association not optimal (LAP solving might be better)
          # association is performed at the same frame, not adjacent ones
          if 1 <= offset <= ttl - track['ttl'] and iou(track['bboxes'][-offset], bbox) >= sigma_iou:
            if offset > 1:
              # remove existing visually tracked boxes behind the matching frame
              track['bboxes'] = track['bboxes'][:-offset+1]
            track['bboxes'] += list(reversed(boxes))[1:]
            track['bboxes'].append(det['bbox'])
            track['max_score'] = max(track['max_score'], det['score'])
            track['classes'].append(det['class'])
            track['ttl'] = ttl
            track['visual_tracker'] = None

            #tracks_extendable.remove(track)
            #if track in tracks_finished:
            #  del tracks_finished[tracks_finished.index(track)]
            updated_tracks.append(track)

            finished = True
            break
        if finished:
          break
      if not finished:
        dets_for_new.append(det)



    # create new tracks
    new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num, 'ttl': ttl,
             'classes': [det['class']], 'det_counter': 1, 'visual_tracker': None, 'scores': []} for det in dets_for_new]
    tracks_active = []
    for track in updated_tracks + new_tracks:
      if track['ttl'] == 0:
        tracks_extendable.append(track)
      else:
        tracks_active.append(track)

    


  # finish all remaining active and extendable tracks
  tracks_finished = tracks_finished + \
            [track for track in tracks_active + tracks_extendable
             if track['max_score'] >= sigma_h and track['det_counter'] >= t_min]

  # remove last visually tracked frames and compute the track classes
  for track in tracks_finished:
    if ttl != track['ttl']:
      track['bboxes'] = track['bboxes'][:-(ttl - track['ttl'])]
    track['class'] = max(set(track['classes']), key=track['classes'].count)

    #del track['visual_tracker']

  return tracks_finished


def associate(tracks, detections, sigma_iou):
  """ perform association between tracks and detections in a frame.
  Args:
    tracks (list): input tracks
    detections (list): input detections
    sigma_iou (float): minimum intersection-over-union of a valid association
  Returns:
    (tuple): tuple containing:
    track_ids (numpy.array): 1D array with indexes of the tracks
    det_ids (numpy.array): 1D array of the associated indexes of the detections
  """
  costs = np.empty(shape=(len(tracks), len(detections)), dtype=np.float32)
  for row, track in enumerate(tracks):
    for col, detection in enumerate(detections):
      costs[row, col] = 1 - iou(track['bboxes'][-1], detection['bbox'])

  np.nan_to_num(costs)
  costs[costs > 1 - sigma_iou] = np.nan
  track_ids, det_ids = solve_dense(costs)
  return track_ids, det_ids



def to_viou_format(detection_results):
  #dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c})
  pass

def nms(boxes, scores, overlapThresh, classes=None):
  """
  perform non-maximum suppression. based on Malisiewicz et al.
  Args:
    boxes (numpy.ndarray): boxes to process
    scores (numpy.ndarray): corresponding scores for each box
    overlapThresh (float): overlap threshold for boxes to merge
    classes (numpy.ndarray, optional): class ids for each box.
  Returns:
    (tuple): tuple containing:
    boxes (list): nms boxes
    scores (list): nms scores
    classes (list, optional): nms classes if specified
  """
  # # if there are no boxes, return an empty list
  # if len(boxes) == 0:
  #   return [], [], [] if classes else [], []

  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  if scores.dtype.kind == "i":
    scores = scores.astype("float")

  # initialize the list of picked indexes
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]
  #score = boxes[:, 4]
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(scores)

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > overlapThresh)[0])))

  if classes is not None:
    return boxes[pick], scores[pick], classes[pick]
  else:
    return boxes[pick], scores[pick]

    

def nms(boxes, scores, overlapThresh, classes=None):
  """
  perform non-maximum suppression. based on Malisiewicz et al.
  Args:
    boxes (numpy.ndarray): boxes to process
    scores (numpy.ndarray): corresponding scores for each box
    overlapThresh (float): overlap threshold for boxes to merge
    classes (numpy.ndarray, optional): class ids for each box.

  Returns:
    (tuple): tuple containing:

    boxes (list): nms boxes
    scores (list): nms scores
    classes (list, optional): nms classes if specified
  """
  # # if there are no boxes, return an empty list
  # if len(boxes) == 0:
  #   return [], [], [] if classes else [], []

  # if the bounding boxes integers, convert them to floats --
  # this is important since we'll be doing a bunch of divisions
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  if scores.dtype.kind == "i":
    scores = scores.astype("float")

  # initialize the list of picked indexes
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = boxes[:, 0]
  y1 = boxes[:, 1]
  x2 = boxes[:, 2]
  y2 = boxes[:, 3]
  #score = boxes[:, 4]
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(scores)

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
                         np.where(overlap > overlapThresh)[0])))

  if classes is not None:
    return boxes[pick], scores[pick], classes[pick]
  else:
    return boxes[pick], scores[pick]


def iou(bbox1, bbox2):
  """
  Calculates the intersection-over-union of two bounding boxes.

  Args:
    bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

  Returns:
    int: intersection-over-onion of bbox1, bbox2
  """

  bbox1 = [float(x) for x in bbox1]
  bbox2 = [float(x) for x in bbox2]

  (x0_1, y0_1, x1_1, y1_1) = bbox1
  (x0_2, y0_2, x1_2, y1_2) = bbox2

  # get the overlap rectangle
  overlap_x0 = max(x0_1, x0_2)
  overlap_y0 = max(y0_1, y0_2)
  overlap_x1 = min(x1_1, x1_2)
  overlap_y1 = min(y1_1, y1_2)

  # check if there is an overlap
  if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
    return 0

  # if yes, calculate the ratio of the overlap to each ROI size and the unified size
  size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
  size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
  size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
  size_union = size_1 + size_2 - size_intersection

  return size_intersection / size_union
