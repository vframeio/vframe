############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click


@click.command('')
@click.option('-i', '--input', 'opt_input', required=True,
  help="Input file")
@click.option('-o', '--output', 'opt_output', required=True,
  help='Output file')
@click.option('--minify', 'opt_minify', is_flag=True,
  default=False,
  help='Minify JSON')
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_minify):
  """Track and interpolate using V-IOU + object tracker"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  from tqdm import tqdm
  from dacite import from_dict

  from vframe.utils import file_utils
  from vframe.settings import app_cfg
  from vframe.models.pipe_item import PipeContextHeader
  from vframe.models.geometry import BBox
  from vframe.models.cvmodels import DetectResults
  from vframe.utils import tracker_utils

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  # load
  data = file_utils.load_json(opt_input)
  frames_data = data[0]['frames_data']
  dets_viou = {}
  #max_frame = 40  # for dev
  data_key_name = 'face'

  # count original detections
  n_dets_orig = 0
  for results in data:
    frames_data = results.get('frames_data')
    for frame_idx, frame_items in frames_data.items():
      for model_name, model_items in frame_items.items():
        n_dets_orig += len(model_items.get('detections', []))

  
  

  # convert to VIOU bbox format
  # {'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, 'class': c}
  for frame_idx, frame_items in frames_data.items():
    dets_viou[frame_idx] = []
    for model_name, model_results in frame_items.items():
      for d in model_results['detections']:
        bbox = from_dict(data=d['bbox'], data_class=BBox)
        obj = {'bbox': tuple(bbox.xyxy_int), 'score': d['confidence'], 'class': d['index']}
        dets_viou[frame_idx].append(obj)

  # convert to VIOU format
  fp_video = data[0]['filepath']
  tracks = tracker_utils.track_viou(fp_video, dets_viou, tracker_type='KCF', 
    sigma_l=0.0, sigma_h=0.5, sigma_iou=0.5, t_min=2, ttl=1, keep_upper_height_ratio=1.0)
  log.debug(f'found {len(tracks)}')



  header = PipeContextHeader(fp_video)
  dim = header.dim

  # convert back to vframe format
  # start with fresh dict
  frames_data_viou = {}
  for item in data:
    frames_data = item.get('frames_data')
    for frame_idx in frames_data.keys():
      frames_data_viou[frame_idx] = {}  # eg "1"
      frames_data_viou[frame_idx][data_key_name] = {}
      frames_data_viou[frame_idx][data_key_name]["detections"] = []
      frames_data_viou[frame_idx][data_key_name]['task_type'] = "detection"

  """
  "203": {
    "face": {
      "detections": [
        {
          "bbox": {
            "dh": 506,
            "dw": 900,
            "x1": 706.5804443359375,
            "x2": 718.270751953125,
            "y1": 287.1981201171875,
            "y2": 301.6626281738281
          },
          "confidence": 0.8241621255874634,
          "index": 0,
          "label": "face"
        }
      ],
      "task_type": "detection"
    }
  },
  """
  """
  {'bboxes': [(884, 491, 947, 555), (883, 490, 946, 554), (882, 489, 945, 555), 
    (881, 490, 942, 554), (879, 490, 941, 555), (877, 489, 938, 555), (870, 491, 932, 558), 
    (866, 490, 927, 557), (860, 492, 922, 560)], 
    'max_score': 0.9995935559272766, 
    'start_frame': 1, 
    'ttl': 10, 
    'classes': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
    'det_counter': 9, 
    'scores': [0.9992990493774414, 0.9992337226867676, 0.9993810653686523, 0.9994375109672546, 0.9995935559272766, 0.9988943934440613, 0.9939717650413513, 0.9609559178352356], 
    'class': 0}
  """

  # count total detects
  n_tracks = 0
  for track in tracks:
    n_tracks += track['det_counter']

  log.debug(f'original dets: {n_dets_orig}')
  log.debug(f'viou dets: {n_tracks}')

  # for each track
  for track in tracks:
    start_idx = track['start_frame'] - 1
    n_dets = track['det_counter']

    # iterate frames and append all detections
    for det_idx in range(n_dets):
      frame_idx = str(det_idx + start_idx)
      xyxy = track['bboxes'][det_idx]
      bbox = BBox(*xyxy, *dim)
      obj = {'bbox': bbox.to_dict()}
      obj['index'] = track['class']
      #obj['confidence'] = track['scores'][det_idx]
      obj['confidence'] = track['max_score']
      obj['label'] = 'face'
      frames_data_viou[frame_idx][data_key_name]["detections"].append(obj)


  data_viou = []
  data_viou.append({'filepath': fp_video, 'frames_data': frames_data_viou})

  # write
  #results_out = list(merge_results.values())
  file_utils.write_json(data_viou, opt_output, minify=opt_minify)

