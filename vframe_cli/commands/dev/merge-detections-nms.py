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
  help="Input JSON detection file with multiple model detections")
@click.option('-o', '--output', 'opt_output', required=True,
  help='Output JSON file')
@click.option('--name', 'opt_name', 
  help='Merge to data key')
@click.option('--minify', 'opt_minify', is_flag=True,
  default=False,
  help='Minify JSON')

@click.pass_context
def cli(ctx, opt_input, opt_output, opt_name, opt_minify):
  """Merge JSON detections"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  from tqdm import tqdm

  from vframe.utils import file_utils
  from vframe.settings import app_cfg
  from vframe.models.cvmodels import DetectResults, DetectResult

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  data = file_utils.load_json(opt_input)
  fp_media = data.get('filepath')
  frames_data = data.get('frames_data')

  for frame_idx, frame_data in frames_data.items():
    
    bboxes = []
    confidences = []
    detect_results_nms = []
    labels = []

    for model_name, model_results in frame_data.items():
      
      detect_results = dacite.from_dict(data=model_results, data_class=DetectResults)
      detect_results_nms.append(detect_results)

      for detection in detect_result.detections:
        bboxes.append(detection.bbox)
        confidences.append(detection.confidence)
        labels.append(detection.label)

    idxs = cv.dnn.NMSBoxes(bboxes, confidences, opt_dnn_thresh, opt_nms_thresh)
    detect_results_nms = [detect_results_nms[i[0]] for i in idxs]



