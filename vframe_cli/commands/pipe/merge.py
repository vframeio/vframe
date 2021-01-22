############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils.click_utils import processor

@click.command('')
@click.option( '-n', '--name', 'opt_data_keys',
  multiple=True,
  help='Data key names to merge from')
@click.option('--to', 'opt_name', default=None, required=True,
  help="Rename merged data-key to this")
@click.option('--nms-threshold', 'opt_nms_thresh', default=0.4,
  help='NMS threshold')
@click.option('--dnn-threshold', 'opt_dnn_thresh', default=0.7,
  help='DNN threshold')
@click.option('--remove/--keep', 'opt_remove_old', is_flag=True,
  default=True,
  help='Remove unmerged keys')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_keys, opt_nms_thresh, opt_dnn_thresh, opt_name, opt_remove_old):
  """Merge bboxes using NMS (single class)"""
  
  import cv2 as cv

  from vframe.settings import app_cfg
  from vframe.models.cvmodels import DetectResults
  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG

  log.warn(f'Multi-class merging not yet supported. Merging all classes to "{opt_name}"')

  # ---------------------------------------------------------------------------
  # process

  while True:

    pipe_item = yield
    header = ctx.obj['header']
    frame_dim = header.dim
    data_keys = opt_data_keys if opt_data_keys else header.get_data_keys()

    bboxes = []
    confidences = []
    detect_results = []
    labels = []

    for data_key in data_keys:
      
      if header.data_key_exists(data_key):
        item_data =  header.get_data(data_key)

        if item_data.detections:
          for face_idx, detect_result in enumerate(item_data.detections):
            bboxes.append(detect_result.bbox.xywh_int)
            confidences.append(float(detect_result.confidence))
            labels.append(detect_result.label)
            detect_results.append(detect_result)

    # merge labels into single string
    label = '-'.join(list(set(labels)))
    
    # run nms
    idxs = cv.dnn.NMSBoxes(bboxes, confidences, opt_dnn_thresh, opt_nms_thresh)
    detect_results_nms = [detect_results[i[0]] for i in idxs]

    # reassign label
    for d in detect_results_nms:
      d.label = label

    # remove old data keys if optioned
    if opt_remove_old:
      for data_key in data_keys:
        if data_key != opt_name:
          header.remove_data(data_key)

    # add/update merged bboxes
    detect_results = DetectResults(detect_results_nms)
    pipe_data = {opt_name: detect_results}
    header.set_data(pipe_data)

    # continue pipe
    pipe.send(pipe_item)