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
def cli(ctx, sink, opt_data_keys, opt_nms_thresh, opt_dnn_thresh, opt_name, opt_remove_old):
  """Merge bboxes using NMS (single class)"""
  
  import cv2 as cv

  from vframe.models.cvmodels import DetectResults
  from vframe.settings.app_cfg import LOG, SKIP_FRAME
  from vframe.models import types


  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    data_keys = opt_data_keys if opt_data_keys else list(M.metadata[M.index].keys())

    bboxes = []
    confidences = []
    detect_results = []
    labels = []

    for data_key in data_keys:
      
      if data_key in M.metadata[M.index].keys():
        item_data = M.metadata[M.index].get(data_key)

        if item_data.detections:
          for face_idx, detect_result in enumerate(item_data.detections):
            bbox = detect_result.bbox.redim((M.width, M.height))
            bboxes.append(detect_result.bbox.xywh_int)
            confidences.append(float(detect_result.conf))
            labels.append(detect_result.label)
            detect_results.append(detect_result)

    # merge labels into single string
    label = '-'.join(list(set(labels)))
    
    # run nms
    idxs = cv.dnn.NMSBoxes(bboxes, confidences, opt_dnn_thresh, opt_nms_thresh)
    # detect_results_nms = [detect_results[i[0]] for i in idxs]
    detect_results_nms = [detect_results[i] for i in idxs]

    # reassign label
    # for d in detect_results_nms:
    #   d.label = label

    # remove old data keys if optioned
    if opt_remove_old:
      for data_key in data_keys:
        if data_key != opt_name:
          try:
            M.metadata[M.index].pop(data_key)
          except Exception as e:
            pass

    # add/update merged bboxes
    detect_results = DetectResults(detect_results_nms)
    M.metadata[M.index].update({opt_name: detect_results})

    # continue pipe
    sink.send(M)
