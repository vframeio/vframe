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
  from vframe.models.geometry import BBox
  from vframe.models.cvmodels import DetectResult, DetectResults
  from vframe.utils import tracker_utils

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  #tracking_cfg = {}
  opt_frame_buffer = 10
  mot_tracker = MultiObjectTracker(frame_buffer_size=opt_frame_buffer)

  # ---------------------------------------------------------------------------
  # process

  while True:

    pipe_item = yield
    im = pipe_item.get_image(opt_frame_type)
    header = ctx.obj.get('header')
    # get all data keys present in current frame's data
    data_keys = opt_data_keys if opt_data_keys else header.get_data_keys()
    # data keys may change during real-time detection eg when new objects are detected
    # preprocessor should create ObjectDetectors on demand as new data_keys are added
    # the image is added to the buffer
    mot_tracker.preprocess(im, data_keys)

    # add current frame's data and append to MOT tracker
    for data_key in data_keys:
      frame_data = header.get_data(data_key)
      mot_tracker.set_frame_data(data_key, header.frame_index, frame_data)

    # run MOT tracker and associate BBoxes within frame buffer to track IDs
    tracker_results = mot_tracker.process()
    
    # update matched tracks, update unmatched, remove expired
    for data_key, tracker_result in tracker_results.items():
      for frame_idx, detections in tracker_result.items():
        pipe_data = {data_key: detections}
        header.set_data(pipe_data, frame_idx=frame_idx)

    # send item
    pipe.send(pipe_item)  