############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.utils.click_utils import generator

@click.command('')
@click.option('-i', '--input', 'opt_input', required=True,
  help='Path to YAML config file')
@click.option('--slice', 'opt_slice', type=(int, int), 
  default=(None, None),
  help="Slice list of inputs")
@click.option('--decimate', 'opt_decimate', type=int, default=None,
  help="Number of frames to skip between processing")
@click.option('--shuffle', 'opt_shuffle', is_flag=True,
  help='Randomly shuffle dataset items')
@click.option('--labels', 'opt_labels', multiple=True, type=str,
  help='Filter by labels')
@generator
@click.pass_context
def cli(ctx, pipe, opt_input, opt_slice, opt_decimate, opt_shuffle, opt_labels):
  """+ Add annotated dataset"""
  
  from pathlib import Path
  from os.path import join
  import random

  from tqdm import tqdm, trange
  import cv2 as cv

  from vframe.settings import app_cfg
  from vframe.models.color import Color
  from vframe.models.geometry import BBox
  from vframe.models.annotation import Annotation
  from vframe.models.pipe_item import PipeContextHeader, PipeFrame
  from vframe.utils import file_utils, draw_utils, display_utils
  from vframe.models.cvmodels import DetectResult, DetectResults
  from vframe.models.training_dataset import YoloProjectConfig

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  ctx.obj['fp_input'] = opt_input
  opt_data_key = 'annotations'

  yolo_cfg = file_utils.load_yaml(opt_input, data_class=YoloProjectConfig)

  # get images labels directory
  fps_txt = file_utils.glob_multi(join(yolo_cfg.output, yolo_cfg.images_labels), exts=['txt'], sort=True)
  fps_im = file_utils.glob_multi(join(yolo_cfg.output, yolo_cfg.images_labels), exts=['jpg', 'png'], sort=True)

  if len(fps_im) != len(fps_txt):
    app_cfg.LOG.error('Images and text labels not balanced. Exiting')
    return

  # slice input
  if any(opt_slice):
    fps_txt = fps_txt[opt_slice[0]:opt_slice[1]]
    fps_im = fps_im[opt_slice[0]:opt_slice[1]]

  # randomize list of images
  if opt_shuffle:
    fps_txt_im = list(zip(fps_txt, fps_im))
    random.shuffle(fps_txt_im)
    fps_txt, fps_im = zip(*fps_txt_im)
  
  # load classes
  labels = file_utils.load_txt(join(yolo_cfg.output, yolo_cfg.classes))

  for n in trange(len(fps_im), desc='Annotate dataset'):

    # choose a random file/image pair
    fp_txt, fp_im = (fps_txt[n], fps_im[n])
    
    # load text file
    anno_lines = file_utils.load_txt(fp_txt)
    detections = []

    # convert annotations into detection results
    if opt_labels:
      labels_found = []
      for anno_line in anno_lines:
        fn = Path(fp_im).name
        if anno_line:
          label_index, cx, cy, w, h = anno_line.split(' ')
          label_enum = labels[int(label_index)]
          labels_found.append(label_enum)
      
      labels_ok = any([label in opt_labels for label in labels_found])
      if not labels_ok:
        continue

    # init pipe header
    ctx.obj['header'] = PipeContextHeader(fp_im)
    header = ctx.obj['header']

    # load image
    frame = cv.imread(fp_im)
    dim = frame.shape[:2][::-1]
    pipe_item = PipeFrame(frame)

    # convert annotations into detection results
    for anno_line in anno_lines:
      fn = Path(fp_im).name
      if anno_line:
        found = True
        label_index, cx, cy, w, h = anno_line.split(' ')
        label_enum = labels[int(label_index)]
        bbox = BBox.from_cxcywh_norm(float(cx), float(cy), float(w), float(h), *dim)
        anno = Annotation(fn, label_index, label_enum, label_enum, bbox, Color.random())
        detect_result = DetectResult(label_index, 1.0, bbox, label_enum)
        detections.append(detect_result)


    # append detect results to pipe frame
    detect_results = DetectResults(detections)
    pipe_data = {opt_data_key: detect_results}
    header.set_data(pipe_data)

    # send data through pipe
    pipe.send(pipe_item)



  
  


