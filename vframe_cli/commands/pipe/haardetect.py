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
from vframe.utils.click_utils import show_help
from vframe.models.types import ModelZooClickVar, ModelZoo, FrameImage
from vframe.models.types import HaarcascadeVar, Haarcascade
from vframe.settings import app_cfg

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  default=app_cfg.DEFAULT_HAARCASCADE,
  type=HaarcascadeVar,
  help=show_help(Haarcascade))
@click.option('--gpu/--cpu', 'opt_gpu', is_flag=True, default=True,
  help='Use GPU or CPU for inference')
@click.option('--min-size', 'opt_min_size', default=None, type=int,
  help='Min detect size')
@click.option('--max-size', 'opt_max_size', default=None, type=int,
  help='Max detect size')
@click.option( '--overlaps', 'opt_overlaps', default=5, type=int,
  help='Minimum neighbor overlaps')
@click.option( '--scale-factor', 'opt_scale_factor', default=1.1,
  help='Scale factor')
@click.option('--name', '-n', 'opt_data_key', default=None,
  help='Name of data key')
@click.option('-r', '--rotate', 'opt_rotate', 
  type=click.Choice(app_cfg.ROTATE_VALS.keys()), 
  default='0',
  help='Rotate image this many degrees in counter-clockwise direction before detection')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@processor
@click.pass_context
def cli(ctx, pipe, opt_model_enum, opt_data_key, opt_gpu, 
  opt_overlaps, opt_scale_factor, opt_min_size, opt_max_size, opt_rotate, opt_verbose):
  """Haarcascade face detection"""
  
  from os.path import join
  from pathlib import Path
  import traceback

  import cv2 as cv

  from vframe.models.dnn import DNN
  from vframe.settings.modelzoo_cfg import modelzoo
  from vframe.image.dnn_factory import DNNFactory
  from vframe.models.geometry import BBox
  from vframe.models.cvmodels import DetectResult, DetectResults

  
  # ---------------------------------------------------------------------------
  # initialize
  model_name = opt_model_enum.name.lower()
  if not opt_data_key:
    opt_data_key = model_name

  # hardcoded class meta
  class_idx = 0
  label = 'Face'
  confidence = 1.0
    
  # rotate cv, np vals
  cv_rot_val = app_cfg.ROTATE_VALS[opt_rotate]
  np_rot_val =  int(opt_rotate) // 90  # counter-clockwise 90 deg rotations

  # create haar cvmodel
  fp_haarcascade = join(app_cfg.DIR_CV2_DATA, f'haarcascade_{model_name}.xml')
  cascade = cv.CascadeClassifier(fp_haarcascade)

  # ---------------------------------------------------------------------------
  # process

  while True:

    # get pipe data
    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(FrameImage.ORIGINAL)
    dim = im.shape[:2][::-1]
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    # rotate if optioned  
    if cv_rot_val is not None:
      im = cv.rotate(im, cv_rot_val)

    # detect
    try:
      if opt_min_size and opt_max_size:
        detections = cascade.detectMultiScale(im, scaleFactor=opt_scale_factor, 
          minNeighbors=opt_overlaps, minSize=(opt_min_size,opt_min_size), maxSize=(opt_max_size, opt_max_size))
      elif opt_max_size:
        detections = cascade.detectMultiScale(im, scaleFactor=opt_scale_factor, 
          minNeighbors=opt_overlaps, maxSize=(opt_max_size,opt_max_size))
      elif opt_min_size:
        detections = cascade.detectMultiScale(im, scaleFactor=opt_scale_factor, 
          minNeighbors=opt_overlaps, minSize=(opt_min_size,opt_min_size))
      else:
        detections = cascade.detectMultiScale(im, scaleFactor=opt_scale_factor, 
          minNeighbors=opt_overlaps)

      bboxes = [BBox.from_xywh(*xywh,*dim) for xywh in detections]
      results = [DetectResult(class_idx, confidence, bbox, label) for bbox in bboxes]
      results = DetectResults(results)

      # rotate if optioned
      if results and np_rot_val != 0:
        for detect_results in results.detections:
          detect_results.bbox = detect_results.bbox.rot90(np_rot_val)

    except Exception as e:
      results = {}
      app_cfg.LOG.error(f'Could not detect: {header.filepath}')
      tb = traceback.format_exc()
      app_cfg.LOG.error(tb)

    # debug
    if opt_verbose:
      app_cfg.LOG.debug(f'{model_name} detected: {len(results.detections)} objects')

    # update data
    if results:
      pipe_data = {opt_data_key: results}
      header.set_data(pipe_data)
    
    # continue processing
    pipe.send(pipe_item)