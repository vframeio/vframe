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

default_scale_factor = 1.1
default_overlaps = 5

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
@click.option( '--overlaps', 'opt_overlaps', default=default_overlaps, 
  type=int, help='Minimum neighbor overlaps')
@click.option( '--scale-factor', 'opt_scale_factor', default=default_scale_factor,
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
def cli(ctx, sink, opt_model_enum, opt_data_key, opt_gpu, 
  opt_overlaps, opt_scale_factor, opt_min_size, opt_max_size, opt_rotate, opt_verbose):
  """Haarcascade face detection"""
  
  from os.path import join
  from pathlib import Path
  import traceback

  import cv2 as cv

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, modelzoo
  from vframe.models.dnn import DNN
  from vframe.image.dnn_factory import DNNFactory
  from vframe.models.geometry import BBox
  from vframe.models.cvmodels import DetectResult, DetectResults

  
  # ---------------------------------------------------------------------------
  # initialize

  model_name = opt_model_enum.name.lower()
  opt_data_key = opt_data_key if opt_data_key else model_name

  # hardcoded class meta
  class_idx = 0
  label = 'Face'
  conf_overlaps = (opt_overlaps / default_overlaps)
    
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
    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue


    im = M.images.get(FrameImage.ORIGINAL)
    dim = im.shape[:2][::-1]
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # create a simulated confidence based on default overlaps and scale factor
    targ_scale = (default_scale_factor / 640) * dim[0]
    conf_scale = (opt_scale_factor / targ_scale)
    confidence = min(1.0, conf_scale * conf_overlaps)
    LOG.debug(conf_scale)
    LOG.debug(conf_overlaps)
    LOG.debug(confidence)
    
    # rotate if optioned  
    if cv_rot_val is not None:
      im_gray = cv.rotate(im_gray, cv_rot_val)

    # detect
    try:
      if opt_min_size and opt_max_size:
        detections = cascade.detectMultiScale(im_gray, scaleFactor=opt_scale_factor, 
          minNeighbors=opt_overlaps, minSize=(opt_min_size,opt_min_size), maxSize=(opt_max_size, opt_max_size))
      elif opt_max_size:
        detections = cascade.detectMultiScale(im_gray, scaleFactor=opt_scale_factor, 
          minNeighbors=opt_overlaps, maxSize=(opt_max_size,opt_max_size))
      elif opt_min_size:
        detections = cascade.detectMultiScale(im_gray, scaleFactor=opt_scale_factor, 
          minNeighbors=opt_overlaps, minSize=(opt_min_size,opt_min_size))
      else:
        detections = cascade.detectMultiScale(im_gray, scaleFactor=opt_scale_factor, 
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
      app_cfg.LOG.error(f'Could not detect: {M.filepath}')
      tb = traceback.format_exc()
      app_cfg.LOG.error(tb)

    # debug
    if opt_verbose:
      app_cfg.LOG.debug(f'{model_name} detected: {len(results.detections)} objects')

    # update data
    if results:
      metadata = {opt_data_key: results}
      M.metadata[M.index].update(metadata)
    
    # continue processing
    sink.send(M)