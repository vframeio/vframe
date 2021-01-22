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
from vframe.settings import app_cfg

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  default=app_cfg.DEFAULT_DETECT_MODEL,
  type=ModelZooClickVar,
  help=show_help(ModelZoo))
@click.option('--gpu/--cpu', 'opt_gpu', is_flag=True, default=True,
  help='Use GPU or CPU for inference')
@click.option('-s', '--size', 'opt_dnn_size', default=(None, None), type=(int, int),
  help='DNN blob image size. Overrides config file')
@click.option('-t', '--threshold', 'opt_dnn_threshold', default=None, type=float,
  help='Detection threshold. Overrides config file')
@click.option('--name', '-n', 'opt_data_key', default=None,
  help='Name of data key')
@click.option('-r', '--rotate', 'opt_rotate', 
  type=click.Choice(app_cfg.ROTATE_VALS.keys()), 
  default='0',
  help='Rotate image this many degrees in counter-clockwise direction before detection')
@click.option('--phash', 'opt_phash_filter', default=0, type=click.IntRange(0,36),
  help='Perceptual hash to skip similar non-face frames (1 - 4 recommended values)')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@processor
@click.pass_context
def cli(ctx, pipe, opt_model_enum, opt_data_key, opt_gpu, opt_dnn_threshold, 
  opt_dnn_size, opt_rotate, opt_phash_filter, opt_verbose):
  """Detect objects"""
  
  from os.path import join
  from pathlib import Path
  import traceback

  import cv2 as cv
  import imagehash

  from vframe.models.dnn import DNN
  from vframe.utils import im_utils
  from PIL import Image
  from vframe.settings.modelzoo_cfg import modelzoo
  from vframe.image.dnn_factory import DNNFactory

  
  # ---------------------------------------------------------------------------
  # initialize
  
  model_name = opt_model_enum.name.lower()
  dnn_cfg = modelzoo.get(model_name)

  # override dnn_cfg vars with cli vars
  dnn_cfg.override(gpu=opt_gpu, size=opt_dnn_size, threshold=opt_dnn_threshold)
    
  # rotate cv, np vals
  cv_rot_val = app_cfg.ROTATE_VALS[opt_rotate]
  np_rot_val =  int(opt_rotate) // 90  # counter-clockwise 90 deg rotations

  # phash filtering to skip same frames
  im_blank = Image.new('RGB', (100,100), (0,0,0))
  phash_pre = imagehash.phash(im_blank)
  n_phashes = 8
  phashes_pre = n_phashes * [phash_pre]
  objects_found = True  # set True to force detection on first frame
  phash_max = 36

  if not opt_data_key:
    opt_data_key = model_name

  # create dnn cvmodel
  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)

  # ---------------------------------------------------------------------------
  # process

  while True:

    # get pipe data
    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(FrameImage.ORIGINAL)
    
    phash_same = False
    if opt_phash_filter > 0:
      phash_cur = imagehash.phash(im_utils.np2pil(im))
      phashes_pre.pop(0)  # remove oldest
      phashes_pre.append(phash_cur)  # add new
      phash_avg = sum([phash_cur - p for p in phashes_pre]) / n_phashes
      if phash_avg < opt_phash_filter:
        phash_same = True

    if phash_same and not objects_found:
      # no faces in previous frame and this frame looks the same
      continue
    else:
      # rotate if optioned  
      if cv_rot_val is not None:
        im = cv.rotate(im, cv_rot_val)
      
      # detect
      try:
        results = cvmodel.infer(im)

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
        app_cfg.LOG.debug(f'{cvmodel.dnn_cfg.name} detected: {len(results.detections)} objects')

      # set flag if faces were found
      if results:
        objects_found = bool(len(results.detections))

      # update data
      if results:
        pipe_data = {opt_data_key: results}
        header.set_data(pipe_data)
    
    # continue processing
    pipe.send(pipe_item)