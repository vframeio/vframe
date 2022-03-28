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
@click.option('-d', '--device', 'opt_device', default=0,
  help='GPU device for inference (use -1 for CPU)')
@click.option('-s', '--size', 'opt_dnn_size', type=(int, int), default=(0,0),
  help='DNN blob image size. Overrides config file')
@click.option('-t', '--threshold', 'opt_dnn_threshold', default=None, type=float,
  help='Detection threshold. Overrides config file')
@click.option('--name', '-n', 'opt_data_key', default=None,
  help='Name of data key')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.option('--batch-size', 'opt_batch_size', default=1,
  type=click.IntRange(1, 64),
  help='Inference batch size')
@click.option('-r', '--rotate', 'opt_rotate', 
  type=click.Choice(app_cfg.ROTATE_VALS.keys()), 
  default='0',
  help='Rotate image this many degrees in counter-clockwise direction before detection')
@processor
@click.pass_context
def cli(ctx, sink, opt_model_enum, opt_data_key, opt_device, opt_dnn_threshold, 
  opt_dnn_size, opt_batch_size, opt_rotate, opt_verbose):
  """Detect objects with batch inference"""

  import cv2 as cv

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, OBJECT_COLORS, SKIP_FILE, USE_DRAW_FRAME
  from vframe.settings.app_cfg import modelzoo
  from vframe.image.dnn_factory import DNNFactory

  
  model_name = opt_model_enum.name.lower()
  dnn_cfg = modelzoo.get(model_name)

  if opt_batch_size > 1 and not dnn_cfg.batch_enabled:
    opt_batch_size = 1
    LOG.warn(f'Batch processing not enabled for this model. Batch size reset to 1')

  if opt_batch_size > 1 and ctx.obj.get(USE_DRAW_FRAME):
    LOG.warn('Using "draw" with "--batch-size" > 1 will result in missed frames')
    LOG.warn('  Use "--batch-size 1" to draw on all frames. Or post-processing JSON.')

  # override dnn_cfg vars with cli vars
  dnn_cfg.override(device=opt_device, dnn_size=opt_dnn_size, threshold=opt_dnn_threshold)
  
  # rotate cv, np vals
  cv_rot_val = app_cfg.ROTATE_VALS[opt_rotate]
  np_rot_val =  int(opt_rotate) // 90  # counter-clockwise 90 deg rotations

  # data key name
  opt_data_key = opt_data_key if opt_data_key else model_name

  # create cvmodel
  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)

  # add globally accessible colors
  ctx.obj.setdefault(OBJECT_COLORS, {})
  ctx.obj[OBJECT_COLORS][opt_data_key] = dnn_cfg.colorlist

  Q = []


  while True:

    # get pipe data
    M = yield

    if M._skip_file or ctx.obj[SKIP_FILE]:
      # corrupt file, abort processing and reset Q
      Q = []
      sink.send(M)
      continue


    if opt_batch_size == 1:

      if ctx.obj.get(SKIP_FRAME):
          sink.send(M)
          continue
          
      # TODO: copy results from previous frame is SIM_FRAME_KEY
      # if ctx.obj[SIM_FRAME_KEY]:
      #   # copy last frame detections if exist
      #   # results = M.inherit_from_last_frame(opt_data_key)
        
      #   if M.index > 1:
      #     results = M.metadata.get(M.index - 1)
      #     if results:
      #       M.metadata[M.index] = results.copy()
        
      im = M.images.get(FrameImage.ORIGINAL)
      
      # rotate if optioned  
      if cv_rot_val is not None:
        im = cv.rotate(im, cv_rot_val)

      # detect
      results = cvmodel.infer(im)
      if isinstance(results, list):
        results = results[0]

      # rotate if optioned
      if results and np_rot_val != 0:
        for detect_results in results.detections:
          detect_results.bbox = detect_results.bbox.rot90(np_rot_val)

      # update data
      if len(results.detections) > 0:
        if opt_verbose:
          LOG.debug(f'{cvmodel.dnn_cfg.name} detected: {len(results.detections)} objects')

        # update media file metadata
        M.metadata[M.index].update({opt_data_key: results})


    else:

      if not ctx.obj.get(SKIP_FRAME):
        # add new frame to queue
        Q.append([M.index, M.images.get(FrameImage.ORIGINAL)])

      # batch inference if Q has items or last frame
      if len(Q) >= opt_batch_size or M.is_last_item:
        ims = [im for idx, im in Q ]
        idxs = [idx for idx, im in Q]

        # rotate if optioned  
        if cv_rot_val is not None:
          ims = [cv.rotate(im, cv_rot_val) for im in ims]

        # inference
        batch_results = cvmodel.infer(ims)

        # convert to result objects
        for results, idx in zip(batch_results, idxs):

          # rotate if optioned
          if results and np_rot_val != 0:
            for detect_results in results.detections:
              detect_results.bbox = detect_results.bbox.rot90(np_rot_val)

          # update data
          if len(results.detections) > 0:
            M.metadata[idx].update({opt_data_key: results})

        # reset
        Q = []

    sink.send(M)