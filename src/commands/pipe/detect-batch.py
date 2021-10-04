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
from vframe.models.types import BatchModelZooClickVar, BatchModelZoo, FrameImage
from vframe.settings import app_cfg

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  default=app_cfg.DEFAULT_DETECT_MODEL,
  type=BatchModelZooClickVar,
  help=show_help(BatchModelZoo))
@click.option('--device', 'opt_device', default=0,
  help='GPU device for inference (use -1 for CPU)')
@click.option('-s', '--size', 'opt_dnn_size', default=(None, None), type=(int, int),
  help='DNN blob image size. Overrides config file')
@click.option('-t', '--threshold', 'opt_dnn_threshold', default=None, type=float,
  help='Detection threshold. Overrides config file')
@click.option('--name', '-n', 'opt_data_key', default=None,
  help='Name of data key')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.option('--batch-size', 'opt_batch_size', default=16,
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

  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY, modelzoo
  from vframe.image.dnn_factory import DNNFactory

  
  model_name = opt_model_enum.name.lower()
  dnn_cfg = modelzoo.get(model_name)

  # override dnn_cfg vars with cli vars
  dnn_cfg.override(device=opt_device, size=opt_dnn_size, threshold=opt_dnn_threshold)

  # rotate cv, np vals
  cv_rot_val = app_cfg.ROTATE_VALS[opt_rotate]
  np_rot_val =  int(opt_rotate) // 90  # counter-clockwise 90 deg rotations

  # data key name
  opt_data_key = opt_data_key if opt_data_key else model_name

  # create cvmodel
  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)

  Q = []


  while True:

    # get pipe data
    M = yield
    
    # create batch
    n = len([skip for idx, im, skip in Q if not skip])
    
    Q.append([M.index, M.images.get(FrameImage.ORIGINAL), ctx.opts[SKIP_FRAME_KEY]])
    
    if n < opt_batch_size and not (M.is_last_item):
      sink.send(M)
    else:
      # setup batch
      ims = [im for idx, im, skip in Q if not skip]
      idxs = [idx for idx, im, skip in Q if not skip]

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
          M.metadata.get(idx).update({opt_data_key: results})

      Q = []
      sink.send(M)