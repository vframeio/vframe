############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils.click_utils import show_help, processor
from vframe.models.types import ModelZooClickVar, ModelZoo, FrameImage
from vframe.settings import app_cfg

@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  default=app_cfg.DEFAULT_CLASSIFICATION_MODEL,
  type=ModelZooClickVar,
  help=show_help(ModelZoo))
@click.option('--device', 'opt_device', default=0,
  help='GPU device for inference (use -1 for CPU)')
@click.option('-s', '--size', 'opt_dnn_size', default=(None, None), type=(int, int),
  help='DNN blob image size. Overrides config file')
@click.option('-t', '--threshold', 'opt_dnn_threshold', default=None, type=float,
  help='Detection threshold. Overrides config file')
@click.option('--name', '-n', 'opt_data_key', default=None,
  help='Name of data key')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@processor
@click.pass_context
def cli(ctx, sink, opt_model_enum, opt_device, opt_dnn_size, opt_dnn_threshold, opt_data_key, opt_verbose):
  """Compute DNN features/embeddings"""
  
  from PIL import Image
  import cv2 as cv
  from sklearn.metrics.pairwise import cosine_similarity

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, modelzoo
  from vframe.utils.im_utils import resize, np2pil
  from vframe.image.dnn_factory import DNNFactory


  model_name = opt_model_enum.name.lower()
  dnn_cfg = modelzoo.get(model_name)

  # override dnn_cfg vars with cli vars
  dnn_cfg.override(device=opt_device, size=opt_dnn_size, threshold=opt_dnn_threshold)
  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)
  
  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue
      
    im = M.images.get(FrameImage.ORIGINAL)
    features = cvmodel.features(im)
    
    # append to metadata
    
    sink.send(M)