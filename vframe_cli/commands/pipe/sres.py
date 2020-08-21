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
from vframe.models.types import ModelZooClickVar, ModelZoo, FrameImage, FrameImageVar


@click.command('')
@click.option('-m', '--model', 'opt_model_enum', 
  required=True,
  default='espcn_x4',
  type=ModelZooClickVar,
  help=show_help(ModelZoo))
@click.option('--gpu/--cpu', 'opt_gpu', is_flag=True, default=True,
  help='Use GPU or CPU for inference')
@click.option('-f', '--frame', 'opt_frame_type', default='draw',
  type=FrameImageVar,
  help=show_help(FrameImage))
@click.option('--no-resize/--resize', 'opt_reset_size', is_flag=True, default=True,
  help="Reset to original size")
@processor
@click.pass_context
def cli(ctx, pipe, opt_model_enum, opt_reset_size, opt_frame_type, opt_gpu):
  """Super resolution upsample"""

  from vframe.settings import app_cfg
  from vframe.utils import im_utils
  from vframe.settings.modelzoo_cfg import modelzoo
  from vframe.image.dnn_factory import DNNFactory

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  
  model_name = opt_model_enum.name.lower()
  dnn_cfg = modelzoo.get(model_name)

  app_cfg.LOG.debug(f'use gpu: {opt_gpu}')
  # override dnn_cfg vars with cli vars
  if opt_gpu:
    dnn_cfg.use_gpu()
  else:
    dnn_cfg.use_cpu()


  # create dnn cvmodel
  cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)

  # ---------------------------------------------------------------------------
  # process

  while True:

    # Get pipe data
    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(FrameImage.ORIGINAL)
    h,w,c = im.shape

    im = cvmodel.upsample(im)
    if opt_reset_size:
      im = im_utils.resize(im, width=w, height=h)

    pipe_item.set_image(opt_frame_type, im)
    
    # Continue processing
    pipe.send(pipe_item)