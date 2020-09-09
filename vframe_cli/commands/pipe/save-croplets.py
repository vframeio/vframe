############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.models import types
from vframe.utils import click_utils
from vframe.utils.click_utils import processor

@click.command('')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output directory')
@click.option('-n', '--name', 'opt_data_key', required=True,
  help="Name of data key for croplets")
@click.option('-e', '--ext', 'opt_ext', default='jpg',
  type=types.ImageFileExtVar,
  help=click_utils.show_help(types.ImageFileExt))
@click.option('--frame', 'opt_frame_type', default='draw',
  type=types.FrameImageVar,
  help=click_utils.show_help(types.FrameImage))
@click.option('--prefix', 'opt_prefix', default='crop',
  help='Filename prefix')
@processor
@click.pass_context
def cli(ctx, pipe, opt_dir_out, opt_ext, opt_frame_type, opt_prefix, opt_data_key):
  """Save to images"""
  
  from os.path import join

  import cv2 as cv
  
  from vframe.settings import app_cfg
  from vframe.utils import file_utils, im_utils


  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  file_utils.ensure_dir(opt_dir_out)
  ext = opt_ext.name.lower()

  # ---------------------------------------------------------------------------
  # process 
  
  while True:
    
    pipe_item = yield

    header = ctx.obj['header']
    im = pipe_item.get_image(opt_frame_type)
    item_data = header.get_data(opt_data_key)

    if item_data:
      for face_idx, detection in enumerate(item_data.detections):
        bbox_norm = detection.bbox
        face_idx_pad = file_utils.zpad(face_idx)
        fn = pipe_item.filename
        fn_croplet = f'{opt_prefix}_{face_idx_pad}_{fn}'
        fn_croplet = file_utils.swap_ext(fn_croplet, ext)
        fp_out = join(opt_dir_out, fn_croplet)
        im_crop = im_utils.crop_roi(im, bbox_norm)
        cv.imwrite(fp_out, im_crop)

    pipe.send(pipe_item)