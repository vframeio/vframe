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
@click.option('-e', '--ext', 'opt_ext', default='jpg',
  type=types.ImageFileExtVar,
  help=click_utils.show_help(types.ImageFileExt))
@click.option('--frame', 'opt_frame_type', default='draw',
  type=types.FrameImageVar,
  help=click_utils.show_help(types.FrameImage))
@click.option('--prefix', 'opt_prefix', default='',
  help='Filename prefix')
@click.option('--suffix', 'opt_suffix', default='',
  help='Filename suffix')
@click.option('--numbered', 'opt_numbered', is_flag=True,
  help='Number files sequentially')
@click.option('-q', '--quality', 'opt_quality', default=90, type=click.IntRange(0,100, clamp=True),
  show_default=True,
  help='JPEG write quality')
@processor
@click.pass_context
def cli(ctx, pipe, opt_dir_out, opt_ext, opt_frame_type, opt_prefix, opt_suffix,
  opt_numbered, opt_quality):
  """Save to images"""
  
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  
  from vframe.settings import app_cfg
  from vframe.models import types
  from vframe.utils import file_utils


  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  file_utils.ensure_dir(opt_dir_out)
  ext = opt_ext.name.lower()
  frame_count = 0


  # ---------------------------------------------------------------------------
  # process 
  
  while True:
    
    pipe_item = yield
    im = pipe_item.get_image(opt_frame_type)

    # filename options
    if opt_numbered:
      stem = file_utils.zpad(frame_count)
      frame_count += 1
    else:
      stem = Path(pipe_item.filename).stem

    # set filename
    fn = f'{opt_prefix}{stem}{opt_suffix}.{ext}'
    fp_out = join(opt_dir_out, fn)

    # write image
    if ext == 'jpg':
      cv.imwrite(fp_out, pipe_item.get_image(opt_frame_type), [int(cv.IMWRITE_JPEG_QUALITY), opt_quality])
    else:
      cv.imwrite(fp_out, pipe_item.get_image(opt_frame_type))

    # continue pipestream
    pipe.send(pipe_item)