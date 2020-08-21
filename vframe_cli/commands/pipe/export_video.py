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
@click.option('-e', '--ext', 'opt_ext', default='mp4',
  type=types.VideoFileExtVar,
  help=click_utils.show_help(types.VideoFileExt))
@click.option('--frame', 'opt_frame_type', default='draw',
  type=types.FrameImageVar,
  help=click_utils.show_help(types.FrameImage))
@click.option('--codec', 'opt_codec', 
  type=click.Choice(['mp4v', 'avc1']),
  default='mp4v',
  help='Four CC codec (TODO: enumerate, check available codecs)')
@processor
@click.pass_context
def cli(ctx, pipe, opt_dir_out, opt_ext, opt_frame_type, opt_codec):
  """Save to video"""
  
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  
  from vframe.settings import app_cfg
  from vframe.utils import file_utils


  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG

  file_utils.ensure_dir(opt_dir_out)
  ext = opt_ext.name.lower()
  four_cc = cv.VideoWriter_fourcc(*f'{opt_codec}')

  frame_count = 0
  filepath = None
  is_writing = False
  video_out = None

  # ---------------------------------------------------------------------------
  # process 
  
  while True:
    
    pipe_item = yield
    header = ctx.obj['header']

    if header.frame_index == header.frame_end and video_out is not None:
      video_out.release()
      is_writing = False
      filepath = None

    # start new video if new headers
    if header.filepath != filepath and header.frame_index == 0:
      filepath = header.filepath
      fn = Path(header.filename).stem
      fp_out = join(opt_dir_out, f'{fn}.{ext}')
      video_out = cv.VideoWriter(fp_out, four_cc, header.fps, header.dim)
      is_writing = True

    if is_writing:
      im = pipe_item.get_image(types.FrameImage.DRAW)
      video_out.write(im)


    pipe.send(pipe_item)