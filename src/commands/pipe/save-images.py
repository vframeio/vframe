############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.settings import app_cfg
from vframe.models import types
from vframe.utils import click_utils
from vframe.utils.click_utils import processor

@click.command('')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output directory')
@click.option('-e', '--ext', 'opt_ext', default=None,
  type=types.ImageFileExtVar,
  help=click_utils.show_help(types.ImageFileExt))
@click.option('-f', '--frame', 'opt_frame_type', 
  default=types.FrameImage.DRAW.name.lower(),
  type=types.FrameImageVar,
  help=click_utils.show_help(types.FrameImage))
@click.option('--prefix', 'opt_prefix', default='',
  help='Filename prefix')
@click.option('--suffix', 'opt_suffix', default='',
  help='Filename suffix')
@click.option('--numbered', 'opt_numbered', is_flag=True,
  help='Number files sequentially')
@click.option('-z', '--zeros', 'opt_n_zeros', default=app_cfg.ZERO_PADDING)
@click.option('-q', '--quality', 'opt_quality', default=1, 
  type=click.FloatRange(0,1, clamp=True), show_default=True,
  help='JPEG write quality')
@click.option('--subdirs', 'opt_keep_subdirs', is_flag=True,
  help='Keep subdirectory structure in output directory')
@processor
@click.pass_context
def cli(ctx, sink, opt_dir_out, opt_ext, opt_frame_type, opt_prefix, opt_suffix,
  opt_numbered, opt_quality, opt_n_zeros, opt_keep_subdirs):
  """Save to images"""
  
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  
  from vframe.models.types import MediaType
  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME, READER
  from vframe.utils.file_utils import zpad, get_ext, ensure_dir


  # ---------------------------------------------------------------------------
  # initialize

  if opt_frame_type == types.FrameImage.DRAW:
    ctx.obj[USE_DRAW_FRAME] = True

  ensure_dir(opt_dir_out)
  frame_count = 0


  # ---------------------------------------------------------------------------
  # process 
  
  while True:
    
    M = yield
    R = ctx.obj[READER]

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME]:
      sink.send(M)
      continue
      
    im = M.images.get(opt_frame_type)

    # filename options
    if opt_numbered:
      stem = zpad(frame_count, z=opt_n_zeros)
      frame_count += 1
    else:
      stem = Path(M.filename).stem

    # add relative subdir to output destination
    if opt_keep_subdirs and Path(M.filepath).parent != Path(R.filepath):
      fp_subdir_rel = Path(M.filepath).relative_to(Path(R.filepath)).parent
    else:
      fp_subdir_rel = ''
    
    # ensure output directory
    fp_dir_out = join(opt_dir_out, fp_subdir_rel)
    ensure_dir(fp_dir_out)

    # output filepath
    if M.type == types.MediaType.IMAGE:
      ext = opt_ext.name.lower() if opt_ext else get_ext(M.filename)
      fn = f'{opt_prefix}{stem}{opt_suffix}.{ext}'
      fp_out = join(fp_dir_out, fn)
      
    elif M.type == types.MediaType.VIDEO:
      ext = opt_ext.name.lower() if opt_ext is not None else 'jpg'
      fn = f'{zpad(M.index, z=opt_n_zeros)}.{ext}'
      fp_out = join(fp_dir_out, Path(M.filepath).stem, fn)

    ensure_dir(fp_out)

    # write image
    if ext == 'jpg':
      cv.imwrite(fp_out, im, [int(cv.IMWRITE_JPEG_QUALITY), int(opt_quality*100)])
    else:
      cv.imwrite(fp_out, im)

    # continue
    sink.send(M)