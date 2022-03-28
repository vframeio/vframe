############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils import click_utils
from vframe.models.types import FrameImage, FrameImageVar
from vframe.utils.click_utils import processor

@click.command('')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output directory')
@click.option('-f', '--frame', 'opt_frame_type', default=FrameImage.DRAW.name.lower(),
  type=FrameImageVar,
  help=click_utils.show_help(FrameImage))
@click.option('--fps', 'opt_fps', default=6,
  help='Override media FPS')
@click.option('--loop', 'opt_loop', default=0,
  help='Number of times to loop GIF (0 = infinite)')
@click.option('--colors', 'opt_colors', default=256,
  help='Number of times to loop GIF (0 = infinite)')
@click.option('--optimize/--no-optimize', 'opt_optimize', 
  is_flag=True, default=True,
  help='Number of times to loop GIF (0 = infinite)')
@click.option('--subdirs', 'opt_keep_subdirs', is_flag=True,
  help='Keep subdirectory structure in output directory')
@click.option('--palette', 'opt_palette', 
  type=click.Choice(['web', 'adaptive']), default='web',
  help='Compression color palette')
@click.option('--verbose', 'opt_verbose', is_flag=True,
  help='Check filesize after writing GIF')
@processor
@click.pass_context
def cli(ctx, sink, opt_dir_out, opt_frame_type, opt_fps, opt_keep_subdirs,
  opt_colors, opt_loop, opt_palette, opt_optimize, opt_verbose):
  """Save to animated GIF"""
  
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  from PIL import Image
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME, READER
  from vframe.models.types import MediaType
  from vframe.utils.im_utils import np2pil, write_animated_gif
  from vframe.utils.file_utils import ensure_dir, filesize


  # ---------------------------------------------------------------------------
  # initialize

  if opt_frame_type == FrameImage.DRAW:
    ctx.obj[USE_DRAW_FRAME] = True
  
  frames = None
  fp_parent = None
  
  opt_palette = Image.WEB if opt_palette == 'web' else Image.ADAPTIVE
  opt_duration = 1000 // opt_fps

  convert_kwargs = {
    'mode': 'P', 
    'dither': None, 
    'palette': opt_palette, 
    'colors': opt_colors,
    }
  gif_kwargs = {
    'format': 'GIF',
    'save_all': True, 
    'optimize': opt_optimize,
    'duration': opt_duration, 
    'loop':opt_loop,
  }


  # ---------------------------------------------------------------------------
  # process 
  
  while True:
    
    M = yield
    R = ctx.obj[READER]

    # Check if last file in a subdir
    if M.parent != fp_parent and frames is not None:
      write_animated_gif(fp_out, frames, verbose=opt_verbose, **gif_kwargs)
      frames = None
      fp_parent = None

    # check if new file in new subdir and start new gif
    if not ctx.obj[SKIP_FRAME] and \
      frames is None and \
      M.parent != fp_parent:

      # configure file io, add relative subdir output dir
      if opt_keep_subdirs and M.parent != Path(R.filepath):
        fp_subdir_rel = Path(M.filepath).relative_to(Path(R.filepath)).parent
      else:
        fp_subdir_rel = ''

      # output directory
      fp_dir_out = join(opt_dir_out, fp_subdir_rel)
      ensure_dir(fp_dir_out)

      # filename
      if M.type == MediaType.VIDEO:
        fn = Path(M.filename).stem  # use video name as dir
      elif M.type == MediaType.IMAGE:
        fn = M.parent.name  # use subdir name

      # output file
      fp_out = join(fp_dir_out, f'{fn}.gif')

      # init frames holder and store reference to current dir
      frames = []
      fp_parent = M.parent
      if opt_verbose:
        LOG.debug(f'Start: {fp_out}')


    # check if frame is usable and add to stack
    if not ctx.obj[SKIP_FRAME] and frames is not None:
      im = M.images.get(opt_frame_type)
      im_pil = np2pil(im).convert(**convert_kwargs)
      frames.append(im_pil)

    # check if last frame of last file
    if R.is_last_item and frames is not None and len(frames):
      write_animated_gif(fp_out, frames, verbose=opt_verbose, **gif_kwargs)
      frames = None
      fp_parent = None


  sink.send(media)