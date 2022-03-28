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
from vframe.models.types import FrameImage, FrameImageVar, VideoFileExt, VideoFileExtVar
from vframe.models.types import MediaType
from vframe.utils.click_utils import processor

@click.command('')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output directory')
@click.option('-e', '--ext', 'opt_ext', default='mp4',
  type=VideoFileExtVar,
  help=click_utils.show_help(VideoFileExt))
@click.option('-f', '--frame', 'opt_frame_type', default=FrameImage.DRAW.name.lower(),
  type=FrameImageVar,
  help=click_utils.show_help(FrameImage))
@click.option('--fps', 'opt_fps', type=int, default=None,
  help='Override media FPS')
@click.option('--codec', 'opt_codec', 
  type=click.Choice(['mp4v', 'avc1']),
  default='mp4v',
  help='Four CC codec')  # TODO: enumerate, check available codecs
@click.option('--subdirs', 'opt_keep_subdirs', is_flag=True,
  help='Keep subdirectory structure in output directory')
@processor
@click.pass_context
def cli(ctx, sink, opt_dir_out, opt_ext, opt_frame_type, opt_codec, opt_fps, opt_keep_subdirs):
  """Save to video"""
  
  from os.path import join
  from pathlib import Path

  import cv2 as cv
  
  from vframe.settings import app_cfg
  from vframe.settings.app_cfg import LOG, USE_DRAW_FRAME, READER
  from vframe.utils.file_utils import ensure_dir


  # ---------------------------------------------------------------------------
  # initialize

  
  if opt_frame_type == FrameImage.DRAW:
    ctx.obj[USE_DRAW_FRAME] = True

  ext = opt_ext.name.lower()
  four_cc = cv.VideoWriter_fourcc(*f'{opt_codec}')

  video_out = None
  filepath = None

  # ---------------------------------------------------------------------------
  # process 
  
  while True:
    
    M = yield
    R = ctx.obj[READER]

    if M.type == MediaType.VIDEO:
      
      # start new video
      if not ctx.obj[app_cfg.SKIP_FRAME] and video_out is None and M.filepath != filepath:
        # configure file io
        # add relative subdir to output destination
        if opt_keep_subdirs and Path(M.filepath).parent != Path(R.filepath):
          fp_subdir_rel = Path(M.filepath).relative_to(Path(R.filepath)).parent
        else:
          fp_subdir_rel = ''
        # ensure output directory
        fp_dir_out = join(opt_dir_out, fp_subdir_rel)
        ensure_dir(fp_dir_out)
        # output file
        fn = Path(M.filename).stem
        fp_out = join(fp_dir_out, f'{fn}.{ext}')
        # video writer settings
        dim = M.images.get(opt_frame_type).shape[:2][::-1]
        fps = opt_fps if opt_fps else M.fps
        # init new video writer
        video_out = cv.VideoWriter(fp_out, four_cc, fps, tuple(dim))
        video_out.set(cv.VIDEOWRITER_PROP_QUALITY, 1)
        # store reference to current file to check for new media
        filepath = M.filepath

      # add frame
      if not ctx.obj[app_cfg.SKIP_FRAME] and video_out is not None:
        im = M.images.get(opt_frame_type)
        video_out.write(im)

      # end video
      if M.is_last_item and video_out is not None:
        video_out.release()
        video_out = None
        filepath = None

    sink.send(M)