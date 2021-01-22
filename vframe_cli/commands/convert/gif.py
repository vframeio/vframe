############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################"""

import click

from vframe.models import types
from vframe.settings import app_cfg
from vframe.utils import click_utils

ops_interpolation = ['nearest', 'antialias']

@click.command()
@click.option('-i', '--input', 'opt_dir_in', required=True)
@click.option('-o', '--output', 'opt_fp_out')
@click.option('--fps', 'opt_fps', type=int, default=12, help='Frames per second')
@click.option('-e', '--ext', 'opt_ext', default='png', help='ext to glob')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('--decimate', 'opt_decimate', default=0,
  help='Number of frames to between each GIF frame')
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help='Force overwrite video file')
@click.option('--width', 'opt_width', type=int, default=None,
  help='Width output size')
@click.option('--interpolation', 'opt_interp', type=click.Choice(ops_interpolation),
  default='nearest',
  help='Type of interpolation for resizing. Use "nearst" for annotation masks.')
@click.pass_context
def cli(ctx, opt_dir_in, opt_fp_out, opt_fps, opt_ext, 
  opt_slice, opt_decimate, opt_width, opt_force, opt_interp):
  """Creates GIF from directory of images"""
  
  from glob import glob
  from pathlib import Path

  from PIL import Image
  import numpy as np
  from tqdm import tqdm
  from moviepy.editor import VideoClip

  from vframe.utils import file_utils, im_utils

  log = app_cfg.LOG
  log.info('Generating animated GIF...')

  if not opt_fp_out:
    opt_fp_out = str(Path(opt_dir_in).parent / f'{Path(opt_dir_in).name}.gif')
    if Path(opt_fp_out).is_dir() and not opt_force:
      log.error(f'{opt_fp_out} exists. Use "-f/--force" to overwrite')

  # glob comp images  
  fps_im = sorted([im for im in glob(str(Path(opt_dir_in) / f'*.{opt_ext}'))])
  if opt_slice:
    fps_im = fps_im[opt_slice[0]:opt_slice[1]]
  if opt_decimate:
    fps_im = [x for i, x in enumerate(fps_im) if i % opt_decimate == 0]

  if len(fps_im) > 100:
    log.warn('Creating GIF with over 100 frames. Could create memory issues')

  # load all images into list
  ims = []
  for fp_im in tqdm(fps_im):
    im = Image.open(fp_im)
    w, h = im.size
    if opt_width:
      h = int(opt_width * h / w)
      #im = im.resize((opt_width, h), Image.BICUBIC)
      #im = im.resize((opt_width, h), Image.NEAREST)
      app_cfg.LOG.debug('resize')
      im = im.resize((opt_width, h))
    ims.append(im)

  num_frames = len(fps_im)
  duration_sec = num_frames / opt_fps

  def make_frame(t):
    frame_idx = int(np.clip(np.round(t * opt_fps), 0, num_frames - 1))
    im = ims[frame_idx]
    im_np_rgb = im_utils.pil2np(im, swap=False)
    return im_np_rgb

  animation = VideoClip(make_frame, duration=duration_sec)
  animation.write_gif(opt_fp_out, fps=opt_fps) # export as GIF (slow)


  # ims[0].save(opt_fp_out,
  #            save_all=True,
  #            append_images=ims[1:],
  #            duration=int((1.0/opt_fps)*1000),
  #            loop=0)

  
  log.info('Done.')