############################################################################# 
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click


@click.command()
@click.option('-i', '--input', 'opt_dir_ims', required=True)
@click.option('-o', '--output', 'opt_fp_out_video',
  help='Path to video output file')
@click.option('--video', 'opt_write_video', is_flag=True,
  help='Writes video to default location as parent directory name MP4"')
@click.option('--fps', 'opt_fps', type=int, default=12, help='Frames per second')
@click.option('--bg', 'opt_bg_color', default=0.75, type=click.FloatRange(0,1),
  help='Mask darkness')
@click.option('--bitrate', 'opt_bitrate', default=16, type=float, help='Video bitrate (Mbp/s')
@click.option('--mp4_codec', 'opt_codec', default='libx264', help='Video bitrate (Mbp/s')
@click.option('--cleanup', 'opt_cleanup', is_flag=True, default=False, show_default=True,
  help='Deletes image sequence files after writing video')
@click.option('-t', '--threads', 'opt_threads', default=12,
  help='Number threads')
@click.pass_context
def cli(ctx, opt_dir_ims, opt_fps, opt_bitrate, opt_codec, opt_fp_out_video, opt_bg_color, 
  opt_write_video, opt_cleanup, opt_threads):
  """Generates real-mask composite images"""
  
  import pandas as pd
  from glob import glob
  from pathlib import Path

  import cv2 as cv
  import numpy as np
  import blend_modes
  from moviepy.editor import VideoClip
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool

  from vframe.settings import app_cfg
  from vframe.utils import file_utils, draw_utils

  log = app_cfg.LOG
  log.info(f'Compositing masks and synthetic images from: {Path(opt_dir_ims).name}')

  # init
  fps_ims_comp = []

  # glob images
  fps_ims_real = sorted([im for im in glob(str(Path(opt_dir_ims) / app_cfg.DN_REAL / '*.png'))])
  fps_ims_mask = sorted([im for im in glob(str(Path(opt_dir_ims) / app_cfg.DN_MASK / '*.png'))])
  if not len(fps_ims_mask) == len(fps_ims_real):
    log.info(f'found {len(fps_ims_mask)} mask images')
    log.info(f'found {len(fps_ims_real)} real images')
    log.warn('Error: number images not same')

  # ensure output dir
  opt_dir_ims_comp = Path(opt_dir_ims) / app_cfg.DN_COMP
  if not Path(opt_dir_ims_comp).is_dir():
    Path(opt_dir_ims_comp).mkdir(parents=True, exist_ok=True)


  def pool_worker(fps_ims):

    fp_im_mask = fps_ims['mask']
    fp_im_real = fps_ims['real']
    
    # add background/black
    #im_mask = cv.cvtColor(cv.imread(fp_im_mask).astype(np.float32), cv.COLOR_BGR2BGRA)
    im_mask = cv.cvtColor(cv.imread(fp_im_mask), cv.COLOR_BGR2BGRA).astype(np.float32)
    bg_color = np.array([0.,0.,0.,255.])  # black fill
    mask_idxs = np.all(im_mask == bg_color, axis=2)
    im_mask[mask_idxs] = [0,0,0,255]

    # add color overlay
    im_real = cv.cvtColor(cv.imread(fp_im_real), cv.COLOR_BGR2BGRA).astype(np.float32)
    im_comp = blend_modes.multiply(im_real, im_mask, opt_bg_color)
    im_comp = blend_modes.addition(im_comp, im_mask, 0.5)
    im_comp = cv.cvtColor(im_comp, cv.COLOR_BGRA2BGR)

    fp_out = Path(opt_dir_ims_comp) / Path(fp_im_mask).name
    cv.imwrite(str(fp_out), im_comp)


  
  # ----------------------------------------------------------------------------------
  # Process images

  fps_ims = [{'mask': fp_mask, 'real': fp_real} for fp_mask, fp_real in zip(fps_ims_mask, fps_ims_real)]
  # Multiprocess/threading use imap instead of map via @hkyi Stack Overflow 41920124
  with Pool(opt_threads) as p:
    d = f'Compositing x{opt_threads}'
    pool_results = list(tqdm(p.imap(pool_worker, fps_ims), total=len(fps_ims), desc=d))

  # ----------------------------------------------------------------------------------
  # Write video 

  if opt_write_video:
  
    if not opt_fp_out_video:
      # create default video file name if none provided
      opt_fp_out_video = str(Path(opt_dir_ims) / f'{Path(opt_dir_ims).name}.mp4')

    # glob comp images  
    fps_ims_comp = sorted([im for im in glob(str(Path(opt_dir_ims_comp) / '*.png'))])

    opt_bitrate = f'{opt_bitrate}M'  # megabits / second
    num_frames = len(fps_ims_comp)
    duration_sec = num_frames / opt_fps

    log.debug(f'num images: {len(fps_ims_comp)}')
    def make_frame(t):
      #global fps_ims_comp
      frame_idx = int(np.clip(np.round(t * opt_fps), 0, num_frames - 1))
      fp_im = fps_ims_comp[frame_idx]
      im = cv.cvtColor(cv.imread(fp_im), cv.COLOR_BGR2RGB)  # Moviepy uses RGB
      return im

    log.info(f'Generating movieclip to: {opt_fp_out_video}')
    VideoClip(make_frame, duration=duration_sec).write_videofile(opt_fp_out_video, fps=opt_fps, codec=opt_codec, bitrate=opt_bitrate)
    log.info('Done.')

  if opt_cleanup:
    # remove all comp images
    log.info('Removing all temporary images...')
    import shutil
    shutil.rmtree(opt_dir_ims_comp)
    log.info(f'Deleted {opt_dir_ims_comp}')