############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.utils.click_utils import generator

@click.command('')
@click.option('-i', '--input', 'opt_input', required=True,
  help='Path to image or image folder')
@click.option('-r', '--recursive', 'opt_recursive', is_flag=True,
  help='Recursive glob')
@click.option('--replace-path', 'opt_replace_path',
  help="Replace file parent path")
@click.option('--width', 'opt_width', default=None,
  help="Draw image width")
@click.option('--height', 'opt_height', default=None,
  help="Draw image height")
@click.option('-e', '--exts', 'opt_exts', default=['jpg', 'jpeg', 'png'],
  multiple=True,
  help='Extensions to glob for')
@click.option('--slice', 'opt_slice', type=(int, int), 
  default=(None, None),
  help="Slice list of inputs")
@click.option('--start', 'opt_frame_start', type=int,
  help="0-indexed frame number to start on")
@click.option('--end', 'opt_frame_end', type=int, default=None,
  help='0-indexed frame number to end on')
@click.option('--decimate', 'opt_decimate', type=int, default=None,
  help="Number of frames to skip between processing")
@generator
@click.pass_context
def cli(ctx, sink, opt_input, opt_recursive, opt_replace_path, opt_width, opt_height, opt_exts, opt_slice,
  opt_frame_start, opt_frame_end, opt_decimate):
  """+ Add images or videos"""
  
  from pathlib import Path
  from os.path import join

  from tqdm import tqdm
  
  import cv2 as cv
  from vframe.settings import app_cfg
  from vframe.models.pipe_item import PipeContextHeader, PipeFrame
  from vframe.utils import file_utils

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG

  # Get input filepaths
  if Path(opt_input).is_dir():
    # glob directory
    exts = opt_exts if opt_exts is not None else app_cfg.VALID_PIPE_EXTS
    # load multiple images
    items = file_utils.glob_multi(opt_input, exts=exts, recursive=opt_recursive)
    if len(items) == 0:
      log.error(f'No {opt_exts} found in {opt_input}')
      log.info('Use "-e/--ext" option to select different glob extension')
      return


    items.sort()

  elif Path(opt_input).is_file():
    
    ext = file_utils.get_ext(opt_input).lower()
    
    # load preprocessed JSON_input
    if ext in app_cfg.VALID_PIPE_DATA_EXTS:
      items = file_utils.load_file(opt_input)
    else:
      # load single image or video
      items = [opt_input]
  else:
    log.error(f'{opt_input} is not a valid file or folder')
    return
    
  # slice input
  if any(opt_slice):
    items = items[opt_slice[0]:opt_slice[1]]


  
  # ---------------------------------------------------------------------------
  # process

  for item in tqdm(items, desc='Files', leave=False):
    

    if type(item) == dict:      
      
      # replace parent filepath if optioned
      if opt_replace_path is not None:
        item['filepath'] = join(opt_replace_path, Path(item['filepath']).name)
      
      fp_item = item['filepath']
      if not Path(fp_item).is_file():
        log.error(f'Does not exist: {fp_item}')
        continue

      ctx.obj['header'] = PipeContextHeader.from_dict(item)
   
    else:
      
      fp_item = item
      if not Path(fp_item).is_file():
        log.error(f'Does not exist: {fp_item}')
        continue
      
      ctx.obj['header'] = PipeContextHeader(fp_item)

    header = ctx.obj['header']

    ext = file_utils.get_ext(fp_item)

    if ext in app_cfg.VALID_PIPE_IMAGE_EXTS:
      frame = cv.imread(fp_item)
      pipe_frame = PipeFrame(frame)
      sink.send(pipe_frame)

    elif ext in app_cfg.VALID_PIPE_VIDEO_EXTS:

      video = header.video

      # init video
      header.set_frame_min_max(opt_frame_start, opt_frame_end, opt_decimate)
      if header.frame_start is not None:
        video.set(cv.CAP_PROP_POS_FRAMES, header.frame_start)
        header.set_frame_index(header.frame_start)

      n_frames = header.frame_end - (header.frame_start)
      pbar = tqdm(total=n_frames, desc=header.filename, initial=header.frame_start, leave=False)

      while video.isOpened():

        if header.frame_index > header.frame_end:
          pbar.close()
          break

        frame_ok, frame = video.read()

        if not frame_ok:
          pbar.close()
          break
        elif opt_decimate and (header.frame_index % opt_decimate):
          pbar.update()
          header.increment_frame()
          continue
        else:
          pbar.update()
          pipe_frame = PipeFrame(frame)      
          sink.send(pipe_frame)
          header.increment_frame()




  
  


