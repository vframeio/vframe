############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.settings.app_cfg import VALID_PIPE_MEDIA_EXTS
from vframe.utils.click_utils import generator

@click.command('')
@click.option('-i', '--input', 'opt_input', required=True,
  help='Path to image or directory')
@click.option('-e', '--exts', 'opt_exts', default=VALID_PIPE_MEDIA_EXTS,
  multiple=True, help='Extensions to glob for')
@click.option('-r', '--recursive', 'opt_recursive', is_flag=True,
  help='Recursive glob')
@click.option('--slice', 'opt_slice', type=(int, int), default=(-1, -1),
  help="Slice list of inputs")
@click.option('--skip-frames', 'opt_skip_frames', is_flag=True,
  help='Skip all frames, only iterate files')
@click.option('--check-exist', 'opt_check_exist', 
  is_flag=True, default=False,
  help='Check files existence before processing')
@click.option('--randomize', 'opt_randomize', is_flag=True, 
  help='Randomize file list before slicing')
@click.option('--media-path', 'opt_new_filepath', type=str,
  default='',
  help='Override JSON filepath')
@generator
@click.pass_context
def cli(ctx, sink, opt_input, opt_recursive, opt_exts, opt_slice, 
  opt_skip_frames, opt_check_exist, opt_randomize, opt_new_filepath):
  """Open media for processing"""

  from tqdm import tqdm
  import dacite

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, READER, SKIP_FILE
  from vframe.settings.app_cfg import USE_PREHASH, USE_DRAW_FRAME
  from vframe.settings.app_cfg import MEDIA_FILTERS, SKIP_MEDIA_FILTERS
  from vframe.models.media import MediaFileReader
  from vframe.utils.sys_utils import SignalInterrupt
  from vframe.utils.file_utils import get_ext

  
  # ---------------------------------------------------------------------------
  # init


  sigint = SignalInterrupt()

  init_obj = {
    'filepath': opt_input,
    'exts': tuple(opt_exts),
    'slice_idxs': opt_slice,
    'recursive': opt_recursive,
    'use_prehash': ctx.obj.get(USE_PREHASH, False),
    'use_draw_frame': ctx.obj.get(USE_DRAW_FRAME, False),
    'media_filters': ctx.obj.get(MEDIA_FILTERS, []),
    'skip_all_frames': opt_skip_frames,
    'opt_check_exist': opt_check_exist,
    'opt_randomize': opt_randomize,
    'opt_new_filepath': opt_new_filepath,
    }

  # init media file reader
  r = dacite.from_dict(data_class=MediaFileReader, data=init_obj)
  ctx.obj[READER] = r
  ctx.obj[SKIP_MEDIA_FILTERS] = get_ext(opt_input) == 'json'

  # error checks
  if not r.n_files:
    LOG.info('No files to process.')
    return

  # process media
  for m in tqdm(r.iter_files(), total=r.n_files, desc='Files', leave=False):
    
    ctx.obj[SKIP_FILE] = False  # reset
    m.skip_all_frames = opt_skip_frames

    if sigint.interrupted:
      m.unload()
      return
    
    for ok in tqdm(m.iter_frames(), total=m.n_frames, desc=m.fn, disable=m.n_frames <= 1, leave=False):
      
      ctx.obj[SKIP_FRAME] = (opt_skip_frames or m.skip_all_frames)

      # TODO: cleanup
      if ctx.obj.get(SKIP_FILE, False) or m._skip_file:
        ctx.obj[SKIP_FILE] = True
        m.set_skip_file()
    
      # check for ctl-c, exit gracefully
      if sigint.interrupted:
        m.unload()
        return
    
      sink.send(m)


  # print stats
  LOG.info(r.stats)