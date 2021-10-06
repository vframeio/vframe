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
@click.option('--no-load', 'opt_no_load', is_flag=True,
  help='Don\'t load frames, only iterate images/videos')
@generator
@click.pass_context
def cli(ctx, sink, opt_input, opt_recursive, opt_exts, opt_slice, opt_no_load):
  """Open media for processing"""

  from tqdm import tqdm
  import dacite

  from vframe.settings.app_cfg import LOG, SKIP_FRAME
  from vframe.settings.app_cfg import USE_PHASH, USE_DRAW_FRAME
  from vframe.models.media import MediaFileReader
  from vframe.utils.sys_utils import SignalInterrupt

  
  # ---------------------------------------------------------------------------
  # init

  print('init open')

  sigint = SignalInterrupt()

  init_obj = {
    'filepath': opt_input,
    'exts': tuple(opt_exts),
    'slice_idxs': opt_slice,
    'recursive': opt_recursive,
    'use_prehash': ctx.obj.get(USE_PHASH, False),
    'use_draw_frame': ctx.obj.get(USE_DRAW_FRAME, False),
    
    }

  # init media file reader
  r = dacite.from_dict(data_class=MediaFileReader, data=init_obj)
  ctx.obj['reader'] = r

  # process media
  for m in tqdm(r.iter_files(), total=r.n_files, desc='Files', leave=False):

    m.skip_all_frames = opt_no_load

    if sigint.interrupted:
      m.unload()
      return
    
    for ok in tqdm(m.iter_frames(), total=m.n_frames, desc=m.fn, disable=m.n_frames <= 1, leave=False):
    
      if not m.vstream.frame_count > 0:
        continue
    
      # check for ctl-c, exit gracefully
      if sigint.interrupted:
        m.unload()
        return
    
      # init frame-iter presets
      ctx.opts[SKIP_FRAME] = False
      sink.send(m)

  # print stats
  LOG.info(r.stats)