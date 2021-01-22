############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils.click_utils import processor

@click.command('')
@click.option('-o', '--output', 'opt_fp_out', required=True,
  help='Path to output file')
@click.option('--minify', 'opt_minify', is_flag=True, default=False,
  help='Minify JSON output')
@processor
@click.pass_context
def cli(ctx, pipe, opt_fp_out, opt_minify):
  """Save data"""
  
  from os.path import join
  
  from vframe.settings import app_cfg
  from vframe.utils import file_utils

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG

  # error check
  ext = file_utils.get_ext(opt_fp_out).lower()
  if not ext == 'json':
    log.error('Only JSON export supported')
    return

  # ---------------------------------------------------------------------------
  # process 

  header_items = []
  
  # accumulate all pipe items
  while True:

    try:
      pipe_item = yield
      header = ctx.obj['header']
      if header.frame_index == header.last_frame_index:
        header_items.append(header.to_dict())
      pipe.send(pipe_item)

    except GeneratorExit as e:

      # FIXME: find better way to detect end of generator?
      if not len(header_items) > 0:
        log.error('No items to save')
        return

      # write to file
      log.info(f'Writing {len(header_items)} item(s) to {opt_fp_out}')
      file_utils.write_json(header_items, opt_fp_out, minify=opt_minify)

      # exit yield loop
      break