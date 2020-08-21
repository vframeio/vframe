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
@click.option( '-n', '--name', 'opt_data_keys', required=True, type=(str,str),
  help='From and To data keys')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_keys):
  """Modify BBoxes"""
  
  from vframe.settings import app_cfg
  import copy
  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  log.debug(f'Copy data from {opt_data_keys[0]} to {opt_data_keys[1]}')


  # ---------------------------------------------------------------------------
  # process

  while True:

    pipe_item = yield

    item_data = pipe_item.get_data(opt_data_keys[0])
    item_data_copy = {opt_data_keys[1]: copy.deepcopy(item_data)}
    pipe_item.add_data(item_data_copy)

    pipe.send(pipe_item)