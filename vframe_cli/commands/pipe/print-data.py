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
@processor
@click.pass_context
def cli(ctx, pipe):
  """Print data"""

  from os.path import join
  from pprint import pprint


  from vframe.settings import app_cfg
  from vframe.utils import file_utils

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG


  # ---------------------------------------------------------------------------
  # process 

  while True:

    pipe_item = yield
    header = ctx.obj['header']
    
    serialized_data = header.to_dict()
    pprint(serialized_data)

    pipe.send(pipe_item)