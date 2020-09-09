############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.models import types
from vframe.utils import click_utils
from vframe.utils.click_utils import processor

@click.command('')
@processor
@click.pass_context
def cli(ctx, pipe):
  """Template processor"""
  
  from time import sleep
  import random

  from vframe.settings import app_cfg
  from vframe.utils import im_utils
  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  log.info('Init processor script. Sleep for between 0 - 1 seconds each iteration.')

  # ---------------------------------------------------------------------------
  # process

  while True:

    pipe_item = yield
    
    t = random.uniform(0, 1.0)  # sleep time
    sleep(t)

    pipe.send(pipe_item)