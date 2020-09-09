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
@click.option('--iters', 'opt_iters', default=10,
  help='Number of iterations')
@generator
@click.pass_context
def cli(ctx, sink, opt_iters):
  """Template generator"""
  
  from tqdm import trange

  from vframe.settings import app_cfg

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG

  
  # ---------------------------------------------------------------------------
  # generate

  for i in trange(opt_iters, desc='Generator Template'):
    sink.send(i)