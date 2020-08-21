############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

@click.command('')
@click.pass_context
def cli(ctx):
  """Simple template"""

  # ------------------------------------------------
  # imports

  from os.path import join

  from vframe.settings import app_cfg

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  log.debug('Simple template')