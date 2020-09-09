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

  log.info('Template info message')
  log.debug('Template debug message')
  log.warn('Template warn message')
  log.error('Template error message')

  log.info(f'{app_cfg.UCODE_OK}')