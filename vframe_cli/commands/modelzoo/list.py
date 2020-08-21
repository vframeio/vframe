############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

@click.command('list')
@click.option('-g', '--group-by', 'opt_group_by', default="output",
  help='Field to group by: [processor, output]')
@click.pass_context
def cli(ctx, opt_group_by):
  """List models in the ModelZoo by attribute"""

  # ------------------------------------------------
  # imports

  from vframe.utils import model_utils
  from vframe.settings import app_cfg, modelzoo_cfg

  # ------------------------------------------------
  # start

  log = app_cfg.LOG
  modelzoo = modelzoo_cfg.modelzoo

  txt = model_utils.list_models(group_by=opt_group_by)
  log.info(txt)
  
