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
def cli(sink, opt_group_by):
  """List models in the ModelZoo by attribute"""


  from vframe.settings.app_cfg import LOG, modelzoo
  from vframe.utils import model_utils


  txt = model_utils.list_models(group_by=opt_group_by)
  LOG.info(txt)
  
