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
@click.option('-n', '--name', 'opt_data_keys', multiple=True,
  help='Include this label')
@processor
@click.pass_context
def cli(ctx, sink, opt_data_keys):
  """Pop (remove) metadata for specified data label"""
  
  from vframe.settings.app_cfg import LOG
    
  
  while True:

    M = yield

    for k in opt_data_keys:
      if k in M.metadata[M.index].keys():
        M.metadata[M.index].pop(k, None)
    
    sink.send(M)