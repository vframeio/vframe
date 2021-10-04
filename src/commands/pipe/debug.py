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

opt_choices = ['detections']

@click.command('')
@click.option('-t', '--type', 'opt_type',
 type=click.Choice(opt_choices), default='detections',
  help='Debug action type')
@processor
@click.pass_context
def cli(ctx, sink, opt_type):
  """Print frame data"""

  from pprint import pprint
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY

  opt_data_keys = None

  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME_KEY]:
      sink.send(M)
      continue
    
    if opt_type == 'detections':

      all_keys = list(M.metadata.get(M.index).keys())
      if not opt_data_keys:
        data_keys = all_keys
      else:
        data_keys = [k for k in opt_data_keys if k in all_keys]


      for data_key in data_keys:
        # draw bbox, labels, mask
        item_data = M.metadata.get(M.index).get(data_key)
        LOG.debug(f'{data_key} detections: {len(item_data.detections)}')

    
    sink.send(M)