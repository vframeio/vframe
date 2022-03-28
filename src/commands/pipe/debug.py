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
from vframe.settings.app_cfg import caption_accessors, compare_accessors
caption_accessors.update(compare_accessors)
accessors = caption_accessors


@click.command('')
@click.option('-a', '--attribute', 'opt_attrs', required=True, multiple=True,
  type=click.Choice(list(accessors.keys())),)
@processor
@click.pass_context
def cli(ctx, sink, opt_attrs):
  """Debug media attributes"""

  from pprint import pprint
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME, READER
  from vframe.models.types import MediaType

  opt_data_keys = None

  while True:

    M = yield
    R = ctx.obj[READER]

    opt_attrs_file = [a for a in opt_attrs if a in compare_accessors]
    opt_attrs_frame = [a for a in opt_attrs if a in caption_accessors]

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue
    
    text = []

    if (M.type == MediaType.VIDEO and M.is_first_item) or M.type == MediaType.IMAGE:
      # beginning of new media file      
      for attr in opt_attrs_file:
        try:
          text.append(f'{attr}: {getattr(M, accessors.get(attr))}')
        except Exception as e:
          LOG.error(f'{attr} is not a valid accessor. Error: {e}')

      LOG.debug(f'File: {", ".join(text)}')

    text = []
    for attr in opt_attrs_frame:
      try:
        text.append(f'{attr}: {getattr(M, accessors.get(attr))}')
      except Exception as e:
        LOG.error(f'{attr} is not a valid accessor. Error: {e}')
    LOG.debug(f'File: {", ".join(text)}')

    
    sink.send(M)