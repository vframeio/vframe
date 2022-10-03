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
@click.option('-i', '--input', 'opt_input', required=True, 
    help="Path to input")
@click.option('-o', '--output', 'opt_output',
    help="Path to output")
@click.pass_context
def cli(ctx, opt_input, opt_output):
  """Remove items without frame detection meta"""

  from pathlib import Path
  from tqdm import tqdm

  from vframe.utils.file_utils import load_json, write_json, add_suffix
  from vframe.settings.app_cfg import LOG

  # load
  items = load_json(opt_input)

  # filter
  items_pruned = [item for item in items if item.get('frames_meta')]

  # save
  opt_output = opt_output if opt_output else add_suffix(opt_input, '_pruned')
  write_json(opt_output, items_pruned)

  # stats
  LOG.info(f'Original: {len(items)}')
  LOG.info(f'New: {len(items_pruned)}')
  LOG.info(f'Removed: {len(items) - len(items_pruned)}')

  