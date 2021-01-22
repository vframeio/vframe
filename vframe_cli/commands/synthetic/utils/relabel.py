#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

import click

from vframe.settings import app_cfg

ext_choices = ['jpg', 'png']

@click.command()
@click.option('-i', '--input', 'opt_input', required=True,
  help='Input file CSV')
@click.option('-o', '--output', 'opt_output',
  help='Input file CSV')
@click.option('--label', 'opt_labels_from_to', required=True, type=(str,str),
  multiple=True, help='Label from, to')
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_labels_from_to):
  """Relabel label enum in annotation CSV"""

  import pandas as pd

  log = app_cfg.LOG

  opt_output = opt_output if opt_output else opt_input
  df_meta = pd.read_csv(opt_input)

  for label_from, label_to in opt_labels_from_to:
    df_meta.loc[(df_meta.label_enum == label_from), 'label_enum'] = label_to

  # write csv
  df_meta.to_csv(opt_output, index=False)
