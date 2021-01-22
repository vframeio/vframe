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

@click.command()
@click.option('-i', '--input', 'opt_input', required=True,
  help='Input file CSV')
@click.option('--dry-run/--confirm', 'opt_dry_run', is_flag=True, default=True,
  help='Dry run, do not delete any files')
@click.pass_context
def cli(ctx, opt_input, opt_dry_run):
  """Deduplicate CSV"""

  import pandas as pd

  df = pd.read_csv(opt_input)
  n_before = len(df)
  df.drop_duplicates(keep='first', inplace=True)
  app_cfg.LOG.info(f'Found {n_before - len(df)} duplicates')
  if not opt_dry_run:
    df.to_csv(opt_input, index=False)
    app_cfg.LOG.info('Dropped dupes.')
  else:
    app_cfg.LOG.info('Dry run. Add "--confirm" to overwrite file.')
