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
@click.option('-i', '--input', 'opt_fp_meta', required=True,
  help='Input metadata or annotation CSV')
@click.option('--labelmap', 'opt_fp_labelmap', required=True,
  help='Labelmap YAML')
@click.pass_context
def cli(ctx, opt_fp_meta, opt_fp_labelmap):
  """Rename files in render subdirectories"""

  import pandas as pd
  from vframe.utils.file_utils import load_yaml
  from vframe.models.annotation import LabelMaps

  log = app_cfg.LOG

  df_meta = pd.read_csv(opt_fp_meta)
  labelmaps = load_yaml(opt_fp_labelmap, data_class=LabelMaps)

  for label in labelmaps.labels:
    df_meta.loc[(df_meta.label_enum == label.enum), 'label_index'] = label.index
    df_meta.loc[(df_meta.label_enum == label.enum), 'label_display'] = label.display

  # write csv
  df_meta.to_csv(opt_fp_meta, index=False)
