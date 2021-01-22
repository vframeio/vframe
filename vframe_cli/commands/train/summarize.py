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
  help='Input annotation CSV')
@click.option('-o', '--output', 'opt_output',
  help='Output annotation CSV')
@click.option('-f', '--force', 'opt_force', is_flag=True)
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_force):
  """Summarize annotations"""

  from pathlib import Path

  import pandas as pd

  from vframe.models.geometry import BBox


  log = app_cfg.LOG

  df = pd.read_csv(opt_input)
  opt_output = opt_output if opt_output else opt_input.replace('.csv', '_summary.csv')

  if Path(opt_output).is_file() and not opt_force:
    log.error('File exists. Use "-f/--force" to overwrite')
    return

  anno_stats = []
  groups = df.groupby('label_enum')

  for label_enum, df_group in groups:
    # filter annos
    annos = df_group[df_group.label_enum == label_enum]
    label_index = df_group.label_index.values[0]

    # create list of bboxes
    bboxes = [BBox(r.x1, r.y1, r.x2, r.y2, r.dw, r.dh) for i, r in annos.iterrows()]
    # conver to dict of w, h
    bboxes_wh = [{'width': b.width, 'height': b.height} for b in bboxes]
    # to df
    df_bboxes = df.from_dict(bboxes_wh)
    # compute stats
    std_wh = df_bboxes.std()
    mean_wh = df_bboxes.mean()
    median_wh = df_bboxes.median()
    max_wh = df_bboxes.max()
    min_wh = df_bboxes.min()

    o = {
      'label_index': label_index,
      'label_count': len(df_group),
      'label_enum': label_enum,
      'min_width': min_wh.width,
      'max_width': max_wh.width,
      'std_width': std_wh.width,
      'mean_width': mean_wh.width,
      'median_width': median_wh.width,
      'min_height': min_wh.height,
      'max_height': max_wh.height,
      'std_height': std_wh.height,
      'mean_height': mean_wh.height,
      'median_height': median_wh.height,
    }
    anno_stats.append(o)


  # create dataframe
  df_out = pd.DataFrame.from_dict(anno_stats)
  # change order of cols
  cols = list(df_out.keys())
  cols.remove('label_enum')
  cols = ['label_enum'] + cols
  df_out.to_csv(opt_output, index=False, columns=cols)

  # totals
  n_bg = df_out[df_out.label_index == -1].label_count.sum()
  n_total = df_out.label_count.sum()
  n_annos = n_total - n_bg
  log.info(f'total annotations: {n_total:,}')
  log.info(f'object annotations: {n_annos:,}')
  log.info(f'background annotations: {n_bg:,}')

  for idx, row in df_out.iterrows():
    log.info(f'{row.label_enum}: {row.label_count:,}')

  # TODO: create plots for each class object size

