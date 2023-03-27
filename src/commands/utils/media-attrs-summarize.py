############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click
from datetime import date

@click.command('')
@click.option('-i', '--input', 'opt_input', required=True)
@click.option('-o', '--output', 'opt_output')
@click.option('--minute-range', 'opt_min_range', type=(int,int), default=(1,5))
@click.option('--minute-intervals', 'opt_min_interval', default=1)
@click.pass_context
def cli(sink, opt_input, opt_output, opt_min_range, opt_min_interval):
  """Summarize media attributes"""

  import pandas as pd
  
  from vframe.settings.app_cfg import LOG, MEDIA_ATTRS_DTYPES
  from vframe.utils.file_utils import ensure_dir
  

  # read csv
  df = pd.read_csv(opt_input, dtype=MEDIA_ATTRS_DTYPES)

  # drop duplicates
  df = df.drop_duplicates(['filename'])

  # clean
  df = df[df.frame_count > 0]
  df = df[df.frame_rate > 0]

  # extend
  df['seconds'] = df.frame_count / df.frame_rate  # add seconds col

  # ---------------------------------------------------------------------------
  # Summarize attribute stats
  # ---------------------------------------------------------------------------

  d = {}
  d['n_days'] = df.seconds.sum() / 60 / 60 / 24
  d['n_hours'] = df.seconds.sum() / 60 / 60
  d['n_minutes'] = df.seconds.sum() / 60
  d['n_seconds'] = df.seconds.sum() 
  d['n_frames'] = df.frame_count.sum()
  d['n_videos'] = len(df)
  for i in range(opt_min_range[0], opt_min_range[1] + 1, opt_min_interval):
    d[f'n_videos_lt_{i}min'] = len(df[df.seconds <= i*60])
  d['n_days_human_8hrs'] = (df.seconds.sum() / 60 / 60 / 8)
  d['n_days_human_24hrs'] = (df.seconds.sum() / 60 / 60 / 24)
  d['n_days_machine_30fps'] = (df.frame_count.sum() / 30 / 60 / 60 / 24)
  d['n_days_machine_60fps'] = (df.frame_count.sum() / 60 / 60 / 60 / 24)
  d['n_days_machine_120fps'] = (df.frame_count.sum() / 120 / 60 / 60 / 24)
  d['n_days_machine_240fps'] = (df.frame_count.sum() / 240 / 60 / 60 / 24)

  # print
  for k,v in d.items():
    if isinstance(v, float):
      print(f'{k}: {v:,.3f}')
    else:
      print(f'{k}: {v:,}')

  # write csv
  if opt_output:
    arr = {'name': [], 'value':[]}
    for k,v in d.items():
      arr['name'].append(k)
      arr['value'].append(v)
    pd.DataFrame.from_dict(arr).to_csv(opt_output, index=False)
