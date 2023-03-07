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
@click.option('-i', '--input', 'opt_input', required=True)
@click.option('-o', '--output', 'opt_output', required=True)
@click.option('--min-seconds', 'opt_min_seconds', type=int)
@click.option('--max-seconds', 'opt_max_seconds', type=int)
@click.option('--min-width', 'opt_min_width', type=int)
@click.option('--max-width', 'opt_max_width', type=int)
@click.option('--min-height', 'opt_min_height', type=int)
@click.option('--max-height', 'opt_max_height', type=int)
@click.option('--min-fps', 'opt_min_fps', type=int)
@click.option('--max-fps', 'opt_max_fps', type=int)
@click.option('--output-txt', 'opt_output_txt')
@click.option('--prefix', 'opt_prefix', type=str)
@click.option('--input-lookup', 'opt_fp_lkup', 
  help='Filepath lookup reference to extract filepath with filename')
@click.pass_context
def cli(sink, opt_input, opt_output, opt_min_seconds, opt_max_seconds,
  opt_min_width, opt_max_width, opt_min_height, opt_max_height,
  opt_min_fps, opt_max_fps, opt_output_txt, opt_prefix, opt_fp_lkup):
  """Filter and clean media attributes to new filelist"""

  # ------------------------------------------------
  # imports

  import os
  from pathlib import Path
  from os.path import join

  import numpy as np
  import pandas as pd
  
  from vframe.settings.app_cfg import LOG, MEDIA_ATTRS_DTYPES
  from vframe.utils.file_utils import ensure_dir, write_txt, load_txt

  # error check
  if opt_prefix and opt_fp_lkup:
    LOG.warn('Using reference file and ignoring prefix but both prefix and lookup reference file were set. ')

  # create output
  ensure_dir(opt_output)

  # read csv
  df = pd.read_csv(opt_input, dtype=MEDIA_ATTRS_DTYPES)

  # drop duplicates
  df = df.drop_duplicates(['filename'])

  # clean
  df = df[df.frame_count > 0]
  df = df[df.frame_rate > 0]

  # filter width
  if opt_min_width:
    df = df[df.width >= opt_min_width]
  if opt_max_width:
    df = df[df.width <= opt_max_width]

  # filter height
  if opt_min_height:
    df = df[df.height >= opt_min_height]
  if opt_max_height:
    df = df[df.height <= opt_max_height]

  # filter fps
  if opt_min_fps:
    df = df[df.frame_rate >= opt_min_fps]
  if opt_max_fps:
    df = df[df.frame_rate <= opt_max_fps]  

  # generate seconds column
  if any([opt_min_seconds, opt_max_seconds]):
    df['seconds'] = df.frame_count / df.frame_rate

  # filter duration seconds  
  if opt_min_seconds:
    df = df[df.seconds >+ opt_min_seconds]
  if opt_max_seconds:
    df = df[df.seconds <= opt_max_seconds]


  # write new data
  df.to_csv(opt_output, index=False)

  #
  # write new filelist
  if opt_output_txt:
    filenames = df.filename.values.tolist()
    if opt_fp_lkup:
      filepaths = load_txt(opt_fp_lkup)
      filepaths_lkup = {Path(x).name:x for x in filepaths}
      filenames_matched = []
      for fn in filenames:
        if fn in filepaths_lkup.keys():
          filenames_matched.append(filepaths_lkup.get(fn))
      filenames = filenames_matched
    elif opt_prefix:
      filenames = [join(opt_prefix, x) for x in filenames]
    write_txt(opt_output_txt, filenames)

  # status
  opt_verbose = True
  if opt_verbose:
    LOG.info(f'Wrote: {len(df):,} lines')
