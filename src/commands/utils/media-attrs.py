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
@click.option('-r', '--recursive', 'opt_recursive', is_flag=True)
@click.option('-e', '--ext', 'opt_exts', default=['mp4', 'jpg'], multiple=True, 
  help='Glob extension')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('-f', '--force', 'opt_force', is_flag=True, 
  help='Overwrite current file')
@click.option('-t', '--threads', 'opt_threads', default=None, type=int)
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.pass_context
def cli(sink, opt_input, opt_output, opt_recursive, opt_exts, opt_slice, 
  opt_verbose, opt_force, opt_threads):
  """Create mediainfo metadata header CSV"""

  import os
  from os.path import join
  from glob import glob
  from pathlib import Path
  from dataclasses import asdict

  import pandas as pd
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count
  
  from vframe.settings.app_cfg import LOG
  from vframe.utils.file_utils import glob_multi, load_txt
  from vframe.utils.video_utils import mediainfo


  if Path(opt_output).is_file() and not opt_force:
    LOG.error('File exists. Use "-f/--force" to overwrite')
    return

  if not opt_threads:
    opt_threads = cpu_count()


  if Path(opt_input).is_file() and Path(opt_input).suffix.lower() == '.txt':
    fp_items = load_txt(opt_input)
  else:
    fp_items = file_utils.glob_multi(opt_input, opt_exts, recursive=opt_recursive)
  
  if any(opt_slice):
    fp_items = fp_items[opt_slice[0]:opt_slice[1]]

  # -----------------------------------------------------------
  # start pool worker

  def pool_worker(fp_item):
    return mediainfo(fp_item, verbose=opt_verbose)

  # end pool worker
  # -----------------------------------------------------------

  # Multiprocess/threading use imap instead of map via @hkyi Stack Overflow 41920124
  with Pool(opt_threads) as p:
    pool_results = list(tqdm(p.imap(pool_worker, fp_items), total=len(fp_items), desc=f'x{opt_threads} threads'))

  # Separate and recast to dict
  records = []
  errors = []

  for pool_result in pool_results:
    if not pool_result.valid:
      errors.append(asdict(pool_result))
    else:
      records.append(asdict(pool_result))

  # save records
  pd.DataFrame.from_dict(records).to_csv(opt_output, index=False)

  # save errors
  if len(errors):
    fp_out_bad = opt_output.replace('.csv', '_errors.csv')
    pd.DataFrame.from_dict(errors).to_csv(fp_out_bad, index=False)

  # status
  if opt_verbose:
    LOG.info(f'Processed {len(fp_items):,}')
    LOG.info(f'Valid: {len(records):,}, Errors: {len(errors):,}')
    