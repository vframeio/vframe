############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.settings.app_cfg import VALID_PIPE_EXTS

FILE_ACTIONS = ['copy', 'move', 'symlink']
OPTS_KEEP = ['first', 'last']

@click.command('')
@click.option('-i', '--input', 'opt_inputs', required=True, multiple=True,
    help="Path to input(s)")
@click.option('-o', '--output', 'opt_output', required=True,
  help='Path to output directory with deduplicated files')
@click.option('--action', 'opt_action', 
  type=click.Choice(FILE_ACTIONS), default='copy',
  help='File action')
@click.option('--cache', 'opt_cache', required=True,
  help='Path to CSV cache file')
@click.option('--force-hash', 'opt_force', is_flag=True,
  help='Rehash if already exists')
@click.option('-e', '--ext', 'opt_exts', multiple=True,
  type=click.Choice(VALID_PIPE_EXTS),
  default=VALID_PIPE_EXTS)
@click.option('--keep', 'opt_keep', 
  type=click.Choice(OPTS_KEEP),
  default='first',
  help='Keep first or last item')
@click.option('-t', '--threads', 'opt_threads', type=int, show_default=True,
  help='Number threads')
@click.pass_context
def cli(ctx, opt_inputs, opt_output, opt_action, opt_cache, 
  opt_exts, opt_keep, opt_force, opt_threads):
  """Deduplicate files using SHA256 hash"""

  # TODO
  # add output for duplicates
  # add option to remove in source folders
  # add warning for move
  # add phash version

  import shutil
  from pathlib import Path
  from os.path import join

  from pathos.multiprocessing import cpu_count
  from pathos.multiprocessing import ProcessingPool as Pool
  from tqdm import tqdm
  import pandas as pd

  from vframe.utils.file_utils import (date_created, glob_multi, mk_sha256)
  from vframe.utils.file_utils import (ensure_dir, load_txt)
  from vframe.settings.app_cfg import LOG

  # inits
  n_cpus = cpu_count()
  opt_threads = opt_threads if opt_threads else n_cpus
  ensure_dir(opt_output)

  # mk list of all file inputs and update cache
  ls_filepaths = []
  for opt_input in opt_inputs:
    if Path(opt_input).suffix.lower() == '.txt':
      ls_filepaths.extend(load_txt(opt_input))
    else:
      ls_filepaths.extend(glob_multi(opt_input, exts=opt_exts, recursive=False))

  LOG.debug(f'{len(ls_filepaths)} files')
  # init list of dict of filepaths
  ls_files = [{'filepath': fp, 'sha256': '', 'date_modified': ''} for fp in ls_filepaths]

  # load cache of existing filepaths and hashes
  if Path(opt_cache).is_file():
    df_cache = pd.read_csv(opt_cache, dtype={'date_modified': str})
    df_files = pd.DataFrame.from_dict(ls_files)
    df_cache.merge(df_files).drop_duplicates(['filepath'], inplace=True)
    ls_cache = list(df_cache.to_dict('records'))
  else:
    ls_cache = ls_files


  # multiprocessor
  def pool_worker_sha(o):
    fp = o['filepath']
    if not o['sha256'] or opt_force:
      o['sha256'] = mk_sha256(fp)
    o['date_modified'] = date_created(fp)
    o['exists'] = Path(fp).is_file()
    return o

  # multiprocess hash
  with Pool(opt_threads) as p:
    d = f'Hashing x{opt_threads}'
    t = len(ls_cache)
    ls_cache_res = list(tqdm(p.imap(pool_worker_sha, ls_cache), total=t, desc=d))

  # create df, sort by date, drop empties, drop dupes, write to csv
  pd.DataFrame.from_dict(ls_cache_res)
  df_cache = pd.DataFrame.from_dict(ls_cache_res)
  df_cache = df_cache[df_cache['exists'] == True]
  df_cache.sort_values(by=('date_modified'), ascending=False, inplace=True)
  df_cache.drop_duplicates(['sha256'], keep=opt_keep, inplace=True)
  df_cache.to_csv(opt_cache, index=False)


  # multiprocessor
  def pool_worker_file(o):
    fp_src = o['filepath']
    fp_dst = join(opt_output, Path(fp_src).name)
    if opt_action == 'copy':
      shutil.copy(fp_src, fp_dst)
    elif opt_action == 'move':
      shutil.mv(fp_src, fp_dst)
    elif opt_action == 'symlink':
      if Path(fp_dst).is_symlink():
        Path(fp_dst).unlink()
      Path(fp_dst).symlink_to(fp_src)

  # process file action
  ls_cache = df_cache.to_dict('records')
  # limit threads to half max
  opt_threads = n_cpus//2 if opt_threads > n_cpus//2 else opt_threads
  # multiprocess
  with Pool(opt_threads) as p:
    d = f'{opt_action} x{opt_threads}'
    t = len(ls_cache)
    ls_cache_res = list(tqdm(p.imap(pool_worker_file, ls_cache), total=t, desc=d))

  # print status
  t = len(ls_files) - len(df_cache)
  LOG.info(f'Removed {t} duplicates')
  LOG.info(f'{len(df_cache)} de-duplicated files "{opt_action}" to {opt_output}')