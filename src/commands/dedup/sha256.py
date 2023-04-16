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

FILE_ACTIONS = ['copy', 'move', 'symlink', 'filelist']
OPTS_KEEP = ['first', 'last']

@click.command('')
@click.option('-i', '--input', 'opt_input', required=True,
    help="Path to input(s)")
@click.option('-o', '--output', 'opt_output',
  help='Path to output directory for CSV and TXT files')
@click.option('--rebuild', 'opt_rebuild', is_flag=True,
  help='Rebuild hash cache if already exists')
@click.option('-e', '--ext', 'opt_exts', multiple=True,
  type=click.Choice(VALID_PIPE_EXTS),
  default=VALID_PIPE_EXTS)
@click.option('--keep', 'opt_keep', 
  type=click.Choice(OPTS_KEEP),
  default='first',
  help='Keep first or last item')
@click.option('-r', '--recursive', 'opt_recursive', is_flag=True,
  help='Recursive globbing')
@click.option('-t', '--threads', 'opt_threads', type=int, show_default=True,
  help='Number threads')
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_recursive, opt_keep, opt_rebuild, opt_exts, opt_threads):
  """Build SHA256 cache"""

  from pathlib import Path
  from os.path import join

  from pathos.multiprocessing import cpu_count
  from pathos.multiprocessing import ProcessingPool as Pool
  from tqdm import tqdm
  import pandas as pd

  from vframe.utils.file_utils import (date_created, glob_multi, mk_sha256)
  from vframe.utils.file_utils import (ensure_dir, load_txt, write_txt)
  from vframe.settings.app_cfg import LOG, FN_CACHE_SHA256, FN_DEDUP_SHA256


  if not opt_output:
    dn_out = opt_input if Path(opt_input).is_dir() else Path(opt_input).parent
  else:
    dn_out = opt_output
  fp_out = join(dn_out, FN_CACHE_SHA256)

  # inits
  n_cpus = cpu_count()
  opt_threads = opt_threads if opt_threads else n_cpus
  ensure_dir(fp_out)

  # mk list of all file inputs and update cache
  filepaths = []
  if Path(opt_input).suffix.lower() == '.txt':
    filepaths.extend(load_txt(opt_input))
  else:
    filepaths.extend(glob_multi(opt_input, exts=opt_exts, recursive=opt_recursive))
  LOG.info(f'{len(filepaths):,} files found')

  # init list of dict of filepaths

  # load cache of existing filepaths and hashes
  if Path(fp_out).is_file() and not opt_rebuild:
    df_cache = pd.read_csv(fp_out, dtype={'date_modified': str})
    LOG.info(f'{len(df_cache):,} items cached')
  else:
    df_cache = pd.DataFrame({"filepath": [], 'sha256': [], 'date_modified': []})
    LOG.info(f'Building new cache file at: {fp_out}')

    
  # remove files from cache that no longer exist in file list
  n_cached = len(df_cache)
  df_cache = df_cache[df_cache.filepath.isin(filepaths)]
  n_rm = n_cached - len(df_cache)

  # remove files from new list that already exist in cache
  filepaths = [x for x in filepaths if x not in df_cache.filepath.values.tolist()]
  n_new = len(filepaths)

  if n_new > 0:
    LOG.info(f'{n_new:,} new files')

  if n_rm > 0:
    LOG.info(f'{n_rm} file(s) removed from cache.')
    if not n_new > 0:
      df_cache.to_csv(opt_output, index=False)
      fp_out = join(Path(opt_output).parent, FN_DEDUP_SHA256)
      write_txt(fp_out, df_cache.filepath.values.tolist())
      LOG.info(f'{len(df_cache):,} deduplicated files written to text file: {fp_out}')
      LOG.info('No new files. Exiting.') 
      return
  elif not n_new > 0:
    LOG.info('No new files. Exiting.') 
    return
  
    
  # create list of files to be processed
  filemeta = [{'filepath': fp, 'sha256': '', 'date_modified': ''} for fp in filepaths]

  # multiprocessor
  def pool_worker(o):
    fp = o['filepath']
    o['sha256'] = mk_sha256(fp)
    o['date_modified'] = str(date_created(fp))
    return o

  # multiprocess hash
  with Pool(opt_threads) as p:
    d = f'Hashing x{opt_threads}/{n_cpus}'
    t = len(filemeta)
    results = list(tqdm(p.imap(pool_worker, filemeta), total=t, desc=d))

  # merge results into cache, sort, write to csv
  df_new = pd.DataFrame.from_dict(results)
  df_cache = pd.concat([df_cache, df_new])
  df_cache.sort_values(by=('date_modified'), ascending=False, inplace=True)
  df_cache.to_csv(fp_out, index=False)

  # dedupe
  df_cache.drop_duplicates(['sha256'], keep=opt_keep, inplace=True)
  fp_out = join(dn_out, FN_DEDUP_SHA256)
  write_txt(fp_out, df_cache.filepath.values.tolist())
  LOG.info(f'{len(df_cache):,} deduplicated files written to text file: {fp_out}')