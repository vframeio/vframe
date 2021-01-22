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
@click.option('--slice', 'opt_slice', type=(int, int), 
  default=(None, None),
  help="Slice list of inputs")
@click.option('-e', '--exts', 'opt_exts', default=['jpg', 'jpeg', 'png'],
  multiple=True,
  help='Extensions to glob for')
@click.option('-t', '--threads', 'opt_threads', default=None, type=int)
@click.option('--confirm', 'opt_confirm', is_flag=True,
  default=False,
  help='Dry run or confirm rename')
@click.pass_context
def cli(ctx, opt_input, opt_slice, opt_exts, opt_threads, opt_confirm):
  """Rename files to SHA256"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  import shutil

  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count

  from vframe.utils import file_utils
  from vframe.settings import app_cfg

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  # set N threads
  if not opt_threads:
    opt_threads = cpu_count()  # maximum


  # -----------------------------------------------------------
  # start pool worker

  def pool_worker(pool_item):
    fp = pool_item['fp']
    sha256 = file_utils.sha256(fp)
    ext = Path(fp).suffix
    fp_sha256 = join(opt_input, f'{sha256}{ext}')
    shutil.move(fp, fp_sha256)
    return {'result': True}

  # end pool worker
  # -----------------------------------------------------------

  # glob files
  fp_items = file_utils.glob_multi(opt_input, exts=opt_exts)
  
  # slice input
  if any(opt_slice):
    fp_items = fp_items[opt_slice[0]:opt_slice[1]]

  # status  
  log.debug(f'Renaming {len(fp_items)} files. Confirm: {opt_confirm}')

  # confirm
  if not opt_confirm:
    log.warn(f"This was a dry run. No files were renamed.")
    log.warn('Add the "--confirm" option to rename files. This can\'t be undone')
    return

  # convert file list into object with 
  pool_items = [{'fp': fp} for fp in fp_items]

  # init processing pool iterator
  # use imap instead of map via @hkyi Stack Overflow 41920124
  desc = f'Rename SHA256 x{opt_threads}'
  with Pool(opt_threads) as p:
    pool_results = list(tqdm(p.imap(pool_worker, pool_items), total=len(pool_items), desc=desc))