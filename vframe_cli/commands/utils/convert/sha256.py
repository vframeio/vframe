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
@click.option('--confirm', 'opt_confirm', is_flag=True,
  default=False,
  help='Dry run or confirm rename')
@click.pass_context
def cli(ctx, opt_input, opt_slice, opt_exts, opt_confirm):
  """Rename files"""

  """
  TODO
  - add suffix, prefix, numerical
  - multiprocessor
  """

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  from urllib.parse import unquote
  from tqdm import tqdm
  import shutil

  from vframe.utils import file_utils
  from vframe.utils.url_utils import download_url
  from vframe.settings import app_cfg

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

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

  # rename
  for fp_item in tqdm(fp_items, desc='Files'):
    sha256 = file_utils.sha256(fp_item)
    ext = Path(fp_item).suffix
    fp_sha256 = join(opt_input, f'{sha256}{ext}')
    shutil.move(fp_item, fp_sha256)
