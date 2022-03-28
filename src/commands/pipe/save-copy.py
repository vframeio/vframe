############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils.click_utils import processor
from vframe.utils.click_utils import show_help

@click.command('')
@click.option('-o', '--output', 'opt_dir_out', required=True,
  help='Path to output directory')
@click.option('--keep-subdirs', 'opt_keep_subdirs', is_flag=True,
  help='Keep subdirectory structure in output directory')
@click.option('-t', '--type', 'opt_type', default='copy',
  type=click.Choice(['move', 'copy', 'symlink']))
@processor
@click.pass_context
def cli(ctx, sink, opt_dir_out, opt_keep_subdirs, opt_type):
  """Move, copy, or symlink media files"""

  from os.path import join
  from pathlib import Path
  import shutil

  from vframe.models.types import MediaType
  from vframe.settings.app_cfg import LOG, SKIP_FRAME
  from vframe.utils.file_utils import ensure_dir

  while True:
    
    M = yield # media

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue
    
    if (M.type == MediaType.VIDEO and M.is_last_item) or M.type == MediaType.IMAGE:
    
      # add relative subdir to output destination
      if opt_keep_subdirs and Path(M.filepath).parent != Path(reader.filepath):
        fp_subdir_rel = Path(M.filepath).relative_to(Path(reader.filepath)).parent
      else:
        fp_subdir_rel = ''

      fp_dir_out = join(opt_dir_out, fp_subdir_rel)
      fp_out = join(fp_dir_out, M.filename)
      ensure_dir(fp_out)
      
      if opt_type == 'symlink':
        if Path(fp_out).is_symlink():
          Path(fp_out).unlink()
        Path(fp_out).symlink_to(M.filepath)
      elif opt_type == 'copy':
        shutil.copy(M.filepath, fp_out)
      elif opt_type == 'move':
        shutil.move(M.filepath, fp_out)

    # continue
    sink.send(M)