############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.models import types
from vframe.utils import click_utils

@click.command('')
@click.option('-m', '--model', 'opt_model_enum',
  type=types.ModelZooClickVar,
  help=click_utils.show_help(types.ModelZoo))
@click.option('--all', 'opt_dl_all', is_flag=True,
  help='Download all models')
@click.option('-f', '--force', 'opt_force', is_flag=True)
@click.pass_context
def cli(sink, opt_model_enum, opt_dl_all, opt_force):
  """Download DNN models"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path

  from tqdm import tqdm

  from vframe.settings.app_cfg import LOG, modelzoo
  from vframe.utils.file_utils import ensure_dir
  from vframe.utils.model_utils import download_model

  # ------------------------------------------------
  # start


  if opt_model_enum:
    model_list = [opt_model_enum.name.lower()]
  elif opt_dl_all:
    model_list = list(modelzoo.keys())
  else:
    LOG.error('No model(s) selected. Choose "-m/--model" or download "--all"')
    return


  for model_name in tqdm(model_list, desc='Model', leave=False):

    dnn_cfg = modelzoo.get(model_name)
    download_model(dnn_cfg, opt_force=opt_force)