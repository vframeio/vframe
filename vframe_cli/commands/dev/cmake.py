############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click
from dataclasses import asdict

from vframe.settings import app_cfg
from vframe.models.cmake import CmakeOpenCV

@click.command()
@click.option('-i', '--input', 'opt_fp_in', default=app_cfg.FP_CMAKE_OPENCV)
@click.option('-o', '--output', 'opt_fp_out', required=True,
  help='Path to opencv build directory (eg ../3rdparty/opencv/build/')
@click.pass_context
def cli(ctx, opt_fp_in, opt_fp_out):
  """Create OpenCV build file"""

  # ------------------------------------------------
  # imports

  from os.path import join
  import GPUtil

  from vframe.utils import file_utils

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  config = file_utils.load_yaml(opt_fp_in, data_class=CmakeOpenCV)
  config_adj = {k:str(v).replace('True','ON').replace('False','OFF') for k,v in asdict(config).items()}
  
  header = ['#!/bin/bash', '', '']

  intro = ['# Usage: ']
  intro += ['# $ sh your-file.sh', '', '']

  cmds = ['cmake \\']
  cmds += [f'-D {k}={v} \\' for k,v in config_adj.items()]
  cmds += ['..', '', '']

  footer = ['echo "------------------------------------------------------------"']
  footer += ['echo "If this looks good, run the make install command:"']
  footer += ['echo "sudo make install -j $(nproc)"']
  footer += ['echo "-----------------------------------------------------------"']

  payload = header + intro + cmds + footer

  file_utils.write_txt(payload, opt_fp_out)
  file_utils.chmod_exec(opt_fp_out)


