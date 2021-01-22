############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from os.path import join
from pathlib import Path
import click

from vframe.models import types
from vframe.settings import app_cfg
from vframe.utils import click_utils


@click.command()
@click.option('--blend', 'opt_fp_blend', help='Path to blender')
@click.option('-n', '--name', 'opt_names', multiple=True, help="Name of objects")
@click.option('-o', '--output', 'opt_fp_out', help="Path to output directory")
@click.option('-a', '--all', 'opt_fp_all', is_flag=True, help="Append all objects")
@click.option('--frames', 'opt_num_frames', type=int, default=100, help="Number of frames")
@click.pass_context
def cli(ctx, opt_fp_blend, opt_names, opt_fp_out, opt_fp_all, opt_num_frames):
  """Turntable with appended objects"""
  
  import os
  import subprocess

  log = app_cfg.LOG
  log.info('Turntable')