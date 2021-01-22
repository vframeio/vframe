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
@click.option('-i', '--input', 'opt_fp_in', required=True,
  help='Path to input SVG')
@click.option('-c', '--cfg', 'opt_fp_cfg', required=True,
  help='Path to output SVG')
@click.option('-o', '--output', 'opt_fp_out', required=True,
  help='Path to output SVG')
@click.pass_context
def cli(ctx, opt_fp_in, opt_fp_cfg, opt_fp_out):
  """Generate synthetic images"""

  # ------------------------------------------------
  # imports

  from os.path import join

  from vframe.settings import app_cfg
  from vframe.utils.file_utils import load_yaml
  from vframe.models.font_glyphs import FontConfig

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  # load config yaml with layer-unicode mappings
  #font_cfg = load_yaml(opt_fp_cfg, data_class=FontConfig)

  # temp, get this from config file
  chars = [c for c in "abcdefghijklmopqrstuvABCDEFGHIJKLMOPQRSTUV0123456789!@#$%^&*()"]

  # load phash scores

  # glob images

  # set image params from cli opts

  # write metadata

  # generate images

  # TODO: multithread

  # for each iter

  # draw text

  # save

  # append annotation and write every image or at end




  