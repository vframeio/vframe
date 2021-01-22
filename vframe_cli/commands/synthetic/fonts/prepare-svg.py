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
  """Preprocess SVG XML into SVG with glyphs"""

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
  font_cfg = load_yaml(opt_fp_cfg, data_class=FontConfig)

  # parse SVG and add path data
  font_cfg.parse_svg(opt_fp_in)
  
  # check if errors
  log.debug(font_cfg.status_report())

  # write to new SVG file
  font_cfg.to_svg(opt_fp_out)