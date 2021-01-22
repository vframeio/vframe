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
  help='Path to YAML config')
@click.option('-o', '--output', 'opt_fp_out', required=True,
  help='Path to output YAML')
@click.option('--font', 'opt_fp_font', required=True,
  help='Path to .ttf font file')
@click.option('--font-size', 'opt_font_size', default=64,
  help='Font size')
@click.option('--font-color', 'opt_font_color', default=(255,255,255),
  help='Font color')
@click.option('--bg-color', 'opt_bg_color', default=(0,0,0),
  help='Background color')
@click.pass_context
def cli(ctx, opt_fp_in, opt_fp_out, opt_fp_font, opt_font_size, opt_font_color, opt_bg_color):
  """Create font score similarities"""

  # ------------------------------------------------
  # imports

  from collections import OrderedDict

  import imagehash

  from vframe.settings import app_cfg
  from vframe.utils.file_utils import load_yaml, write_yaml
  from vframe.utils import draw_utils
  from vframe.models.color import Color
  from vframe.models.font_glyphs import FontConfig

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  # load config yaml with layer-unicode mappings
  font_cfg = load_yaml(opt_fp_in, data_class=FontConfig)
  char_targ = font_cfg.metadata.target_glyph
  opt_bg_color = Color.from_rgb_int(opt_bg_color)
  opt_font_color = Color.from_rgb_int(opt_font_color)

  # load font
  font_name = font_cfg.metadata.font_name
  font_mngr = draw_utils.font_mngr
  font_mngr.add_font(font_name, opt_fp_font, font_size=opt_font_size)

  # draw all chars into images
  ims = {}
  for g in font_cfg.glyphs:
    if not g.active:
      continue
    im = draw_utils.mk_font_chip(g.unicode, font_name, 
      font_size=opt_font_size, font_color=opt_font_color, bg_color=opt_bg_color)
    ims[g.unicode] = im
  
  # compare all images to current image
  scores = {}
  max_score = 36  # imagehash default
  hash_targ = imagehash.phash(ims[char_targ])
  for g in font_cfg.glyphs:
    if not g.active:
      continue
    hash_query = imagehash.phash(ims[g.unicode])
    score = 1.0 - ((hash_targ - hash_query) / max_score)
    scores[g.layer] = score

  # write updated yaml with scores
  font_cfg.add_glyph_score(scores)
  data = OrderedDict(font_cfg.to_dict())
  write_yaml(data, opt_fp_out, comment=app_cfg.LICENSE_HEADER)