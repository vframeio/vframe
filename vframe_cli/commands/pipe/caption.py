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

@click.command('')
@click.option('-c', '--color', 'opt_color', 
  default=(255,255,255),
  help='font color in RGB int (eg 0 255 0)')
@click.option('-t', '--text', 'opt_caption', required=True,
  help='Caption')
@click.option('--xy', 'opt_xy', required=True, default=(20, -20),
  help='XY position in pixels. Use negative for distance from bottom.')
@click.option('-a', '--alpha', 'opt_alpha', default=1.0,
  help='Opacity of font')
@click.option('--font-size', 'opt_font_size', default=14,
  help='Font size for labels')
@click.option('--knockout', 'opt_knockout', default=(1,1),
  help='Knockout pixel distance')
@click.option('--knockout-color', 'opt_color', 
  type=(int, int, int), default=(0, 0, 0),
  help='font color in RGB int (eg 0 255 0)')
@processor
@click.pass_context
def cli(ctx, pipe, opt_caption, opt_xy, opt_color, opt_font_size, opt_alpha, opt_knockout):
  """Add text caption"""
  
  import cv2 as cv

  from vframe.settings import app_cfg
  from vframe.models import types
  from vframe.models.bbox import PointNorm
  from vframe.utils import im_utils
  from vframe.utils.draw_utils import DrawUtils
  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG

  draw_utils = DrawUtils()

  # ---------------------------------------------------------------------------
  # Example: process images as they move through pipe

  while True:

    pipe_item = yield
    header = ctx.obj['header']
    im = pipe_item.get_image(types.FrameImage.DRAW)

    # draw text
    h,w,c = im.shape
    xy = list(opt_xy)

    if xy[0] < 0:
      xy[0] = w + xy[0]
    
    if xy[1] < 0:
      xy[1] = h + xy[1]

    xy_norm = PointNorm(xy[0] / w, xy[1] / h)
    im = draw_utils.draw_text_pil(im, xy_norm, opt_caption, 
      font_size=opt_font_size, color=opt_color, knockout=opt_knockout)
    
    pipe_item.set_image(types.FrameImage.DRAW, im)
    pipe.send(pipe_item)

