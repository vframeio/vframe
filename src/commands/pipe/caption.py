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
from vframe.settings.app_cfg import caption_accessors as accessors


@click.command('')
@click.option('-t', '--text', 'opt_text', required=True,
  help=f'Caption text. Attributes: {", ".join(accessors.keys())}')
@click.option('-x', '--x', 'opt_x', required=True, default=0,
  help='X position in pixels. Use negative for distance from bottom.')
@click.option('-y', '--y', 'opt_y', required=True, default=0,
  help='Y position in pixels. Use negative for distance from bottom.')
@click.option('-c', '--color', 'opt_color', 
  default=(0,255,0),
  help='font color in RGB int (eg 0 255 0)')
@click.option('--font-size', 'opt_font_size', default=16,
  help='Font size for labels')
@click.option('--bg/--no-bg', 'opt_bg', is_flag=True, default=True,
  help='Add text background')
@click.option('--bg-color', 'opt_color_bg', default=(0,0,0))
@click.option('--bg-padding', 'opt_padding_text', default=None, type=int)
@processor
@click.pass_context
def cli(ctx, sink, opt_text, opt_x, opt_y, opt_color, opt_font_size,
  opt_bg, opt_color_bg, opt_padding_text):
  """Add text caption"""

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME
  from vframe.models.types import FrameImage
  from vframe.models.color import Color
  from vframe.models.geometry import Point
  from vframe.utils import im_utils, draw_utils
  

  """
  - Replace $text with attribute if exists, else warn
  - $filename: M.filename
  - $frame_index: M.frame_index
  """

  # ---------------------------------------------------------------------------
  # initialize
  
  ctx.obj[USE_DRAW_FRAME] = True

  color_text = Color.from_rgb_int(opt_color)
  color_bg = Color.from_rgb_int(opt_color_bg)


  # ---------------------------------------------------------------------------
  # process

  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    im = M.images.get(FrameImage.DRAW)
    text = opt_text

    for k,v in accessors.items():
      if k in text:
        try:
          text = text.replace(k, str(getattr(M, v)))
          LOG.debug(text)
        except Exception as e:
          LOG.error(f'{k}:{v} is not a valid text accessor. Error: {e}')


    # draw text
    h,w,c = im.shape
    xy = [opt_x, opt_y]

    if xy[0] < 0:
      xy[0] = w + xy[0]
    
    if xy[1] < 0:
      xy[1] = h + xy[1]

    dim = im.shape[:2][::-1]
    pt = Point(*xy, *dim)
    
    # draw text
    im = draw_utils.draw_text(im, text, pt, font_color=color_text, font_size=opt_font_size, 
      bg=opt_bg, padding_text=opt_padding_text, color_bg=color_bg, upper=False)
    
    M.images[FrameImage.DRAW] = im
    sink.send(M)

