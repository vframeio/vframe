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
from vframe.settings.app_cfg import DEFAULT_DETECT_MODEL
from vframe.models.types import ModelZooClickVar, ModelZoo
from vframe.utils.click_utils import show_help

# TODO: enumerate
color_styles = ['preset', 'custom']

@click.command('')
@click.option( '-n', '--name', 'opt_data_keys', 
  multiple=True,
  help='Name of data key for ROIs')
@click.option('--bbox/--no-bbox', 'opt_bbox', is_flag=True, default=True,
  help='Draw bbox')
@click.option('--no-labels', 'opt_no_labels', is_flag=True,
  help='Disable labels')
@click.option('--label-class/--no-label-class', 'opt_label', is_flag=True, default=True,
  help='Draws label')
@click.option('--label-data/--no-label-data', 'opt_key', is_flag=True, default=False,
  help='Draws data key')
@click.option('--label-conf/--no-label-conf', 'opt_conf', is_flag=True, default=True,
  help='Draws confidence score text')
@click.option('--mask/--no-mask', 'opt_mask', is_flag=True, default=False,
  help='Draws mask (if available)')
@click.option('--rbbox/--no-rbbox', 'opt_rbbox', is_flag=True, default=False,
  help='Draw rotated bbox (for scene text detectors)')
@click.option('--stroke', 'opt_stroke', default=4,
  help='Size of border. Use -1 for fill.')
@click.option('--expand', 'opt_expand', default=None, 
  type=click.FloatRange(0.0, 1.0, clamp=True),
  help='Percentage to expand bbox')
@click.option('-c', '--color', 'opt_color', 
  type=(int, int, int), default=(0, 255, 0),
  help='Color in RGB int (eg 0 255 0)')
@click.option('--color-source', 'opt_color_source', default='preset', 
  type=click.Choice(color_styles),
  help="Assign color to bbox and label background")
@click.option('--font-size', 'opt_size_font', default=12,
  help='Text size')
@click.option('--label-color', 'opt_color_label', 
  type=(int, int, int), default=(None, None, None),
  help='Color in RGB int (eg 0 255 0)')
@click.option('--label-padding', 'opt_padding_label', 
  type=int, default=0,
  help='Label padding')
@click.option('--label-index', 'opt_label_index', 
  is_flag=True,
  help='Label padding')
@click.option('-m', '--model-colors', 'opt_model_enum', 
  default=None,
  type=ModelZooClickVar,
  help=f'Use class colors from: {show_help(ModelZoo)}')
@click.option('--exclude-label', 'opt_exclude_labels', multiple=True, default=[''])
@processor
@click.pass_context
def cli(ctx, sink, opt_data_keys, opt_bbox, opt_no_labels, opt_label, opt_key, opt_conf, 
  opt_mask, opt_rbbox, opt_stroke, opt_size_font, opt_expand, 
  opt_color_source, opt_color_label, opt_color, opt_padding_label,
  opt_label_index, opt_exclude_labels, opt_model_enum):
  """Draw bboxes and labels"""
  
  from os.path import join

  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME
  from vframe.settings.app_cfg import USE_DRAW_FRAME, OBJECT_COLORS
  from vframe.settings.app_cfg import modelzoo
  from vframe.models.types import FrameImage
  from vframe.models.color import Color
  from vframe.utils import draw_utils


  ctx.obj[USE_DRAW_FRAME] = True

  # load model colors if using detection priors
  if opt_model_enum:
    model_name = opt_model_enum.name.lower()
    dnn_cfg = modelzoo.get(model_name)
    ctx.obj.setdefault(OBJECT_COLORS, {})
    ctx.obj[OBJECT_COLORS][model_name] = dnn_cfg.colorlist


  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    im = M.images.get(FrameImage.DRAW)
    dim = im.shape[:2][::-1]
    
    all_keys = list(M.metadata[M.index].keys())
    if not opt_data_keys:
      data_keys = all_keys
    else:
      data_keys = [k for k in opt_data_keys if k in all_keys]

    for data_key in data_keys:

      color_presets = ctx.obj.get(OBJECT_COLORS, {}).get(data_key, None)

      item_data = M.metadata[M.index].get(data_key)

      if item_data:
        # draw bbox, labels, mask
        for obj_idx, detection in enumerate(item_data.detections):
          
          if opt_exclude_labels and detection.label in opt_exclude_labels:
            continue
            
          bbox = detection.bbox.redim(dim)

          if opt_bbox:

            if opt_color_source == 'preset' and color_presets:
              try:
                color = color_presets[detection.index]
              except Exception as e:
                LOG.warn(f'No color for index: {detection.index} in {color_presets}')
                color = Color.from_rgb_int(opt_color)  # default
            else:
              color = Color.from_rgb_int(opt_color)
            
            # prepare label
            if not opt_no_labels:
              labels = []
              if opt_key:
                labels.append(data_key)
              if opt_label:
                labels.append(detection.label)
              if opt_conf:
                labels.append(f'{detection.conf * 100:.1f}%')
              if opt_label_index:
                labels.append(f'Index: {detection.index}')
              label = ': '.join(labels) if labels else None
            else:
              label = ''

            # draw bbox and optional labeling
            try:
              im = draw_utils.draw_bbox(im, bbox, color=color,
                stroke=opt_stroke, expand=opt_expand,
                label=label, font_size=opt_size_font, padding=opt_padding_label)
            except Exception as e:
              LOG.error(e)
              LOG.debug(bbox)
              LOG.debug(M.filename)

    # update pipe with modified image
    M.images[FrameImage.DRAW] = im

    sink.send(M)