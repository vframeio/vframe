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

# TODO: enumerate
color_styles = ['random', 'preset', 'fixed']

@click.command('')
@click.option( '-n', '--name', 'opt_data_keys', 
  multiple=True,
  help='Name of data key for ROIs')
@click.option('--bbox/--no-bbox', 'opt_bbox', is_flag=True, default=True,
  help='Draw bbox')
@click.option('--label/--no-label', 'opt_label', is_flag=True, default=True,
  help='Draws label')
@click.option('--key/--no-key', 'opt_key', is_flag=True, default=False,
  help='Draws data key')
@click.option('--confidence/--no-confidence', 'opt_conf', is_flag=True, default=False,
  help='Draws confidence score text')
@click.option('--mask/--no-mask', 'opt_mask', is_flag=True, default=False,
  help='Draws mask (if available)')
@click.option('--rbbox/--no-rbbox', 'opt_rbbox', is_flag=True, default=False,
  help='Draw rotated bbox (for scene text detectors)')
@click.option('--stroke', 'opt_stroke_weight', default=4,
  help='Size of border. Use -1 for fill.')
@click.option('--expand', 'opt_expand', default=0.0,
  help='Percentage to expand bbox')
@click.option('--font-size', 'opt_font_size', default=14,
  help='Font size for labels')
@click.option('--mask-alpha', 'opt_mask_alpha', default=0.6,
  help='Mask color weight')
@click.option('--color-source', 'opt_color_source', default='random', 
  type=click.Choice(color_styles),
  help="Assign color to bbox and label background")
@click.option('--font-color', 'opt_font_color', 
  type=(int, int, int), default=(None, None, None),
  help='Color in RGB int (eg 0 255 0)')
@click.option('-c', '--color', 'opt_color', 
  type=(int, int, int), default=(None, None, None),
  help='Color in RGB int (eg 0 255 0)')
@click.option('--backend', 'opt_backend', 
  type=click.Choice(['pil', 'cv', 'np']),
  default='pil')
@processor
@click.pass_context
def cli(ctx, pipe, opt_data_keys, opt_bbox, opt_label, opt_key, opt_conf, 
  opt_mask, opt_rbbox, opt_stroke_weight, opt_font_size, opt_expand, 
  opt_mask_alpha, opt_color_source, opt_font_color, opt_color, opt_backend):
  """Draw bboxes, labels, and masks"""
  
  from os.path import join

  from vframe.settings import app_cfg
  from vframe.models import types
  from vframe.models.color import Color
  from vframe.utils.draw_utils import DrawUtils


  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  draw_utils = DrawUtils()
  if all(v is not None for v in opt_color):
    opt_color_source = 'fixed'

  # ---------------------------------------------------------------------------
  # process

  while True:

    pipe_item = yield
    header = ctx.obj['header']
    
    im = pipe_item.get_image(types.FrameImage.DRAW)

    if not opt_data_keys:
      data_keys = header.get_data_keys()
    else:
      data_keys = opt_data_keys

    for data_key in data_keys:
      
      if data_key not in header.get_data_keys():
        log.error(f'data_key: {data_key} not found')
        
      item_data = header.get_data(data_key)

      if item_data:
        # draw bbox, labels, mask
        for obj_idx, detection in enumerate(item_data.detections):
          bbox_norm = detection.bbox
          
          if opt_expand:
            bbox_norm = detection.bbox.expand_per(opt_expand)

          if opt_color_source == 'random':
            color = Color.random()
          elif opt_color_source == 'fixed':
            color = Color.from_rgb_int(opt_color)
          elif opt_color_source == 'preset':
            # TODO load JSON colors from .yaml
            log.warn('Not yet implemented')
            color = Color.from_rgb_int((255,0,0))
          
          # draw mask
          if opt_mask and item_data.task_type == types.Processor.SEGMENTATION:
            mask = detection.mask
            im = draw_utils.draw_mask(im, bbox_norm, mask, 
              color=color, color_weight=opt_mask_alpha)

          # draw rotated bbox
          if opt_rbbox and item_data.task_type == types.Processor.DETECTION_ROTATED:
            im = draw_utils.draw_rotated_bbox_pil(im, detection.rbbox, 
              stroke_weight=opt_stroke_weight, color=color)

          # draw bboxes
          # TODO: clean up norm labeled bbox
          if opt_bbox:
            fn = header.filename
            label_index = detection.index
            labels = []
            if opt_label:
              labels.append(detection.label)
            if opt_key:
              labels.append(data_key)
            if opt_conf:
              labels.append(f'{detection.confidence * 100:.1f}')

            label = ': '.join(labels)

            bbox_nlc = bbox_norm.to_labeled_colored(label, label_index, fn, color)
        
            if opt_label or opt_key or opt_conf:
              im = draw_utils.draw_bbox_labeled_pil(im, bbox_nlc, 
                stroke_weight=opt_stroke_weight, font_size=opt_font_size)
            else:
              if opt_backend == 'pil':
                im = draw_utils.draw_bbox_pil(im, bbox_norm, color, 
                  stroke_weight=opt_stroke_weight)
              elif opt_backend == 'cv':
                im = draw_utils.draw_bbox_cv(im, bbox_norm, color, 
                  stroke_weight=opt_stroke_weight)
              elif opt_backend == 'np':
                im = draw_utils.draw_bbox_np(im, bbox_norm, color)


    pipe_item.set_image(types.FrameImage.DRAW, im)
    pipe.send(pipe_item)