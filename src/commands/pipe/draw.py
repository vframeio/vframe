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
@click.option( '-d', '--data', 'opt_data_keys', 
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
@click.option('--mask-alpha', 'opt_mask_alpha', default=0.6,
  help='Mask color weight')
@click.option('-c', '--color', 'opt_color', 
  type=(int, int, int), default=(0, 0, 255),
  help='Color in RGB int (eg 0 255 0)')
@click.option('--color-source', 'opt_color_source', default='random', 
  type=click.Choice(color_styles),
  help="Assign color to bbox and label background")
@click.option('--label-size', 'opt_size_label', default=12,
  help='Text size')
@click.option('--label-color', 'opt_color_label', 
  type=(int, int, int), default=(None, None, None),
  help='Color in RGB int (eg 0 255 0)')
@click.option('--label-padding', 'opt_padding_label', 
  type=int, default=None,
  help='Label padding')
@click.option('--label-index', 'opt_label_index', 
  is_flag=True,
  help='Label padding')
@click.option('-t', '--threshold', 'opt_threshold', default=0.0,
  help='Minimum detection confidence to draw')
@processor
@click.pass_context
def cli(ctx, sink, opt_data_keys, opt_bbox, opt_no_labels, opt_label, opt_key, opt_conf, 
  opt_mask, opt_rbbox, opt_stroke, opt_size_label, opt_expand, 
  opt_mask_alpha, opt_color_source, opt_color_label, opt_color, opt_padding_label,
  opt_label_index, opt_threshold):
  """Draw bboxes and labels"""
  
  from os.path import join

  from vframe.settings.app_cfg import LOG, SKIP_FRAME_KEY, USE_DRAW_FRAME_KEY
  from vframe.settings.app_cfg import USE_DRAW_FRAME_KEY
  from vframe.models.types import FrameImage
  from vframe.models.color import Color
  from vframe.utils import draw_utils


  # ---------------------------------------------------------------------------
  # initialize

  ctx.obj[USE_DRAW_FRAME_KEY] = True

  if all(v is not None for v in opt_color):
    opt_color_source = 'fixed'


  # ---------------------------------------------------------------------------
  # process

  while True:

    M = yield

    # skip frame if flagged
    if ctx.opts[SKIP_FRAME_KEY]:
      sink.send(M)
      continue

    im = M.images.get(FrameImage.DRAW)
    dim = im.shape[:2][::-1]
    
    all_keys = list(M.metadata.get(M.index).keys())
    if not opt_data_keys:
      data_keys = all_keys
    else:
      data_keys = [k for k in opt_data_keys if k in all_keys]

    for data_key in data_keys:

      item_data = M.metadata.get(M.index).get(data_key)

      if item_data:
        # draw bbox, labels, mask
        for obj_idx, detection in enumerate(item_data.detections):
          bbox = detection.bbox.redim(dim)

          # FIXME
          if opt_color_source == 'random':
            color = Color.random()
          elif opt_color_source == 'fixed':
            color = Color.from_rgb_int(opt_color)
          elif opt_color_source == 'preset':
            # TODO: load JSON colors from .yaml
            # TODO: add tracking ID based colors
            LOG.warn('Not yet implemented')
            color = Color.from_rgb_int((255,0,0))
          
          # TODO: implement mask-segmentation drawing
          # # draw mask
          # if opt_mask and item_data.task_type == types.Processor.SEGMENTATION:
          #   mask = detection.mask
          #   im = draw_utils.draw_mask(im, bbox, mask, 
          #     color=color, color_weight=opt_mask_alpha)

          # TODO: implement rotated BBox drawing
          # # draw rotated bbox
          # if opt_rbbox and item_data.task_type == types.Processor.DETECTION_ROTATED:
          #   im = draw_utils.draw_rotated_bbox_pil(im, detection.rbbox, 
          #     stroke=opt_stroke, color=color)

          if opt_bbox:
            
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
            im = draw_utils.draw_bbox(im, bbox, color=color,
              stroke=opt_stroke, expand=opt_expand,
              label=label, size_label=opt_size_label, padding_label=opt_padding_label,
              )

    # update pipe with modified image
    M.images[FrameImage.DRAW] = im

    sink.send(M)