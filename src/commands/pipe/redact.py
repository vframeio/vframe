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

redact_types = ['pixellate', 'blur', 'softblur']

@click.command('')
@click.option( '-n', '--name', 'opt_data_keys', multiple=True,
  help='Data key for ROIs')
@click.option('--fac', 'opt_factor', default=0.5, show_default=True,
  help='Strength to apply redaction filter')
@click.option('--iters', 'opt_iters', default=2, show_default=True,
  help='Blur iterations')
@click.option('--expand', 'opt_expand', default=0.05, show_default=True,
  help='Percentage to expand')
@click.option('-t', '--type', 'opt_redact_type', type=click.Choice(redact_types),
  show_default=True, default='blur', help='Redact type')
@processor
@click.pass_context
def cli(ctx, sink, opt_data_keys, opt_factor, opt_iters, opt_expand, opt_redact_type):
  """Blur-redact BBoxes"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME, USE_DRAW_FRAME
  from vframe.models.types import FrameImage
  from vframe.utils import im_utils


  ctx.obj[USE_DRAW_FRAME] = True


  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    im = M.images.get(FrameImage.DRAW)
    dim = im.shape[:2][::-1]
    
    # get data keys
    all_keys = list(M.metadata.get(M.index, {}).keys())
    if not opt_data_keys:
      data_keys = all_keys
    else:
      data_keys = [k for k in opt_data_keys if k in all_keys]

    # iterate data keys
    bboxes = []

    for data_key in data_keys:
      
      # get data
      item_data = M.metadata.get(M.index, {}).get(data_key)

      # blur data
      if item_data:
        for obj_idx, detection in enumerate(item_data.detections):
          bbox = detection.bbox.expand_per(opt_expand).redim(dim)
          bboxes.append(bbox)


    # redact method
    if opt_redact_type == 'pixellate':
      im = im_utils.pixellate_bboxes(im, bboxes)
    elif opt_redact_type == 'blur':
      im = im_utils.blur_bboxes(im, bboxes, iters=opt_iters)
    elif opt_redact_type == 'softblur':
      im = im_utils.blur_bbox_soft(im, bboxes, iters=2, expand_per=-0.15, 
        mask_k_fac=0.25, im_k_fac=0.995, multiscale=True)
    

    # update and resume
    M.images[FrameImage.DRAW] = im

    sink.send(M)