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
@click.option( '-n', '--name', 'opt_data_key', required=True,
  help='Data key for ROIs')
@click.option('--square', 'opt_square', is_flag=True,
  help='Square bbox dimensions')
@click.option('-e', '--expand', 'opt_expand', type=float,
  help='Expand percent')
@processor
@click.pass_context
def cli(ctx, sink, opt_data_key, opt_square, opt_expand):
  """Modify BBoxes"""
  
  from vframe.settings.app_cfg import LOG, SKIP_FRAME
  from vframe.models import types


  while True:

    M = yield

    # skip frame if flagged
    if ctx.obj[SKIP_FRAME]:
      sink.send(M)
      continue

    frame_data = M.metadata.get(opt_data_key)
    im = M.images.get(types.FrameImage.DRAW)
    dim = im.shape[:2][::-1]

    if frame_data:
      for obj_idx, detection in enumerate(frame_data.detections):
        bbox_norm = detection.bbox
        if opt_square:
          bbox_norm = bbox_norm.to_bbox_dim(dim).to_square().to_bbox_norm()
        if opt_expand:
          bbox_norm = bbox_norm.expand(opt_expand)
        detection.bbox = bbox_norm

    sink.send(M)