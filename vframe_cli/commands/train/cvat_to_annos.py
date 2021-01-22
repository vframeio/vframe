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
@click.option('-i','--input','opt_fp_in',required=True)
@click.option('-o','--output','opt_fp_out')
@click.option('-z', '--zeros', 'opt_n_zeros', default=6)
@click.option('-e','--ext','opt_ext', default='PNG')
@click.option('--prefix', 'opt_prefix',default='frame')
@click.option('--decimate', 'opt_decimate',default=1)
@click.option('--limit', 'opt_limit', default=0)
@click.pass_context
def cli(ctx, opt_fp_in, opt_fp_out, opt_n_zeros, opt_ext, opt_prefix, opt_decimate, opt_limit):
  """CVAT XML to VFRAME CSV"""

  from pathlib import Path
  import math

  import pandas as pd
  import xmltodict
  from dacite import from_dict

  from vframe.settings.app_cfg import LOG
  from vframe.models.geometry import BBox
  from vframe.models.annotation import Annotation
  from vframe.models.color import Color
  from vframe.utils.file_utils import zpad

  LOG.debug(f'Process {opt_fp_in}')

  if not opt_fp_out:
    dot_ext = Path(opt_fp_in).suffix
    opt_fp_out = opt_fp_in.replace(dot_ext, '.csv')

  with open(opt_fp_in) as fp:
    cvat_dict = xmltodict.parse(fp.read())

  cvat_annos = cvat_dict.get('annotations')
  size = cvat_annos.get('meta').get('task').get('original_size')
  w,h = (int(float(size.get('width'))), int(float(size.get('height'))))
  tracks = cvat_annos.get('track')
  if not isinstance(tracks, list):
    tracks = [tracks]

  # defaults
  color = Color(0,0,0)  # irrelevant, placeholder
  label_idx = 0  # remapped in other script

  # iterate boxes into annos
  annos = []
  for track in tracks:
    label_enum = track.get('@label')
    boxes = track.get('box')
    boxes = boxes if isinstance(boxes, list) else [boxes]
    for box_idx, b in enumerate(boxes):
      if box_idx % opt_decimate:
        # skip every N decimate frames
        continue
      xyxy = (b.get('@xtl'), b.get('@ytl'), b.get('@xbr'), b.get('@ybr'))
      # default CVAT output name frame_000000.PNG
      frame_idx = zpad(int(b.get('@frame')), zeros=opt_n_zeros)
      filename = f'{opt_prefix}_{frame_idx}.{opt_ext}'
      xyxy = list(map(int,map(float, xyxy)))
      bbox = BBox(*xyxy,w,h)
      o = {
        'filename': filename,
        'bbox': bbox,
        'label_display': label_enum,
        'label_enum': label_enum,
        'label_index': label_idx,
        'color': color,
      }
      anno = from_dict(data_class=Annotation, data=o)
      annos.append(anno.to_dict())

  # to dataframe
  df = pd.DataFrame.from_dict(annos)

  # limit images, grouped by filename
  if opt_limit and len(annos) > opt_limit:
    groups = df.groupby('filename')
    df_groups = [df_group for i, df_group in groups]
    n_interval = math.ceil(len(annos) / opt_limit)
    df_groups = df_groups[::n_interval]
    df = pd.concat(df_groups)

  # write csv
  df.to_csv(opt_fp_out, index=False)
  LOG.info(f'Wrote {len(df):,} annotations to {opt_fp_out}')
