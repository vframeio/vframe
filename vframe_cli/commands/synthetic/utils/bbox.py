#############################################################################
#
# VFRAME Synthetic Data Generator
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

import click

from vframe.settings import app_cfg


opts_sources = [app_cfg.DN_REAL, app_cfg.DN_MASK, app_cfg.DN_COMP, app_cfg.DN_BBOX]

@click.command()
@click.option('-i', '--input', 'opt_dir_render', required=True)
@click.option('--type', 'opt_type', type=click.Choice(opts_sources),
  default=app_cfg.DN_COMP,
  help='Output dir')
@click.option('--slice', 'opt_slice', type=(int, int), default=(None, None),
  help='Slice list of files')
@click.option('-t', '--threads', 'opt_threads', type=int,
  help='Number threads')
@click.option('--font-size', 'opt_font_size', default=14)
@click.option('--from-norm', 'opt_use_bbox_norm', is_flag=True,
  help="Use old annotation bbox norm format")
@click.option('-e','--ext','opt_ext', default='png')
@click.option('--skip-blanks/--draw-blanks', 'opt_skip_blank', is_flag=True)
@click.option('--label-color', 'opt_label_color', is_flag=True)
@click.pass_context
def cli(ctx, opt_dir_render, opt_type, opt_slice, opt_threads, 
  opt_font_size, opt_use_bbox_norm, opt_ext, opt_skip_blank, opt_label_color):
  """Generates bounding box images"""

  from os.path import join

  from PIL import Image
  import pandas as pd
  from glob import glob
  from pathlib import Path

  import cv2 as cv
  import numpy as np
  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count

  from vframe.utils import file_utils, draw_utils
  from vframe.models.color import Color
  from vframe.models.geometry import BBox

  log = app_cfg.LOG
  log.info('Draw annotations')

  file_utils.ensure_dir(join(opt_dir_render, app_cfg.DN_BBOX))

  # set N threads
  if not opt_threads:
    opt_threads = cpu_count()  # maximum

  # glob images
  dir_glob = str(Path(opt_dir_render) / opt_type / f'*.{opt_ext}')
  fps_ims = sorted(glob(dir_glob))
  if any(opt_slice):
    fps_ims = fps_ims[opt_slice[0]:opt_slice[1]]
  log.info(f'found {len(fps_ims)} images in {dir_glob}')

  # load annotation meta
  fp_annos = join(opt_dir_render, app_cfg.FN_ANNOTATIONS)
  log.debug(f'Load: {fp_annos}')
  try:
    df_annos = pd.read_csv(fp_annos)
  except Exception as e:
    log.warn('No annotations. Exiting')
    return

  def pool_worker(item):
    
    fp_im = item['fp']
    fn = Path(fp_im).name
    # group by filename
    df_fn = df_annos[df_annos.filename == fn]
    if not len(df_fn) > 0:
      # log.warning(f'No annotations in: {fn}')
      if item['opt_skip_blank']:
        return False
    
    # load image
    im = Image.open(fp_im)
    dim = im.size

    # draw bboxes
    for rf in df_fn.itertuples():
      if rf.label_enum == 'background':
        continue
      if opt_use_bbox_norm:
        bbox = BBox.from_xyxy_norm(rf.x1, rf.y1, rf.x2, rf.y2, *dim)
      else:
        bbox = BBox(rf.x1, rf.y1, rf.x2, rf.y2, *dim)

      color_anno = Color.from_rgb_hex((rf.color_hex))
      label = f'{rf.label_enum}'
      if opt_label_color:
        r,g,b = color_anno.to_rgb_int()
        rgb_str = f'{r}, {g}, {b}'
        label = f'{label} ({rgb_str})'
      color_bbox = Color.from_rgb_int((255,255,255))
      im = draw_utils.draw_bbox(im, bbox, color=color_bbox, label=label, size_label=opt_font_size)

    # write file
    fp_out = join(opt_dir_render, app_cfg.DN_BBOX, Path(fp_im).name)
    im.save(fp_out)
    return True

  items = [{'fp': fp, 'opt_skip_blank': opt_skip_blank} for fp in fps_ims]
  n_items = len(items)
  with Pool(opt_threads) as p:
    d = f'Render bbox x{opt_threads}'
    pool_results = list(tqdm(p.imap(pool_worker, items), total=n_items, desc=d))

  n_wrote = sum(pool_results)
  log.info(f'Wrote {n_wrote:,} images. Skipped {n_items - n_wrote:,}')