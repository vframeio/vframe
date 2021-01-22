############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

@click.command()
@click.option('-i', '--input', 'opt_dir_in', required=True,
  help='Path to project folder (metadata.csv, mask, real)')
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help='Force overwrite annotations file')
@click.option('--threshold', 'opt_thresh', default=800, show_default=True,
  help='Minimum total pixels for annotation (25x25 = 625)')
@click.option('--label', 'opt_labels', multiple=True, default=None,
  help='Labels to filter for')
@click.option('-t', '--threads', 'opt_threads', default=12, show_default=True,
  help='Number threads')
@click.option('--force-negative', 'opt_force_negative', is_flag=True,
  help='Force negative null annotations')
@click.option('-e', '--ext', 'opt_ext', default='png')
@click.pass_context
def cli(ctx, opt_dir_in, opt_force, opt_thresh, opt_labels, opt_threads, 
  opt_force_negative, opt_ext):
  """Generate annotations"""
  
  from os.path import join
  from glob import glob
  from pathlib import Path
  from dataclasses import asdict

  import dacite
  from pathos.multiprocessing import ProcessingPool as Pool
  from numba import jit, njit
  from PIL import Image
  import pandas as pd
  import cv2 as cv
  import numpy as np
  from tqdm import tqdm

  from vframe.settings import app_cfg
  from vframe.utils import file_utils, im_utils
  from vframe.models.geometry import BBox
  from vframe.models.annotation import Annotation
  from vframe.models.color import Color
  from vframe.utils import anno_utils


  # init log
  log = app_cfg.LOG

  # post-process input
  opt_threads = opt_threads if opt_threads else pathos.multiprocessing.cpu_count()

  # output file
  fp_annotations = join(opt_dir_in, app_cfg.FN_ANNOTATIONS)
  if Path(fp_annotations).exists() and not opt_force:
    log.error(f'File exists: {fp_annotations}. Use "-f/--force" to overwrite')
    return

  # load the color coded CSV
  fp_metadata = join(opt_dir_in, app_cfg.FN_METADATA)
  df_objects = pd.read_csv(fp_metadata)
  log.info(f'Metadata file contains {len(df_objects):,} objects')
  label_groups = df_objects.groupby('label_enum')
  for label, df in label_groups:
    log.info(f'{label} contains {len(df)} colors')

  # filter labels
  if opt_labels:
    for opt_label in opt_labels:
      df_objects = df_objects[df_objects['label_enum'] == opt_label]
    log.info(f'Metadata file contains {len(df_objects):,} filtered objects')

  # glob mask
  fp_dir_im_reals = join(opt_dir_in, app_cfg.DN_REAL)
  fp_dir_im_masks = join(opt_dir_in, app_cfg.DN_MASK)
  fps_reals = glob(join(fp_dir_im_reals, f'*.{opt_ext}'))
  fps_masks = glob(join(fp_dir_im_masks, f'*.{opt_ext}'))
  
  if len(fps_masks) != len(fps_reals):
    log.error(f'Directories not balanced: {len(fps_masks)} masks != {len(fps_reals)} real')
    return
  
  log.info(f'Converting {len(fps_masks)} mask images to annotations...')


  @jit(nopython=True, parallel=True)
  def fast_np_trim(im):
    '''Trims ndarray of blackspace/zeros
    :param im: np.ndarray image in BGR or RGB
    :returns np.ndarray image in BGR or RGB
    '''
    # Warning: does not throw error when used in @jit mode
    # this can cause early termination of pool processes
    # comment out @jit if early termination
    npza = np.array([0,0,0], dtype=np.uint8)
    w, h = im.shape[:2][::-1]
    im = im.reshape((w * h, 3))
    idxs = np.where(im > npza)
    if len(idxs[0]):
      return im[min(idxs[0]):max(idxs[0])]
    else:
      return im


  def pool_worker(fp_mask):
    fn_mask = Path(fp_mask).name
    im_mask = cv.imread(fp_mask)
    w, h = im_mask.shape[:2][::-1]

    # flatten image and find unique colors
    im_mask_rgb = cv.cvtColor(im_mask, cv.COLOR_BGR2RGB)

    # Opt 1: use Numpy unique
    #im_flat_rgb = im_mask_rgb.reshape((w * h, 3))
    #rgb_unique = np.unique(im_flat_rgb, axis=0)

    # Opt 2: use numba then set list (faster)
    n_colors_found = 0
    results = []

    if not opt_force_negative:
      try:
        im_flat_rgb_trim = fast_np_trim(im_mask_rgb)
      except Exception as e:
        log.error(fp_mask)
        log.error(e)
      im_flat_rgb_trim = im_flat_rgb_trim.tolist()
      rgb_unique = set(tuple(map(tuple, im_flat_rgb_trim)))
      # iterate through all colors for all objects

      for df in df_objects.itertuples():
        # if the color is found in the image with a large enough area, append bbox
        color = Color.from_rgb_int((df.r, df.g, df.b)) # RGB uint8 (255,255,255)
        rgb_int = color.to_rgb_int()
        if rgb_int == (0,0,0) and df.label_index == 0:
          # skip background
          continue
        # quick check for color exist in image
        color_range = anno_utils.get_color_boundaries(rgb_int)
        color_test_results = []
        color_found = False
        for color_test in color_range:
          if any([(tuple(color_test) == tuple(c)) for c in rgb_unique]):
            color_found = True
            break

        if color_found:
          #color_hex = f'0x{color_utils.rgb_int_to_hex(rgb_int)}'
          n_colors_found += 1
          #color_hex = color.to_rgb_hex()

          # find bbox by color search
          bbox = anno_utils.color_mask_to_rect(im_mask_rgb, rgb_int, threshold=opt_thresh)
          if bbox is not None:
            #bbox_nlc = bbox_norm.to_labeled(df.label, df.label_index, fn_mask).to_colored(color_hex)
            init_obj = {
              'filename': fn_mask,
              'label_enum': df.label_enum,
              'label_index': df.label_index,
              'label_display': df.label_display,
              'bbox': bbox,
              'color': color,
            }
            annotation = dacite.from_dict(data=init_obj, data_class=Annotation)    
            results.append(annotation.to_dict())

    if not n_colors_found:
      # FIXME: improve negative annotations
      # if no colors were found, image has no annotations. use as negative data
      #log.info(f'No annotations in: {fn_mask}')
      init_obj = {
        'filename': fn_mask,
        'label_enum': 'background',
        'label_display': 'Background',
        'label_index': 0,
        'bbox': BBox(0,0,0,0,0,0),
        'color': Color.from_rgb_hex('0x000000'),
      }
      annotation = dacite.from_dict(data=init_obj, data_class=Annotation)
      results.append(annotation.to_dict())

    return results

  
  # Multiprocess/threading use imap instead of map via @hkyi Stack Overflow 41920124
  with Pool(opt_threads) as p:
    d = f'Annotating x{opt_threads}'
    t = len(fps_masks)
    pool_results = list(tqdm(p.imap(pool_worker, fps_masks), total=t, desc=d))


  # iterate through all images
  records = []
  for pool_result in pool_results:
    #records += [asdict(bbox_nlc) for bbox_nlc in pool_result]
    records += [bbox_dict for bbox_dict in pool_result]  # ???

  # Convert to dataframe
  df_annos = pd.DataFrame.from_dict(records)
  #df_annos = df_annos[df_annos['label_index'] != 0]  #  remove background

  # write CSV
  df_annos.to_csv(fp_annotations, index=False)

  # status
  log.info(f'Wrote {len(df_annos)} annotations to {fp_annotations} ')

