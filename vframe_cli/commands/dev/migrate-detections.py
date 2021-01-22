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
@click.option('-i', '--input', 'opt_input', required=True,
  help="Input file to migrate")
@click.option('-o', '--output', 'opt_output',
  help='Output file')
@click.option('--minify/--no-minify', 'opt_minify', is_flag=True,
  default=False,
  help='Minify JSON')
@click.option('-f', '--force', 'opt_force', is_flag=True,
  help='Overwrite output file')
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_minify, opt_force):
  """Migrate JSON detections to BBox.v2"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  from dataclasses import asdict

  from tqdm import tqdm

  from vframe.utils import file_utils
  from vframe.settings import app_cfg
  from vframe.models.geometry import BBox
  from vframe.models.pipe_item import PipeContextHeader

  # ------------------------------------------------
  # start

  log = app_cfg.LOG
  opt_output = opt_output if opt_output else opt_input.replace('.json', '_v2.json')
  fp_out = opt_input.replace('.json', '_migrated.json')
  if Path(fp_out).is_file() and not opt_force:
    log.error(f'{fp_out}  exists. Use "-f/--force to overwrite')
    return
  else:
    log.debug(f'Output file: {fp_out}')

  # load
  log.debug(f'Loading {opt_input}...')
  items = file_utils.load_json(opt_input)

  # migrate bboxes
  for item in tqdm(items):
    fp = item['filepath']
    if Path(fp).is_file():
      header = PipeContextHeader(fp)
      frames_data = item['frames_data']
      for frame_idx, frame_data in frames_data.items():
        for model_name, model_results in frame_data.items():
          if 'duration' in model_results.keys():
            model_results.pop('duration')
          for detection in model_results['detections']:
            bbox = detection['bbox']
            x1,y1,x2,y2 = (bbox['x1'],bbox['y1'], bbox['x2'], bbox['y2'])
            bbox_new = BBox.from_xyxy_norm(x1,y1,x2,y2,*header.dim)
            detection['bbox'] = asdict(bbox_new)

  log.debug(f'Writing {Path(fp_out).name}...')
  file_utils.write_json(items, fp_out, minify=opt_minify)