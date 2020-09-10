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
@click.option('-i', '--input', 'opt_inputs', required=True,
  multiple=True,
  help="Input files to merge")
@click.option('-o', '--output', 'opt_output', required=True,
  help='Output file')
@click.option('--minify', 'opt_minify', is_flag=True,
  default=False,
  help='Minify JSON')
@click.pass_context
def cli(ctx, opt_inputs, opt_output, opt_minify):
  """Merge JSON detections"""

  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  from tqdm import tqdm

  from vframe.utils import file_utils
  from vframe.settings import app_cfg

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  # load first file
  opt_inputs = list(opt_inputs)
  fp_first = opt_inputs.pop(0)
  data =  file_utils.load_json(fp_first)

  # merge 
  for fp_in in tqdm(opt_inputs, desc='Files'):
    
    results = file_utils.load_json(fp_in)

    for result_idx, result in enumerate(results):
      
      for frame_index, frame_data in result['frames_data'].items():
        data[result_idx]['frames_data'][frame_index].update(frame_data)

  # write
  file_utils.write_json(data, opt_output, minify=opt_minify)

