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
@click.option('--replace-path', 'opt_replace_path',
  help="Replace file parent path")
@click.pass_context
def cli(ctx, opt_inputs, opt_output, opt_replace_path, opt_minify):
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
  merge_results = {}

  # merge 
  for fp_in in tqdm(opt_inputs, desc='Files'):

    # load json
    log.debug(f'load: {fp_in}')
    detection_results = file_utils.load_json(fp_in)

    # add all the current detections to cumulative detections
    for detection_result in detection_results:
        # replaced place in item data
      if opt_replace_path is not None:
        detection_result['filepath'] = join(opt_replace_path, Path(detection_result['filepath']).name)
      filepath = detection_result['filepath']
      
      if not filepath in merge_results.keys():
        merge_results[filepath] = {'filepath': filepath}
      
      for frame_idx, frame_data in detection_result['frames_data'].items():
        if not 'frames_data' in merge_results[filepath].keys():
          merge_results[filepath]['frames_data'] = {}
        if not frame_idx in merge_results[filepath]['frames_data'].keys():
          merge_results[filepath]['frames_data'][frame_idx] = {}
        
        for model_name, model_results in frame_data.items():
          merge_results[filepath]['frames_data'][frame_idx][model_name] = model_results

  # write
  results_out = list(merge_results.values())
  file_utils.write_json(results_out, opt_output, minify=opt_minify)

