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
    help="Path to input")
@click.option('-o', '--output', 'opt_output',
    help="Path to output. Defaults to input as CSV")
@click.option('--slice', 'opt_slice', type=(int, int), default=(-1, -1),
  help="Slice list of inputs")
@click.option('--label', 'opt_labels', multiple=True,
  help='Labels to include in detection summary count')
@click.option('--min', 'opt_threshold_lt', 
  type=click.FloatRange(0,1), default=0.0,
  help='Skip detections less than this threshold')
@click.option('--max', 'opt_threshold_gt', 
  type=click.FloatRange(0,1), default=1.0,
  help='Skip detections greater than this threshold')
@click.option('--min-detections', 'opt_min_detections', default=None, type=int)
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_slice, opt_labels, opt_threshold_gt,
  opt_threshold_lt, opt_min_detections):
  """Summarize detection JSON"""

  from tqdm import tqdm

  import dacite
  import pandas as pd

  from vframe.models.cvmodels import ProcessedFile
  from vframe.models.media import MediaFile
  from vframe.utils.file_utils import load_json
  from vframe.utils.file_utils import swap_ext, add_suffix
  from vframe.settings.app_cfg import LOG


  results = []
  thresholds = [opt_threshold_lt, opt_threshold_gt]
  if not opt_labels:
    LOG.error('"--label" is required')
    return

  # load
  items = load_json(opt_input)
  if all([x > -1 for x in opt_slice]):
    items = items[opt_slice[0]:opt_slice[1]]
  
  # create list of dict results
  for item in tqdm(items):
    pf = dacite.from_dict(data=item, data_class=ProcessedFile)
    mf = MediaFile.from_processed_file(pf)
    d = mf.to_dict()['file_meta']
    d.pop('sha256')
    d['frame_count'] = int(d['frame_count'])
    for opt_label in opt_labels:
      d[opt_label] = mf.n_detections_filtered_total(labels=[opt_label], thresholds=thresholds)
    results.append(d)
          
  # to data frame
  df = pd.DataFrame.from_dict(results)

  if opt_min_detections:
    for opt_label in opt_labels:
      df = df[df[opt_label] >= opt_min_detections]

  # to csv
  fp_out = opt_output if opt_output else swap_ext(opt_input, 'csv')
  df.to_csv(fp_out, index=False)


  # -----------------------------------------------------------------
  # output summary stats

  results = []

  # summarize number of videos
  filters = {
    'n_videos_dets_gt0': 0,
    'n_videos_dets_gt1': 1,
    'n_videos_dets_gt4': 4,
    'n_videos_dets_gt8': 8,
    'n_videos_dets_gt16': 16,
    'n_videos_dets_gt32': 32,
    'n_videos_dets_gt64': 64,
    'n_videos_dets_gt128': 128,
  }

  for fk, fv in filters.items():
    result = {}
    result['label'] = fk
    for opt_label in opt_labels:
      result[opt_label] = len(df[df[opt_label] > fv])
    results.append(result)

  # to csv
  fp_out = add_suffix(fp_out, '_summary')
  pd.DataFrame.from_dict(results).to_csv(fp_out, index=False)