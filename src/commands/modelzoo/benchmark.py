############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

from vframe.models import types
from vframe.utils import click_utils


@click.command('')
@click.option('-m', '--model', 'opt_model_enums',
  type=types.ModelZooClickVar,
  default={'coco':'COCO'},
  multiple=True, required=True,
  help=click_utils.show_help(types.ModelZoo))
@click.option('-o', '--output', 'opt_fp_out',
  help='Filepath to output CSV')
@click.option('-d', '--device', 'opt_device', default=0,
  help='GPU device for inference (use -1 for CPU)')
@click.option('-s', '--size', 'opt_dnn_sizes', type=(int, int),
  default=[(0, 0)], show_default=True,
  multiple=True,
  help='DNN blob image size. Overrides config file')
@click.option('--iterations', 'opt_n_iters',
  default=10, show_default=True,
  type=click.IntRange(1,1000),
  help='Number of iterations')
@click.option('--size', 'opt_size', type=(int, int), default=(640,480),
  help='Image inference size')
@click.option('--batch-size', 'opt_batch_size', default=1,
  help='Image inference size')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.pass_context
def cli(sink, opt_model_enums, opt_device, opt_dnn_sizes, 
  opt_n_iters, opt_fp_out, opt_size, opt_batch_size, opt_verbose):
  """Benchmark models"""

  from os.path import join
  from pathlib import Path
  from dataclasses import asdict

  from dacite import from_dict
  from tqdm import tqdm
  import pandas as pd

  from vframe.models.cvmodels import BenchmarkResult
  from vframe.image.dnn_factory import DNNFactory
  from vframe.utils import file_utils, im_utils
  from vframe.settings.app_cfg import LOG, modelzoo


  model_names = [x.name.lower() for x in opt_model_enums]
  
  # init args
  benchmarks = []

  # iterate models
  for model_name in model_names:
    
    for dnn_size in opt_dnn_sizes:
      
      dnn_cfg = modelzoo.get(model_name)
      # override dnn_cfg vars with cli vars
      dnn_cfg.override(device=opt_device)

      if all(dnn_size):
        # override if non-zero
        dnn_cfg.override(size=dnn_size)
      
      processor = 'gpu' if opt_device > -1 else 'cpu'

      # init cvmodel
      cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)
      
      w,h = opt_size

      if opt_batch_size > 1 and not getattr(cvmodel, 'fps_batch', None):
        LOG.warn("Batch inference not available. Using single image.")

      if opt_batch_size > 1 and getattr(cvmodel, 'fps_batch', None):
        fps = cvmodel.fps_batch(n_iters=opt_n_iters, dim=opt_size, batch_size=opt_batch_size)
      else:
        fps = cvmodel.fps(n_iters=opt_n_iters, dim=opt_size)
      
      # force height to zero if cfg does not declare
      if not dnn_cfg.width:
        dnn_cfg.width = 0
      if not dnn_cfg.height:
        dnn_cfg.height = 0
      
      o = {
        'model': model_name,
        'fps': float(f'{fps:.2f}'),
        'iterations': opt_n_iters,
        'image_width': w,
        'image_height': h,
        'dnn_width': dnn_cfg.width,
        'dnn_height': dnn_cfg.height,
        'user_width': dnn_size[0],
        'user_height': dnn_size[1],
        'processor': processor
      }

      # append data as a list of dicts
      benchmarks.append(asdict(from_dict(data=o, data_class=BenchmarkResult)))

  # write
  if opt_fp_out:
    if opt_verbose:
      LOG.info(f'Wrote data to {opt_fp_out}')
    pd.DataFrame.from_dict(benchmarks).to_csv(opt_fp_out, index=False)
  else:
    for benchmark in benchmarks:
      LOG.info(benchmark)
    LOG.info('Use "-o/--output" to write results to CSV file')