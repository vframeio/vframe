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
@click.option('--image-size', 'opt_im_sizes', 
  default=[(640,480)], type=(int, int), multiple=True,
  help='Image inference size')
@click.option('--iterations', 'opt_n_iters',
  default=10, show_default=True,
  type=click.IntRange(1,1000),
  help='Number of iterations')
@click.option('--batch-size', 'opt_batch_sizes', default=[1], multiple=True,
  help='Image inference size')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.pass_context
def cli(sink, opt_model_enums, opt_device, opt_dnn_sizes, 
  opt_n_iters, opt_fp_out, opt_batch_sizes, opt_im_sizes, opt_verbose):
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

  # add models
  new_opts = []
  for opt_model_enum in opt_model_enums:
      new_opt = {'model': opt_model_enum.name.lower()}
      new_opts.append(new_opt)
  opts = new_opts

  # add batch sizes
  new_opts = []
  for batch_size in opt_batch_sizes:
    for opt in opts:
      new_opt = opt.copy()
      new_opt.update({'batch_size': batch_size})
      new_opts.append(new_opt)
  opts = new_opts

  # add dnn sizes
  if all([all(x) for x in opt_dnn_sizes]):
    new_opts = []
    for dnn_size in opt_dnn_sizes:
      for opt in opts:
        new_opt = opt.copy()
        new_opt.update({'dnn_size': dnn_size})
        new_opts.append(new_opt)
    opts = new_opts

  # add initial image sizes
  if any(opt_im_sizes):
    new_opts = []
    for im_size in opt_im_sizes:
      for opt in opts:
        new_opt = opt.copy()
        new_opt.update({'im_size': im_size})
        new_opts.append(new_opt)
    opts = new_opts
        
  # add device
  for opt in opts:
    opt.update({'device': opt_device})
  
  # init args
  benchmarks = []

  # iterate models
  for opt in opts:
    
    dnn_cfg = modelzoo.get(opt['model'])
    dnn_cfg.override(**opt)
    
    processor = 'gpu' if opt_device > -1 else 'cpu'

    # init cvmodel
    cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)
    
    w,h = opt['im_size']
    opt['batch_size'] = int(opt['batch_size'])
    
    if opt['batch_size'] > 1 and not getattr(cvmodel, 'fps_batch', None):
      LOG.warn("Batch inference not available. Using single image.")

    if opt['batch_size'] > 1 and getattr(cvmodel, 'fps_batch', None):
      fps = cvmodel.fps_batch(n_iters=opt_n_iters, dim=(w,h), batch_size=opt['batch_size'])
    else:
      fps = cvmodel.fps(n_iters=opt_n_iters, dim=(w,h))
    
    # force height to zero if cfg does not declare
    if not dnn_cfg.width:
      dnn_cfg.width = 0
    if not dnn_cfg.height:
      dnn_cfg.height = 0
    
    o = {
      'model': opt['model'],
      'fps': float(f'{fps:.2f}'),
      'iterations': opt_n_iters,
      'image_width': w,
      'image_height': h,
      'dnn_width': dnn_cfg.width,
      'dnn_height': dnn_cfg.height,
      'processor': processor,
      'batch_size': opt['batch_size'],
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