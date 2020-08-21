############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click
from dacite import from_dict

from vframe.models import types
from vframe.utils import click_utils


@click.command('')
@click.option('-m', '--model', 'opt_model_enums',
  type=types.ModelZooClickVar,
  multiple=True,
  required=True,
  help=click_utils.show_help(types.ModelZoo))
@click.option('-o', '--output', 'opt_fp_out',
  help='Filepath to output CSV')
@click.option('--gpu/--cpu', 'opt_gpu', is_flag=True, default=False,
  help='GPU index, overrides config file')
@click.option('-s', '--size', 'opt_dnn_size', default=(None, None), type=(int, int),
  help='DNN blob image size. Overrides config file. TODO add multiple sizes')
@click.option('--iters', 'opt_n_iters',
  default=10,
  type=click.IntRange(1,1000),
  help='Number of iterations')
@click.option('-i', '--input', 'opt_fp_in',
  help='Path to input image')
@click.option('--gpu-name', 'opt_gpu_name', default='',
  help='Type in your GPU name (e.g. RTX 2070)')
@click.pass_context
def cli(ctx, opt_model_enums, opt_gpu, opt_dnn_size, 
  opt_n_iters, opt_fp_in, opt_gpu_name, opt_fp_out):
  """Benchmark models"""


  # ------------------------------------------------
  # imports

  from os.path import join
  from pathlib import Path
  from dataclasses import asdict

  from dacite import from_dict
  from tqdm import tqdm
  import pandas as pd

  from vframe.models.cvmodels import BenchmarkResult
  from vframe.image.dnn_factory import DNNFactory
  from vframe.utils import file_utils, im_utils
  from vframe.settings import app_cfg, modelzoo_cfg

  # ------------------------------------------------
  # start

  log = app_cfg.LOG
  

  model_names = [x.name.lower() for x in opt_model_enums]
  
  # init ars
  benchmarks = []

  # iterate models
  for model_name in model_names:
    log.debug(f'Benchmark: {model_name}')
    dnn_cfg = modelzoo_cfg.modelzoo.get(model_name)

    # create blank image if none provided
    if not opt_fp_in:
      im = im_utils.create_blank_im(640, 480)
    else:
      im = cv.imread(opt_fp_in)

    # update dnn cfg
    if opt_gpu:
      dnn_cfg.use_gpu()
      processor = 'gpu'
    else:
      dnn_cfg.use_cpu()
      processor = 'cpu'

    if all(opt_dnn_size):
      dnn_cfg.width = opt_dnn_size[0]
      dnn_cfg.height = opt_dnn_size[1]

    # init cvmodel
    cvmodel = DNNFactory.from_dnn_cfg(dnn_cfg)
    
    h,w = im.shape[:2]
    fps = cvmodel.fps(n_iters=opt_n_iters, im=im)
    o = {
      'model': model_name,
      'fps': float(f'{fps:.2f}'),
      'iterations': opt_n_iters,
      'image_width': w,
      'image_height': h,
      'dnn_width': dnn_cfg.width,
      'dnn_height': dnn_cfg.height,
      'processor': processor
    }

    # append data as a list of dicts
    benchmarks.append(asdict(from_dict(data=o, data_class=BenchmarkResult)))

    log.info(f'FPS: {fps:.2f}, for {opt_n_iters} iterations')

  # Write to CSV or print
  if opt_fp_out:
    df = pd.DataFrame.from_dict(benchmarks)
    df.to_csv(opt_fp_out, index=False)
    log.info(f'Saved file to: {opt_fp_out}')
    log.info('NB: run "plot_benchmark" to visualize data.')
  else:
    for benchmark in benchmarks:
      log.info(benchmark)