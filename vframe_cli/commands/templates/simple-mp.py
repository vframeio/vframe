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
@click.option('-n', '--iters', 'opt_iters', default=100)
@click.option('-t', '--threads', 'opt_threads', default=None, type=int)
@click.pass_context
def cli(ctx, opt_iters, opt_threads):
  """Multiprocessor simple template"""

  # ------------------------------------------------
  # imports

  from time import sleep
  import random

  from tqdm import tqdm
  from pathos.multiprocessing import ProcessingPool as Pool
  from pathos.multiprocessing import cpu_count

  from vframe.settings import app_cfg


  log = app_cfg.LOG

  # set N threads
  if not opt_threads:
    opt_threads = cpu_count()  # maximum

  # -----------------------------------------------------------
  # start pool worker

  def pool_worker(pool_item):

    x = pool_item['x']
    # add your processor intensive task here
    sleep(x)

    # return a dict of the result here
    return {'result': True, 'seconds': x}

  # end pool worker
  # -----------------------------------------------------------


  # convert file list into object with 
  pool_items = [{'x': random.uniform(0, 1.0)} for x in range(opt_iters)]

  # init processing pool iterator
  # use imap instead of map via @hkyi Stack Overflow 41920124
  desc = f'simple-mp x{opt_threads}'
  with Pool(opt_threads) as p:
    pool_results = list(tqdm(p.imap(pool_worker, pool_items), total=opt_iters, desc=desc))