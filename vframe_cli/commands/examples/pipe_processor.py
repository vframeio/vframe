############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import click

from vframe.utils.click_utils import processor
#from vframe.utils.click_utils import generator

@click.command('')
@processor
@click.pass_context
def cli(ctx, pipe):
  """Template"""
  
  from vframe.settings import app_cfg

  
  # ---------------------------------------------------------------------------
  # initialize

  log = app_cfg.LOG
  log.debug('This is a pipe processor template')


  # ---------------------------------------------------------------------------
  # Example: process images as they move through pipe

  while True:

    # get data
    pipe_item = yield
    header = ctx.obj['header']
    item_data = header.get_data(opt_data_key)
    
    # get image
    im = pipe_item.get_image(types.FrameImage.DRAW)

    # do something to image
    # im = do_something(im)  # for example
    
    # update pipe images
    pipe_item.set_image(types.FrameImage.DRAW, im)

    # send image throug pipestream
    pipe.send(pipe_item)