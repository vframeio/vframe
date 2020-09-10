#!/usr/bin/env python

############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import os
from os.path import join
import sys
from functools import update_wrapper, wraps
from pathlib import Path
import time
from dataclasses import dataclass
from pdb import set_trace as bp
import importlib
from glob import iglob

import argparse
import click

from vframe.settings import app_cfg, plugins_cfg
from vframe.utils import log_utils


# -----------------------------------------------------------------------------
#
# Argparse pre-process
#
# -----------------------------------------------------------------------------

import argparse
def choicesDescriptions(plugins):
  """Generates custom help menu
  """
  clr_h = '\033[1m\033[94m'
  clr_t = '\033[0m'
  sp_max = 30 + len(clr_h) + len(clr_t)
  t = ['Command options:']
  for plugin in plugins:
    t_cli = f'{clr_h}./cli.py {plugin.name}{clr_t}'
    sp = sp_max - len(t_cli)
    t.append(f'\t{t_cli}{" "*sp}{plugin.description}')
  t = "\n".join(t)
  return t

# intercept the first argument using argparse to select command group
argv_tmp = sys.argv
sys.argv = sys.argv[:2]
ap = argparse.ArgumentParser(prog='./cli.py [command]',
  usage='./cli.py [command]',
  description='\033[1m\033[94mVFRAME CLI (beta)\033[0m',
  formatter_class=argparse.RawDescriptionHelpFormatter,
  epilog=choicesDescriptions(plugins_cfg.plugins.plugins))
ap.add_argument('commands', choices=plugins_cfg.plugins.keys(), 
metavar='[command]')
args = ap.parse_args()
sys.argv = argv_tmp
sys.argv.pop(1)  # remove arg

# create plugin config
plugin_group = plugins_cfg.plugins.get(args.commands)


# -----------------------------------------------------------------------------
#
# Click CLI
#
# -----------------------------------------------------------------------------

@click.group(chain=plugin_group.pipe, no_args_is_help=True)
@click.option('--pipe', 'opt_pipe', type=bool, default=plugin_group.pipe)
@click.pass_context
def cli(ctx, opt_pipe):
  """\033[1m\033[94mVFRAME\033[0m
  """
  opt_verbosity = int(os.environ.get("VERBOSITY", 4)) # 1 - 5
  # store reference to opt_pipe for access in callback
  ctx.opts = {'opt_pipe': opt_pipe}
  # store user object variables
  ctx.ensure_object(dict)
  ctx.obj['start_time'] = time.time()
  # init global logger
  log_utils.Logger.create(verbosity=opt_verbosity)



@cli.resultcallback()
def process_commands(processors, opt_pipe):
    """This result callback is invoked with an iterable of all the chained
    subcommands. As in this example each subcommand returns a function
    we can chain them together to feed one into the other, similar to how
    a pipe on UNIX works. Copied from Click's docs.
    """
    
    if not opt_pipe:
      return

    def sink():
      """This is the end of the pipeline"""
      while True:
        yield

    sink = sink()
    sink.__next__()

    # Compose all of the coroutines, and prime each one
    for processor in reversed(processors):
      sink = processor(sink)
      sink.__next__()

    sink.close()



# -----------------------------------------------------------------------------
#
# Setup commands
#
# -----------------------------------------------------------------------------

# append files to click groups
for plugin_script in plugin_group.scripts:
  
  # append plugin directory to import paths
  fp_root = '/'.join(plugin_script.filepath.split('/')[:2])  # eg plugins/vframe_custom_plugin
  if fp_root not in sys.path:
    sys.path.append(fp_root)

  # glob for python files inside command directory
  fp_dir_glob = join(plugin_script.filepath, '*.py')
  for fp_py in iglob(fp_dir_glob):

    # skip files starting with "_"
    fn = Path(fp_py).name
    if plugin_script.include_hidden == False and fn[0] == '_':
      continue
      
    fp_module = fp_py.replace('/', '.').replace('.py','')
    try:
      print(fp_module)
      module = importlib.import_module(fp_module)
      fn = Path(fp_py).stem
      cli.add_command(module.cli, name=fn)
    except Exception as e:
      msg = f'Could not import "{fn}": {e}'
      print(f"{app_cfg.TERM_COLORS.FAIL}{msg}{app_cfg.TERM_COLORS.ENDC}")


# -----------------------------------------------------------------------------
#
# Start CLI application
#
# -----------------------------------------------------------------------------

cli()
