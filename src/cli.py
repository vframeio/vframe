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
from pathlib import Path
import time
import importlib
from glob import iglob

import argparse
import click

from vframe.settings import app_cfg
from vframe.utils import log_utils

# -----------------------------------------------------------------------------
#
# Argparse pre-process
#
# -----------------------------------------------------------------------------


def choices_description(plugins):
  """Generate custom help menu with colored text
  """
  clr_h: str = '\033[1m\033[94m'
  clr_t: str = '\033[0m'
  sp_max: int = 20 + len(clr_h) + len(clr_t)
  t = ['Commands and plugins:']
  for plugin in plugins:
    t_cli = f'{clr_h}{plugin.name}{clr_t}'
    sp = sp_max - len(t_cli)
    t.append(f'\t{t_cli}{" " * sp}{plugin.description}')
  result: str = "\n".join(t)
  return result


# intercept first argument using argparse to select command group
argv_tmp = sys.argv
sys.argv = sys.argv[:2]
help_desc = f"\033[1m\033[94mVFRAME CLI ({app_cfg.VERSION})\033[0m"
ap = argparse.ArgumentParser(usage="vf [command]",
                             description=help_desc,
                             formatter_class=argparse.RawDescriptionHelpFormatter,
                             epilog=choices_description(app_cfg.plugins.plugins))
ap.add_argument('commands', choices=app_cfg.plugins.keys(), metavar='[command]')

# exit and how help if no command group supplied
if len(sys.argv) < 2:
  ap.print_help()
  sys.exit(1)

args = ap.parse_args()
sys.argv = argv_tmp
sys.argv.pop(1)  # remove first argument (command group)
plugin_group = app_cfg.plugins.get(args.commands) # create plugin


# -----------------------------------------------------------------------------
#
# Click CLI
#
# -----------------------------------------------------------------------------

# @click.option('--pipe', 'opt_pipe', type=bool, default=plugin_group.pipe)
@click.group(chain=plugin_group.pipe, no_args_is_help=True, help=help_desc)
@click.pass_context
def cli(ctx, opt_pipe=True):
  """\033[1m\033[94mVFRAME\033[0m
  """
  # print("plugin_group.pipe", plugin_group.pipe)
  # opt_pipe = plugin_group.pipe
  opt_verbosity = int(os.environ.get("VFRAME_VERBOSITY", 4))  # 1 - 5
  # store reference to opt_pipe for access in callback
  ctx.opts = {'opt_pipe': plugin_group.pipe}
  # store user object variables
  ctx.ensure_object(dict)
  ctx.obj['start_time'] = time.time()
  # init global logger
  log_utils.Logger.create(verbosity=opt_verbosity)


# def process_commands(processors, opt_pipe):
@cli.resultcallback()
def process_commands(processors):
  """This result callback is invoked with an iterable of all the chained
  subcommands. As in this example each subcommand returns a function
  we can chain them together to feed one into the other, similar to how
  a pipe on UNIX works. Copied from Click's docs.
  """
  if not plugin_group.pipe:
    return

  def sink():
    """This is the end of the pipeline
    """
    while True:
      yield

  sink = sink()
  sink.__next__()

  # Compose and prime processors
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
import vframe.utils.im_utils

for plugin_script in plugin_group.scripts:

  fp_root = '/'.join(plugin_script.filepath.split('/')[:2])  # eg plugins/vframe_custom_plugin
  fp_root = join(app_cfg.DIR_SRC, fp_root)

  # print(fp_root)
  if not Path(fp_root).is_dir():
    print(f'{50 * "*"}\nWARNING: {fp_root} does not exist\n{50 * "*"}')
    continue

  # append plugin directory to import paths
  if fp_root not in sys.path:
    sys.path.append(fp_root)

  # glob for python files inside command directory
  fp_dir_glob = join(app_cfg.DIR_SRC, plugin_script.filepath, '*.py')
  for fp_py in iglob(fp_dir_glob):
    fn = Path(fp_py).stem

    # skip files starting with "_"
    if plugin_script.include_hidden is False and fn.startswith('_'):
      continue

    fp_module = str(Path(fp_py).relative_to(Path(app_cfg.DIR_SRC)))
    fp_import = fp_module.replace('/', '.').replace('.py', '')
    try:
      module = importlib.import_module(fp_import)
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