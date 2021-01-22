############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

@click.command()
@click.option('-i', '--input', 'opt_input', required=True,
  help='Path to logfile')
@click.option('-o', '--output', 'opt_output',)
@click.option('-x', '--x', 'opt_x_lim', type=(int, int), default=(None, None))
@click.option('-y', '--y', 'opt_y_lim', type=(float, float), default=(0, 18))
@click.option('--dpi', 'opt_dpi', default=72,
  help="Pixels per inch resolution for output")
@click.option('--figsize', 'opt_figsize', default=(1280, 720),
  help="matplotlib figure size (pixels")
@click.pass_context
def cli(ctx, opt_input, opt_output, opt_x_lim, opt_y_lim, opt_dpi, opt_figsize):
  """Plots YOLO training logfile"""

  import re
  import matplotlib.pyplot as plt
  from pathlib import Path

  from vframe.utils import log_utils, file_utils
  from vframe.settings import app_cfg
  from vframe.utils.draw_utils import pixels_to_figsize, set_matplotlib_style

  log = app_cfg.LOG

  log.info(f'Generate YOLO logfile plot: {opt_input}')

  if not opt_output:
    ext = Path(opt_input).suffix
    opt_output = opt_input.replace(ext, f'_plot.png')

  # set styles
  set_matplotlib_style(plt)

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  lines = file_utils.load_txt(opt_input)
  numbers = {'1','2','3','4','5','6','7','8','9'}

  iters = []
  losses = []
  
  for line in lines:
    result = re.match(r'^[\s]?([0-9]+):', line)
    if result:
      n_iter = int(result[1])
      loss_match = re.search(r'([0-9.]+)\s\bavg loss\b', line)  # 0.095066 avg loss
      if loss_match:
        loss = float(loss_match[1])
        iters.append(n_iter)
        losses.append(loss)
           
  opt_x_lim = opt_x_lim if any(opt_x_lim) else [min(iters), max(iters)]
  log.debug(opt_x_lim)
  ax.set_xlim(opt_x_lim)
  ax.set_ylim(opt_y_lim)
  project_name = Path(opt_input).parent.name
  plt.title(f'Config: {project_name}')
  plt.xlabel('Iteration')
  plt.ylabel('Average Loss')
  # Show the major grid lines with dark grey lines
  plt.grid(b=True, which='major', color='#666666', linestyle='-')

  # Show the minor grid lines with very faint and almost transparent grey lines
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

  #plt.grid()
  plt.scatter(iters, losses, s=2)
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(opt_output, dpi=opt_dpi)