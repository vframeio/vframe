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
@click.option('-i', '--input', 'opt_input', required=True)
@click.option('-o', '--output', 'opt_output', required=True)
@click.option('--dpi', 'opt_dpi', default=72,
  help="Pixels per inch resolution for output")
@click.option('--figsize', 'opt_figsize', type=(int,int), default=(1280, 720),
  help="matplotlib figure size (pixels")
@click.option('--prefix', 'opt_prefix', default='plot',
  help='Filename prefix')
@click.option('--title/--no-title', 'opt_title', is_flag=True, default=True,
  help='Show title')
@click.option('--clusters', 'opt_n_clusters', default=6,
  help='Number of K-Means clusters')
@click.option('--num-videos', 'opt_n_videos', type=int,
  help='Override actual video number to show and approximated number.')
@click.pass_context
def cli(sink, opt_input, opt_output, opt_dpi, opt_figsize, opt_prefix, 
  opt_title, opt_n_clusters, opt_n_videos):
  """Plot media attributes"""

  # ------------------------------------------------
  # imports

  import os
  from os.path import join
  from glob import glob
  from dataclasses import asdict
  from operator import itemgetter

  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.use('Agg')
  #import matplotlib
  import numpy as np
  import pandas as pd
  from sklearn.cluster import KMeans
  
  from vframe.settings.app_cfg import LOG
  from vframe.utils import log_utils, file_utils, video_utils
  from vframe.utils.draw_utils import pixels_to_figsize, set_matplotlib_style


  # create output
  file_utils.ensure_dir(opt_output)

  # set styles
  set_matplotlib_style(plt)

  # read csv
  dtypes = {
    'filename': str,
    'ext': str,
    'valid': np.bool,
    'width': int,
    'height': int,
    'aspect_ratio': float,
    'frame_count': int,
    'codec': str,
    'duration': float,  # int, but pandas doesn't have int na
    'frame_rate': float
    }
  df = pd.read_csv(opt_input, dtype=dtypes)

  # patch
  df.width.fillna(0, inplace=True)
  df.codec.fillna('', inplace=True)
  df.duration.fillna(0, inplace=True)
  df.frame_rate.fillna(0, inplace=True)

  # drop images
  df = df[df.frame_count > 1]
  n_videos = len(df)
  n_videos_display = n_videos if not opt_n_videos else opt_n_videos
  LOG.debug(f'Items: {n_videos}')


  # ---------------------------------------------------------------------------
  # Plot width
  # ---------------------------------------------------------------------------

  LOG.debug('Plot width...')

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)


  if opt_title:
    plt.title(f'Video Width Distribution for {n_videos_display:,} Videos')
  plt.ylabel("Videos")
  plt.xlabel("Width (pixels)")

  # set bins
  bins = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100)

  # plot data
  x = df['width'].values
  plt.hist([x], bins, label=['Width'])
  plt.legend(loc='upper right')

  # save
  fp_out = join(opt_output, f'{opt_prefix}_width.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)

  # ---------------------------------------------------------------------------
  # Plot height
  # ---------------------------------------------------------------------------

  LOG.debug('Plot height...')

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  if opt_title:
    plt.title(f'Video Height Distribution for {n_videos_display:,} Videos')
  plt.ylabel("Videos")
  plt.xlabel("Height (pixels)")

  # set bins
  bins = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100)

  # plot data
  x = df['height'].values
  plt.hist([x], bins, label=['height'])
  plt.legend(loc='upper right')

  # save
  fp_out = join(opt_output, f'{opt_prefix}_height.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)


  # ---------------------------------------------------------------------------
  # Plot FPS
  # ---------------------------------------------------------------------------

  LOG.debug('Plot FPS...')

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  if opt_title:
    plt.title(f'Frames Per Second Distribution for {n_videos_display:,} Videos')
  plt.ylabel("Video")
  plt.xlabel("Frames Per Second")

  # set bins
  n_bins = 30
  bin_size = 1
  bins = list(range(15, n_bins * bin_size, bin_size))

  # plot data
  x = df['frame_rate'].values
  plt.hist([x], bins, label=['FPS'])
  plt.legend(loc='upper right')

  # save
  fp_out = join(opt_output, f'{opt_prefix}_fps.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)


  # ---------------------------------------------------------------------------
  # Plot duration distribution
  # ---------------------------------------------------------------------------
  
  LOG.debug('Plot duration...')

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  if opt_title:
    plt.title(f'Duration Distribution for {n_videos_display:,} Videos')
  plt.ylabel("Videos")
  plt.xlabel("Duration (seconds)")

  # set bins
  bins = list(range(0,15*30, 15))

  # plot data
  x = [x//1000 for x in df['duration'].values]
  plt.hist([x], bins, label=['Seconds'])
  plt.legend(loc='upper right')

  # save
  fp_out = join(opt_output, f'{opt_prefix}_duration.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)


  # ---------------------------------------------------------------------------
  # Plot video dimension distribution
  # ---------------------------------------------------------------------------
  
  LOG.debug('Plot dimension...')

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  heights = list(df.height.values)
  widths = list(df.width.values)

  #rom sklearn.datasets.samples_generator import make_blobs
  X = np.array([np.array([w,h]) for w,h in zip(widths, heights)])

  if opt_title:
    plt.title('Video Dimension Distribution')
  plt.ylabel("Height (pixels)")
  plt.xlabel("Width (pixels)")
  plt.scatter(X[:, 0], X[:, 1], s=2)

  kmeans = KMeans(n_clusters=opt_n_clusters)
  kmeans.fit(X)
  y_kmeans = kmeans.predict(X)

  # plot kemans

  plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
  centers = kmeans.cluster_centers_
  plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
  plt.ylabel("Height")
  plt.xlabel("Width")
  if opt_title:
    plt.title(f'K-Means Clusters for {n_videos_display:,} Videos')

  # save
  fp_out = join(opt_output, f'{opt_prefix}_kmeans.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)

  
  # ---------------------------------------------------------------------------
  # Plot aspect ratios
  # ---------------------------------------------------------------------------

  LOG.debug('Plot aspect ratios...')

  cluster_ids = list(y_kmeans)
  centers = kmeans.cluster_centers_
  size_results = []
  for cluster_id in range(opt_n_clusters):
    n_found = cluster_ids.count(cluster_id)
    center = centers[cluster_id]
    dim = int(round(round(center[0])/10)*10), int(round(round(center[1])/10)*10)
    LOG.info(f'Cluster: {cluster_id}, count: {n_found:,} Dimensions: {dim}')
    o = {
      'width': dim[0],
      'height': dim[1],
      'count': n_found,
      'cluster_id': cluster_id,
      'label': f'{dim[0]}x{dim[1]}',
    }
    size_results.append(o)
  
  # sort sm -> lg
  size_results = sorted(size_results, key=itemgetter('width')) 

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  if opt_title:
    plt.title(f'Size Distribution for {n_videos_display:,} Videos')
  plt.ylabel("Videos")
  plt.xlabel("Aspect Ratio")

  x = list(range(len(size_results)))
  y = [x['count'] for x in size_results]
  labels = [x['label'] for x in size_results]

  # plot data
  plt.bar(x, y, label='Aspect Ratio')
  plt.legend(loc='upper left')
  plt.xticks(x, labels)

  # save
  fp_out = join(opt_output, f'{opt_prefix}_ratio.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)


  # ---------------------------------------------------------------------------
  # Save records to CSV
  # ---------------------------------------------------------------------------

  LOG.debug('Generate summary CSV...')
  records = []
  for size_result in size_results:
    dim = int(round(round(center[0])/10)*10), int(round(round(center[1])/10)*10)
    o = {
      'width': int(size_result['width']),
      'height': int(size_result['height']), 
      'count': size_result['count']
    }
    records.append(o)

  fp_out = join(opt_output, 'clustered_size_summary.csv')
  df_summary = pd.DataFrame.from_dict(records)
  df_summary.to_csv(fp_out, index=False)


  # ---------------------------------------------------------------------------
  # Quick Stats
  # ---------------------------------------------------------------------------

  LOG.info(f'Videos under 1 minute: {(len(df[df.duration < 60*1000]) / len(df)):.2%}')
  LOG.info(f'Videos under 1.5 minutes: {(len(df[df.duration < 90*1000]) / len(df)):.2%}')
  LOG.info(f'Videos under 2 minutes: {(len(df[df.duration < 120*1000]) / len(df)):.2%}')

  LOG.info(f'Human Work days @8h: {(df.duration.sum() / 1000 / 60 / 60 / 8):.2f}')
  LOG.info(f'Human Days @24h: {(df.duration.sum() / 1000 / 60 / 60 / 24):.2f}')
  LOG.info(f'Computer Work days @30FPS: {(df.frame_count.sum() / 30 / 60 / 60 / 24):.2f}')
  LOG.info(f'Computer Work days @60FPS: {(df.frame_count.sum() / 60 / 60 / 60 / 24):.2f}')
  LOG.info(f'Computer Work days @120FPS: {(df.frame_count.sum() / 120 / 60 / 60 / 24):.2f}')

  LOG.info(f'Frames: {df.frame_count.sum():,}')
  LOG.info(f'Hours: {(df.duration.sum() / 1000 / 60 / 60):,.2f}')


