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
@click.option('--bins', 'opt_n_bins', default=10)
@click.option('--split-years', 'opt_split_years', is_flag=True,
  help='Split years into separate plots')
@click.option('--verbose', 'opt_verbose', is_flag=True)
@click.option('--daily/--no-daily', 'opt_daily', is_flag=True, default=False,
  help='Generate daily media counts')
@click.option('--monthly/--no-monthly', 'opt_monthly', is_flag=True, default=True,
  help='Generate monthly media counts')
@click.option('--yearly/--no-yearly', 'opt_yearly', is_flag=True, default=True,
  help='Generate yearly media counts')
@click.pass_context
def cli(sink, opt_input, opt_output, opt_dpi, opt_figsize, opt_prefix,
 opt_title, opt_n_clusters, opt_n_bins, opt_split_years, opt_verbose,
 opt_daily, opt_monthly, opt_yearly):
  """Plot media attributes"""

  import os
  from os.path import join
  from operator import itemgetter
  import calendar
  from datetime import datetime

  from tqdm import tqdm
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.dates as mdates
  from matplotlib.ticker import MaxNLocator
  import numpy as np
  import pandas as pd
  from sklearn.cluster import KMeans
  
  from vframe.settings.app_cfg import LOG, MEDIA_ATTRS_DTYPES
  from vframe.utils.file_utils import ensure_dir
  from vframe.utils.draw_utils import pixels_to_figsize, set_matplotlib_style


  # create output
  ensure_dir(opt_output)

  # set styles
  set_matplotlib_style(plt)

  # read csv
  df = pd.read_csv(opt_input, dtype=MEDIA_ATTRS_DTYPES)

  # aux funcs
  def make_bins(x, n_bins, verbose=False, prefix='', dtype=np.uint16):
    interval = int((max(x) - min(x)) // n_bins)
    return list(range(int(min(x)), int(max(x)) + interval, interval))
  

  # patch csv data
  df.width.fillna(0, inplace=True)
  df.codec.fillna('', inplace=True)
  df.duration.fillna(0, inplace=True)
  df.frame_rate.fillna(0, inplace=True)
  df.duration = df.duration.astype(np.uint16) // 1000  # ms to seconds
  df['seconds'] = df.frame_count / df.frame_rate  # add seconds col

  # inits
  n_videos = len(df)
  n_videos_str = f'(total={n_videos:,})'

  # setup tqdm
  n_plots = 6
  p_bar = tqdm(range(n_plots), desc='Generating plots', leave=False)

  def inc_pbar():
    p_bar.update(1)
    p_bar.refresh()

  
  # ---------------------------------------------------------------------------
  # Plot videos per month, year
  # ---------------------------------------------------------------------------
  
  df['created_at_dt'] = pd.to_datetime(df['created_at']).dt.date
  df = df.sort_values(by=['created_at_dt'], ascending=True)

  if opt_yearly:
    df['created_at_yr'] = df['created_at_dt'].map(lambda dt: int(dt.strftime('%Y')))
    yr_min, yr_max = (df['created_at_yr'].min(), df['created_at_yr'].max())

    # sum videos per year
    years = range(yr_min, yr_max + 1)
    counts = {i:len(df[df['created_at_yr'] == i]) for i in years}

    # setup plot
    fig, ax = plt.subplots()
    figsize = pixels_to_figsize(opt_figsize, opt_dpi)
    fig.set_size_inches(figsize)

    # plot data
    x = counts.keys()
    y = counts.values()
    plt.bar(x, y, align='center', label='Videos')

    # format
    if opt_title:
      plt.title(f'Videos/Year {yr_min}-{yr_max} (total={len(df):,})')
    plt.legend(loc='upper right')
    plt.ylabel("Videos")
    plt.xlabel("Year")
    plt.xticks(years)
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # save
    fn = f"{opt_prefix}_years.png"
    fp_out = join(opt_output, fn)
    plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
    plt.savefig(fp_out, dpi=opt_dpi)
    plt.close()
  
  if opt_monthly:

    # group by year
    df['created_at_yr'] = df['created_at_dt'].map(lambda dt: int(dt.strftime('%Y')))
    df_gr_years = df.groupby('created_at_yr')
    
    # iterate years
    for g, df_gr_year in tqdm(df_gr_years, desc='Years'):
      # add column for YYYY-MM
      df_gr_year['created_at_mm_str'] = df_gr_year['created_at_dt'].map(lambda dt: dt.strftime('%m'))

      # sum videos per month
      counts = {}
      for i in range(1,13):
        counts[calendar.month_name[i]] = len(df_gr_year[df_gr_year['created_at_mm_str'] == str(i).zfill(2)])
      
      # setup plot
      fig, ax = plt.subplots()
      figsize = pixels_to_figsize(opt_figsize, opt_dpi)
      fig.set_size_inches(figsize)

      # plot data
      x = counts.keys()
      y = counts.values()
      plt.bar(x, y, align='center', label='Videos')

      # format
      if opt_title:
        plt.title(f'Videos/Month {g} (total={len(df_gr_year):,})')
      plt.legend(loc='upper right')
      plt.ylabel("Videos")
      ax.yaxis.set_major_locator(MaxNLocator(integer=True))
      plt.xlabel(f"Month ({g})")
      plt.xticks(rotation=45, ha='right')

      # save
      fn = f"{opt_prefix}_date_{str(g).replace('-','_')}.png"
      fp_out = join(opt_output, fn)
      plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
      plt.savefig(fp_out, dpi=opt_dpi)
      plt.close()


  if opt_daily:

    # group by month
    df['created_at_ym_str'] = df['created_at_dt'].map(lambda dt: dt.strftime('%Y-%m'))  
    df_gr_yms = df.groupby('created_at_ym_str')

    # iterate months
    for g, df_gr_ym in tqdm(df_gr_yms, desc='Month'):
      # group by day
      df_gr_ym['created_at_day'] = df_gr_ym['created_at_dt'].map(lambda dt: dt.strftime('%d'))
      date = datetime.strptime(g, '%Y-%m')
      date_str = date.strftime('%b %Y')
      n_days = calendar.monthrange(date.year, date.month)[1]
      n_vid_gr = len(df_gr_ym)
      month_name = calendar.month_name[date.month]
      counts = {}
      for i in range(1, n_days + 1):
        x = str(i).zfill(2)
        counts[x] = len(df_gr_ym[df_gr_ym['created_at_day'] == x])

      # setup plot
      fig, ax = plt.subplots()
      figsize = pixels_to_figsize(opt_figsize, opt_dpi)
      fig.set_size_inches(figsize)

      # plot data
      x = counts.keys()
      y = counts.values()
      plt.bar(x, y, align='center', label='Videos')

      # format
      if opt_title:
        plt.title(f'Videos/Day {date_str} (total={n_vid_gr:,})')
      plt.legend(loc='upper right')
      plt.ylabel("Videos")
      plt.xticks(rotation=45, ha='right')
      ax.yaxis.set_major_locator(MaxNLocator(integer=True))

      plt.xlabel(f"Date ({date_str})")
      # save
      fn = f"{opt_prefix}_date_{g.replace('-','_')}.png"
      fp_out = join(opt_output, fn)
      plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
      plt.savefig(fp_out, dpi=opt_dpi)
      plt.close()


  # ---------------------------------------------------------------------------
  # Plot width
  # ---------------------------------------------------------------------------

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  if opt_title:
    plt.title(f'Width Distribution {n_videos_str} (bins={opt_n_bins})')
  plt.ylabel("Videos")
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  plt.xlabel("Width (pixels)")

  # set bins
  x = df.width.values.tolist()
  bins = make_bins(x, opt_n_bins, verbose=opt_verbose, prefix='width')

  # plot data
  plt.hist([x], bins, label=['Videos'], align='left')
  plt.legend(loc='upper right')
  plt.xticks(bins)

  # save
  fp_out = join(opt_output, f'{opt_prefix}_width.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)
  plt.close()
  inc_pbar()

  # ---------------------------------------------------------------------------
  # Plot height
  # ---------------------------------------------------------------------------

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  if opt_title:
    plt.title(f'Height Distribution {n_videos_str} (bins={opt_n_bins})')
  plt.ylabel("Videos")
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  plt.xlabel("Height (pixels)")

  # set bins
  x = df.height.values.tolist()
  bins = make_bins(x, opt_n_bins, verbose=opt_verbose, prefix='height')

  # plot data
  x = df['height'].values
  plt.hist([x], bins, label=['Videos'], align='left')
  plt.xticks(bins)
  plt.legend(loc='upper right')

  # save
  fp_out = join(opt_output, f'{opt_prefix}_height.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)
  plt.close()
  inc_pbar()


  # ---------------------------------------------------------------------------
  # Plot FPS
  # ---------------------------------------------------------------------------

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  if opt_title:
    plt.title(f'Frames Per Second Distribution {n_videos_str} (bins={opt_n_bins})')
  plt.ylabel("Video")
  plt.xlabel("Frames Per Second")

  # set bins
  x = df['frame_rate'].values
  bins = make_bins(x, opt_n_bins, verbose=opt_verbose, prefix='frame rate', dtype=np.float64)

  # plot data
  x = df['frame_rate'].values
  plt.hist([x], bins, label=['Videos'], align='left')
  plt.legend(loc='upper right')
  plt.xticks(bins)

  # save
  fp_out = join(opt_output, f'{opt_prefix}_fps.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)
  plt.close()
  inc_pbar()


  # ---------------------------------------------------------------------------
  # Plot duration distribution
  # ---------------------------------------------------------------------------

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  if opt_title:
    plt.title(f'Duration Distribution {n_videos_str} (bins={opt_n_bins})')
  plt.ylabel("Videos")
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
  plt.xlabel("Duration (seconds)")

  x = df.seconds.values.tolist()
  bins = make_bins(x, opt_n_bins, verbose=opt_verbose, prefix='duration')

  # plot data
  plt.hist([x], bins, label=['Videos'], align='left')
  plt.legend(loc='upper right')

  # save
  fp_out = join(opt_output, f'{opt_prefix}_duration.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)
  plt.close()
  inc_pbar()

  # ---------------------------------------------------------------------------
  # Plot video dimension distribution
  # ---------------------------------------------------------------------------

  # setup plot
  fig, ax = plt.subplots()
  figsize = pixels_to_figsize(opt_figsize, opt_dpi)
  fig.set_size_inches(figsize)

  heights = list(df.height.values)
  widths = list(df.width.values)

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
    plt.title(f'K-Means Clusters {n_videos_str} (k={opt_n_clusters})')

  # save
  fp_out = join(opt_output, f'{opt_prefix}_kmeans.png')
  plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
  plt.savefig(fp_out, dpi=opt_dpi)
  plt.close()
  inc_pbar()

  
  # ---------------------------------------------------------------------------
  # Plot aspect ratios
  # ---------------------------------------------------------------------------

  cluster_ids = list(y_kmeans)
  centers = kmeans.cluster_centers_
  size_results = []
  for cluster_id in range(opt_n_clusters):
    n_found = cluster_ids.count(cluster_id)
    center = centers[cluster_id]
    dim = int(round(round(center[0])/10)*10), int(round(round(center[1])/10)*10)
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
    plt.title(f'Size Distribution {n_videos_str} (k={opt_n_clusters})')
  plt.ylabel("Videos")
  ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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
  plt.close()
  inc_pbar()


  # ---------------------------------------------------------------------------
  # Save records to CSV
  # ---------------------------------------------------------------------------

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

  if opt_verbose:
    LOG.info(f'Videos under 1 minute: {(len(df[df.seconds <= 60]) / len(df)):.2%}')
    LOG.info(f'Videos under 2 minutes: {(len(df[df.seconds <= 120]) / len(df)):.2%}')
    LOG.info(f'Videos under 4 minutes: {(len(df[df.seconds <= 240]) / len(df)):.2%}')

    LOG.info(f'Human Work days @8h: {(df.seconds.sum() / 60 / 60 / 8):.2f}')
    LOG.info(f'Human Days @24h: {(df.seconds.sum() / 60 / 60 / 24):.2f}')

    LOG.info(f'Computer Work days @30FPS: {(df.frame_count.sum() / 30 / 60 / 60 / 24):.2f}')
    LOG.info(f'Computer Work days @60FPS: {(df.frame_count.sum() / 60 / 60 / 60 / 24):.2f}')
    LOG.info(f'Computer Work days @120FPS: {(df.frame_count.sum() / 120 / 60 / 60 / 24):.2f}')

    LOG.info(f'Frames: {df.frame_count.sum():,}')
    LOG.info(f'Hours: {(df.duration.sum() / 60 / 60):,.2f}')