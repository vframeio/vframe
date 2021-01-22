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
@click.option('-i', '--input', 'opt_fp_cfg', required=True,
  help='Path YAML job config')
@click.option('--skip-images', 'opt_skip_images', is_flag=True)
@click.option('--skip-labels', 'opt_skip_labels', is_flag=True)
@click.pass_context
def cli(ctx, opt_fp_cfg, opt_skip_images, opt_skip_labels):
  """YOLO PyTorch project"""

  from os.path import join
  from pathlib import Path
  import shutil

  from dataclasses import asdict
  from tqdm import tqdm
  import pandas as pd

  from vframe.settings import app_cfg
  from vframe.utils.file_utils import ensure_dir, load_yaml, write_yaml
  from vframe.utils.file_utils import write_txt, replace_ext, chmod_exec
  from vframe.utils.dataset_utils import split_train_val_test
  from vframe.models.annotation import Annotation
  from vframe.models.training_dataset import YoloPyTorch

  log = app_cfg.LOG

  # load config
  cfg = load_yaml(opt_fp_cfg, data_class=YoloPyTorch)

  # provision output
  ensure_dir(cfg.fp_output)
  dir_images = join(cfg.fp_output, cfg.fn_images)
  dir_labels = join(cfg.fp_output, cfg.fn_labels)
  ensure_dir(dir_images)
  ensure_dir(dir_labels)

  # write to yaml
  fp_out = join(cfg.fp_output, cfg.fn_hyp)
  comment = '\n'.join([app_cfg.LICENSE_HEADER,'# Hyperparameter'])
  write_yaml(asdict(cfg.hyperparameters), fp_out, comment=comment)

  # load annos
  df = pd.read_csv(cfg.fp_annotations)
  df_pos = df[df.label_index != -1]
  # df_neg = df[df.label_enum == app_cfg.LABEL_BACKGROUND or df.label_index == -1]
  df_neg = df[df.label_index == -1]

  # count
  log.info(f'positive annotations: {len(df_pos):,}')
  log.info(f'background annotations: {len(df_neg):,}')
  log.info(f'total annotations: {len(df):,}')
  log.info(f'positive images: {len(df_pos.groupby("filename")):,}')
  log.info(f'negative images: {len(df_neg.groupby("filename")):,}')
  log.info(f'total images: {len(df.groupby("filename")):,}')

  # get class-label list sorted by class index
  df_sorted = df_pos.sort_values(by='label_index', ascending=True)
  df_sorted.drop_duplicates(['label_enum'], keep='first', inplace=True)
  class_labels = df_sorted.label_enum.values.tolist()
  # write to txt
  write_txt(class_labels, join(cfg.fp_output, app_cfg.FN_LABELS))

  # update config
  cfg.classes = class_labels

  # Generate one label per file with all bboxes and classes
  # <object-class> <x_center> <y_center> <width> <height>
  labels_data = {}
  file_list = []
  df_groups = df_pos.groupby('filename')
  for fn, df_group in df_groups:
    annos = []
    file_list.append(join(dir_images, fn))
    for row_idx, row in df_group.iterrows():
      annos.append(Annotation.from_anno_series_row(row).to_darknet_str())
    labels_data.update({fn: annos})

  # write txt files for train, val
  splits = split_train_val_test(file_list, splits=cfg.splits, seed=1)
  write_txt(splits['train'], join(cfg.fp_output, cfg.fn_train))
  write_txt(splits['val'], join(cfg.fp_output, cfg.fn_val))
  write_txt(splits['test'], join(cfg.fp_output, cfg.fn_test))

  # write metadata
  fp_out = join(cfg.fp_output, cfg.fn_metadata)
  comment = '\n'.join([app_cfg.LICENSE_HEADER, '# Metadata'])
  write_yaml(cfg.to_metadata(), fp_out, comment=comment)

  # copy postive images
  if not opt_skip_labels:
    for fn, annos in tqdm(labels_data.items()):
      # write all annos for this image to txt file
      fp_label = join(dir_labels, replace_ext(fn, 'txt'))
      write_txt(annos, fp_label)

  # symlink/copy images
  if not opt_skip_images:
    df_groups = df.groupby('filename')
    for fn, df_group in tqdm(df_groups):
      fpp_im_dst = Path(join(dir_images, fn))
      fpp_im_src = Path(join(cfg.fp_images, fn))
      if not fpp_im_src.is_file():
          app_cfg.LOG.error(f'{fpp_im_dst} missing')
          continue
      if cfg.symlink:
        if fpp_im_dst.is_symlink():
          fpp_im_dst.unlink()
        fpp_im_dst.symlink_to(fpp_im_src)
      else:
        shutil.copy(fpp_im_src, fpp_im_dst)

  # write model yaml, but print k:v pairs instead of dump
  model_cfg = load_yaml(cfg.fp_model_cfg)
  fp_out = join(cfg.fp_output, cfg.fn_model_cfg)
  model_cfg['nc'] = len(cfg.classes)
  with open(fp_out, 'w') as f:
    for k,v in model_cfg.items():
     f.write(f'{k}: {v}\n')

  # shell scripts
  args = cfg.arguments
  py_cmds = ['python','train.py','']
  cli_opts = cfg.to_cli_args()
  # join strings
  sh_header_str = '\n'.join(['#!/bin/bash','','# training', ''])
  py_cmds_str = list(map(str, py_cmds))
  cli_opts_str = list(map(str, cli_opts))
  sh_script  = sh_header_str + ' '.join(py_cmds_str) + ' '.join(cli_opts_str)
  # write
  fp_sh = join(cfg.fp_output, app_cfg.FN_TRAIN_INIT)
  write_txt(sh_script, fp_sh)
  # make executable
  chmod_exec(fp_sh)

  # TODO: add tensorboard script
  # tensorboard --logdir runs/exp0 --bind_all
  if args.device and len(args.device) > 1:
    n_gpus = len(args.device)
    # multi GPU cmd
    py_cmds = ['python', '-m', 'torch.distributed.launch', '--nproc_per_node', f'{n_gpus}', 'train.py', '']
    # join strings
    sh_header_str = '\n'.join(['#!/bin/bash','','# multi gpu training', ''])
    py_cmds_str = list(map(str, py_cmds))
    cfg.arguments.batch_size *= 2
    cli_opts = cfg.to_cli_args()
    cli_opts_str = list(map(str, cli_opts))
    sh_script  = sh_header_str + ' '.join(py_cmds_str) + ' '.join(cli_opts_str)
    # write
    fp_sh = join(cfg.fp_output, app_cfg.FN_TRAIN_MULTI)
    write_txt(sh_script, fp_sh)
    # make executable
    chmod_exec(fp_sh)
