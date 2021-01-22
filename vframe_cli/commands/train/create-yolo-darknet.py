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
@click.pass_context
def cli(ctx, opt_fp_cfg):
  """New YOLO Darknet project"""

  from os.path import join
  import random
  from pathlib import Path
  import operator
  import shutil

  import pandas as pd
  from tqdm import tqdm

  from vframe.utils import file_utils
  from vframe.models.annotation import Annotation

  from vframe.settings import app_cfg
  from vframe.utils.file_utils import load_yaml
  from vframe.models.training_dataset import YoloDarknet


  log = app_cfg.LOG
  log.info(f'Create YOLO project from: {opt_fp_cfg}')

  # load config file
  cfg = load_yaml(opt_fp_cfg, data_class=YoloDarknet)

  # load annotations and convert to BBox class
  df = pd.read_csv(cfg.annotations)

  # class indices must start from zero
  label_min = label_min = df.label_index.min()
  if label_min > 0:
      log.error('Label index minimum must start from zero. Remap annotations.')
      return

  df.label_index -= df.label_index.min()
  n_bg_annos = len(df[df.label_enum == 'background'])
  if n_bg_annos > 0:
    log.debug(f'Annotations contain {n_bg_annos} negative images. Removing 0th index')
    # subtract the 0th index if Background (negative data is used)
    df.label_index -= 1

  # ensure output directory
  file_utils.ensure_dir(cfg.output)

  # .sh train script
  fp_sh_train = join(cfg.output, app_cfg.FN_TRAIN_INIT)

  # .sh testing script
  fp_sh_test = join(cfg.output, app_cfg.FN_TEST_INIT)

  # .sh resume [multi] GPU script
  fp_sh_resume = join(cfg.output, app_cfg.FN_TRAIN_RESUME)

  # .data file
  fp_metadata = join(cfg.output, app_cfg.FN_META_DATA)

  # .cfg file
  fp_cfg_train = join(cfg.output, 'yolov4.cfg')
  fp_cfg_deploy = join(cfg.output, 'yolov4_deploy.cfg')

  # images/
  dir_images = join(cfg.output, cfg.images_labels)
  dir_labels = dir_images
  file_utils.ensure_dir(dir_images)

  # .data and deps filepaths
  fp_classes = join(cfg.output, app_cfg.FN_LABELS)
  fp_valid_list = join(cfg.output, app_cfg.FN_VALID)
  fp_train_list = join(cfg.output, app_cfg.FN_TRAIN)
  dir_backup = join(cfg.output, app_cfg.DN_BACKUP)
  file_utils.ensure_dir(dir_backup)

  # create dict of classes, then sort by index, 0 - N
  class_labels = {}
  for idx, record in df.iterrows():
    if record.label_enum == app_cfg.LABEL_BACKGROUND:
      continue
    if record.label_enum not in class_labels.keys():
      class_labels.update({record.label_enum: record.label_index})
  class_labels = sorted(class_labels.items(), key=operator.itemgetter(1))
  class_labels = [x[0] for x in class_labels]

  # Create training ".cfg"
  num_classes = len(class_labels)  # class in annotation file
  num_masks = 3  # assuming 3
  num_filters = (num_classes + 5) * num_masks

  # max_batches: classes*2000 but not less than number of training images
  max_batches = max(6000, max(len(df), 2000 * num_classes))
  max_batches = min(cfg.batch_ceiling, max_batches)
  # change line steps to 80% and 90% of max_batches
  batch_steps = (int(0.8 * max_batches), int(0.9 * max_batches))

  # Generate meta.data file
  data = []
  data.append(f'classes = {num_classes}')
  data.append(f'train = {fp_train_list}')
  data.append(f'valid = {fp_valid_list}')
  data.append(f'names = {fp_classes}')
  data.append('backup = {}'.format(dir_backup))
  file_utils.write_txt(data, fp_metadata)

  # Create training .cfg
  subs_all = []

  # [net]
  subs_all.append(('{width}', str(cfg.width)))
  subs_all.append(('{height}', str(cfg.height)))
  subs_all.append(('{classes}', str(num_classes)))
  subs_all.append(('{num_filters}', str(num_filters)))
  subs_all.append(('{max_batches}', str(max_batches)))
  subs_all.append(('{saturation}', str(cfg.saturation)))
  subs_all.append(('{exposure}', str(cfg.exposure)))
  subs_all.append(('{hue}', str(cfg.hue)))

  subs_all.append(('{batch_normalize}', str(int(cfg.batch_normalize))))
  subs_all.append(('{steps_min}', str(batch_steps[0])))
  subs_all.append(('{steps_max}', str(batch_steps[1])))
  subs_all.append(('{focal_loss}', f'{int(cfg.focal_loss)}'))
  subs_all.append(('{resize}', f'{cfg.resize}'))
  subs_all.append(('{learning_rate}', f'{cfg.learning_rate}'))

  # Data augmentation
  subs_all.append(('{cutmix}', f'{int(cfg.cutmix)}'))
  subs_all.append(('{mosaic}', f'{int(cfg.mosaic)}'))
  subs_all.append(('{mosaic_bound}', f'{int(cfg.mosaic_bound)}'))
  subs_all.append(('{mixup}', f'{int(cfg.mixup)}'))
  subs_all.append(('{blur}', f'{int(cfg.blur)}'))
  subs_all.append(('{flip}', f'{int(cfg.flip)}'))
  subs_all.append(('{gaussian_noise}', f'{int(cfg.gaussian_noise)}'))
  subs_all.append(('{jitter}', f'{str(cfg.jitter)}'))
  # not well tested, too experimental
  #subs_all.append(('{adversarial_lr}', f'{int(cfg.adversarial_lr)}'))
  #subs_all.append(('{attention}', f'{int(cfg.attention)}'))

  # images per class
  groups = df.groupby('label_enum')
  ipc = ','.join([str(len(groups.get_group(label))) for label in class_labels])
  subs_all.append(('{counters_per_class}', ipc))

  # batch size train
  subs_train = subs_all.copy()
  subs_train.append(('{batch_size}', str(cfg.batch_size)))
  subs_train.append(('{subdivisions}', str(cfg.subdivisions)))

  # batch size test
  subs_test = subs_all.copy()
  subs_test.append(('{batch_size}', '1'))
  subs_test.append(('{subdivisions}', '1'))

  # load original cfg into str
  cfg_orig = '\n'.join(file_utils.load_txt(cfg.cfg))

  # search and replace train
  cfg_train = cfg_orig  # str copy
  for placeholder, value in subs_train:
    app_cfg.LOG.debug(f'{placeholder}, {value}')
    cfg_train = cfg_train.replace(placeholder, value)

  # search and replace test
  cfg_test = cfg_orig # str copy
  for placeholder, value in subs_test:
    cfg_test = cfg_test.replace(placeholder, value)

  # write .cfg files
  file_utils.write_txt(cfg_train, fp_cfg_train)
  file_utils.write_txt(cfg_test, fp_cfg_deploy)

  # write train .sh
  sh_base = []
  sh_base.append('#!/bin/bash')
  sh_base.append(f'DARKNET={cfg.darknet}')
  sh_base.append(f'DIR_PROJECT={cfg.output}')
  sh_base.append(f'FP_META={fp_metadata}')
  sh_base.append('MAP="-map"')  # add mAP to chart
  if cfg.show_output:
    sh_base.append(f'VIZ=""')
  else:
    sh_base.append(f'VIZ="-dont_show"')  # don't show viz, if running in docker
  if cfg.show_images:
    sh_base.append(f'SHOW_IMGS="-show_imgs"')  # show images while training
  else:
    sh_base.append(f'SHOW_IMGS=""')

  # init training
  sh_train = sh_base.copy()
  sh_train.append(f'FP_CFG={fp_cfg_train}')
  sh_train.append(f'FP_WEIGHTS={cfg.weights}')
  sh_train.append('CMD="detector train"')
  gpus_init_str = ','.join(list(map(str, cfg.gpu_idxs_resume)))
  sh_train.append(f'GPUS="-gpus {gpus_init_str}"')
  sh_train.append(f'$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $GPUS $VIZ $MAP $SHOW_IMGS 2>&1 | tee {cfg.logfile}')
  file_utils.write_txt(sh_train, fp_sh_train)
  file_utils.chmod_exec(fp_sh_train)

  # resume training
  sh_resume = sh_base.copy()
  sh_resume.append(f'FP_CFG={fp_cfg_train}')
  cfg_name = Path(fp_cfg_train).stem
  fp_weights_last = join(dir_backup, f'{cfg_name}_last.weights')
  sh_resume.append(f'FP_WEIGHTS={fp_weights_last}')
  sh_resume.append('CMD="detector train"')
  gpus_resume_str = ','.join(list(map(str, cfg.gpu_idxs_resume)))
  sh_resume.append(f'GPUS="-gpus {gpus_resume_str}"')
  sh_resume.append(f'$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $GPUS $VIZ $MAP $SHOW_IMGS 2>&1 | tee -a {cfg.logfile}')
  file_utils.write_txt(sh_resume, fp_sh_resume)
  file_utils.chmod_exec(fp_sh_resume)

  # test
  sh_test = sh_base.copy()
  sh_test.append(f'FP_CFG={fp_cfg_deploy}')
  cfg_name = Path(fp_cfg_train).stem
  sh_test.append('# Edit path to weights')
  fp_weights_best = join(dir_backup, f'{cfg_name}_best.weights')
  sh_test.append(f'FP_WEIGHTS={fp_weights_best}')
  sh_test.append('CMD="detector test"')
  sh_test.append('$DARKNET $CMD $FP_META $FP_CFG $FP_WEIGHTS $1')
  file_utils.write_txt(sh_test, fp_sh_test)
  file_utils.chmod_exec(fp_sh_test)

  # Generate classes.txt
  file_utils.write_txt(class_labels, fp_classes)

  # Generate the labels data
  # one label per file with all bboxes and classes
  # <object-class> <x_center> <y_center> <width> <height>
  labels_data = {}
  file_list = []
  df_im_groups = df.groupby('filename')
  for fn, df_im_group in df_im_groups:
    darknet_annos = []
    file_list.append(join(dir_images, fn))
    for row_idx, row in df_im_group.iterrows():
      anno = Annotation.from_anno_series_row(row)
      #if anno.label_enum == 'background' and int(anno.label_index) == -1:
      if anno.label_enum == 'background':
        # negative data
        darknet_anno = ''  # empty entry for negative data
      else:
        darknet_anno = anno.to_darknet_str()
      darknet_annos.append(darknet_anno)
    labels_data.update({fn: darknet_annos})

  # write labels and symlink images
  for fn, darknet_annos in tqdm(labels_data.items()):
    fp_label = join(dir_labels, file_utils.replace_ext(fn, 'txt'))
    file_utils.write_txt(darknet_annos, fp_label)
    fpp_im_dst = Path(join(dir_images, fn))
    fpp_im_src = Path(join(cfg.images, fn))
    if cfg.use_symlinks:
        if fpp_im_dst.is_symlink():
          fpp_im_dst.unlink()
        fpp_im_dst.symlink_to(fpp_im_src)
    else:
      shutil.copy(fpp_im_src, fpp_im_dst)

  # Generate training list of images
  random.shuffle(file_list)
  n_train = int(0.8 * len(file_list))
  training_list = file_list[:n_train]
  validation_list = file_list[n_train:]

  # write txt files for training
  file_utils.write_txt(training_list, fp_train_list)
  # write txt files for val
  file_utils.write_txt(validation_list, fp_valid_list)
