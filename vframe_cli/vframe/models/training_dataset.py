#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################
from os.path import join
from dataclasses import dataclass, field, asdict
from typing import List

from vframe.settings import app_cfg

# -----------------------------------------------------------------------------
# YOLO V5 PyTorch project
# -----------------------------------------------------------------------------
@dataclass
class HyperParameters:
  # hyperparameters (default from ultralytics)
  lr0: float=0.0032
  lrf: float=0.12
  momentum: float=0.843
  weight_decay: float=0.00036
  warmup_epochs: float=2.0
  warmup_momentum: float=0.5
  warmup_bias_lr: float=0.05
  box: float=0.0296
  cls: float=0.243
  cls_pw: float=0.631
  obj: float=0.301
  obj_pw: float=0.911
  iou_t: float=0.2
  anchor_t: float=2.91
  # anchors: float=3.63
  fl_gamma: float=0.0
  hsv_h: float=0.0138
  hsv_s: float=0.664
  hsv_v: float=0.464
  degrees: float=0.373
  translate: float=0.245
  scale: float=0.898
  shear: float=0.602
  perspective: float=0.0
  flipud: float=0.00856
  fliplr: float=0.5
  mosaic: float=1.0
  mixup: float=0.243

@dataclass
class YoloPyTorchArgs:
  # transfer weights relative to yolov4/weights directory
  weights: str
  # CLI opts
  epochs: int=300
  # total batch size for all GPUs
  batch_size: int=16
  # [train, test] image sizes
  img_size_train: int=640
  img_size_test: int=640
  # rectangular training
  rect: bool=False
  # resume most recent training
  resume: bool=False
  # only save final checkpoint
  no_save: bool=False
  # only test final epoch
  no_test: bool=False
  # disable autoanchor check
  no_autoanchor: bool=False
  # evolve hyperparameters
  evolve: bool=False
  # gsutil bucket
  bucket: str=''
  # cache images for faster training
  cache_images: bool=False
  # use weighted image selection for training
  image_weights: bool=False
  # renames experiment folder exp{N} to exp{N}_{name} if supplied
  name: str=''
  # cuda device, i.e. 0 or 0,1,2,3 or cpu
  device: List = field(default_factory=lambda: [0])
  # vary img-size +/- 50%
  multi_scale: bool=False
  # train as single-class dataset
  single_cls: bool=False
  # use torch.optim.Adam() optimizer
  adam: bool=False
  # use SyncBatchNorm, only available in DDP mode
  sync_bn: bool=False
  # DDP parameter, do not modify
  local_rank: int=-1
  # logging directory
  project: str=''
  # number of images for W&B logging, max 100
  log_imgs: int=10
  # maximum number of dataloader workers
  workers: int=8

  def to_dict(self):
    d = asdict(self)
    # replace vars with "-" in name to "_
    d['multi-scale'] = d.pop('multi_scale')
    d['single-cls'] = d.pop('single_cls')
    d['sync-bn'] = d.pop('sync_bn')
    d['image-size'] = d.pop('image_size')
    d['batch-size'] = d.pop('batch_size')
    d['image-weights'] = d.pop('image_weights')
    d['log-imgs'] = d.pop('log_imgs')
    d['cache-images'] = d.pop('cache_images')
    return d


@dataclass
class YoloPyTorch:
  # input
  fp_annotations: str  # path to annotations csv
  fp_images: str  # path to all images
  fp_model_cfg: str  # path to base model file
  # cli args
  arguments: YoloPyTorchArgs
  hyperparameters: HyperParameters
  # output
  fp_output: str
  fn_train: str='train.txt'
  fn_val: str='val.txt'
  fn_test: str='test.txt'
  fn_hyp: str='hyp.yaml'
  fn_metadata: str='metadata.yaml'
  fn_model_cfg: str='model.yaml'
  fn_images: str='images'
  fn_labels: str='labels'
  symlink: bool=True
  download: str=''
  n_classes: int=0
  classes: List = field(default_factory=lambda: [])
  splits: List = field(default_factory=lambda: [0.6, 0.2, 0.2])

  def __post_init__(self):
    if not self.arguments.project:
      self.arguments.project = join(self.fp_output, 'runs')

  def set_classes(self, classes):
    self.classes = classes

  def to_metadata(self):
    d = {
      'download': self.download,
      'train': join(self.fp_output, self.fn_train),
      'val': join(self.fp_output, self.fn_val),
      'test': join(self.fp_output, self.fn_test),
      'nc': len(self.classes),
      'names': self.classes,
    }
    return d

  def to_cli_args(self):
    args = self.arguments
    opts = []
    opts.extend(['--weights', args.weights])
    opts.extend(['--cfg', join(self.fp_output, self.fn_model_cfg)])
    opts.extend(['--data', join(self.fp_output, self.fn_metadata)])
    opts.extend(['--hyp', join(self.fp_output, self.fn_hyp)])
    opts.extend(['--epochs', str(args.epochs)])
    opts.extend(['--batch', str(args.batch_size)])
    opts.extend(['--img-size', str(args.img_size_train)])
    if args.rect:
      opts.extend(['--rect', args.rect])
    if args.resume:
      opts.extend(['--resume', args.resume])
    if args.no_save:
      opts.extend(['--nosave'])
    if args.no_test:
      opts.extend(['--notest'])
    if args.no_autoanchor:
      opts.extend(['--noautoanchor'])
    if args.evolve:
      opts.extend(['--evolve'])
    opts.extend(['--bucket', args.local_rank])
    if args.cache_images:
      opts.extend(['--cache-images'])
    if args.image_weights:
      opts.extend(['--image-weights'])
    if args.name:
      opts.extend(['--name', args.name])
    if args.device:
      device_str = [str(x) for x in args.device]
      opts.extend(['--device', str(','.join(device_str))])
    if args.multi_scale:
      opts.extend(['--multi-scale'])
    if args.adam:
      opts.extend(['--adam'])
    if args.single_cls:
      opts.extend(['--single-cls'])
    if args.sync_bn:
      opts.extend(['--sync-bn'])
    opts.extend(['--local_rank', args.local_rank])
    opts.extend(['--project', args.project])
    opts.extend(['--log-imgs', args.log_imgs])
    opts.extend(['--workers', args.workers])
    return opts


# -----------------------------------------------------------------------------
# Darknet project
# -----------------------------------------------------------------------------
@dataclass
class YoloDarknet:
  # YOLO file i/o
  annotations: str
  output: str
  images: str
  cfg: str
  weights: str
  darknet: str=app_cfg.FP_DARKNET_BIN
  logfile: str=app_cfg.FN_LOGFILE
  show_output: bool=False
  show_images: bool=True
  gpu_idx_init: List = field(default_factory=lambda: [0])
  gpu_idxs_resume: List = field(default_factory=lambda: [0])
  labels: str=app_cfg.FN_LABELS
  images_labels: str=app_cfg.DN_IMAGES_LABELS
  use_symlinks: bool=True

  # YOLO network config
  subdivisions: int=16
  batch_size: int=64
  batch_normalize: bool=True
  width: int=608
  height: int=608
  focal_loss: bool=False
  learning_rate: float=0.001  # yolov4
  batch_ceiling: int=50000  # total max batches, overrides suggested values

  # Data augmentation
  flip: bool=True
  resize: float=1.0
  jitter: float=0.3
  exposure: float=1.5
  saturation: float=1.5
  hue: float=0.1
  cutmix: bool=False
  mosaic: bool=False
  mosaic_bound: bool=False
  mixup: bool=False
  blur: bool=False
  gaussian_noise: int=0

  def __post_init__(self):
    #learning_rate = 0.00261 / GPUs
    #self.learning_rate = self.learning_rate / len(self.gpu_idxs_resume)
    # force mosaic bound false if not using mosaic augmentation
    self.mosaic_bound = False if not self.mosaic else self.mosaic_bound
