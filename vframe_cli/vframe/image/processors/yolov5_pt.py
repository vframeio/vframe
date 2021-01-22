#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################
import os
import math
import time
from pathlib import Path

from vframe.settings import app_cfg
from vframe.models.geometry import BBox
from vframe.image.processors.base import DetectionProc
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.utils.file_utils import load_txt

# -----------------------------------------------------------------------------
# Start YoloV5
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn

class Conv(nn.Module):
  # Standard convolution
  def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
    super(Conv, self).__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
    self.bn = nn.BatchNorm2d(c2)
    self.act = nn.Hardswish() if act else nn.Identity()

  def forward(self, x):
    return self.act(self.bn(self.conv(x)))

  def fuseforward(self, x):
    return self.act(self.conv(x))

class Ensemble(nn.ModuleList):
  # Ensemble of models
  def __init__(self):
    super(Ensemble, self).__init__# Images()

  def forward(self, x, augment=False):
    y = []
    for module in self:
      y.append(module(x, augment)[0])
    # y = torch.stack(y).max(0)[0]  # max ensemble
    # y = torch.cat(y, 1)  # nms ensemble
    y = torch.stack(y).mean(0)  # mean ensemble
    return y, None  # inference, train output

# -----------------------------------------------------------------------------
# END YoloV5
# -----------------------------------------------------------------------------



class YOLOV5Proc(DetectionProc):

  def __init__(self, dnn_cfg):
    """Instantiate an DNN network model
    """
    self.log = app_cfg.LOG
    self.dnn_cfg = dnn_cfg
    cfg = self.dnn_cfg
    self.device = self.select_device(str(0))
    half = self.device.type != 'cpu'  # half precision only supported on CUDA
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    self.log.debug(cfg.fp_model)
    fp = str(Path(cfg.fp_model).parent)
    self.log.debug(fp)
    self.model = torch.hub.load(fp, cfg.model, source='local')
    #self.model = self.load_model(cfg.fp_model, map_location=self.device).autoshape()
    if half:
      self.model.half()  # to FP16
    self.model.conf = cfg.threshold
    self.model.iou = cfg.iou
    # load class labels
    if self.dnn_cfg.labels_exist:
      self.labels = load_txt(dnn_cfg.fp_labels)  # line-delimited class labels
    else:
      self.log.debug(f'Labels file missing: {dnn_cfg.fp_labels}')
      self.labels = []


  # -------------------------------------------------------------------------
  # Start copied from YoloV5
  # -------------------------------------------------------------------------

  def load_model(self, weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
      model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    # Compatibility updates
    for m in model.modules():
      if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        m.inplace = True  # pytorch 1.7.0 compatibility
      elif type(m) is Conv:
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
      return model[-1]  # return model
    else:
      print('Ensemble created with %s\n' % weights)
      for k in ['names', 'stride']:
        setattr(model, k, getattr(model[-1], k))
      return model  # return ensemble

  def autopad(self, k, p=None):  # kernel, padding
      # Pad to 'same'
      if p is None:
          p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
      return p

  def select_device(self, device='', batch_size=None):
      # device = 'cpu' or '0' or '0,1,2,3'
      self.log.debug(device)
      cpu_request = device.lower() == 'cpu'
      if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

      cuda = False if cpu_request else torch.cuda.is_available()
      if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" % (s, i, x[i].name, x[i].total_memory / c))
      else:
        print('Using CPU')

      return torch.device('cuda:0' if cuda else 'cpu')


  def make_divisible(self, x, divisor):
      # Returns x evenly divisible by divisor
      return math.ceil(x / divisor) * divisor


  def check_img_size(self, img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = self.make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        self.log.debug('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

  def non_max_suppression(self, prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
      """Performs Non-Maximum Suppression (NMS) on inference results

      Returns:
           detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
      """

      nc = prediction[0].shape[1] - 5  # number of classes
      xc = prediction[..., 4] > conf_thres  # candidates

      # Settings
      min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
      max_det = 300  # maximum number of detections per image
      time_limit = 10.0  # seconds to quit after
      redundant = True  # require redundant detections
      multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

      t = time.time()
      output = [None] * prediction.shape[0]
      for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = self.xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
          i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
          x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
          conf, j = x[:, 5:].max(1, keepdim=True)
          x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
          x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
          continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
          i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
          try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
          except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
            print(x, i, x.shape, i.shape)
            pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
          break  # time limit exceeded

      return output

  def box_iou(self, box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
      # box = 4xn
      return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


  # -------------------------------------------------------------------------
  # End copied from YoloV5
  # -------------------------------------------------------------------------

  def _pre_process(self, im):
    imgsz = self.dnn_cfg.width
    imgsz = self.check_img_size(imgsz, s=self.model.stride.max())
    self.im_pt = torch.zeros((1, 3, imgsz, imgsz), device=self.device)
    _ = self.model(im.half() if self.half else im) if self.device.type != 'cpu' else None
    self.dim = im.shape[:2][::-1]
    self.imgs = [self.im]


  def infer(self):
    """Runs pre-processor, inference, and post-processor
    """
    # TODO: enable batch inference
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    outputs = self.model(self.imgs, size=self.dnn_cfg.width)  # includes NMS
    results = self._post_process(outputs)[0]  # limit to 1 in non-batch mode
    return results


  def _post_process(self, outputs):
    """Post process net output for YOLOV5 object detection.
    Code based on YOLOV5 Ultralytics by Glenn Jocher
    """

    detect_results = []

    for output in outputs:
      output = output.tolist()
      xyxy = list(map(int, output[:4]))
      class_idx = int(output[5])
      conf = output[4]
      bbox = BBox(*xyxy, *self.dim)
      label = self.labels[class_idx]
      detect_results.append(DetectResult(class_idx, conf, bbox, label))

    return DetectResults(detect_results)
