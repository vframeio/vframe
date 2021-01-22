#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

from dacite import from_dict

from vframe.settings import app_cfg, modelzoo_cfg
from vframe.models.dnn import DNN
from vframe.utils import model_utils
from vframe.image.processors.base import DetectionProc, ClassificationProc
from vframe.image.processors.yolo import YOLOProc
from vframe.image.processors.ssd import SSDProc
from vframe.image.processors.retinaface_mxnet import RetinaFaceMXNetProc
from vframe.image.processors.ultralight import UltralightRetinaFaceProc
from vframe.image.processors.pose import COCOPoseFaceProc
from vframe.image.processors.yolov5_onnx import YOLOV5ONNXProc
# from vframe.image.processors.mask_rcnn import MaskRCNNProc
# from vframe.image.processors.human_pose import HumanPoseProc
# from vframe.image.processors.east import EASTProc
# conditional imports
if app_cfg.SRES_ENABLED:
    from vframe.image.processors.sres import SuperResolution

# ---------------------------------------------------------------------------
# DNN CV Model factory
# ---------------------------------------------------------------------------

class DNNFactory:

  processors = {
    'detection': DetectionProc,  # generic
    'classify': ClassificationProc,  # generic FIXME - which should we use?
    'classification': ClassificationProc,  # generic
    'ssd': SSDProc,
    'yolo': YOLOProc,
    'yolov5_onnx': YOLOV5ONNXProc,
    'retinaface_mxnet': RetinaFaceMXNetProc,
    'ultralight': UltralightRetinaFaceProc,
    'bbox_features': ClassificationProc,
    'coco_poseface': COCOPoseFaceProc,
    # 'mask_rcnn': MaskRCNNProc,
    # 'east_text': EASTProc,
  }
  # conditional processors
  if app_cfg.SRES_ENABLED:
    processors['sres'] = SuperResolution


  @classmethod
  def from_dnn_cfg(cls, dnn_cfg):
    """Creates DNN model based on configuration from ModelZoo
    :param dnn_cfg: DNN object for the model
    :returns (NetProc):
    """
    processor = cls.processors.get(dnn_cfg.processor)
    # download model if files not found
    model_utils.download_model(dnn_cfg, opt_verbose=False)
    return processor(dnn_cfg)


  @classmethod
  def from_enum(cls, enum_obj):
    """Loads DNN model based on enum name. Use from_dnn_cfg for custom props.
    :param enum_obj: enum name of model in the ModelZoo configuration YAML
    :returns (NetProc):
    """
    name = enum_obj.name.lower()
    dnn_cfg = modelzoo_cfg.modelzoo.get(name)
    return cls.from_dnn_cfg(dnn_cfg)
