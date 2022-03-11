#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

from dacite import from_dict

from vframe.settings.app_cfg import modelzoo
from vframe.models.dnn import DNN
from vframe.utils.model_utils import download_model
from vframe.image.processors.ssd import SSD
from vframe.image.processors.yolo_darknet import YOLODarknet
from vframe.image.processors.yolov5_onnx import YOLOV5ONNX
from vframe.image.processors.yolov5_pytorch import YOLOV5PyTorch

# ---------------------------------------------------------------------------
# DNN CV Model factory
# ---------------------------------------------------------------------------

class DNNFactory:

  processors = {
    'ssd': SSD,
    'yolo_darknet': YOLODarknet,
    'yolov5_onnx': YOLOV5ONNX,
    'yolov5_pytorch': YOLOV5PyTorch,
  }


  @classmethod
  def from_dnn_cfg(cls, dnn_cfg):
    """Creates DNN model based on configuration from ModelZoo
    :param dnn_cfg: DNN object for the model
    :returns (NetProc):
    """
    processor = cls.processors.get(dnn_cfg.processor)
    download_model(dnn_cfg, opt_verbose=False) # auto-download if not found
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