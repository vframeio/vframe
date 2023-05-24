#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

import onnxruntime as ort
import torchvision  # explicit import to avoid nms error
import torch
import numpy as np
import cv2 as cv

from vframe.image.processors.utils.yolov5_utils import letterbox, non_max_suppression
from vframe.settings.app_cfg import LOG
from vframe.models.geometry import BBox, Dimension
from vframe.image.processors.base import Detection
from vframe.models.cvmodels import DetectResult, DetectResults, PreProcImDim
from vframe.utils.file_utils import load_txt
from vframe.utils.im_utils import resize


class YOLOV5ONNX(Detection):
    def __init__(self, dnn_cfg):
        """Instantiate an ONNX DNN network model"""
        self.dnn_cfg = dnn_cfg
        cfg = self.dnn_cfg

        # model
        providers_list = ort.get_available_providers()
        # TOOD: add tensorrt provider
        if dnn_cfg.device < 0:
            providers = [("CPUExecutionProvider", {})]
        else:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": cfg.device,
                    },
                ),
            ]

        self.model = ort.InferenceSession(cfg.fp_model, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.dim_model = Dimension(*self.model.get_inputs()[0].shape[2:4][::-1])
        self.batch_size = self.model.get_inputs()[0].shape[0]


        # load class labels
        if cfg.labels_exist:
            self.labels = load_txt(cfg.fp_labels)  # line-delimited class labels
        else:
            LOG.debug(f"Labels file missing: {cfg.fp_labels}")
            self.labels = []

    def _pre_process(self, im):
        """Pre-process image
        :param im: (numpy.ndarray) in BGR
        """

        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = resize(
            im,
            width=self.dnn_cfg.width,
            height=self.dnn_cfg.height,
            interp=cv.INTER_AREA,
        )

        # TODO: fallback to user input height if model size is dynamic
        im_dst, ratio, self.pad = letterbox(
            im, new_shape=self.dim_model.wh, auto=False, scaleup=False
        )
        im_dst = np.transpose(im_dst, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
        im_dst = np.expand_dims(im_dst, axis=0)
        im_dst /= 255.0
        # image dims
        dim_orig = Dimension(*im.shape[:2][::-1])
        dim_letterbox = Dimension(self.dim_model.w, self.dim_model.w - self.pad[1] * 2)
        return PreProcImDim(im_dst, dim_orig, self.dim_model, dim_letterbox)

    def _post_process(self, outputs, ppimd):
        """Post process net output for YOLOV5 object detection.
        Post-process outputs of ONNX model inference into DetectResults
        Code based on YOLOV5 Ultralytics by Glenn Jocher
        :param outputs: ONNX model inference output
        :param ppimd: PreProcImDim
        :returns (DetectResult): of object detection resultss
        """

        detect_results = []

        detections = torch.from_numpy(np.array(outputs[0]))
        detections = non_max_suppression(
            detections,
            conf_thres=self.dnn_cfg.nms_threshold,
            iou_thres=self.dnn_cfg.iou,
            max_det=self.dnn_cfg.max_det,
        )

        # benchmark
        if detections[0] is not None:
            outputs = np.array([(d.tolist()) for d in detections[0]])

            for output in outputs:
                output = output.tolist()
                conf = output[4]
                if conf >= self.dnn_cfg.threshold:
                    xyxy = list(map(int, output[:4]))
                    bbox = (
                        BBox(*xyxy, *ppimd.dim_model.wh)
                        .translate(0, -self.pad[1])
                        .to_dim(ppimd.dim_letterbox.wh)
                        .redim(ppimd.dim_orig.wh)
                    )
                    class_idx = int(output[5])
                    LOG.debug(f'class_idx: {class_idx}')
                    label = self.labels[class_idx]
                    detect_results.append(DetectResult(class_idx, conf, bbox, label))

        return DetectResults(detect_results)

    def infer(self, im):
        """Runs pre-processor, inference, and post-processor
        :param im: (numpy.ndarry) image in BGR
        :returns (ProcessorResult)
        """
        ppimd = self._pre_process(im)
        outputs = self.model.run(None, {self.input_name: ppimd.im})
        return self._post_process(outputs, ppimd)
