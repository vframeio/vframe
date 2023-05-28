#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################


import os
from os.path import join
import math
import time
from pathlib import Path
import logging
import random

import torch
from ultralytics import YOLO

from vframe.settings.app_cfg import LOG
from vframe.models.geometry import BBox
from vframe.utils.im_utils import np2pil, create_random_im
from vframe.image.processors.base import Detection
from vframe.models.cvmodels import DetectResult, DetectResults
from vframe.utils.file_utils import load_txt
from vframe.utils.im_utils import create_random_im


class YOLOV8(Detection):
    def __init__(self, cfg):
        """Instantiate DNN network model"""
        self.cfg = cfg
        device = "cpu" if cfg.device == -1 else cfg.device
        
        self.model = YOLO(cfg.fp_model, task='detect')
        im_rand = create_random_im(cfg.width, cfg.height)
        self.args = dict(half=cfg.use_half_precision,
                    conf=cfg.threshold,
                    iou=cfg.iou,
                    device=device,
                    agnostic_nms=cfg.agnostic_nms,
                    imgsz=cfg.width,
                    verbose=False)

        # load labels
        self.labels = load_txt(cfg.fp_labels) if cfg.labels_exist else []
        if not self.labels:
            LOG.debug(f"No labels or missing file: {cfg.fp_labels}")

    def fps_batch(self, n_iters=10, dim=(960, 960), batch_size=12):
        """Benchmark model FPS on image
        :param im: (np.ndarray) image
        :pram n_iters: (int) iterations
        """
        ims = [create_random_im(*dim) for i in range(batch_size)]
        _ = self.infer(random.choice(ims))  # warmup
        st = time.perf_counter()
        for i in range(n_iters):
            _ = self.infer(ims)
        fps = batch_size * n_iters / (time.perf_counter() - st)
        return fps


    def infer(self, ims):
        """Runs pre-processor, inference, and post-processor
        This processing uses batch inference
        :param ims: List[numpy.ndarray]
        """
        if not isinstance(ims, list):
            ims = [ims]
        results = self.model(ims, **self.args)
        return self._post_process(results)


    def _post_process(self, batch_results):
        """Convert to list DetectResults"""
        batch_dets = []
        for batch_idx, batch_result in enumerate(batch_results):
          batch_n_dets = []
          yolo_results = batch_result.cpu().numpy()
          for yolo_result_idx, yolo_result in enumerate(yolo_results):
            yolo_box = yolo_result.boxes
            bbox = BBox(*yolo_box.xyxy[0], *yolo_box.orig_shape[:2][::-1])
            class_idx = int(yolo_box.cls[0])
            det = DetectResult(class_idx, yolo_box.conf[0], bbox, self.labels[class_idx])
            batch_n_dets.append(det)
          batch_dets.append(DetectResults(batch_n_dets))
        return batch_dets