############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import logging
from dataclasses import dataclass

@dataclass
class MediaMeta:
  filename: str=''
  ext: str=''
  valid: bool=True
  width: int=0
  height: int=0
  aspect_ratio: float=0.0
  frame_count: int=1
  codec: str=''
  duration: int=0
  frame_rate: float=0.0
  created_at: str=''

  def __post_init__(self):
    if not self.aspect_ratio and self.valid:
      self.aspect_ratio = self.width / self.height


@dataclass
class KeyframeMediaMeta(MediaMeta):
  sha256: str=''


