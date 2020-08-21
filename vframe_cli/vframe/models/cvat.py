############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from dataclasses import dataclass, field
from typing import Dict, Tuple, List


@dataclass
class Size:
	width: int
  height: int


@dataclass
class User:
	username: str
  email: str


@dataclass
class User:
	username: str
  email: str


@dataclass
class Segments:
	id: int
  start: int
  stop: int
  url: str


@dataclass
class LabelAttributes:
	name: str
	attributes: str


@dataclass
class Label:
	label: LabelAttributes
   

@dataclass
class Task:
	id: int
  name: str
  size: str
  mode: str
  overlap: int
  bugtracker: str
  created: str
  updated: str
  start_frame: int
  stop_frame: int
  frame_filter: str
  z_order: bool
  labels: List[Label]
  segments: List[Segment]
  owner: User
  assignee: User
  original_size: Size


@dataclass
class Meta:
	task: List[Task]
	dumped: str
	source: str