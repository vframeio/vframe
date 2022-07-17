############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict
from enum import Enum
from functools import lru_cache
import time
import traceback
from random import shuffle
from os.path import join

import cv2 as cv
from PIL import Image
from dataclasses import asdict
import dacite
from tqdm import tqdm
import numpy as np

from vframe.settings import app_cfg
from vframe.settings.app_cfg import DN_REAL
from vframe.settings.app_cfg import LOG
from vframe.models.types import MediaType, FrameImage
from vframe.models.cvmodels import ClassifyResults, DetectResults, ProcessedFile, FileMeta
from vframe.utils.file_utils import glob_multi, load_json, load_txt, get_ext, date_modified
from vframe.utils.file_utils import get_sha256
from vframe.utils.video_utils import FileVideoStream
from vframe.models.geometry import BBox, Point


# -----------------------------------------------------------------------------
#
# MediaFileReader
# represents a collection of media files
#
# -----------------------------------------------------------------------------
@dataclass
class MediaFileReader:
  """Video processing pipeline that instantiates new PipeFile for each file
  """
  filepath: str  # file or directory
  exts: Tuple[str, ...]  # eg mp4, ignored if single file
  slice_idxs: Tuple[int, int]  # 
  recursive: bool=False  # recursively glob filepath
  ext_upper: bool=True  # glob lower and upper exts
  use_prehash: bool=False
  use_draw_frame: bool=False
  skip_if_expression: str=None
  media_filters: List=field(default_factory=lambda:[])
  skip_all_frames: bool=False
  opt_check_exist: bool=False
  opt_randomize: bool=False
  opt_new_filepath: str=''  # override item filepath in priors

  def __post_init__(self):
    fp = self.filepath
    self.index = 0
    self._n_frames_processed = []
    self._files = []

    # JSON: pre-processed files
    if get_ext(fp).lower() == 'json':
      
      # load 
      items = load_json(fp)

      # slice
      idx_from = max(0, self.slice_idxs[0])
      idx_to = len(items) if self.slice_idxs[1] == -1 else self.slice_idxs[1]
      items = items[idx_from:idx_to]

      # init MediaFiles
      # TODO: make multithreaded
      for item in tqdm(items, desc='Initializing files'):

        if self.opt_new_filepath:
          # replace filepath
          fp = item['file_meta']['filepath']
          item['file_meta']['filepath'] = join(self.opt_new_filepath, Path(fp).name)

        pf = dacite.from_dict(data=item, data_class=ProcessedFile)

        if self.opt_check_exist and not Path(pf.file_meta.filepath).is_file():
          continue

        # filter media
        skip_results = []
        for media_filter in self.media_filters:
          val = getattr(pf.file_meta, media_filter.attribute)
          skip_results.append(media_filter.evaulate(val))

        if not any(skip_results):
          self._files.append(MediaFile.from_processed_file(pf))

      # randomize if optioned
      if self.opt_randomize:
        shuffle(self._files)

      # # slice
      # if len(self._files) > 0:
      #   idx_from = max(0, self.slice_idxs[0])
      #   idx_to = len(self._files) if self.slice_idxs[1] == -1 else self.slice_idxs[1]
      #   self._files = self._files[idx_from:idx_to]

    # TXT: filelist
    elif get_ext(fp).lower() == 'txt':

      # load list, removing empty
      self._files = load_txt(fp)
      if self.opt_check_exist:
        self._files = [f for f in tqdm(self._files, desc='Checking files') if Path(f).is_file()]
      else:
        self._files = [f for f in self._files if f]
      
      # randomize if optioned
      if self.opt_randomize:
        shuffle(self._files)

      # slice
      if len(self._files) > 0:
        idx_from = max(0, self.slice_idxs[0])
        idx_to = len(self._files) if self.slice_idxs[1] == -1 else self.slice_idxs[1]
        self._files = self._files[idx_from:idx_to]

      # init MediaFiles
      self._files = [MediaFile(fp) for fp in self._files]

    # Files and directories
    else:

      # Directory
      if Path(fp).is_dir():
        self._files = glob_multi(fp, exts=self.exts, recursive=self.recursive, sort=True, multi_case=self.ext_upper)

      # File
      elif Path(fp).is_file() and get_ext(fp).lower() in app_cfg.VALID_PIPE_MEDIA_EXTS:
        self._files = [fp]  # single media file as list

      # Missing
      elif not Path(fp).exists():
        LOG.error(f'File {fp} does not exist')

      # randomize if optioned
      if self.opt_randomize:
        shuffle(self._files)

      # slice
      if len(self._files) > 0:
        idx_from = max(0, self.slice_idxs[0])
        idx_to = len(self._files) if self.slice_idxs[1] == -1 else self.slice_idxs[1]
        self._files = self._files[idx_from:idx_to]
        # create empty metadata placeholders

      # init MediaFiles
      self._files = [MediaFile(fp) for fp in self._files]

  

  def split_into_threads(self, n):
    return np.array_split(np.array(self._files), n)


  def iter_files(self):
    """Generator yielding next MediaFile
    """
    # init
    self._start_time = time.perf_counter()
    
    if not len(self._files):
      return
    
    # iter
    for i in range(len(self._files)):
      self.index = i
      
      if i > 0:
        self._n_frames_processed.append(self._files[i - 1].n_frames)
        self._files[i - 1].unload()  # unload previous

      self._files[i].load(use_draw_frame=self.use_draw_frame, 
        use_prehash=self.use_prehash,
        skip_all_frames=self.skip_all_frames)
      yield self._files[i]
    
    # post iter
    self._n_frames_processed.append(self._files[i].n_frames)
    self._files[i].unload()


  def to_dict(self):
    return [mf.to_dict() for mf in self.files]


  @property
  def n_files(self):
    """Number of files
    """
    return len(self._files)

  
  @property
  def is_last_item(self):
    return self.index >= self.n_files -1


  @property
  def stats(self):
    et = time.perf_counter() - self._start_time
    nf = sum(self._n_frames_processed)
    fps = nf / et
    fpm = (self.index + 1) / ((time.perf_counter() - self._start_time) / 60)
    t = f'Files: {self.n_files:,}. '
    t += f'Frames: {nf:,}. '
    t += f'Files/min: {fpm:.2f}. '
    t += f'FPS: {fps:.2f}. '
    t += (f'Time: {time.strftime("%H:%M:%S", time.gmtime(et))}. ')
    t += f'Seconds: {et:.6f}'
    return t
  
  


# ---------------------------------------------------------------------------
#
# MediaFile 
# represents an image or video and its frames
#
# ---------------------------------------------------------------------------

@dataclass
class MediaFile:
  """Image or video file processed by an ImagePipe
  """
  filepath: str
  # metadata_priors: Dict[DetectResults]=field(default_factory=lambda: {})
  metadata_priors: Dict=field(default_factory=lambda: {})
  file_meta: FileMeta=None
  skip_all_frames: bool=False
  use_sha256: bool=False


  def __post_init__(self):
    self.metadata = self.metadata_priors or {}
    self.images = {}  # stores image data
    self._datetime = None  # TODO: move to file_meta
    if self.ext in app_cfg.VALID_PIPE_IMAGE_EXTS:
      self.type = MediaType.IMAGE
    elif self.ext in app_cfg.VALID_PIPE_VIDEO_EXTS:
      self.type = MediaType.VIDEO
    self.vstream = None
    self._skip_file = False



  @classmethod
  def from_processed_file(cls, pf):
    return cls(pf.file_meta.filepath, pf.detections, pf.file_meta)


  def load(self, use_draw_frame=False, use_prehash=False, skip_all_frames=False):
    """Loads the media file
    """
    self.use_prehash = use_prehash
    self.use_draw_frame = use_draw_frame
    if self.file_meta and (self._skip_file or skip_all_frames):
      self.vstream = None
      return

    try:
      self.vstream = FileVideoStream(self.filepath, use_prehash=use_prehash)
      
      self.frame_idx_end = self.n_frames - 1
      self.images = {}

      if not self.vstream.start():
        self._skip_file = True
        return False
      else:
        self.processed_fps = 0

    except Exception as e:
      LOG.error(f'Could not load file: {self.filepath}. Error: {e}')
      # LOG.error(traceback.format_exc())
      self.vstream = None
      self._skip_file = True
      self.skip_all_frames = True
      return False


  def unload(self):
    """Unloads media and delete image/frame vars. Called on media file release
    """

    if self.vstream is not None:
      self.vstream.stop()
      self.vstream.release()
    del self.images
    del self.metadata

  def set_skip_file(self):
    self._skip_file = True
    self._n_frames_processed = 0


  def to_dict(self):
    """Serializes metadata for export to JSON
    """
    frames_meta = {}
    for frame_idx, frame_meta in self.metadata.items():
      meta = {k:v.to_dict() for k,v in frame_meta.items() if v.detections}
      if meta:
        frames_meta[frame_idx] = meta

    # only get date if requested
    return {
      'file_meta': {
        'filepath': self.filepath,
        'width': self.width,
        'height': self.height,
        'frame_count': self.n_frames,
        'datetime': str(self.datetime),
        'sha256': self.sha256,
      },
      'frames_meta': frames_meta
      }


  def iter_frames(self):
    """Generator yielding next frame
    """

    # init
    self._start_time = time.perf_counter()

    # skip frames for dummy processing
    if self.skip_all_frames or self._skip_file:
      LOG.debug(f'skip file break iter: {self.filename}, {self.n_frames}')
      self.metadata = {}
      yield
      return

    # iter
    for index in range(self.n_frames):
      # check if video stopped running before iterating all frames
      if not self.vstream.running():
        LOG.warn(f'Corrupt video: {self.filepath}. Exited at frame: {index}/{self.n_frames}. No data saved.')
        self._skip_file = True
        self.metadata = {}
        yield
        return
      
      if self.use_prehash:
        im, self.phash = self.vstream.read_frame_phash()
      else:
        im = self.vstream.read_frame()

      self.images = {}  # stores image data
      self.metadata.setdefault(index, {})  # blank if no precomputed meta
      self.images[FrameImage.ORIGINAL] = im
      if self.use_draw_frame:
        self.images[FrameImage.DRAW] = im.copy()

      yield True

    # post iter
    self.processed_fps = (self.n_frames + 1) / (time.perf_counter() - self._start_time)



  def inherit_from_last_frame(self, opt_data_key):
    if self.index > 0:
      return self.metadata.get(self.index -1).get(opt_data_key)


  @property
  def detected_labels(self):
    # Get list of all labels detected
    labels = []
    for i in range(self.n_frames):
      for k, dr in self.metadata.get(i).items():
        labels.extend([d.label for d in dr.detections])
    return labels


  def includes_labels(self, labels, opt_all=False):
    """Returns True if any frame does not contain any labels
    """
    if opt_all:
      return all([label in self.detected_labels for label in labels])
    else:
      return any([label in self.detected_labels for label in labels])
    

  def excludes_labels(self, labels, opt_all=False):
    """Returns True if any frame does not contain any labels
    """
    if opt_all:
      return not all([label in self.detected_labels for label in labels])
    else:
      return not any([label in self.detected_labels for label in labels])

  # def remove_detections(self, labels=None, thresholds=None):
  #   if self.metadata:
  #     for k, dr in self.metadata.get(self.index).items():
  #       dets = dr.detections
  #       if threshold:
  #         dets = [x for x in dets if x.conf >= threshold]
  #       if labels:
  #         dets = [x for x in dets if x.label in labels]


  def frame_detections_exist(self, labels=None, thresholds=None):
    """Returns True if any frame contains any detection
    """
    n = self.n_detections_filtered(labels=labels, thresholds=thresholds)
    return n > 0


  def n_detections_filtered(self, labels=None, thresholds=None, index=None):
    """Returns True if any frame contains any detection at current index
    """
    index = index or self.index
    d = []
    if self.metadata:
      for k, dr in self.metadata.get(index).items():
        d = dr.detections
        if thresholds:
          d = [x for x in d if x.conf >= thresholds[0] and x.conf <= thresholds[1]]
        if labels:
          d = [x for x in d if x.label in labels]

    return len(d)

  def n_detections_filtered_total(self, labels=None, thresholds=None):

    n = 0
    for i in range(self.n_frames):
      if i in self.metadata.keys():
        n += self.n_detections_filtered(labels=labels, thresholds=thresholds, index=i)
    return n

  @property
  def n_detections(self):
    """Returns True if any frame contains any detection
    """

    if self.metadata:
      return sum([len(v.detections) for k,v in self.metadata.get(self.index).items()])
    else:
      return 0


  @property
  def file_detections_exist(self):
    """Returns True if any frame contains any detection
    """
    for i in range(self.n_frames):
      for k, dr in self.metadata.get(i).items():
        if len(dr.detections):
          return True
    return False
  

  @property
  def parent(self):
    return Path(self.filepath).parent

  @property
  def parentname(self):
    return Path(self.filepath).parent.name

  @property
  def filename(self):
    return Path(self.filepath).name

  @property
  def filestem(self):
    return Path(self.filepath).stem

  @property
  def fn(self):
    return self.filename

  @property
  def ext(self):
    return get_ext(self.filename)
  
  @property
  def index(self):
    if self.vstream:
      return self.vstream.index
    else:
      return 0

  @property
  def width(self):
    if self.file_meta:
      return self.file_meta.width
    elif self.vstream is not None:
      return self.vstream.width
    else:
      LOG.warn('media width not initialized')
      return 0

  @property
  def height(self):
    if self.file_meta:
      return self.file_meta.height
    elif self.vstream is not None:
      return self.vstream.height
    else:
      LOG.warn('media height not initialized')
      return 0
  
  @property
  def n_frames(self):
    if self.file_meta:
      return self.file_meta.frame_count
    elif self.vstream is not None:
      return self.vstream.frame_count
    else:
      return 0

  @property
  def frame_count(self):
    return self.n_frames

  @property
  def datetime(self):
    if self.file_meta:
      return self.file_meta.datetime
    else:
      return date_modified(self.filepath)  # datetime format

  @property
  def sha256(self):
    if self.use_sha256 and self.file_meta and self.file_meta.sha256:
      return self.file_meta.sha256
    elif self.use_sha256:
      return get_sha256(self.filepath)
    else:
      return ''

  @property
  def date(self):
    return self.datetime.date()

  @property
  def year(self):
    return self.datetime.year

  @property
  def is_last_item(self):
    return any([self.skip_all_frames, (self.index >= self.n_frames - 1)])

  @property
  def is_first_item(self):
    return any([self.skip_all_frames, (self.index == 1)])

  @property
  def fps(self):
    return self.vstream.fps
  
  

# ---------------------------------------------------------------------------
#
# Used for media summary analysis
#
# ---------------------------------------------------------------------------

@dataclass
class MediaMeta:
  filename: str=''
  ext: str=''
  valid: bool=True
  width: int=0
  height: int=0
  aspect_ratio: float=0.0
  frame_count: int=0
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
