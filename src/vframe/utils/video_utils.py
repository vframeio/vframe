############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from pathlib import Path
from datetime import datetime
import time
from threading import Thread
from queue import Queue

from pymediainfo import MediaInfo
from PIL import Image
from PIL import ImageFile
import cv2 as cv
import dacite

from vframe.settings import app_cfg
from vframe.settings.app_cfg import LOG
from vframe.models.mediameta import MediaMeta
from vframe.utils import file_utils
from vframe.utils.im_utils import pil2np, np2pil, resize, phash



# --------------------------------------------------------------
# based on code from jrosebr1 (PyImageSearch)
# from imutils.video
# https://raw.githubusercontent.com/jrosebr1/imutils/master/imutils/video/filevideostream.py
# --------------------------------------------------------------

"""
TODO
- improve error handling on empty container MP4s by inspecting video properties
- improve image/video separation

"""

class FileVideoStream:

  frame_count = 0
  frame_read_index = 0
  height = 0
  width = 0

  def __init__(self, fp, queue_size=512, use_prehash=False):
    """Threaded video reader
    """
    # TODO: cv.CAP_FFMPEG, cv.CAP_GSTREAMER
    # self.vcap = cv.VideoCapture(str(fp), cv.CAP_FFMPEG)

    # override PIL error if images are slightly corrupted
    # TODO: verify integrity of images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    self.is_image = Path(fp).suffix[1:].lower() in ['jpg', 'png']
    self.use_prehash = use_prehash
    # TODO: explore further. currently not working
    # self.vcap.set(cv.CAP_PROP_HW_ACCELERATION, 0.0)
    # self.vcap.set(cv.CAP_PROP_HW_DEVICE, 0.0)
    # LOG.debug(f'CAP_PROP_HW_ACCELERATION: {self.vcap.get(cv.CAP_PROP_HW_ACCELERATION)}')
    # LOG.debug(f'CAP_PROP_HW_DEVICE: {self.vcap.get(cv.CAP_PROP_HW_DEVICE)}')
    # LOG.debug(f'Using backend:: {self.vcap.getBackendName()}')
    # self.vcap.set(cv.CAP_PROP_BUFFERSIZE, 1024*20)

    if self.is_image:
      self.fps = 25.0  # default 25.0 for still image
      im = Image.open(fp)
      self.width, self.height = im.size
      self.dim = (self.width, self.height)
      self.index = -1
      self.stopped = True
      self.frame_count = 1
      self.queue = Queue(maxsize=1)
      self.queue.put(im)

    else:
      self.vcap = cv.VideoCapture(str(fp), cv.CAP_FFMPEG)
      self.queue = Queue(maxsize=queue_size)
      try:
        self.height = int(self.vcap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.vcap.get(cv.CAP_PROP_FRAME_WIDTH))
        if file_utils.get_ext(fp).lower() in app_cfg.VALID_PIPE_IMAGE_EXTS:
          self.frame_count = 1  # force set image to 1 frame
        else:  
          self.frame_count = int(self.vcap.get(cv.CAP_PROP_FRAME_COUNT))
          # Bug: the frame count is not always correct
          # TODO: set a variable to perform this check
          #   calculate overhead
          check_idx = self.frame_count
          while check_idx > 0:
            # seek to frame and check if valid
            self.vcap.set(cv.CAP_PROP_POS_FRAMES, check_idx - 1)
            frame_ok, vframe = self.vcap.read()
            if not frame_ok:
              check_idx -= 1  # rewind
            else:
              break
          if check_idx != self.frame_count:
            LOG.warn(f'Partial video: rewinding from {self.frame_count} to: {check_idx}')
          self.frame_count = check_idx

        # rewind
        self.vcap.set(cv.CAP_PROP_POS_FRAMES, 0)
        self.vcap_cc = self.vcap.get(cv.CAP_PROP_FOURCC)  
        self.fps = self.vcap.get(cv.CAP_PROP_FPS)  # default 25.0 for still image
        self.stopped = False
        self.index = -1
        # initialize queue used to store frames
        if self.use_prehash:
          self.queue_phash = Queue(maxsize=queue_size)
        # initialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.spf = 1 / self.fps if self.fps else 0  # seconds per frame
        self.mspf = self.spf * 1000  # milliseconds per frame
      except Exception as e:
        # TODO: add error logging
        LOG.error(f'Skipping file: {fp}. Error: {e}. FPS: {self.fps}')

    self.dim = (self.width, self.height)


  def start(self):
    # start a thread to read frames from the file video stream
    if self.width > 0 and self.height > 0 \
      and self.frame_count > 0 \
      and self.fps > 0:
      if not self.is_image:
        self.thread.start()
      return self
    else:
      return None


  def update(self):
    # keep looping infinitely
    while True:
      if self.stopped:
        break
      if not self.queue.full():
        (frame_ok, frame) = self.vcap.read()
        if not frame_ok:
          self.stopped = True
          break
        else:
          # frame
          self.frame_read_index += 1
          self.queue.put(frame)
          # add phash
          if self.use_prehash:
            h = phash(frame)
            self.queue_phash.put(h)
      else:
        time.sleep(0.1)

    if not self.is_image:
      self.vcap.release()


  def release(self):
    if self.frame_count > 0:
      del self.queue
      if self.use_prehash:
        del self.queue_phash
      if not self.is_image:
        self.vcap.release()


  def read_frame(self):
    # return next frame in the queue
    self.index += 1
    if self.is_image:
      return pil2np(self.queue.get())
    else:
      return self.queue.get()


  def read_frame_phash(self):
    # return next frame in the queue
    self.index += 1
    if self.is_image:
      return (pil2np(self.queue.get()), self.queue_phash.get())
    else:
      return (self.queue.get(), self.queue_phash.get())


  # check if all available video frames read
  def running(self):
    return self.more() or not self.stopped


  def more(self):
    # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
    tries = 0
    while self.queue.qsize() == 0 and not self.stopped and tries < 5:
      time.sleep(0.1)
      tries += 1

    return self.queue.qsize() > 0


  def stop(self):
    if self.frame_count > 0:
      # indicate that the thread should be stopped
      self.stopped = True
      # wait until stream resources are released (producer thread might be still grabbing frame)
      if not self.is_image:
        self.thread.join()





def mediainfo(fp_in):
  """Returns abbreviated video/audio metadata for video files
  :param fp_in: filepath"""
  
  # extension and media type
  ext = file_utils.get_ext(fp_in)
  if ext in app_cfg.VALID_PIPE_IMAGE_EXTS:
    media_type = 'image'
  elif ext in app_cfg.VALID_PIPE_VIDEO_EXTS:
    media_type = 'video'
  else:
    media_type = 'invalid'

  # init data
  data = {
    'filename': Path(fp_in).name,
    'ext': ext,
    'media_type': media_type
  }

  if media_type == 'image':
    
    # extend image metadata
    try:
      im = Image.open(fp_in)
      width, height = im.size
      data.update({'width': width, 'height': height})
    except Exception as e:
      log.error(f'{fp_in} not valid. Skipping.')
      data.update({'valid': False})

  elif media_type == 'video':
    
    # extend video metadata if available
    attrs = MediaInfo.parse(fp_in).to_data()
    video_attrs = [x for x in attrs['tracks'] if x['track_type'] == 'Video']
    general_attrs = [x for x in attrs['tracks'] if x['track_type'] == 'General']
    #audio_attrs = [x for x in attrs['tracks'] if x['track_type'] == 'Audio']
    if video_attrs and general_attrs:
      video_attrs = video_attrs[0]
      general_attrs = general_attrs[0]
      # use float, then int to avoid "invalid literal for int() errors"
      if 'encoded_date' in general_attrs.keys():
        encoded_date = str(general_attrs.get('encoded_date', ''))
        created_at = str(datetime.strptime(encoded_date, '%Z %Y-%m-%d %H:%M:%S'))
      elif 'file_last_modification_date' in general_attrs.keys():
        encoded_date = str(general_attrs.get('file_last_modification_date', ''))
        created_at = str(datetime.strptime(encoded_date, '%Z %Y-%m-%d %H:%M:%S'))
      else:
        created_at = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        app_cfg.LOG.warn(f'No date available for {fp_in}. Using now()')

      data.update({
        'codec': video_attrs.get('codec_id', ''),
        'duration': int(float(video_attrs.get('duration', 0))),
        'aspect_ratio': float(video_attrs.get('display_aspect_ratio', 0)),
        'width': int(video_attrs.get('width', 0)),
        'height': int(video_attrs.get('height', 0)),
        'frame_rate': float(video_attrs.get('frame_rate', 0)),
        'frame_count': int(float(video_attrs.get('frame_count', 0))),
        'created_at': created_at
        })
    else:
      log.error(f'{fp_in} not valid. Skipping')
      data.update({'valid': False})

  mediameta = dacite.from_dict(data=data, data_class=MediaMeta)
  
  return mediameta

