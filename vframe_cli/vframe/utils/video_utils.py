from pathlib import Path

from pymediainfo import MediaInfo
import dacite
import threading
from datetime import datetime

from PIL import Image
import cv2 as cv

from vframe.settings import app_cfg
from vframe.models.mediameta import MediaMeta
from vframe.utils import file_utils


log = app_cfg.LOG


# --------------------------------------------------------------
# jrosebr1
# from imutils.video
# https://raw.githubusercontent.com/jrosebr1/imutils/master/imutils/video/filevideostream.py
# --------------------------------------------------------------

# import the necessary packages
from threading import Thread
import time

from queue import Queue


class FileVideoStream:
  def __init__(self, path, transform=None, queue_size=256):
    # initialize the file video stream along with the boolean
    # used to indicate if the thread should be stopped or not
    self.stream = cv.VideoCapture(path)
    self.height = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
    self.width = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
    self.stopped = False
    self.transform = transform

    # initialize the queue used to store frames read from
    # the video file
    self.Q = Queue(maxsize=queue_size)
    # intialize thread
    self.thread = Thread(target=self.update, args=())
    self.thread.daemon = True

  def start(self):
    # start a thread to read frames from the file video stream
    self.thread.start()
    return self

  def update(self):
    # keep looping infinitely
    while True:
      # if the thread indicator variable is set, stop the
      # thread
      if self.stopped:
        break

      # otherwise, ensure the queue has room in it
      if not self.Q.full():
        # read the next frame from the file
        (grabbed, frame) = self.stream.read()

        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
          self.stopped = True
          
        # if there are transforms to be done, might as well
        # do them on producer thread before handing back to
        # consumer thread. ie. Usually the producer is so far
        # ahead of consumer that we have time to spare.
        #
        # Python is not parallel but the transform operations
        # are usually OpenCV native so release the GIL.
        #
        # Really just trying to avoid spinning up additional
        # native threads and overheads of additional
        # producer/consumer queues since this one was generally
        # idle grabbing frames.
        if self.transform:
          frame = self.transform(frame)

        # add the frame to the queue
        self.Q.put(frame)
      else:
        time.sleep(0.01)  # Rest for 10ms, we have a full queue

    self.stream.release()

  def read(self):
    # return next frame in the queue
    return self.Q.get()

  # Insufficient to have consumer use while(more()) which does
  # not take into account if the producer has reached end of
  # file stream.
  def running(self):
    return self.more() or not self.stopped

  def more(self):
    # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
    tries = 0
    while self.Q.qsize() == 0 and not self.stopped and tries < 5:
      time.sleep(0.1)
      tries += 1

    return self.Q.qsize() > 0

  def stop(self):
    # indicate that the thread should be stopped
    self.stopped = True
    # wait until stream resources are released (producer thread might be still grabbing frame)
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

