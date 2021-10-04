from pathlib import Path


from slugify import slugify
from PIL import Image, ExifTags

from src.utils import file_utils, logger_utils

class MediaInfoExtractor:

  video_exts = ['mp4', 'mov', 'avi']
  image_exts = ['jpg', 'png']

  def __init__(self):
    self.log = logger_utils.Logger.getLogger()

  def extract(self, fp_in, opt_mediainfo=False, opt_exif=False, opt_hash=False, 
    opt_faces=False, opt_objects=False):
    '''Extract mediainfo about an image
    '''
    fp_inp = Path(fp_in)
    ext = file_utils.get_ext(fp_in)

    # determine media type and extract info
    
    result = {}

    # file system data
    
    # video
    if ext in self.video_exts:
      if opt_mediainfo:
        meta = self._video(fp_in)
        result.update(meta)

    # image
    if ext in self.image_exts:
      meta = self._image(fp_in, opt_exif=opt_exif)
      result.update(meta)
    
    if opt_hash:
      meta = {'sha256': file_utils.sha256(fp_in)}
      result.update(meta)

    return result

  
  def _image(self, fp_in, opt_exif=False):
    '''Get metadata about an image. Currently only supported for JPGs
    '''
    im = Image.open(fp_in)
    data = {
      'image_width': im.size[0],
      'image_height': im.size[1]
    }
    if opt_exif:
      try:
        exif_data = im._getexif()
        if exif_data:
          exif_data = {f'exif_{slugify(ExifTags.TAGS[k])}': v for k, v in exif_data.items() if k in ExifTags.TAGS}
          self.log.debug(data)
          try:
            _ = data.pop('exif_makernote')  # ignore maker note (often proprietary byte code)
          except Exception as e:
            self.log.debug('maker note')
            pass
          data.update(exif_data)
      except Exception as e:
        self.log.info(f'No exif data for {Path(fp_in).name}. Error: {e}')
    return data

  def _video(fp_in):
    """Get media info using pymediainfo"""
    mi_data = MediaInfo.parse(fp_in).to_data()
    data = {}
    for track in mi_data['tracks']:
      if track['track_type'] == 'Video':
        data = {
          'video_codec_cc': track['codec_cc'],
          'video_duration': int(track['duration']),
          'video_aspect_ratio': float(track['display_aspect_ratio']),
          'video_width': int(track['width']),
          'video_height': int(track['height']),
          'video_frame_rate': float(track['frame_rate']),
          'video_frame_count': int(track['frame_count']),
          }
    return data


  """
  Mediainfo:
  codec_cc
  display_aspect_ratio
  frame_count
  width
  height
  frame_rate
  duration (in ms)
  """

  """
  format_url
  proportion_of_this_stream
  frame_count
  stream_identifier
  other_scan_type
  count_of_stream_of_this_kind
  interlacement
  codec_settings__cabac
  codec_id_info
  chroma_subsampling
  other_maximum_bit_rate
  other_kind_of_stream
  codec_cc
  track_type
  count
  codec_settings
  encoded_date
  format_settings__cabac
  other_bit_depth
  stored_height
  other_format_settings__reframes
  bits__pixel_frame
  format_profile
  other_stream_size
  other_track_id
  resolution
  format
  color_space
  sampled_height
  other_display_aspect_ratio
  other_width
  rotation
  codec_family
  framerate_mode_original
  other_interlacement
  other_height
  codec
  display_aspect_ratio
  duration
  bit_rate
  frame_rate_mode
  height
  sampled_width
  maximum_bit_rate
  pixel_aspect_ratio
  codec_id
  scan_type
  codec_url
  codec_info
  other_duration
  codec_settings_refframes
  streamorder
  tagged_date
  track_id
  other_format_settings__cabac
  format_settings__reframes
  other_codec
  bit_depth
  format_info
  other_frame_rate
  commercial_name
  frame_rate
  stream_size
  colorimetry
  other_frame_rate_mode
  internet_media_type
  format_settings
  kind_of_stream
  codec_id_url
  other_resolution
  codec_profile
  width
  other_bit_rate
  mediainfo
  None
  """

  