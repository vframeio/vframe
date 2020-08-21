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
from pathlib import Path
from dataclasses import dataclass
from glob import glob
import logging

import dacite
import boto3


@dataclass
class S3Config:
  S3_BUCKET: str
  S3_KEY: str
  S3_SECRET: str
  S3_ENDPOINT: str
  S3_REGION: str


class RemoteStorageS3:

  def __init__(self):

    self.log = logging.getLogger('vframe')

    self.s3_cfg = dacite.from_dict(data_class=S3Config, data=os.environ)

    self.session = boto3.session.Session()

    self.s3_client = self.session.client(
      service_name='s3',
      aws_access_key_id=self.s3_cfg.S3_KEY,
      aws_secret_access_key=self.s3_cfg.S3_SECRET,
      endpoint_url=self.s3_cfg.S3_ENDPOINT,
      region_name=self.s3_cfg.S3_REGION,
    )


  def list_dir(self, fp_dir_remote):
    """Sync local directory to remote directory
    """
    
    obj_list_remote = self.s3_client.list_objects(
      Bucket=self.s3_cfg.S3_BUCKET, 
      Prefix=fp_dir_remote)


    for obj in obj_list_remote.get('Contents', []):
      s3_fn = obj['Key']
      self.log.debug(s3_fn)


  def sync_dir(self, fp_dir_local, fp_dir_remote):
    """Sync local directory to remote directory
    """
    
    # get list of local files
    fps_local = glob(join(fp_dir_local, '*'))
    fp_local_lkup = {}
    for fp in fps_local:
      fp_local_lkup[Path(fp).name] = fp

    # get list of remote files
    obj_list_remote = self.s3_client.list_objects(Bucket=self.s3_cfg.S3_BUCKET, Prefix=fp_dir_remote)

    # check if remove files exist locally
    if 'Contents' in obj_list_remote:
      for obj in obj_list_remote['Contents']:
        s3_fn = obj['Key']
        fn_remote = Path(s3_fn).name
        if fn_remote in fp_local_lkup.keys():
          # remove from queue
          # compare timestamps
          fp_local = fp_local_lkup[fn_remote]
          del fp_local_lkup[fn_remote]
          if obj['LastModified'].timestamp() < os.path.getmtime(fp_local):
            self.log.debug("Update s3 with newer local file: {}".format(s3_fn))
            self.s3_client.upload_file(
              fp_local,
              self.s3_cfg.S3_BUCKET,
              s3_fn,
              ExtraArgs={'ACL': 'public-read' })
          else:
            self.log.debug(f'Skipping. Same file: {s3_fn}')
        else:
          self.log.debug(f'Orphaned remote file: {s3_fn}')
          self.log.debug("s3 delete {}".format(s3_fn))
          response = self.s3_client.delete_object(
            Bucket=self.s3_cfg.S3_BUCKET,
            Key=s3_fn,
          )
    else:
      pass
      #self.log.debug(f'No "Contents" in {obj_list_remote.keys()}')

    # put the remaining files to S3
    for fn_local, fp_local in fp_local_lkup.items():
      s3_fn = join(fp_dir_remote, fn_local)
      self.log.debug("s3 create {}".format(s3_fn))
      self.s3_client.upload_file(
        fp_local,
        os.getenv('S3_BUCKET'),
        s3_fn,
        ExtraArgs={ 'ACL': 'public-read' })


  def sync_file(self, fp_local, fp_remote):
    """Sync local file to remove file
    """
    self.log.warn('Not yet implemented')