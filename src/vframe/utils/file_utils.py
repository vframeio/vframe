#############################################################################
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io
#
#############################################################################

"""
File utilities for reading, writing, and formatting data
"""

import sys
import os
import platform
from os.path import join
import stat
import json
import csv
from glob import glob, iglob
from datetime import datetime
import time
import pickle
import shutil
import collections
import pathlib
from pathlib import Path
import dataclasses
import hashlib
from operator import itemgetter
from collections import OrderedDict

from typing import (Dict, List, Tuple, Union)
from dacite import from_dict
import xmltodict
import click
import ruamel.yaml as yaml
import numpy as np

from vframe.settings import app_cfg
from vframe.settings.app_cfg import LOG
from vframe.models.types import HexInt


# ----------------------------------------------------------------------
# Encode/Decode
# ----------------------------------------------------------------------

def mk_sha256(fp: str, block_size: int=65536):
  """Generates SHA256 hash for a file
  :param fp: (str) filepath
  :param block_size: (int) byte size of block
  :returns: (str) hash
  """
  sha256 = hashlib.sha256()
  with open(fp, 'rb') as f:
    for block in iter(lambda: f.read(block_size), b''):
      sha256.update(block)
  return sha256.hexdigest()

def check_md5(fp, checksum):
  with open(fp, 'rb') as f:
    data = f.read()    
    return hashlib.md5(data).hexdigest() == checksum
  return False

# ----------------------------------------------------------------------
# Path
# ----------------------------------------------------------------------

def mkdirs(fp: str):
  """Ensure parent directories exist for a filepath
  :param fp: string, Path, or click.File
  """
  fpp = ensure_posixpath(fp)
  fpp = fpp.parent if fpp.suffix else fpp
  fpp.mkdir(parents=True, exist_ok=True)


def ensure_dir(fp: str):
  mkdirs(fp)

# ----------------------------------------------------------------------
# Creation and modified date
# ----------------------------------------------------------------------

def date_modified(fp: str, milliseconds: bool=False):
  """Returns file modified time falling back to modified time on Linux
  """
  t = Path(fp).stat().st_mtime
  if not milliseconds:
    t = int(t)
  return datetime.fromtimestamp(t)
    
    
def date_created(fp: str, milliseconds: bool=False):
  """Returns file creation time falling back to modified time on Linux
  """
  stat = Path(fp).stat()
  if platform.system() == 'Windows':
    t = stat.st_ctime
  else:
    try:
      t = stat.st_ctime
    except AttributeError:
      # Fallback to modified time on Linux
      t = stat.st_mtime
  if not milliseconds:
    t = int(t)
  return datetime.fromtimestamp(t)


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def check_file_exists(fp: str):
  if not Path(fp).exists():
    LOG.error('file does not exist: {}'.format(fp))
    return None

def load_yaml(fp: str, data_class: object=None, loader=yaml.SafeLoader):
  """Loads YAML file (Use .yaml, not .yml)
  """
  check_file_exists(fp)
  with open(fp, 'r') as f:
    cfg = yaml.load(f, Loader=loader)
  if data_class:
    cfg = from_dict(data_class=data_class, data=cfg)
  return cfg


def load_csv(fp: str, data_class: object=None, as_list: bool=True):
  """Loads CSV and retuns list of items
  :param fp: string filepath to CSV
  :returns: list of all CSV data
  """
  check_file_exists(fp)
  if not Path(fp).exists():
    LOG.info('not found: {}'.format(fp))
  LOG.info('loading: {}'.format(fp))
  with open(fp, 'r') as f:
    items = csv.DictReader(fp)
    if as_list:
      items = [x for x in items]
    LOG.info('returning {:,} items'.format(len(items)))
    return items


def load_txt(fp: str, data_class: object=None, as_list: bool=True):
  check_file_exists(fp)
  with open(fp, 'rt') as f:
    lines = f.read().rstrip('\n')
  if as_list:
    lines = lines.split('\n')
  if data_class:
    lines = from_dict(data_class=data_class, data=lines)
  return lines


def load_xml(fp: str, data_class: object=None):
  """Loads XML and returns dict of items
  :param fp: String filepath to XML
  :param data_class: DataClass data model
  returns: OrderedDict of XML values
  """
  check_file_exists(fp)
  with open(fp, 'rt') as f:
    lines = f.read()
  data = xmltodict.parse(lines)
  if data_class:
    data = from_dict(data_class=data_class, data=data)
  return data


class NumpyEncoder(json.JSONEncoder):
  """ Special json encoder for numpy types """
  # HT https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689
  def default(self, obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
      return float(obj)
    elif isinstance(obj,(np.ndarray,)): #### This is the fix
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


class EnhancedJSONEncoder(json.JSONEncoder):
  def default(self, o):
    if dataclasses.is_dataclass(o):
      return dataclasses.asdict(o)
    return super().default(o)


def load_json(fp: str, data_class: object=None):
  """Loads JSON and returns items
  :param fp: (str) filepath
  :returns: (dict) data from JSON
  """
  check_file_exists(fp)
  with open(str(fp), 'r') as f:
    data = json.load(f)
  if data_class:
    data = from_dict(data_class=data_class, data=data)
  return data


def load_pkl(fp: str, data_class: object=None):
  """Loads Pickle and returns items
  :param fp: (str) filepath
  :returns: (dict) data from JSON
  """
  check_file_exists()
  with open(str(fp), 'rb') as f:
    data = pickle.load(f)
  if data_class:
    data = from_dict(data_class=data_class, data=data)
  return data

def load_file(fp: str, data_class: object=None):
  """Load file and auto-infer type using extension
  """
  check_file_exists(fp)
  ext = get_ext(fp)
  if ext == 'json':
    return load_json(fp, data_class=data_class)
  elif ext == 'pkl':
    return load_pkl(fp, data_class=data_class)
  elif ext == 'csv':
    return load_csv(fp, data_class=data_class)
  elif ext == 'txt':
    return load_txt(fp, data_class=data_class)
  elif ext == 'xml':
    return load_xml(fp, data_class=data_class)
  elif ext == 'yaml' or ext == 'yml':
    return load_yaml(fp, data_class=data_class)
  else:
    LOG.error(f'Invalid extension: {ext}')
    return None


def jsonify(data):
  """JSONifies data with Numpy converter for Numpy datatypes
  # Source: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
  """
  return json.dumps(data, cls=NumpyEncoder)


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

def write_txt(fp: str, data: object, ensure_path: bool=True, 
  split_lines: bool=True,  empty_ok: bool=False):
  """Writes text file
  :param fp: (str) filepath
  :param ensure_path: (bool) create path if not exist
  """
  if not data and not empty_ok:
    LOG.error('No data. Use "empty_ok=True" to write empty file')
    return

  if ensure_path:
    mkdirs(fp)
  with open(fp, 'w') as f:
    if type(data) == list:
      f.write('\n'.join(data))
    else:
      f.write(data)


def write_xml(fp: str, data: object, ensure_path: bool=True):
  """Writes text file
  :param fp: filepath
  :param ensure_path: (bool) create path if not exist
  """
  if not data:
    LOG.error('No data')
    return
  if ensure_path:
    mkdirs(fp)
  with open(fp, 'w') as f:
    f.write(data)


def write_pkl(fp: str, data: object, ensure_path: bool=True):
  """Writes Pickle file
  :param fp: filepath
  :param ensure_path: (bool) create path if not exist
  """
  if ensure_path:
    mkdirs(fp)
  with open(fp, 'wb') as f:
    pickle.dump(data, f)


def write_json(fp: str, data: object, minify: bool=True, ensure_path: bool=True, 
  sort_keys: bool=True, verbose: bool=False, indent: int=2):
  """Writes JSON file
  :param fp: filepath
  :param minify: minify JSON
  :param verbose: print status
  :param ensure_path: create path if not exist
  """
  if ensure_path:
    mkdirs(fp)
  with open(fp, 'w') as f:
    if minify:
      json.dump(data, f, separators=(',',':'), sort_keys=sort_keys, cls=NumpyEncoder)
    else:
      json.dump(data, f, indent=indent, sort_keys=sort_keys, cls=NumpyEncoder)
  if verbose:
    LOG.info(f'Wrote {len(data)} items to: {fp}')


def write_csv(fp: str, data: object, header: str=None):
  """Writes CSV file
  :param data: list of lists, or dict (writes list of key-value pairs)
  :param fp: filepath
  :param header: column headings """
  with open(fp, 'w') as f:
    if type(data) is dict:
      writer = csv.DictWriter(f, fieldnames=["key","value"])
      writer.writeheader()
      for k, v in data.items():
        writer.writerow([ k, v ])
    elif type(data[0]) is dict:
      writer = csv.DictWriter(f, fieldnames=header)
      writer.writeheader()
      for row in data:
        writer.writerow(data)
    else:
      writer = csv.writer(f)
      if header is not None:
        writer.writerow(header)
      for row in data:
        writer.writerow(row)


def setup_yaml():
  """Format non-standard YAML data types
  https://stackoverflow.com/a/8661021
  https://dustinoprea.com/2018/04/15/python-writing-hex-values-into-yaml/
  """

  # format hex 0xFFFFFF for YAML output
  def representer(dumper, data):
    return yaml.ScalarNode('tag:yaml.org,2002:int', '0x{:06x}'.format(data))
 
  yaml.add_representer(HexInt, representer)
  
  represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
  yaml.add_representer(OrderedDict, represent_dict_order)

setup_yaml()



def write_yaml(fp: str, data: object, indent:int=2, comment: str=None, 
  verbose: bool=False, default_flow_style:bool=None):
  """Writes YAML file. Use OrderedDict to maintain order.
  :param fp: filepath
  :param data: of serialized data
  :param indent: indent
  :param comment: add comment header
  :param verbose: log output
  """
  with open(fp, 'w') as f:
    if comment:
      f.write(f'{comment}\n')
    # yaml.safe_dump(data, f, indent=indent, default_flow_style=default_flow_style, sort_keys=sort_keys)
    # yaml.safe_dump(data, f, indent=indent, default_flow_style=default_flow_style)
    yaml.dump(data, f, indent=indent, default_flow_style=default_flow_style)
  if verbose:
    LOG.info(f'Wrote {fp}')


def write_file(fp: str, data: object, **kwargs: dict):
  ext = get_ext(fp)
  if ext == 'json':
    return write_json(data, fp, **kwargs)
  elif ext == 'pkl':
    return write_pkl(data, fp)
  elif ext == 'csv':
    return write_csv(data, fp)
  elif ext == 'txt':
    return write_txt(data, fp)
  else:
    LOG.error(f'Invalid extension: {ext}')
    return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def sort_dict(d: dict, reverse: bool=True, element_idx: int=1):
  """Sorts dict by value or key
  :param d: of serialized ata
  :param reverse: reverse for ascending
  :returns (OrderedDict)
  """
  return OrderedDict(sorted(d.items(), key=itemgetter(1)))

def timestamp_to_str():
  return datetime.now().strftime("%Y%m%d%H%M%S")


def zpad(n: str, z: int=app_cfg.ZERO_PADDING):
  """Zero-pad string
  """
  return str(n).zfill(z)


def add_suffix(fp:str, suffix: str):
  fpp = Path(fp)
  return join(fpp.parent, f'{fpp.stem}{suffix}{fpp.suffix}')


def swap_ext(fp:str, ext:str):
  """Swaps file extension (eg: file.jpg to file.png)
  :param ext: extension without dot (eg "jpg")
  """
  fpp = Path(fp)
  return join(fpp.parent, f'{fpp.stem}.{ext}')


def get_ext(fpp:str, lower: bool=True):
  """Retuns the file extension w/o dot
  :param fpp: (Pathlib.path) filepath
  :param lower: (bool) force lowercase
  :returns: (str) file extension (ie 'jpg')
  """
  fpp = ensure_posixpath(fpp)
  ext = fpp.suffix.replace('.', '')
  return ext.lower() if lower else ext


def replace_ext(fpp:str, ext:str):
  fpp = ensure_posixpath(fpp)
  ext = ext.replace('.', '')
  fpp = f'{fpp.stem}.{ext}'
  return fpp


def ensure_posixpath(fp: str):
  """Ensures filepath is pathlib.Path
  :param fp: a (str, LazyFile, PosixPath)
  :returns: a PosixPath filepath object
  """
  if type(fp) == str:
    fpp = Path(fp)
  elif type(fp) == click.utils.LazyFile:
    fpp = Path(fp.name)
  elif type(fp) == pathlib.PosixPath:
    fpp = fp
  else:
    raise TypeError('{} is not a valid filepath type'.format(type(fp)))
  return fpp


def filesize(fp: str):
  """Returns filesize in KB
  """
  return Path(fp).stat().st_size // 1000


def is_file_empty(fp: str):  
  return Path(fp).is_file() and Path(fp).stat().st_size == 0


def glob_multi(dp: str, exts: List=['jpg', 'png'], recursive: bool=True, sort: bool=False, 
  sort_reverse: bool=False, multi_case: bool=False):
  files = []
  if multi_case:
    exts = [ext.upper() for ext in exts] + [ext.lower() for ext in exts]
  for ext in exts:
    if recursive:
      files.extend(glob(join(dp, f'**/*.{ext}'), recursive=True))
    else:
      files.extend(glob(join(dp, f'*.{ext}')))
  if sort:
    files = sorted(files, reverse=sort_reverse)
  return files



# ----------------------------------------------------------------------
# chmod utils
# ----------------------------------------------------------------------

def chmod_exec(fp: str):
  """Runs chmod +x on file
  :param fp: script filepath
  """
  st = os.stat(fp)
  os.chmod(fp, st.st_mode | stat.S_IEXEC)