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

from dacite import from_dict
import xmltodict
import click
import ruamel.yaml as yaml
import pandas as pd
import numpy as np

from vframe.settings import app_cfg
from vframe.settings.app_cfg import LOG
from vframe.models.types import HexInt


# ----------------------------------------------------------------------
# Encode/Decode
# ----------------------------------------------------------------------

def get_sha256(fp: str, block_size: int=65536):
  """Generates SHA256 hash for a file
  :param fp: (str) filepath
  :param block_size: (int) byte size of block
  :returns: (str) hash
  """
  sha256 = hashlib.sha256()
  with open(fp, 'rb') as fp:
    for block in iter(lambda: fp.read(block_size), b''):
      sha256.update(block)
  return sha256.hexdigest()


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

def ensure_dir(fp):
  """Alias for mkdirs"""
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

def load_yaml(fp: str, data_class: object=None, loader=yaml.SafeLoader):
  """Loads YAML file (Use .yaml, not .yml)
  """
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
  if not Path(fp).exists():
    LOG.info('not found: {}'.format(fp))
  LOG.info('loading: {}'.format(fp))
  with open(fp, 'r') as f:
    items = csv.DictReader(fp)
    if as_list:
      items = [x for x in items]
    LOG.info('returning {:,} items'.format(len(items)))
    return items


def load_txt(fp_in, data_class=None, as_list=True):
  with open(fp_in, 'rt') as fp:
    lines = fp.read().rstrip('\n')
  if as_list:
    lines = lines.split('\n')
  if data_class:
    lines = from_dict(data_class=data_class, data=lines)
  return lines


def load_xml(fp_in, data_class=None):
  """Loads XML and returns dict of items
  :param fp_in: String filepath to XML
  :param data_class: DataClass data model
  returns: OrderedDict of XML values
  """
  with open(fp_in, 'rt') as fp:
    lines = fp.read()
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



def load_json(fp_in, data_class=None):
  """Loads JSON and returns items
  :param fp_in: (str) filepath
  :returns: (dict) data from JSON
  """
  if not Path(fp_in).exists():
    LOG.error('file does not exist: {}'.format(fp_in))
    return {}
  with open(str(fp_in), 'r') as fp:
    data = json.load(fp)
  if data_class:
    data = from_dict(data_class=data_class, data=data)
  return data


def load_pkl(fp_in, data_class=None):
  """Loads Pickle and returns items
  :param fp_in: (str) filepath
  :returns: (dict) data from JSON
  """
  if not Path(fp_in).exists():
    LOG.error('file does not exist: {}'.format(fp_in))
    return {}
  with open(str(fp_in), 'rb') as fp:
    data = pickle.load(fp)
  if data_class:
    data = from_dict(data_class=data_class, data=data)
  return data

def load_file(fp_in, data_class=None):
  if fp_in is None:
    LOG.error(f'Empty filepath: {fp_in}')
  ext = get_ext(fp_in)
  if ext == 'json':
    return load_json(fp_in, data_class=data_class)
  elif ext == 'pkl':
    return load_pkl(fp_in, data_class=data_class)
  elif ext == 'csv':
    return load_csv(fp_in, data_class=data_class)
  elif ext == 'txt':
    return load_txt(fp_in, data_class=data_class)
  elif ext == 'xml':
    return load_xml(fp_in, data_class=data_class)
  elif ext == 'yaml' or ext == 'yml':
    return load_yaml(fp_in, data_class=data_class)
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

def write_txt(fp_out, data, ensure_path=True, split_lines=True, empty_ok=False):
  """Writes text file
  :param fp_out: (str) filepath
  :param ensure_path: (bool) create path if not exist
  """
  if not data and not empty_ok:
    LOG.error('no data')
    return

  if ensure_path:
    mkdirs(fp_out)
  with open(fp_out, 'w') as fp:
    if type(data) == list:
      fp.write('\n'.join(data))
    else:
      fp.write(data)


def write_xml(fp_out, data, ensure_path=True):
  """Writes text file
  :param fp_out: (str) filepath
  :param ensure_path: (bool) create path if not exist
  """
  if not data:
    LOG.error('no data')
    return
  if ensure_path:
    mkdirs(fp_out)
  with open(fp_out, 'w') as fp:
    fp.write(data)


def write_pkl(fp_out, data, ensure_path=True):
  """Writes Pickle file
  :param fp_out: (str) filepath
  :param ensure_path: (bool) create path if not exist
  """
  if ensure_path:
    mkdirs(fp_out) # mkdir
  with open(fp_out, 'wb') as fp:
    pickle.dump(data, fp)


def write_json(fp_out, data, minify=True, ensure_path=True, sort_keys=True, verbose=False, indent=2):
  """Writes JSON file
  :param fp_out: (str)filepath
  :param minify: (bool) minify JSON
  :param verbose: (bool) print status
  :param ensure_path: (bool) create path if not exist
  """
  if ensure_path:
    mkdirs(fp_out)
  with open(fp_out, 'w') as fp:
    if minify:
      json.dump(data, fp, separators=(',',':'), sort_keys=sort_keys, cls=NumpyEncoder)
    else:
      json.dump(data, fp, indent=indent, sort_keys=sort_keys, cls=NumpyEncoder)
  if verbose:
    LOG.info(f'Wrote {len(data)} items to: {fp_out}')


def write_csv(fp_out, data, header=None):
  """Writes CSV file
  :param data: (str) list of lists, or dict (writes list of key-value pairs)
  :param fp_out: (str) filepath
  :param header: (list) column headings """
  with open(fp_out, 'w') as fp:
    if type(data) is dict:
      writer = csv.DictWriter(fp, fieldnames=["key","value"])
      writer.writeheader()
      for k, v in data.items():
        writer.writerow([ k, v ])
    elif type(data[0]) is dict:
      writer = csv.DictWriter(fp, fieldnames=header)
      writer.writeheader()
      for row in data:
        writer.writerow(data)
    else:
      writer = csv.writer(fp)
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



def write_yaml(fp_out, data, indent=2, comment=None, verbose=False, default_flow_style=None):
  """Writes YAML file. Use OrderedDict to maintain order.
  :param fp_out_out: filepath (str)
  :param data: (dict) of serialized data
  :param indent: indent
  :param comment: (str) add comment header
  :param verbose: (bool) log output
  """
  with open(fp_out, 'w') as f:
    if comment:
      f.write(f'{comment}\n')
    # yaml.safe_dump(data, f, indent=indent, default_flow_style=default_flow_style, sort_keys=sort_keys)
    # yaml.safe_dump(data, f, indent=indent, default_flow_style=default_flow_style)
    yaml.dump(data, f, indent=indent, default_flow_style=default_flow_style)
  if verbose:
    LOG.info(f'Wrote {fp_out}')


def write_file(fp_out, data, **kwargs):
  ext = get_ext(fp_out)
  if ext == 'json':
    return write_json(data, fp_out, **kwargs)
  elif ext == 'pkl':
    return write_pkl(data, fp_out)
  elif ext == 'csv':
    return write_csv(data, fp_out)
  elif ext == 'txt':
    return write_txt(data, fp_out)
  else:
    LOG.error(f'Invalid extension: {ext}')
    return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def sort_dict(d, reverse=True, element_idx=1):
  """Sorts dict by value or key
  :param d: (dict) of serialized ata
  :param reverse: (bool) reverse for ascending
  :returns (OrderedDict)
  """
  return OrderedDict(sorted(d.items(), key=itemgetter(1)))

def timestamp_to_str():
  return datetime.now().strftime("%Y%m%d%H%M%S")


def zpad(n, z=app_cfg.ZERO_PADDING):
  return str(n).zfill(z)


def add_suffix(fp, suffix):
  fpp = Path(fp)
  return join(fpp.parent, f'{fpp.stem}{suffix}{fpp.suffix}')


def swap_ext(fp, ext):
  """Swaps file extension (eg: file.jpg to file.png)
  :param ext: extension without dot (eg "jpg")
  """
  fpp = Path(fp)
  return join(fpp.parent, f'{fpp.stem}.{ext}')


def get_ext(fpp, lower=True):
  """Retuns the file extension w/o dot
  :param fpp: (Pathlib.path) filepath
  :param lower: (bool) force lowercase
  :returns: (str) file extension (ie 'jpg')
  """
  fpp = ensure_posixpath(fpp)
  ext = fpp.suffix.replace('.', '')
  return ext.lower() if lower else ext


def replace_ext(fpp, ext):
  fpp = ensure_posixpath(fpp)
  ext = ext.replace('.', '')
  fpp = f'{fpp.stem}.{ext}'
  return fpp


def ensure_posixpath(fp):
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


def filesize(fp):
  """Returns filesize in KB
  """
  return Path(fp).stat().st_size // 1000

def is_file_empty(fp):  
  return Path(fp).is_file() and Path(fp).stat().st_size == 0


def glob_multi(dir_in, exts=['jpg', 'png'], recursive=True, sort=False, sort_reverse=False, multi_case=False):
  files = []
  if multi_case:
    exts = [ext.upper() for ext in exts] + [ext.lower() for ext in exts]
  for ext in exts:
    if recursive:
      files.extend(glob(join(dir_in, f'**/*.{ext}'), recursive=True))
    else:
      files.extend(glob(join(dir_in, f'*.{ext}')))
  if sort:
    files = sorted(files, reverse=sort_reverse)
  return files


# def glob_subdirs_limit(fp_dir_in, ext='jpg', limit=3, random=False):
#   """Globs one level subdirectories and limits files returned
#   """
#   files = []
#   for subdir in iglob(join(fp_dir_in, '*')):
#     glob_files = glob(join(subdir, f'*.{ext}'))
#     if glob_files:
#       files.extend(glob_files[:limit])
#   return files


# def order_items(records):
#   """Orders records by ASC SHA256"""
#   return collections.OrderedDict(sorted(records.items(), key=lambda t: t[0]))


# ------------------------------------------
# chmod utils
# ------------------------------------------

def chmod_exec(fp_script):
  """Runs chmod +x on file"""
  st = os.stat(fp_script)
  os.chmod(fp_script, st.st_mode | stat.S_IEXEC)
