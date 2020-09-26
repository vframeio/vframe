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
import logging

from dacite import from_dict
import xmltodict
import click
import yaml
import pandas as pd
import numpy as np

from vframe.settings import app_cfg


log = app_cfg.LOG


# ----------------------------------------------------------------------
# Encode/Decode
# ----------------------------------------------------------------------

def sha256(fp_in, block_size=65536):
  """Generates SHA256 hash for a file
  :param fp_in: (str) filepath
  :param block_size: (int) byte size of block
  :returns: (str) hash
  """
  sha256 = hashlib.sha256()
  with open(fp_in, 'rb') as fp:
    for block in iter(lambda: fp.read(block_size), b''):
      sha256.update(block)
  return sha256.hexdigest()


# ----------------------------------------------------------------------
# Path
# ----------------------------------------------------------------------

def mkdirs(fp):
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
# Loaders
# ----------------------------------------------------------------------

def load_yaml(fp_in, data_class=None, loader=yaml.Loader):
  '''Loads YAML file (Use .yaml, not .yml)'''
  with open(fp_in, 'r') as fp:
    cfg = yaml.load(fp, Loader=loader)
  if data_class:
    cfg = from_dict(data_class=data_class, data=cfg)
  return cfg


def load_csv(fp_in, data_class=None, as_list=True):
  """Loads CSV and retuns list of items
  :param fp_in: string filepath to CSV
  :returns: list of all CSV data
  """ 
  if not Path(fp_in).exists():
    log.info('not found: {}'.format(fp_in))
  log.info('loading: {}'.format(fp_in))
  with open(fp_in, 'r') as fp:
    items = csv.DictReader(fp)
    if as_list:
      items = [x for x in items]
    log.info('returning {:,} items'.format(len(items)))
    return items


def load_txt(fp_in, data_class=None):
  with open(fp_in, 'rt') as fp:
    lines = fp.read().rstrip('\n').split('\n')
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
    log.error('file does not exist: {}'.format(fp_in))
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
    log.error('file does not exist: {}'.format(fp_in))
    return {}
  with open(str(fp_in), 'rb') as fp:
    data = pickle.load(fp)
  if data_class:
    data = from_dict(data_class=data_class, data=data)
  return data

def load_file(fp_in, data_class=None):
  if fp_in is None:
    log.error(f'Empty filepath: {fp_in}')
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
    log.error(f'Invalid extension: {ext}')
    return None


def jsonify(data):
  """JSONifies data with Numpy converter for Numpy datatypes
  # Source: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
  """
  return json.dumps(data, cls=NumpyEncoder)


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

def write_txt(data, fp_out, ensure_path=True):
  """Writes text file
  :param fp_out: (str) filepath
  :param ensure_path: (bool) create path if not exist
  """
  if not data:
    log.error('no data')
    return
    
  if ensure_path:
    mkdirs(fp_out)
  with open(fp_out, 'w') as fp:
    if type(data) == list:
      fp.write('\n'.join(data))
    else:
      fp.write(data)


def write_pkl(data, fp_out, ensure_path=True):
  """Writes Pickle file
  :param fp_out: (str) filepath
  :param ensure_path: (bool) create path if not exist
  """
  if ensure_path:
    mkdirs(fp_out) # mkdir
  with open(fp_out, 'wb') as fp:
    pickle.dump(data, fp)


def write_json(data, fp_out, minify=True, ensure_path=True, sort_keys=True, verbose=False, indent=2):
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
    log.info('Wrote JSON: {}'.format(fp_out))


def write_csv(data, fp_out, header=None):
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


def write_file(data, fp_in, **kwargs):
  ext = get_ext(fp_in)
  if ext == 'json':
    return write_json(data, fp_in, **kwargs)
  elif ext == 'pkl':
    return write_pkl(data, fp_in)
  elif ext == 'csv':
    return write_csv(data, fp_in)
  elif ext == 'txt':
    return write_txt(data, fp_in)
  else:
    log.error(f'Invalid extension: {ext}')
    return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def timestamp_to_str():
  return datetime.now().strftime("%Y%m%d%H%M%S")
  

def zpad(n, zeros=app_cfg.ZERO_PADDING):
  return str(n).zfill(zeros)


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

def glob_multi(dir_in, exts=['jpg', 'png'], recursive=True, sort=False, sort_reverse=False):
  files = []
  for ext in exts:
    if recursive:
      fp_glob = join(dir_in, '**/*.{}'.format(ext))
      files +=  glob(fp_glob, recursive=True)
    else:
      fp_glob = join(dir_in, '*.{}'.format(ext))
      files += glob(fp_glob)
  if sort:
    files = sorted(files, reverse=sort_reverse)
  return files

def glob_subdirs_limit(fp_dir_in, ext='jpg', limit=3, random=False):
  '''Globs one level subdirectories and limits files returned
  '''
  files = []
  for subdir in iglob(join(fp_dir_in, '*')):
    glob_files = glob(join(subdir, f'*.{ext}'))
    if glob_files:
      files.extend(glob_files[:limit])
  return files
  

def order_items(records):
  """Orders records by ASC SHA256"""
  return collections.OrderedDict(sorted(records.items(), key=lambda t: t[0]))


# ------------------------------------------
# chmod utils
# ------------------------------------------

def chmod_exec(fp_script):
  '''Runs chmod +x on file'''
  st = os.stat(fp_script)
  os.chmod(fp_script, st.st_mode | stat.S_IEXEC)

