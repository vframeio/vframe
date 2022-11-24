from typing import List
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

from vframe.settings.app_cfg import LOG, modelzoo
from vframe.utils.file_utils import ensure_dir
from vframe.utils.url_utils import download_url



def download_model(dnn_cfg, opt_force=False, opt_verbose=False) -> bool:
  """Auto-download model files
  :param dnn_cfg: DNN configuration object
  :param opt_force: Force download and overwrite existing files
  :param opt_verbose: Print verbose statements
  :returns bool status
  """

  dl_files = []

  # Model
  if dnn_cfg.model:
    dl_files.append({'url': dnn_cfg.url_model, 'fp':dnn_cfg.fp_model})

  # Config
  if dnn_cfg.labels:
    dl_files.append({'url': dnn_cfg.url_labels, 'fp':dnn_cfg.fp_labels})

  # Classes
  if dnn_cfg.config:
    dl_files.append({'url': dnn_cfg.url_config, 'fp':dnn_cfg.fp_config})

  # license
  if dnn_cfg.license:
    dl_files.append({'url': dnn_cfg.url_license, 'fp':dnn_cfg.fp_license})

  results = []

  for dl_file in dl_files:
    
    fp, url = dl_file['fp'], dl_file['url']

    if Path(fp).is_file() and not opt_force:
      if opt_verbose:
        LOG.warn(f'{fp} already exists. Use "-f/--force" to overwrite')
    else:
      if opt_verbose:
        LOG.info(f'Downloading: {url} to {fp}')
      ensure_dir(fp)
      try:
        result = download_url(url, fp)
        results.append(result)
      except Exception as e:
        results.append(False)
        if opt_verbose:
          LOG.error(f'Could not download {url}. Error {e}')

    return all(results)



def list_models(group_by:str='output') -> str:
  """List classification models
  :param group_by: DNN key to group configurations
  """
  processors = defaultdict(list)

  for dnn_key, dnn_cfg in modelzoo.items():
    dnn_cfg_dict = asdict(dnn_cfg)
    processor = dnn_cfg_dict.get(group_by)
    processors[processor].append(dnn_key)

  txt = '\n'
  for processor in sorted(processors.keys()):
    txt += f"[{processor}]\n"
    for dnn_key in sorted(processors[processor]):
      txt += f"  - {dnn_key}\n"

  return txt
