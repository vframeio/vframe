############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import importlib
import logging

from vframe.utils import file_utils

log = logging.getLogger('vframe')

def load_presets(cfg):
  """Load preset yaml files as dicts
  """
  return {k: file_utils.load_file(v) for k,v in cfg.items()}


def patch_presets(obj_cfg, preset_cfg):
  """Recursively load presets and replace text with dict values from preset files
  """

  def recusive_replace(obj_cfg, preset_cfg):
    
    preset_key = 'preset:'

    if type(obj_cfg) == list:
      new_obj_cfg = obj_cfg.copy()
      return [recusive_replace(attr_item, preset_cfg) for attr_item in new_obj_cfg if type(attr_item) == dict]
    
    elif type(obj_cfg) == dict:
      
      new_obj_cfg = obj_cfg.copy()
      for attr_name, attr_value in obj_cfg.items():    
        if type(attr_value) == str:
          #log.debug(f'Process str: {attr_name}:{attr_value}')
          if preset_key in attr_value and attr_value.count(':') == 2:
            log.debug(attr_value)
            splits = attr_value.split(':')
            preset_names = splits[2].split('+')  # combine attributes with '+'
            #log.debug(f'Processing preset names: {preset_names}')
            presets_data = type(preset_cfg.get(splits[1], {}).get(preset_names[0], ''))()

            for preset_name in preset_names:
              #log.debug(f'Get preset name: {preset_name}')
              preset_data = preset_cfg.get(splits[1], {}).get(preset_name, '')
              
              if type(presets_data) == list:
                presets_data.extend(preset_data)
              elif type(presets_data) == dict:
                presets_data.update(preset_data)
              elif type(presets_data) == str:
                presets_data = recusive_replace(preset_data, preset_cfg)

            if presets_data:
              new_obj_cfg[attr_name] = recusive_replace(presets_data, preset_cfg)
            else:
              log.error(f'{attr_value} does not exist for preset value: {preset_names}')
              # FIXME: handle error values in DataClass construct
              new_obj_cfg.pop(attr_name)
              #return
        elif type(attr_value) == dict:
          new_obj_cfg[attr_name] = recusive_replace(attr_value, preset_cfg)
        elif type(attr_value) == list:
          new_obj_cfg[attr_name] = [recusive_replace(attr_item, preset_cfg) for attr_item in attr_value]
        
      return new_obj_cfg

    elif type(obj_cfg) == str:

      if preset_key in obj_cfg and obj_cfg.count(':') == 2:
        splits = obj_cfg.split(':')
        preset_names = splits[2].split('+')
        presets_data = type(preset_cfg.get(splits[1], {}).get(preset_names[0], ''))()

        for preset_name in preset_names:
          preset_data = preset_cfg.get(splits[1], {}).get(preset_name, '')
          log.debug(preset_data)
          if type(presets_data) == list:
            presets_data.extend(preset_data)
          elif type(presets_data) == dict:
            presets_data.update(preset_data)
          elif type(presets_data) == str:
            presets_data = recusive_replace(preset_data, preset_cfg)

        if presets_data:
          new_obj_cfg = presets_data
          return recusive_replace(new_obj_cfg, preset_cfg)
        else:
          log.error(f'{obj_cfg} does not exist')
          return
    else:
      return obj_cfg  # FIXME: not sure if this needed?

    return obj_cfg  # FIXME: this seems redundant

  return recusive_replace(obj_cfg, preset_cfg)
