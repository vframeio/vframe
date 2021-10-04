############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from dataclasses import dataclass, field
from typing import List


@dataclass
class PluginScriptGroup:
  filepath: str  # filepath to directory containing scripts
  include_hidden: bool=False


@dataclass
class Plugin:
  name: str
  scripts: List[PluginScriptGroup]
  pipe: bool=False
  active: bool=True
  description: str=''


@dataclass
class Plugins:  
  plugins: List[Plugin]=field(default_factory=[])

  def __post_init__(self):
    self.plugin_lookup = {x.name:x for x in self.plugins}  # key lookup

  def get(self, key):
    return self.plugin_lookup.get(key)

  def keys(self):
    return self.plugin_lookup.keys()