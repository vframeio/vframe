############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from functools import update_wrapper, wraps

import click

from dataclasses import dataclass, field
from typing import List
import logging

#from vframe.utils.click_utils import ClickSimple, ClickComplex


# -----------------------------------------------------------------------------
#
# Dataclasses to mdoel modelzoo config
#
# -----------------------------------------------------------------------------

@dataclass
class PluginPath:
  commands: str
  root: str=None


@dataclass
class CommandGroup:
  name: str
  plugins: List[PluginPath]=field(default_factory=[])
  chain: bool=False


@dataclass
class CommandGroups:
  
  commands: List[CommandGroup]=field(default_factory=[])

  def __post_init__(self):
    self.commands_lookup = {x.name:x for x in self.commands}  # key lookup

  def get(self, key):
    return self.commands_lookup.get(key)

  def keys(self):
    return self.commands_lookup.keys()



# -----------------------------------------------------------------------------
#
# Processor and generator
#
# -----------------------------------------------------------------------------

def processor(f):
    """Helper decorator to rewrite a function so that it returns another function
    Copied from click's documentation
    """
    def processor_wrapper(*args, **kwargs):
      @wraps(f)
      def processor(sink):
        return f(sink, *args, **kwargs)
      return processor
    return update_wrapper(processor_wrapper, f)


def generator(f):
    """Similar to the :func:`processor` but passes through old values unchanged.
    """
    @processor
    def _generator(sink, *args, **kwargs):
      try:
        while True:
          sink.send((yield))
      except GeneratorExit:
        f(sink, *args, **kwargs)
        sink.close()
    return update_wrapper(_generator, f)



# --------------------------------------------------------
#
# Click command helpers
#
# --------------------------------------------------------

def enum_to_names(enum_type):
  return {x.name.lower(): x for x in enum_type}
  
def show_help(enum_type):
  names = enum_to_names(enum_type)
  return 'Options: "{}"'.format(', '.join(list(names.keys())))

def get_default(opt):
  return opt.name.lower()


# --------------------------------------------------------
# Custom Click parameter class
# --------------------------------------------------------

class ParamVar(click.ParamType):

  name = 'default_type'

  def __init__(self, param_type):
    self.ops =  {x.name.lower(): x for x in param_type}
  
  def convert(self, value, param, ctx):
    """converts (str) repr to Enum hash"""
    try:
      return self.ops[value.lower()]
    except:
      self.fail('{} is not a valid option'.format(value, param, ctx))