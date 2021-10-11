############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

from functools import update_wrapper, wraps
import operator
from datetime import date, datetime

import click

from dataclasses import dataclass, field
from typing import List
import logging

#from vframe.utils.click_utils import ClickSimple, ClickComplex
from vframe.settings.app_cfg import compare_accessors as accessors
from vframe.settings.app_cfg import LOG


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


# --------------------------------------------------------
# Comparison operator options
# --------------------------------------------------------

operators = {
  '+' : operator.add,
  '-' : operator.sub,
  '*' : operator.mul,
  '/' : operator.truediv,
  '%' : operator.mod,
  '^' : operator.xor,
  '<': operator.lt,
  '>': operator.gt,
  '>=': operator.ge,
  '<=': operator.le,
  '==': operator.eq,
  '!=': operator.ne,
}


@dataclass
class OptionOperator:
  attribute: str
  operator: operator
  value: str
  is_skip: bool=True

  def __post_init__(self):
    if self.attribute == 'date':
      try:
        self.date = date.fromisoformat(self.value)
      except ValueError as e:
        e = f'{self.value}\nUse ISO format YYYY-MM-DD'
        raise ValueError(e)
    else:
      self.value_int = int(self.value)

  def evaulate(self, val):
    if self.attribute == 'date':
      result = self.operator(val, self.date)
    else:
      result = self.operator(val, self.value_int)
    return result if self.is_skip else not result


  @classmethod
  def from_opt_val(cls, val):
    a, o, v = val.split(' ')
    if not a in accessors.keys():
      e = f'{a}\nUse: {", ".join(list(accessors.keys()))}'
      raise ValueError(e)
    if not o in operators.keys():
      e = f'{o}\nUse: {", ".join(list(operators.keys()))}'
      raise ValueError(e)
    return cls(accessors.get(a), operators.get(o), v)


def operator_validator(ctx, param, value):
  try:
    return OptionOperator.from_opt_val(value)
  except ValueError as e:
    raise click.BadParameter(e)
  return False

def operator_validator_multi(ctx, param, values):
  results = []
  for value in values:
    try:
      results.append(OptionOperator.from_opt_val(value))
    except ValueError as e:
      raise click.BadParameter(e)
  return results
  return False