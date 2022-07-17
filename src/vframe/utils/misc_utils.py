############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2019 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import random

from vframe.settings.app_cfg import LOG


def evenify(n):
  """Ensure number is even by incrementing if odd
  """
  return n if n % 2 == 0 else n + 1


def oddify(n):
  """Ensure number is odd by incrementing if even
  """
  return n if n % 2 else n + 1


def random_range(value, variance=0.0):
  """Return a random value within range with variance
  """
  value_range = [value * (1.0 - variance), value]
  return random.uniform(*value_range)


def rand_bool():
  return random.random() < 0.5