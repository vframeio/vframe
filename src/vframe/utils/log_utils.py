############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

"""
Logger instantiator for use with Click utlity scripts
"""
import sys
import os
import logging

import colorlog

def get_logger():
  cur_log = logging.getLogger(Logger.LOGGER_NAME)
  if not cur_log or cur_log == '':
    Logger.create()
  return logging.getLogger(Logger.LOGGER_NAME)
  
class Logger:
  
  LOGGER_NAME = 'VFRAME'
  LOGFILE_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(cyan)s%(filename)s:%(lineno)s:%(bold_cyan)s%(funcName)s() %(reset)s%(message)s"

  def __init__(self):
    pass

  @staticmethod
  def create(verbosity=4, logfile=None):
    """Configures a logger from click params
    :param verbosity: (int) between 0 and 5
    :param logfile: (str) path to logfile
    :returns: logging root object
    """
    
    loglevel = (5 - (max(0, min(verbosity, 5)))) * 10 # where logging.DEBUG = 10
    date_format = '%Y-%m-%d %H:%M:%S'
    if 'colorlog' in sys.modules and os.isatty(2):
      cformat = '%(log_color)s' + Logger.LOGFILE_FORMAT
      f = colorlog.ColoredFormatter(cformat, date_format,
        log_colors = { 'DEBUG'   : 'yellow',       'INFO' : 'white',
        'WARNING' : 'bold_yellow', 'ERROR': 'bold_red',
        'CRITICAL': 'bold_red' })
    else:
      f = logging.Formatter(Logger.LOGFILE_FORMAT, date_format)
    
    logger = logging.getLogger(Logger.LOGGER_NAME)
    logger.setLevel(loglevel)
    
    # remove existing handlers
    for handler in logger.handlers:
      logger.removeHandler(handler)

    if logfile:
      # create file handler which logs even debug messages
      fh = logging.FileHandler(logfile)
      fh.setLevel(loglevel)
      logger.addHandler(fh)

    # add colored handler
    ch = logging.StreamHandler()
    ch.setFormatter(f)
    logger.addHandler(ch)

    if verbosity == 0:
      logger.disabled = True

    # test
    # logger.debug('Hello Debug')
    # logger.info('Hello Info')
    # logger.warn('Hello Warn')
    # logger.error('Hello Error')
    # logger.critical('Hello Critical')

    return logger

  @staticmethod
  def getLogger():
    return logging.getLogger(Logger.LOGGER_NAME)