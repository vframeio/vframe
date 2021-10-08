############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################


import sys
import logging

import cv2 as cv

from vframe.settings import app_cfg
from vframe.settings.app_cfg import LOG, PAUSED



# -----------------------------------------------------------------------------
#
# Display standalone
#
# -----------------------------------------------------------------------------

def display_frame(im, delay=1, window_name=''):

  window_name = window_name if window_name else app_cfg.CV_WINDOW_NAME
  cv.imshow(window_name, im)
  
  while True:

    k = cv.waitKey(delay) & 0xFF
    
    if k == 27 or k == ord('q'):  # ESC
      # exits the app
      cv.destroyAllWindows()
      sys.exit('Exiting because Q or ESC was pressed')
    elif k != 255:
      LOG.debug(f'k: {k}')


# -----------------------------------------------------------------------------
#
# Display in pipe processor
#
# -----------------------------------------------------------------------------

class DisplayUtils:


  def show_ctx(self, ctx, im, delay=1):
    if ctx.opts.get(PAUSED):
      self.pause_ctx(ctx)
    cv.imshow(app_cfg.CV_WINDOW_NAME, im)
    self.handle_keyboard_ctx(ctx, delay)


  def handle_keyboard(self, delay_amt=1):
    """Handle key presses
    """
    while True:
      k = cv.waitKey(delay_amt) & 0xFF
      if k == 27 or k == ord('q'):  # ESC
        cv.destroyAllWindows()
        sys.exit()
      elif k == 32 or k == 83:  # 83 = right arrow
        break
      elif k != 255:
        LOG.debug(f'k: {k}')


  def handle_keyboard_video(self, delay_amt=1):
    key = cv.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
      return True


  def pause_ctx(self, ctx):
    """Handle pause during pipeline processing
    """
    ctx.opts[PAUSED] = True


  def handle_keyboard_ctx(self, ctx, opt_delay):
    """Handle keyboard input during pipeline processing
    """

    ctx.opts.setdefault('display_previous', False)
    ctx.opts.setdefault(PAUSED, False)

    while True:

      k = cv.waitKey(opt_delay) & 0xFF
      
      if k == 27 or k == ord('q'):  # ESC
        # exits the app
        cv.destroyAllWindows()
        sys.exit('Exiting because Q or ESC was pressed')
      elif k == ord(' '):
        if ctx.opts.get(PAUSED):
          ctx.opts[PAUSED] = False
          break
        else:
          self.pause_ctx(ctx)
      if k == 81: # left arrow key
        LOG.info('Previous not yet working')
        break
      elif k == 83: # right arrow key
        #LOG.info('next')
        break
      elif k == ord('h'):
          LOG.info("""
            keyboard controls:
            q           : quit
            h           : show this help
            space       : pause/unpause
        """)
      if not ctx.opts.get(PAUSED):
        break