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

log = logging.getLogger('vframe')

class DisplayUtils:


  def show_ctx(self, ctx, im, pause=False, delay=1):
    if pause:
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
        log.debug(f'k: {k}')


  def handle_keyboard_video(self, delay_amt=1):
    key = cv.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
      return True


  def pause_ctx(self, ctx):
    """Handle pause during pipeline processing
    """
    ctx.opts['paused'] = True
    log.info("paused - hit space to resume, or h for help")



  def handle_keyboard_ctx(self, ctx, opt_delay):
    """Handle keyboard input during pipeline processing
    """

    ctx.opts.setdefault('display_previous', False)
    ctx.opts.setdefault('paused', False)

    while True:

      k = cv.waitKey(opt_delay) & 0xFF
      
      if k == 27 or k == ord('q'):  # ESC
        # exits the app
        cv.destroyAllWindows()
        sys.exit('Exiting because Q or ESC was pressed')
      elif k == ord(' '):
        if ctx.opts['paused']:
          ctx.opts['paused'] = False
          break
        else:
          self.pause_ctx(ctx)
      if k == 81: # left arrow key
        log.info('previvous not yet working')
        break
      elif k == 83: # right arrow key
        #log.info('next')
        break
      elif k == ord('h'):
          log.info("""
            keyboard controls:
            q           : quit
            h           : show this help
            space       : pause/unpause
        """)
      if not ctx.opts['paused']:
        break