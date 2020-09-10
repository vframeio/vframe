############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import click

@click.command('')
@click.option('-i', '--input', 'opt_input', required=True,
  help='Input JSON file or directory')
@click.option('-o', '--output', 'opt_output', required=True,
  help='Input JSON file or directory')
@click.pass_context
def cli(ctx, opt_input, opt_output):
  """Convert VFRAME JSON to CVAT XML"""

  # ------------------------------------------------
  # imports

  from os.path import join

  from vframe.utils import file_utils
  from vframe.settings import app_cfg
  from vframe.models.pipe_item import PipeContextHeader

  # ------------------------------------------------
  # start

  log = app_cfg.LOG

  items = file_utils.load_json(opt_input)
  
  for item in items:
    pipe_header = PipeContextHeader.from_dict(item)
    log.debug(pipe_header._frames_data)

"""
   <?xml version="1.0" encoding="utf-8"?>
  <annotations>
    <version>1.1</version>
    <meta>
      <task>
        <id>21</id>
        <name>new task</name>
        <size>288</size>
        <mode>interpolation</mode>
        <overlap>5</overlap>
        <bugtracker></bugtracker>
        <created>2020-07-03 10:19:35.173991+00:00</created>
        <updated>2020-07-03 10:20:18.127172+00:00</updated>
        <start_frame>0</start_frame>
        <stop_frame>287</stop_frame>
        <frame_filter></frame_filter>
        <z_order>False</z_order>
        <labels>
          <label>
            <name>face</name>
            <attributes>
            </attributes>
          </label>
        </labels>
        <segments>
          <segment>
            <id>8</id>
            <start>0</start>
            <stop>287</stop>
            <url>http://localhost:8080/?id=8</url>
          </segment>
        </segments>
        <owner>
          <username>vframe</username>
          <email></email>
        </owner>
        <assignee></assignee>
        <original_size>
          <width>320</width>
          <height>240</height>
        </original_size>
      </task>
      <dumped>2020-07-03 10:21:46.421037+00:00</dumped>
      <source>c6BGAl561No.mp4</source>
    </meta>
    <track id="0" label="face">
      <box frame="0" outside="0" occluded="0" keyframe="1" xtl="115.72" ytl="78.10" xbr="220.16" ybr="163.69">
      </box>
      <box frame="1" outside="1" occluded="0" keyframe="1" xtl="115.72" ytl="78.10" xbr="220.16" ybr="163.69">
      </box>
    </track>
  </annotations>
"""