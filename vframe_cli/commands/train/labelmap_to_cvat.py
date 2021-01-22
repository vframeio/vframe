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
@click.option('-i','--input','opt_fp_in',required=True)
@click.option('-o','--output','opt_fp_out')
@click.pass_context
def cli(ctx, opt_fp_in, opt_fp_out):
  """Labelmap YAML to CVAT JSON"""

  from dataclasses import asdict
  from pathlib import Path
  from vframe.utils.file_utils import load_yaml, write_json
  from vframe.models.annotation import LabelMaps

  if not opt_fp_out:
    dot_ext = Path(opt_fp_in).suffix
    opt_fp_out = opt_fp_in.replace(dot_ext, f'_cvat.json')

  # load labelmap
  labelmap_cfg = load_yaml(opt_fp_in, data_class=LabelMaps)
  labelmaps = [asdict(label.to_cvat_label()) for label in labelmap_cfg.labels]
  # write to json
  write_json(labelmaps, opt_fp_out, minify=False)
