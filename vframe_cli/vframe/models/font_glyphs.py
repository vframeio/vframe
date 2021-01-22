from typing import Dict, Tuple, List
from dataclasses import dataclass, field, asdict
import xml.etree.ElementTree as etree
import logging

from vframe.utils.file_utils import write_txt

log = logging.getLogger('vframe')

@dataclass
class FontMetadata:
  font_family: str
  font_name: str
  human_name: str=None
  weight: str='Regular'
  version: str='1.0'
  author: str=''
  copyright: str=''
  license: str='MIT'
  notes: str='Generated with VFRAME.io FontGen Plugin'
  target_glyph: str=None
  


@dataclass
class FontGlyph:
  unicode: str
  layer: str
  path: str=''
  active: bool=True
  score: float=0.0
    
  def add_path_data(self, path):
    self.path = path
  

  def to_xml(self):
    xml = f'<glyph glyph-name="{self.unicode}" id="{self.layer}" unicode="{self.unicode}" d="{self.path}" />'
    return xml

  def to_dict(self):
    return {
      'unicode': self.unicode,
      'layer': self.layer,
      'score': self.score,
      'path': self.path,
      'active': self.active
    }
  


@dataclass
class FontConfig:
  metadata: FontMetadata
  glyphs: List[FontGlyph]
  chars: List[str] = field(default_factory=lambda: [])

  def __post_init__(self):
    self.glyph_dict = {g.layer: g for g in self.glyphs}
    self.chars = [g.unicode for g in self.glyphs]

  def to_dict(self):
    return {
    'metadata': dict(asdict(self.metadata)),
    'glyphs': [asdict(g) for g in self.glyphs]
    }
  
  def add_glyph_score(self, scores):
    for k,v in scores.items():
      self.glyph_dict[k].score = v

  def status_report(self):
    msg = []
    for name, g in self.glyph_dict.items():
      if g.path is None:
        msg.append(f'{g.unicode}: no path data')
    return '\n'.join(msg)
  
  
  def layer_to_glyph(self, name):
    return self.glyph_dict.get(name, None)
  

  def parse_svg(self, fp_svg):
    """Parses SVG XML data and adds path data to glyphs 
    according to layer names in YAML config
    """
    # load SVG and get namesapce
    tree = etree.parse(fp_svg)
    root = tree.getroot()
    xmlns = root.tag
    xmlns = xmlns[0:xmlns.index('}')+1]
    paths = tree.findall(f'.//{xmlns}path')
    
    # iterate paths, appending data to glyphs
    for path in paths:
      path_id = path.get('id')
      if path_id is None:
        continue
      if path_id.endswith('_1_'):
        # replace this suffix automatically added by Illustrator
        path_id = path_id.replace('_1_','')
      if path_id not in self.glyph_dict.keys():
        continue
      d = path.get('d')
      # strip whitespace from .ai --> .svg files
      while '  ' in d:
        d = d.replace('  ', ' ')
      g = self.glyph_dict.get(path_id, None)
      if g is not None and g.active:
        g.add_path_data(d)
      else:
        print('skipping', g)
    

  def to_svg(self, fp_out):
    """Writes SVG to file
    """

    # TODO verify 1000 or 1024 template size
    # assumes font file is based on 1000 x 1000 px template
    svg_head = """<?xml version="1.0" encoding="utf-8"?>
      <!-- Generator: VFRAME.io FontGen 0.1 -->
      <svg 
      version="1.0" 
      xmlns:svg="http://www.w3.org/2000/svg"
      xmlns="http://www.w3.org/2000/svg"
      xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:cc="http://creativecommons.org/ns#"
      xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      x="0px"
      y="0px"
      width="1000px"
      height="1000px"
      viewBox="0 0 1000 1000"
      style="enable-background:new 0 0 1000 1000;">
      """

    # add glyph XML
    glyph_list = [g.to_xml() for g in self.glyph_dict.values() if g.path is not None]
    n_glyphs = len(glyph_list)
    glyphs = "\n\t\t".join(glyph_list)

    defs_head = f"""<defs id="defs{n_glyphs}">
        <font horiz-adv-x="1024" id="{self.metadata.font_name}" inkscape:label="{self.metadata.font_name}">
          <font-face units-per-em="1024" id="{self.metadata.font_name}" font-family="{self.metadata.font_family}" />
          <missing-glyph d="M0,0h1000v1000h-1000z" id="missing-glyph" />
        """

    defs_tail = """</font>
    </defs>"""

    # static font metadata
    metadata = """<metadata id="metadata">
      <rdf:RDF>
        <cc:Work rdf:about="">
          <dc:format>image/svg+xml</dc:format>
          <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage" />
          <dc:title />
        </cc:Work>
      </rdf:RDF>
    </metadata>
    """

    footer = "</svg>"

    xml = svg_head + defs_head + glyphs + defs_tail + metadata + footer

    # write to .svg file
    try:
      tree = etree.ElementTree(etree.fromstring(xml))
      tree.write(fp_out)
    except Exception as e:
      log.error(e)
