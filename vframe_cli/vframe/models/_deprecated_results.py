from dataclasses import dataclass

@dataclass
class FaceDetectionRecord:
  filepath: str
  x1: float=0.0
  y1: float=0.0
  x2: float=0.0
  y2: float=0.0
  h: int=0
  w: int=0
  confidence: float=0.0
  status: bool=False

  def __post_init__(self):
    self.status = (self.x1 > 0 and self.y1 > 0 and self.x2 <= self.w and self.y2 <= self.h)

  def asdict(self):
    return {
      'filepath': self.filepath,
      'x1': self.x1,
      'x2': self.x2,
      'y1': self.y1,
      'y2': self.y2,
      'h': self.h,
      'w': self.w,
      'confidence': self.confidence,
      'status': int(self.status),
    }
