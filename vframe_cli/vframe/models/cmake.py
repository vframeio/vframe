############################################################################# 
#
# VFRAME
# MIT License
# Copyright (c) 2020 Adam Harvey and VFRAME
# https://vframe.io 
#
#############################################################################

import sys
from pathlib import Path
from os.path import join, expanduser
from dataclasses import dataclass

import GPUtil

from vframe.settings.app_cfg import GPU_ARCHS


@dataclass
class CmakeOpenCV:


  # Python
  PYTHON_EXECUTABLE: str=None
  PYTHON3_EXECUTABLE: str=None
  PYTHON_DEFAULT_EXECUTABLE: str=None
  PYTHON3_NUMPY_INCLUDE_DIRS: str=None
  PYTHON3_LIBRARY: str=None
  OPENCV_PYTHON3_INSTALL_PATH: str=None
  HAVE_OPENCV_PYTHON3: bool=True
  INSTALL_PYTHON_EXAMPLES: bool=False

  # GPU
  CUDA_ARCH_BIN: str=None  # Leave None for auto-config
  #CUDNN_LIBRARY: str='your cudnn path'
  #CUDNN_INCLUDE_DIR="your cudnn include path"
  
  # options
  CMAKE_BUILD_TYPE: str='Release'
  CMAKE_INSTALL_PREFIX: str='/usr/local'
  OPENCV_GENERATE_PKGCONFIG: bool=True  # generate .pc file
  INSTALL_C_EXAMPLES: bool=False
  OPENCV_ENABLE_NONFREE: bool=True
  
  # Video options
  WITH_FFMPEG: bool=True
  BUILD_VIDEO: bool=True
  BUILD_VIDEOIO: bool=True
  BUILD_VIDEOSTAB: bool=False
  WITH_V4L: bool=True
  
  # CUDA/CUDNN
  WITH_CUDA: bool=True
  WITH_CUDNN: bool=True
  OPENCV_DNN_CUDA: bool=True
  ENABLE_FAST_MATH: bool=True
  CUDA_FAST_MATH: bool=True
  WITH_CUBLAS: bool=True

  # OpenCV contrib
  ENABLE_CONTRIB: bool=True
  OPENCV_EXTRA_MODULES_PATH: str='../../opencv_contrib/modules/'

  # Python
  BUILD_opencv_python2: bool=False
  BUILD_opencv_python3: bool=True
  BUILD_opencv_world: bool=False
  BUILD_opencv_apps: bool=False
  BUILD_SHARED_LIBS: bool=False
  BUILD_PYTHON3: bool=True
  # Tests
  BUILD_TESTS: bool=False
  BUILD_PERF_TESTS: bool=False
  BUILD_DOCS: bool=False
  # Accelerations
  WITH_IPP: bool=False
  ENABLE_PRECOMPILED_HEADERS: bool=False
  CMAKE_CXX_FLAGS: str="-U__STRICT_ANSI__"
  # modules
  BUILD_opencv_bgsegm: bool=False  # optional
  BUILD_opencv_bioinspired: bool=False  # optional
  BUILD_opencv_calib3d: bool=True  # required
  BUILD_opencv_ccalib: bool=True  # optional
  BUILD_opencv_core: bool=True  # required
  BUILD_opencv_cudaarithm: bool=True  # required
  BUILD_opencv_cudabgsegm: bool=True  # required for cuda
  BUILD_opencv_cudacodec: bool=True  # required for cuda
  BUILD_opencv_cudafeatures2d: bool=True  # required for cuda
  BUILD_opencv_cudafilters: bool=True  # required for cuda
  BUILD_opencv_cudaimgproc: bool=True  # required for cuda
  BUILD_opencv_cudalegacy: bool=True  # required for cuda
  BUILD_opencv_cudaobjdetect: bool=True  # required for cuda
  BUILD_opencv_cudaoptflow: bool=True   # ?
  BUILD_opencv_cudastereo: bool=True  # ?
  BUILD_opencv_cudawarping: bool=True  # ?
  BUILD_opencv_cudev: bool=True  # ?
  BUILD_opencv_datasets: bool=False   # optional
  BUILD_opencv_dnn: bool=True  # required
  BUILD_opencv_dnn_objdetect: bool=True  # required
  BUILD_opencv_dnn_superres: bool=True  # required
  BUILD_opencv_dpm: bool=False  # optional
  BUILD_opencv_face: bool=False  # optional
  BUILD_opencv_features2d: bool=True  # required
  BUILD_opencv_flann: bool=True  # ?
  BUILD_opencv_freetype: bool=True  # ?
  BUILD_opencv_fuzzy: bool=True  # ?
  BUILD_opencv_gapi: bool=True  # ?
  BUILD_opencv_hfs: bool=True  # ?
  BUILD_opencv_highgui: bool=True  # required
  BUILD_opencv_img_hash: bool=True  # required
  BUILD_opencv_imgcodecs: bool=True  # required
  BUILD_opencv_imgproc: bool=True  # required
  BUILD_opencv_intensity_transform: bool=True  # ?
  BUILD_opencv_line_descriptor: bool=True  # ?
  BUILD_opencv_ml: bool=True  # required
  BUILD_opencv_objdetect: bool=True
  BUILD_opencv_optflow: bool=True  # optional
  BUILD_opencv_phase_unwrapping: bool=True  # optional
  BUILD_opencv_photo: bool=True  # required for videostab xphoto
  BUILD_opencv_plot: bool=True  # required for tracking
  BUILD_opencv_quality: bool=False
  BUILD_opencv_rapid: bool=False
  BUILD_opencv_reg: bool=False
  BUILD_opencv_rgbd: bool=False
  BUILD_opencv_saliency: bool=False
  BUILD_opencv_shape: bool=False
  BUILD_opencv_stereo: bool=False  # optional
  BUILD_opencv_stitching: bool=False  # optional
  BUILD_opencv_structured_light: bool=False  # optional
  BUILD_opencv_superres: bool=True  # required
  BUILD_opencv_surface_matching: bool=True  # optional
  BUILD_opencv_text: bool=True  # required
  BUILD_opencv_tracking: bool=True  # required
  BUILD_opencv_xfeatures2d: bool=True  # optional
  BUILD_opencv_ximgproc: bool=True  # optional
  BUILD_opencv_xobjdetect: bool=True  # optional
  BUILD_opencv_xphoto: bool=True  # optional
  BUILD_opencv_examples: bool=True  # optional



  def __post_init__(self):
    
    p = sys.executable
    conda_env = str(Path(p).parent.parent)
    ver_obj = sys.version_info
    ver = f'{ver_obj.major}.{ver_obj.minor}'

    if not self.PYTHON_EXECUTABLE:
      self.PYTHON_EXECUTABLE = p
    if not self.PYTHON3_EXECUTABLE:
      self.PYTHON3_EXECUTABLE = p
    if not self.PYTHON_DEFAULT_EXECUTABLE:
      self.PYTHON_DEFAULT_EXECUTABLE = p
    if not self.PYTHON3_NUMPY_INCLUDE_DIRS:
      self.PYTHON3_NUMPY_INCLUDE_DIRS = join(conda_env, f'lib/python{ver}/site-packages/numpy/core/include')
    if not self.PYTHON3_LIBRARY:
      self.PYTHON3_LIBRARY = join(conda_env, f'lib/libpython{ver}m.so')
    if not self.OPENCV_PYTHON3_INSTALL_PATH:
      self.OPENCV_PYTHON3_INSTALL_PATH = join(conda_env, f'lib/python{ver}/site-packages')


    if self.CUDA_ARCH_BIN is None:
      # work in progress, auto-find the GPU architecture
      gpus = GPUtil.getGPUs()
      archs = []

      # match name to list of architectures
      for gpu in gpus:
        gpu_name = gpu.name.lower().replace('geforce','').strip()
        for arch, gpu_names in GPU_ARCHS.items():
          for gpu_version in gpu_names:
            if gpu_version in gpu_name:
              archs.append(arch)
        
        # in case GPU was not found, print error
        if not archs:
          app_cfg.LOG.error(f'Could not determine GPU architecture for: {gpu_name}')
      
      # concat to string
      self.CUDA_ARCH_BIN = ','.join(set(archs))