# OpenCV

This OpenCV installation guide is for Linux Ubuntu 18. It should also work for 16 and 20.

## Clone OpenCV and OpenCV Contrib

```
cd vframe/3rdparty
git clone https://github.com/opencv
git clone https://github.com/opencv_contrib
```

## Install Dependencies

First, install dependencies: 
```
sudo apt install -y \
	ffmpeg \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	libgstreamer-plugins-base1.0-dev \
	libgstreamer1.0-dev \
	libgtk-3-dev \
	libpng-dev \
	libjpeg-dev \
	libopenexr-dev \
	libtiff-dev \
	libwebp-dev
```

## Example build file

Generate a cmake file using `./cli.py dev cmake ...` or copy/edit from below

```
cmake \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
-D CUDA_ARCH_BIN=7.5 \
-D PYTHON_EXECUTABLE=/home/ubuntu/miniconda3/envs/vframe/bin/python \
-D PYTHON3_EXECUTABLE=/home/ubuntu/miniconda3/envs/vframe/bin/python \
-D PYTHON_DEFAULT_EXECUTABLE=/home/ubuntu/miniconda3/envs/vframe/bin/python \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/home/ubuntu/miniconda3/envs/vframe/lib/python3.7/site-packages/numpy/core/include \
-D PYTHON3_LIBRARY=/home/ubuntu/miniconda3/envs/vframe/lib/libpython3.7m.so \
-D OPENCV_PYTHON3_INSTALL_PATH=/home/ubuntu/miniconda3/envs/vframe/lib/python3.7/site-packages \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D HAVE_OPENCV_PYTHON3=ON \
-D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \fire
-D OPENCV_ENABLE_NONFREE=ON \
-D WITH_FFMPEG=ON \
-D BUILD_VIDEO=ON \
-D BUILD_VIDEOIO=ON \
-D BUILD_VIDEOSTAB=OFF \
-D WITH_V4L=ON \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D WITH_CUBLAS=ON \
-D ENABLE_CONTRIB=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_opencv_python3=ON \
-D BUILD_opencv_world=OFF \
-D BUILD_opencv_apps=OFF \
-D BUILD_SHARED_LIBS=OFF \
-D BUILD_PYTHON3=ON \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_DOCS=OFF \
-D WITH_IPP=OFF \
-D ENABLE_PRECOMPILED_HEADERS=OFF \
-D CMAKE_CXX_FLAGS=-U__STRICT_ANSI__ \
-D BUILD_opencv_bgsegm=OFF \
-D BUILD_opencv_bioinspired=OFF \
-D BUILD_opencv_calib3d=ON \
-D BUILD_opencv_ccalib=ON \
-D BUILD_opencv_core=ON \
-D BUILD_opencv_cudaarithm=ON \
-D BUILD_opencv_cudabgsegm=ON \
-D BUILD_opencv_cudacodec=ON \
-D BUILD_opencv_cudafeatures2d=ON \
-D BUILD_opencv_cudafilters=ON \
-D BUILD_opencv_cudaimgproc=ON \
-D BUILD_opencv_cudalegacy=ON \
-D BUILD_opencv_cudaobjdetect=ON \
-D BUILD_opencv_cudaoptflow=ON \
-D BUILD_opencv_cudastereo=ON \
-D BUILD_opencv_cudawarping=ON \
-D BUILD_opencv_cudev=ON \
-D BUILD_opencv_datasets=OFF \
-D BUILD_opencv_dnn=ON \
-D BUILD_opencv_dnn_objdetect=ON \
-D BUILD_opencv_dnn_superres=ON \
-D BUILD_opencv_dpm=OFF \
-D BUILD_opencv_face=OFF \
-D BUILD_opencv_features2d=ON \
-D BUILD_opencv_flann=ON \
-D BUILD_opencv_freetype=ON \
-D BUILD_opencv_fuzzy=ON \
-D BUILD_opencv_gapi=ON \
-D BUILD_opencv_hfs=ON \
-D BUILD_opencv_highgui=ON \
-D BUILD_opencv_img_hash=ON \
-D BUILD_opencv_imgcodecs=ON \
-D BUILD_opencv_imgproc=ON \
-D BUILD_opencv_intensity_transform=ON \
-D BUILD_opencv_line_descriptor=ON \
-D BUILD_opencv_ml=ON \
-D BUILD_opencv_objdetect=ON \
-D BUILD_opencv_optflow=ON \
-D BUILD_opencv_phase_unwrapping=ON \
-D BUILD_opencv_photo=ON \
-D BUILD_opencv_plot=ON \
-D BUILD_opencv_quality=OFF \
-D BUILD_opencv_rapid=OFF \
-D BUILD_opencv_reg=OFF \
-D BUILD_opencv_rgbd=OFF \
-D BUILD_opencv_saliency=OFF \
-D BUILD_opencv_shape=OFF \
-D BUILD_opencv_stereo=OFF \
-D BUILD_opencv_stitching=OFF \
-D BUILD_opencv_structured_light=OFF \
-D BUILD_opencv_superres=ON \
-D BUILD_opencv_surface_matching=ON \
-D BUILD_opencv_text=ON \
-D BUILD_opencv_tracking=ON \
-D BUILD_opencv_xfeatures2d=ON \
-D BUILD_opencv_ximgproc=ON \
-D BUILD_opencv_xobjdetect=ON \
-D BUILD_opencv_xphoto=ON \
-D BUILD_opencv_examples=ON \
..


echo "------------------------------------------------------------"
echo "If this looks good, run the make install command:"
echo "sudo make install -j $(nproc)"
echo "-----------------------------------------------------------"
```

Which wil

```
-- 
-- General configuration for OpenCV 4.4.0-pre =====================================
--   Version control:               4.3.0-636-g1fabe92ace
-- 
--   Extra modules:
--     Location (extra):            /work/vframe/3rdparty/opencv_contrib/modules
--     Version control (extra):     4.3.0-98-g5fae4082
-- 
--   Platform:
--     Timestamp:                   2020-07-17T13:55:32Z
--     Host:                        Linux 5.4.0-40-generic x86_64
--     CMake:                       3.16.3
--     CMake generator:             Unix Makefiles
--     CMake build tool:            /usr/bin/make
--     Configuration:               Release
-- 
--   CPU/HW features:
--     Baseline:                    SSE SSE2 SSE3
--       requested:                 SSE3
--     Dispatched code generation:  SSE4_1 SSE4_2 FP16 AVX AVX2 AVX512_SKX
--       requested:                 SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX
--       SSE4_1 (15 files):         + SSSE3 SSE4_1
--       SSE4_2 (1 files):          + SSSE3 SSE4_1 POPCNT SSE4_2
--       FP16 (0 files):            + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 AVX
--       AVX (4 files):             + SSSE3 SSE4_1 POPCNT SSE4_2 AVX
--       AVX2 (29 files):           + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2
--       AVX512_SKX (4 files):      + SSSE3 SSE4_1 POPCNT SSE4_2 FP16 FMA3 AVX AVX2 AVX_512F AVX512_COMMON AVX512_SKX
-- 
--   C/C++:
--     Built as dynamic libs?:      NO
--     C++ standard:                11
--     C++ Compiler:                /usr/bin/c++  (ver 9.3.0)
--     C++ flags (Release):         -U__STRICT_ANSI__   -fsigned-char -ffast-math -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -O3 -DNDEBUG  -DNDEBUG
--     C++ flags (Debug):           -U__STRICT_ANSI__   -fsigned-char -ffast-math -W -Wall -Werror=return-type -Werror=non-virtual-dtor -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wundef -Winit-self -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Winit-self -Wsuggest-override -Wno-delete-non-virtual-dtor -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -fvisibility-inlines-hidden -g  -O0 -DDEBUG -D_DEBUG
--     C Compiler:                  /usr/bin/cc
--     C flags (Release):           -fsigned-char -ffast-math -W -Wall -Werror=return-type -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Winit-self -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -O3 -DNDEBUG  -DNDEBUG
--     C flags (Debug):             -fsigned-char -ffast-math -W -Wall -Werror=return-type -Werror=address -Werror=sequence-point -Wformat -Werror=format-security -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self -Wpointer-arith -Wshadow -Wuninitialized -Winit-self -Wno-comment -Wimplicit-fallthrough=3 -Wno-strict-overflow -fdiagnostics-show-option -Wno-long-long -pthread -fomit-frame-pointer -ffunction-sections -fdata-sections  -msse -msse2 -msse3 -fvisibility=hidden -g  -O0 -DDEBUG -D_DEBUG
--     Linker flags (Release):      -Wl,--gc-sections -Wl,--as-needed  
--     Linker flags (Debug):        -Wl,--gc-sections -Wl,--as-needed  
--     ccache:                      NO
--     Precompiled headers:         NO
--     Extra dependencies:          ade /usr/lib/x86_64-linux-gnu/libgtk-3.so /usr/lib/x86_64-linux-gnu/libgdk-3.so /usr/lib/x86_64-linux-gnu/libpangocairo-1.0.so /usr/lib/x86_64-linux-gnu/libpango-1.0.so /usr/lib/x86_64-linux-gnu/libatk-1.0.so /usr/lib/x86_64-linux-gnu/libcairo-gobject.so /usr/lib/x86_64-linux-gnu/libcairo.so /usr/lib/x86_64-linux-gnu/libgdk_pixbuf-2.0.so /usr/lib/x86_64-linux-gnu/libgio-2.0.so /usr/lib/x86_64-linux-gnu/libgobject-2.0.so /usr/lib/x86_64-linux-gnu/libglib-2.0.so /usr/lib/x86_64-linux-gnu/libgthread-2.0.so /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib/x86_64-linux-gnu/libwebp.so /usr/lib/x86_64-linux-gnu/libpng.so /usr/lib/x86_64-linux-gnu/libz.so /usr/lib/x86_64-linux-gnu/libtiff.so /usr/lib/x86_64-linux-gnu/libImath.so /usr/lib/x86_64-linux-gnu/libIlmImf.so /usr/lib/x86_64-linux-gnu/libIex.so /usr/lib/x86_64-linux-gnu/libHalf.so /usr/lib/x86_64-linux-gnu/libIlmThread.so /usr/lib/x86_64-linux-gnu/libfreetype.so /usr/lib/x86_64-linux-gnu/libharfbuzz.so m pthread cudart_static -lpthread dl rt nppc nppial nppicc nppicom nppidei nppif nppig nppim nppist nppisu nppitc npps cublas cudnn cufft -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu
--     3rdparty dependencies:       ittnotify libprotobuf libjasper quirc
-- 
--   OpenCV modules:
--     To be built:                 aruco calib3d ccalib core cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev dnn dnn_objdetect dnn_superres features2d flann freetype fuzzy gapi hfs highgui img_hash imgcodecs imgproc intensity_transform line_descriptor ml objdetect optflow phase_unwrapping photo plot python3 superres surface_matching text tracking video videoio videostab xfeatures2d ximgproc xobjdetect xphoto
--     Disabled:                    bgsegm bioinspired datasets dpm face quality rapid reg rgbd saliency shape stereo stitching structured_light world
--     Disabled by dependency:      -
--     Unavailable:                 alphamat cnn_3dobj cvv hdf java js julia matlab ovis python2 sfm ts viz
--     Applications:                -
--     Documentation:               NO
--     Non-free algorithms:         YES
-- 
--   GUI: 
--     GTK+:                        YES (ver 3.24.20)
--       GThread :                  YES (ver 2.64.3)
--       GtkGlExt:                  NO
--     VTK support:                 NO
-- 
--   Media I/O: 
--     ZLib:                        /usr/lib/x86_64-linux-gnu/libz.so (ver 1.2.11)
--     JPEG:                        /usr/lib/x86_64-linux-gnu/libjpeg.so (ver 80)
--     WEBP:                        /usr/lib/x86_64-linux-gnu/libwebp.so (ver encoder: 0x020e)
--     PNG:                         /usr/lib/x86_64-linux-gnu/libpng.so (ver 1.6.37)
--     TIFF:                        /usr/lib/x86_64-linux-gnu/libtiff.so (ver 42 / 4.1.0)
--     JPEG 2000:                   build Jasper (ver 1.900.1)
--     OpenEXR:                     /usr/lib/x86_64-linux-gnu/libImath.so /usr/lib/x86_64-linux-gnu/libIlmImf.so /usr/lib/x86_64-linux-gnu/libIex.so /usr/lib/x86_64-linux-gnu/libHalf.so /usr/lib/x86_64-linux-gnu/libIlmThread.so (ver 2_3)
--     HDR:                         YES
--     SUNRASTER:                   YES
--     PXM:                         YES
--     PFM:                         YES
-- 
--   Video I/O:
--     DC1394:                      NO
--     FFMPEG:                      YES
--       avcodec:                   YES (58.54.100)
--       avformat:                  YES (58.29.100)
--       avutil:                    YES (56.31.100)
--       swscale:                   YES (5.5.100)
--       avresample:                NO
--     GStreamer:                   YES (1.16.2)
--     v4l/v4l2:                    YES (linux/videodev2.h)
-- 
--   Parallel framework:            pthreads
-- 
--   Trace:                         YES (with Intel ITT)
-- 
--   Other third-party libraries:
--     Lapack:                      NO
--     Eigen:                       NO
--     Custom HAL:                  NO
--     Protobuf:                    build (3.5.1)
-- 
--   NVIDIA CUDA:                   YES (ver 10.2, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:             75
--     NVIDIA PTX archs:
-- 
--   cuDNN:                         YES (ver 7.6.5)
-- 
--   OpenCL:                        YES (no extra features)
--     Include path:                /work/vframe/3rdparty/opencv/3rdparty/include/opencl/1.2
--     Link libraries:              Dynamic load
-- 
--   Python 3:
--     Interpreter:                 /home/ubuntu/miniconda3/envs/vframe/bin/python (ver 3.7.6)
--     Libraries:                   /home/ubuntu/miniconda3/envs/vframe/lib/libpython3.7m.so (ver 3.7.6)
--     numpy:                       /home/ubuntu/miniconda3/envs/vframe/lib/python3.7/site-packages/numpy/core/include (ver 1.18.3)
--     install path:                /home/ubuntu/miniconda3/envs/vframe/lib/python3.7/site-packages/cv2/python-3.7
-- 
--   Python (for build):            /home/ubuntu/miniconda3/envs/vframe/bin/python
-- 
--   Java:                          
--     ant:                         NO
--     JNI:                         NO
--     Java wrappers:               NO
--     Java tests:                  NO
-- 
--   Install to:                    /usr/local
-- -----------------------------------------------------------------
-- 
-- Configuring done
-- Generating done
-- Build files have been written to: /work/vframe/3rdparty/opencv/build
------------------------------------------------------------
If this looks good, run the make install command:
sudo make install -j 12
-----------------------------------------------------------
```