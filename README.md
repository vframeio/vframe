# VFRAME: Visual Forensics, Redaction, and Metadata Extraction

VFRAME is a computer vision toolkit designed for analyzing large media archives of images and videos. It includes a ModelZoo and a customizable plugin architecture to develop custom CLI tools. 

VFRAME is still under development and code is subject to major changes.

The recommended way to use this VFRAME is with a custom OpenCV build. This utilizes NVIDIA GPUs for DNN inference and omits unused modules. The Conda environment yaml only includes a general CPU version of OpenCV, however CPU inference is too slow for production. Several [Docker](docker/) options are also included. Follow instructions below to setup VFRAME with OpenCV CUDA DNN enabled.

## Setup Conda Environment

```
# Clone this repo
git clone https://github.com/vframeio/vframe

# Create Conda environment
conda env create -f environment-linux.yml  # Linux CPU (Another step required for GPU)
#conda env create -f environment-osx.yml  # MacOS CPU
```

Rebuild OpenCV for GPU DNN inference:
```
# pip wheels for opencv with CUDA DNN are not available due to NVIDIA licensing issues. Temporary workaround is to rebuild OpenCV with CUDA support then remove the pip-installed opencv packages

# clone opencv and contrib
git clone https://github.com/opencv 3rdparty/
git clone https://github.com/opencv_contrib 3rdparty/

# use vframe cmake generator
./cli.py dev cmake -o ../3rdparty/opencv/build/build.sh
cd ../3rdparty/opencv/build/

# run build script
sh ../3rdparty/opencv/build/build.sh

# if build script runs OK then run 
sudo make install -j $(nproc)

# uninstall pip opencv
pip uininstall opencv-python -y
```


## Run ModelZoo Test Script
```
# CD to CLI root
cd vframe_cli

# Test a model (auto-downloads model)
./cli.py modelzoo test -m yolov3_coco

# Speed test model for 20 iterations
./cli.py modelzoo fps -m yolov3_coco --iters 20 --cpu  # use CPU
./cli.py modelzoo fps -m yolov3_coco --iters 20 --gpu  # use GPU if available
```

Read more about the [ModelZoo](docs/modelzoo.md)

## Detect Basic Objects:
```
# Detect COCO objects in an image
./cli.py pipe import -i ../data/media/examples/horse.jpg \
              detect -m yolov3_coco \
              draw \
              display
```

Read more about [classification and detection](docs/examples.md)

## Blur Faces
```
# Detect and blur faces in directory of images
./cli.py pipe open -i path/to/your/images/ --exts jpg \
              detect -m yoloface \
              blur \
              save_image -o path/to/your/images_redacted/
```

Read more about [redaction](docs/redaction.md)

## Plugins

Plugins extend the core scripts. The plugins are located inside `vframe/vframe_cli/plugins`. The commands can be used in combination with other plugins or with the core VFRAME commands. 

Read more about [VFRAME plugins](docs/plugins.md)

