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

# Show list of commands
./cli.py -h

# Show list of commands in subgroup
./cli.py utils.modelzoo

# Test a model (auto-downloads model)
./cli.py utils.modelzoo test -m yolo3-coco

# Speed test model for 20 iterations
./cli.py utils.modelzoo fps -m yolo3-coco --iters 20 --cpu  # use CPU
./cli.py utils.modelzoo fps -m yolo3-coco --iters 20 --gpu  # use GPU if available
```

Read more about the [ModelZoo](docs/modelzoo.md)

## Detect Basic Objects

Under development.

## Blur Faces
```
# Detect and blur faces in directory of images
./cli.py pipe \
    open -i path/to/input/ \
    detect -m yoloface \
    blur \
    save_image -o path/to/output/
```

Read more about [redaction](docs/redaction.md)

## Plugins

Plugins extend the core scripts. The plugins are located inside `vframe/vframe_cli/plugins`. The commands can be used in combination with other plugins or with the core VFRAME commands. 

Read more about [VFRAME plugins](docs/plugins.md)

## Acknowledgments

VFRAME development during 2019-2021 is being supported with a three-year grant by [Meedan](https://meedan.com) / Check Global. With this grant, we have developed tools to integrate computer vision in to Check's infrastructure, allowing computer vision to be deployed in the effort to verify breaking news, and carried out research and development of the synthetic data generation and training environment.

VFRAME development in 2018 and 2019 was supported with a grant from the German Federal Ministry of Education and Research (Bundesministerium für Bildung und Forschung) and the [Prototype Fund](https://prototypefund.de). This funding allowed VFRAME to research computer vision applications in human rights, prototype annotation and processing applications, implement a large-scale visual search engine, and prototype the synthetic 3D data generation environment.

Read more about supporting VFRAME on the website [vframe.io/about](https://vframe.io/about)