# VFRAME: Visual Forensics, Redaction, and Metadata Extraction

VFRAME is a computer vision toolkit designed for analyzing large media archives of images and videos. It includes a ModelZoo and a customizable plugin architecture to develop custom CLI tools. 

VFRAME is still under development and code is subject to major changes.

The recommended way to use this repo is with building a custom version of OpenCV to utilize NVIDIA GPUs for DNN inference. See the [OpenCV CMake](docs/opencv.md) instructions for building locally. The Conda environemnts only include a CPU version of OpenCV.

## Setup Conda Environment

```
# Clone this repo
git clone https://github.com/vframeio/vframe

# Create Conda environment
conda env create -f environment-linux.yml  # Linux GPU
#conda env create -f environment-osx.yml  # MacOS CPU

```


```
# CD to CLI root
cd vframe_cli

# Test a model (auto-downloads model)
./cli.py modelzoo test -m yolov3_coco

# Speed test model for 20 iterations
./cli.py modelzoo fps -m yolov3_coco --iters 20
```


## Detect Basic Objects:
```
# Detect COCO objects in an image
./cli.py pipe import -i ../data/media/examples/horse.jpg \
              detect -m yolov3_coco \
              draw \
              display
```

See more [examples](docs/examples.md)

## Plugins

Plugins extend the core scripts. The plugins are located inside `vframe/vframe_cli/plugins`. The commands can be used in combination with other plugins or with the core VFRAME commands. Follow the [plugins guide](docs/plugins.md) to create a custom plugin or add more VFRAME plugins. 

## ModelZoo

The ModelZoo enables VFRAME to detect more custom objects

## Troubleshooting

View the most commons problems during setup
- conda environment issues
- opencv issues
