# VFRAME: Visual Forensics, Redaction, and Metadata Extraction

VFRAME is a computer vision toolkit designed for analyzing large media archives of images and videos. It includes a ModelZoo and a customizable plugin architecture to develop custom CLI tools. 

VFRAME is still under development and code is subject to major changes.


## Setup

```
# Clone this repo
git clone https://github.com/vframeio/vframe

# Create Conda environment
cd vframe
conda env create -f environment.yml
# on osx, substitute environment-osx.yml

# CD to CLI root
cd vframe_cli

# Test a model (auto-downloads model)
./cli.py modelzoo test -m yolov3_coco

# Speed test model for 20 iterations
./cli.py modelzoo fps -m yolov3_coco
```


## Detect Basic Objects:
```
# Detect COCO objects in an image
./cli.py pipe open -i ../data_store/media/input/samples/horse.jpg \
              detect -m yolo_coco \
              draw --labels \
              display
```


## Example 1: Blur Faces

VFRAME is designed to 
Open an image and blur all the faces:

```
# Blur faces in a single image
./cli.py pipe open -i ../data_store/media/input/samples/test.jpg \
              detect -m yoloface \
              blur \
              display
```

Blur all faces in a directory of images and save redacted images:

```
# Blur faces in a single image
./cli.py pipe open -i ../data_store/media/input/samples/test.jpg \
              detect -m yoloface \
              blur \
              display
```


## Example 2: Detect Basic Objects

Detect basic objects in an image and draw labels

```
./cli.py pipe open -i ../data_store/media/input/samples/horse.jpg \
              detect -m yolo_coco \
              draw --labels \
              display
```



## Example 2: Detect Objects in an Image
```
# Detect COCO objects in an image
./cli.py pipe open -i ../data_store/media/input/samples/horse.jpg \
              detect -m yoloface \
              draw --bbox -d yoloface \
              display
```


## Plugins

Plugins extend the core scripts. The plugins are located inside `vframe/vframe_cli/plugins`. The commands can be used in combination with other plugins or with the core VFRAME commands. Follow the [plugins guide](docs/plugins.md) to create a custom plugin or add more VFRAME plugins. 

## ModelZoo

The ModelZoo enables VFRAME to detect more custom objects

## Troubleshooting

View the most commons problems during setup
- conda environment issues
- opencv issues
