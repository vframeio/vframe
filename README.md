# VFRAME: Visual Forensics, Redaction, and Metadata Extraction

VFRAME is a computer vision toolkit designed for analyzing large media archives of images and videos. It includes a ModelZoo and a customizable plugin architecture to develop custom CLI tools. 

VFRAME is still under development and code is subject to major changes.

The recommended way to use this VFRAME is with a custom OpenCV build. This utilizes NVIDIA GPUs for DNN inference and omits unused modules. The Conda environment yaml only includes a general CPU version of OpenCV, however CPU inference is too slow for production. Several [Docker](docker/) options are also included. Follow instructions below to setup VFRAME with OpenCV CUDA DNN enabled. 

If you're having issues installing, read the [troubleshooting](docs/troubleshooting.md) guide before filing an issue.

## Setup Conda Environment

```
# Clone this repo
git clone https://github.com/vframeio/vframe

# Create Conda environment
conda env create -f environment-linux.yml  # Linux CPU (Another step required for GPU)
#conda env create -f environment-osx.yml  # MacOS CPU

# Copy and edit .env variables
cp .env-sample .env
```

Setup [OpenCV DNN inference](docs/opencv.md) for GPU acceleration (requires NVIDIA GPU)


## Test Installation
```
# cd to CLI root
cd vframe_cli

# Show list of commands
./cli.py -h
```

## ModelZoo
```
# Show list of modelzoo commands
./cli.py modelzoo

# Test a model (auto-downloads model)
./cli.py modelzoo test -m coco

# Speed test model for 20 iterations
./cli.py modelzoo benchmark -m coco --iters 20 --cpu  # use CPU
./cli.py modelzoo benchmark -m coco --iters 20 --gpu  # use GPU if available
```

Read more about the [ModelZoo](docs/modelzoo.md)

## Detect Objects
```
# detect objects using COCO model (replace "image.jpg" with your image)
./cli.py pipe open -i image.jpg detect -m coco draw display

# detect objects using OpenImages model
./cli.py pipe open -i image.jpg detect -m openimages draw display
```

Read more about [object detection](docs/object-detection.md) and the [ModelZoo](docs/modelzoo.md)

## Blur Faces
```
# Detect and blur faces in directory of images
./cli.py pipe open -i input/ detect -m yoloface redact save_image -o output/
```

Read more about [redaction](docs/redaction.md)

### Under Development

- train object detector
- synthetic data generator
- face blur with tracking
- search engine interface
- cvat management


## Acknowledgments

VFRAME gratefully acknowledges support  from the following organizations and grants:

![](docs/assets/spacer_white_10.png)

![](docs/assets/nlnet.jpg)

VFRAME received support from the NLNet Foundation and Next Generation Internet (NGI0) supported research and development of face blurring and biometric redaction tools during 2019 - 2021. Funding was provided through the NGI0 Privacy Enhancing Technologies Fund, a fund established by NLnet with financial support from the European Commission’s Next Generation Internet program. 

![](docs/assets/spacer_white_10.png)

![](docs/assets/meedan.jpg)

VFRAME development during 2019-2021 is being supported with a three-year grant by [Meedan](https://meedan.com) / Check Global. With this grant, we have developed tools to integrate computer vision in to Check's infrastructure, allowing computer vision to be deployed in the effort to verify breaking news, and carried out research and development of the synthetic data generation and training environment.

![](docs/assets/spacer_white_10.png)

![](docs/assets/bmbf.jpg)

VFRAME development in 2018 and 2019 was supported with a grant from the German Federal Ministry of Education and Research (Bundesministerium für Bildung und Forschung) and the [Prototype Fund](https://prototypefund.de). This funding allowed VFRAME to research computer vision applications in human rights, prototype annotation and processing applications, implement a large-scale visual search engine, and prototype the synthetic 3D data generation environment.

Read more about supporting VFRAME on the website [vframe.io/about](https://vframe.io/about)