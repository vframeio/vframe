# VFRAME: Visual Forensics, Redaction, and Metadata Extraction

VFRAME is a computer vision framework designed for analyzing large media archives of images and videos. It includes a model library and a customizable plugin architecture to develop custom CLI tools. VFRAME is still under development and code is subject to major changes.


## Setup

```
# Clone this repo
git clone https://github.com/vframeio/vframe

# Create Python virtual environment, activate, upgrade
python -m venv
source venv/bin/activate
pip install pip -U

# Install VFRAME CLI with "vf" alias
pip install -e .
# or
python setup.py develop
```


## Test Installation
```
# Show list of commands
vf

# Show list of image processing commands
vf pipe

# Test model inference
vf models test
```


## Models
```
# List of models available
vf models list

# Download a test model
vf models download -m coco

# Speed test model for 20 iterations
vf models test -m coco --iters 20 --device -1  # use CPU
vf models test -m coco --iters 20 --device 0 #  use GPU 0

# Test model for 100 iterations and output CSV
vf models test -m coco -o /path/to/output.csv -d 0 --iterations 100

# Plot FPS results
vf models plot -i /path/to/output.csv

```

Read more about the [models](docs/models.md)


## Detect Objects
```
# detect objects using COCO model (replace "image.jpg" with your image)
vf pipe open -i image.jpg detect -m coco draw display
```

Read more about [object detection](docs/object-detection.md) and the [models](docs/models.md)


## Redacting (Blurring) Faces
```
# Detect and blur faces in directory of images
vf pipe open -i input/ detect -m yoloface redact save-images -o output/
```

Read more about [redaction](docs/redaction.md)


## Batch Object Detection

Convert a directory of images or video to JSON summary of detections
```
vf pipe open -i $d detect save-detections -o output/
```


## Primary TODOs

- [ ] Convert pip to poetry and publish to PyPi
- [ ] Add torchscript/tensorrt/coreml inference, remove 3rd party deps
- [ ] Add shell autocompletion
- [ ] add confidence-gradient bbox drawing
- [ ] add checksum and improved error handling for model downloads
- [ ] upgrade processors to include yolov8

## Additional TODOs

- [ ] Add OCR processor
- [ ] upgrade processors to include segmentation
- [ ] upgrade processors to include classification
- [ ] add/debug ONNX tensorrt provider
- [ ] upgrade annotation format
- [ ] create custom metrics with csv annotation format
- [ ] upgrade codebase to Python 3.10
- [ ] fix/improve skip-cnn features
- [ ] overhaul skip-* logic
- [ ] overhaul mediafile logic
- [ ] open issues/prs

---

## Acknowledgments

VFRAME gratefully acknowledges support from the following organizations and grants:

![](docs/assets/nlnet.jpg)

VFRAME received support from the NLNet Foundation and Next Generation Internet (NGI0) supported research and development of face blurring and biometric redaction tools during 2019 - 2021. Funding was provided through the NGI0 Privacy Enhancing Technologies Fund, a fund established by NLnet with financial support from the European Commission’s Next Generation Internet program. 

![](docs/assets/meedan.jpg)

VFRAME development during 2019-2021 is being supported with a three-year grant by [Meedan](https://meedan.com) / Check Global. With this grant, we have developed tools to integrate computer vision in to Check's infrastructure, allowing computer vision to be deployed in the effort to verify breaking news, and carried out research and development of the synthetic data generation and training environment.

![](docs/assets/bmbf.jpg)

VFRAME development in 2018 and 2019 was supported with a grant from the German Federal Ministry of Education and Research (Bundesministerium für Bildung und Forschung) and the [Prototype Fund](https://prototypefund.de). This funding allowed VFRAME to research computer vision applications in human rights, prototype annotation and processing applications, implement a large-scale visual search engine, and prototype the synthetic 3D data generation environment.

Read more about supporting VFRAME on the website [vframe.io/about](https://vframe.io/about)