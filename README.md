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

# Or manually
pip install -r requirements.txt

# Test installation
vf
```

```
usage: vf [command]

VFRAME CLI (0.2.0)

positional arguments:
  [command]

optional arguments:
  -h, --help  show this help message and exit

Commands and plugins:
	pipe                Image processing pipeline
	models              Model scripts
	utils               Utility scripts
	dedup               Deduplication scripts
```


**Disable 3rd Party Library Analytics**

Ultralytics uses Sentry for app usage analytics. Disable it by using:
`yolo settings config=False`
or if that doesn't work edit the `settings.yaml` file in
`/home/#$SER/.config/Ultralytics/settings.yaml` (Linux) and set `sync: False`



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


## TODOs

Models
- [ ] add checksum and improved error handling for model downloads

Usability
- [ ] Ensure MacOS and M1/M2 compatibility
- [ ] Improve model format selection
- [ ] Add CSV output for simple spreadsheet sync/import
- [ ] overhaul skip-* logic
- [ ] overhaul mediafile logic
- [ ] Add shell autocompletion

Inference
- [x] add/debug ONNX tensorrt provider
- [ ] Add multi-GPU processing
- [ ] upgrade annotation format
- [ ] create custom metrics with csv annotation format
- [ ] upgrade codebase to Python 3.10

Visuals
- [ ] Add confidence-coloring
- [ ] Clean/update drawing code

New
- [ ] Add image segmentation
- [ ] Add image classification
- [ ] Add OCR
- [ ] Add skip-cnn features

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