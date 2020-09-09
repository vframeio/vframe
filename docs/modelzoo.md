*This module is still under development. Code is subject to major changes.*

# Model Zoo

The Model Zoo is a collection of neural network computer vision models useful for inferring metadata about videos. It includes object detection, image classification, semantic segmentation, and superresolution with many more to come from VFRAME and contributors.

For compatibility and long term support, the Model Zoo aims to only use models that are compatible with OpenCV. For now this includes Caffe, Darknet, and TensorFlow models. More work is needed to port PyTorch/ONNX models. A few other model types are included for comparison (eg MXNet).


## ModelZoo Utility Scripts

```
# list modelzoo commands
./cli.py modelzoo

# list models
./cli.py modelzoo list

# list models and group by attribute
./cli.py modelzoo list -g output

# download model (models also auto-download)
./cli.py modelzoo download -m caffe-imagenet-bvlc-alexnet

# download all models
./cli.py modelzoo download --all

# run basic inference test
./cli.py modelzoo test -m caffe-imagenet-bvlc-alexnet

# benchmark model fps
./cli.py modelzoo fps -m caffe-imagenet-bvlc-alexnet

# benchmark multiple models to csv
./cli.py modelzoo fps \
    -m caffe-imagenet-bvlc-alexnet \
    -m caffe-imagenet-bvlc-googlenet \
    -m caffe-imagenet-bvlc-googlenet \
    -m caffe-places365-vgg16 \
    -m caffe-places365-imagenet1k-vgg16 \
    -o ../data/modelzoo_fps.csv
```


## Adding new models:

Config files for object detection will need the unconnected layers. Run the layers script to get a list of connected layers and their output size. For object detection use `--type unconnected`. For classification networks use `--type connected`. 

```
# object detection connected layers
./cli.py modelzoo layers -m yolov3_coco --type unconnected

# image classification unconnected layers
./cli.py modelzoo layers -m caffe_places365_googlenet --type connected
```


## Benchmarking

Inference on CPUs will likely be slow. To benchmark speeds on your computer run:
```
# Benchmark AlexNet Places365 classifications for gpu/cpu
./cli.py modelzoo fps -m  caffe_places365_alexnet --gpu
./cli.py modelzoo fps -m caffe_places365_alexnet --cpu

# Benchmark YOLO V3 COCO for gpu/cpu
./cli.py modelzoo fps -m yolov3_coco --gpu
./cli.py modelzoo fps -m yolov3_coco --cpu

# Output to CSV
./cli.py modelzoo fps -m yolov3_coco --cpu  # for CPU
```


## YAML Config Files

- Add the license tag code: https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository#choosing-the-right-license



## Uploading New Models

If you want to host your own Model Zoo distribution server, use the upload script to synchronize models to your S3 server:
```
# upload (requires S3 account credentials in your .env)
./cli.py upload -m caffe-imagenet-bvlc-alexnet
```