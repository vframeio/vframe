*This module is still under development. Code is subject to major changes.*

# Model Zoo

The Model Zoo is a collection of neural network computer vision models useful for inferring metadata about videos. It includes object detection and image classification with many more to come from VFRAME and contributors.

For compatibility and long term support, the Model Zoo aims to only use models that are compatible with OpenCV. For now this includes Caffe, Darknet, and TensorFlow models. More work is needed to port PyTorch/ONNX models. A few other model types are included for comparison (eg MXNet).


## ModelZoo Utility Scripts

```
# list modelzoo commands
./cli.py utils.modelzoo

# list models
./cli.py utils.modelzoo list

# list models and group by attribute
./cli.py utils.modelzoo list -g output

# download model (models also auto-download)
./cli.py utils.modelzoo download -m caffe-imagenet-bvlc-alexnet

# download all models
./cli.py utils.modelzoo download --all

# run basic inference test
./cli.py utils.modelzoo test -m caffe-imagenet-bvlc-alexnet

# benchmark model fps
./cli.py utils.modelzoo fps -m caffe-imagenet-bvlc-alexnet

# benchmark model fps to csv
./cli.py utils.modelzoo fps -m caffe-imagenet-bvlc-alexnet -o ~/Downloads/fps.csv

# benchmark multiple models to csv
./cli.py utils.modelzoo fps \
    -m caffe-imagenet-bvlc-alexnet \
    -m caffe-imagenet-bvlc-googlenet \
    -m caffe-imagenet-bvlc-googlenet \
    -m caffe-places365-vgg16 \
    -m caffe-places365-imagenet1k-vgg16 \
    -o ~/Downloads/fps.csv
```


## Adding new models:

Config files for object detection will need the unconnected layers. Run the layers script to get a list of connected layers and their output size. For object detection use `--type unconnected`. For classification networks use `--type connected`. 

```
# object detection connected layers
./cli.py utils.modelzoo layers -m yolo3-coco --type unconnected

# image classification unconnected layers
./cli.py utils.modelzoo layers -m caffe-places365-googlenet --type connected
```


## YAML Config Files

- Add the license tag code: https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository#choosing-the-right-license


## Uploading New Models

If you want to host your own Model Zoo distribution server, use the upload script to synchronize models to your S3 server:
```
# upload (requires S3 account credentials in your .env)
./cli.py upload -m caffe-imagenet-bvlc-alexnet
```


## Converting TensorFlow Models to OpenCV DNN

Under development. Further reading:
- https://medium.com/@sathualab/how-to-use-tensorflow-graph-with-opencv-dnn-module-3bbeeb4920c5
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
- https://github.com/nvnnghia/opencv-Image_classification
- https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API