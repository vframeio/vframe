# Model Zoo

The Model Zoo is a collection of nueral network computer vision models useful for inferring metadata about videos. It includes object detection, image classification, semantic segmentation, and superresolution with many more to come from VFRAME and contributors.

For compatability and long term support, the Model Zoo aims to only use models that are compatible with OpenCV. For now this includes Caffe, Darknet, and TensorFlow models. More work is needed to port PyTorch/ONNX models. A few other model types are included for comparison (eg MXNet).

If you would like to see your models included in the VFRAME Model Zoo, contact us on keybase @vframeio.


## Testing and Benchmarking

```
# list modelzoo commands
./cli.py modelzoo

# list models
./cli.py modelzoo list

# list models and group by attribute
./cli.py modelzoo list -g output

# download model (models also auto-download)
./cli.py modelzoo download -m caffe_imagenet_bvlc_alexnet

# download all models
./cli.py modelzoo download --all

# run basic inference test
./cli.py modelzoo test -m caffe_imagenet_bvlc_alexnet

# benchmark model fps
./cli.py modelzoo fps -m caffe_imagenet_bvlc_alexnet

# benchmark multiple models to csv
./cli.py modelzoo fps -m caffe_imagenet_bvlc_alexnet -m caffe_imagenet_bvlc_alexnet -o ../data/modelzoo_fps.csv

# benchmark multiple models to csv using custom sizes, with gpu, for 100 iterations.  
./cli.py modelzoo fps -m caffe_imagenet_bvlc_alexnet -m caffe_imagenet_bvlc_alexnet --size 512 512 --gpu --iters 100 -o ../data/modelzoo_fps_multi.csv
```


## Adding new models:

Config files for object detection will need the unconnected layers. Run the layers script to get a list of connected or unconnected layers and their output size:
```
# connected layers
./cli.py modelzoo layers --connected -m yolov3_coco

# unconnected layers
./cli.py modelzoo layers --unconnected -m yolov3_coco
```


## Benchmarking

Inference on CPUs will likely be slow. To benchmark speeds on your computer run:
```
# Benchmark AlexNet Places365 classifications
#./cli.py modelzoo benchmark -m  caffe_places365_alexnet --gpu  # for GPU
#./cli.py modelzoo benchmark -m caffe_places365_alexnet --cpu  # for CPU

# Benchmark YOLO V3 COCO
#./cli.py modelzoo benchmark -m yolov3_coco --gpu  # for GPU
#./cli.py modelzoo benchmark -m yolov3_coco --cpu  # for CPU

# Output to CSV
#./cli.py modelzoo benchmark -m yolov3_coco --cpu  # for CPU
```


## YAML Config Files

- Add the license tag code: https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/licensing-a-repository#choosing-the-right-license



## Uploading New Models

If you want to host your own Model Zoo distribution server, use the upload script to synchronize models to your S3 server:
```
# upload (requires S3 account credentials in your .env)
./cli.py upload -m caffe_imagenet_bvlc_alexnet
```