# Examples


## Blur Faces

First set environment variables:
```
source ../data/media/examples/examples.env
export $=../data/media/examples/faces.jpg
export $im_dir=../data/media/examples/faces/
export $im_out=../data/media/examples/faces.jpg
```

Open an image and blur all faces:
```
./cli.py pipe open -i $face_image detect -m yoloface blur display
```

Open a directory of images and blur all faces:
```
./cli.py pipe open -i $face_dir detect -m yoloface blur display
```

Open a directory of images, blur all faces, and save blurred images:
```
./cli.py pipe open -i  $face_dir detect -m yoloface blur export_images -o $face_dir_out
```

## Face Redaction Options

Pixellate faces in single image:

```
./cli.py pipe open -i ../data/media/examples/faces.jpg detect -m yoloface blur display
```

## Visualize Object Detections

Detect basic objects in an image and draw labels

```
./cli.py pipe open -i ../data_store/media/examples/horse.jpg \
              detect -m yolo_coco \
              draw --labels \
              display
```

Save the output to an image
```
./cli.py pipe open -i ../data_store/media/examples/horse.jpg \
              detect -m yolo_coco \
              draw --labels \
              display
```


## Example 2: Detect Objects in an Image
```
# Detect COCO objects in an image
./cli.py pipe open -i ../data_store/media/examples/horse.jpg \
              detect -m yoloface \
              draw --bbox -d yoloface \
              display
```
