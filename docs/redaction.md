*This module is still under development. Code is subject to major changes.*

# Face Redaction

![](assets/face-snowden-x1.gif)

Detect and redact faces in images and videos. Save the results to JPG, PNG, MP4, or as JSON data files. 


## Examples

First, source the filepaths environment variables
```
# source environment variables used for examples
source data/examples/filepaths.sh
```

Simple face detection and blurring for images
```
vf pipe open -i $FACE_IMAGE detect -m yoloface redact draw display
```

Simple face detection and blurring for videos
```
 # Detect and blur faces in a video
vf pipe open -i $FACE_VIDEO detect -m yoloface redact draw display --auto
```

Save images or video
```
# Save images
vf pipe open -i $FACE_IMAGE detect -m yoloface blur save-images -o $DIR_IMAGES_OUT

# Save video
vf pipe open -i $FACE_VIDEO detect -m yoloface redact draw save-video -o $DIR_VIDEO_OUT
```

Blur the detect face regions
```
# Detect and blur faces and save to new file
vf pipe open -i $FACE_IMAGE detect -m yoloface blur save-images -o $DIR_IMAGES_OUT

# Detect and blur faces and save to new file, draw face bbox
vf pipe open -i $FACE_IMAGE detect -m yoloface redact draw save-images -o $DIR_IMAGES_OUT

# rewrite as multi-line command for clarity
vf pipe \
  open -i $FACE_IMAGE \
  detect -m yoloface \
  redact \
  draw \
  save-images -o $DIR_IMAGES_OUT
```

Add `save-images` or `save-video` to the pipe commands to save redacted frames
```
# Save blurred face image
vf pipe open -i $FACE_IMAGE detect -m yoloface draw save-images $DIR_IMAGES_OUT

# Save blurred face video
vf pipe open -i $FACE_VIDEO detect -m yoloface redact draw save-video

```

## More Advanced Usage

Blur Directory of Images or Videos 
```
# Directory of JPG images
vf pipe \
  open -i ../data/images/ --exts jpg \
  detect -m yoloface \
  redact \
  save-image -o ~/Downloads/ --suffix _redacted

# Directory of MP4 videos
vf pipe
  open -i ../data/images/ --exts mp4 \
  detect -m yoloface \
  redact \
  save-video -o ~/Downloads/ --suffix _redacted
```

Expand BBox (eg: to cover ears)
```
# Expand the bounding box to blur more than the face
vf pipe \
  open -i ../data/media/input/samples/faces.jpg \
  detect -m yoloface \
  blur --expand 0.5 \
  display
```

Blur all faces in a video and export detections to JSON file
```
# Blur faces in a single image
vf pipe \
  open -i ../data/media/input/samples/faces.mp4 \
  detect -m yoloface \
  redact \
  save_data -o ../data/media/output/
```

Blur faces and save detections to JSON
```
# Blur faces in a single image
vf \
  pipe open -i ../data/media/input/samples/faces.mp4 \
  detect -m yoloface \
  redact \
  save_data -o ../data/media/output/faces.json
```

Detector ensemble
```
# Merge detections from multiple models
vf  pipe \
  open -i ../data/media/input/samples/faces.jpg \
  detect -m yoloface \
  detect -m yoloface \
  merge --to face \
  blur -n face \
  draw -c 255 255 255 -n face \
  display
```

Multi-detector ensemble with post-processing NMS
```
#!/bin/bash
# save this to detect.sh and run "bash detect.sh"

FP_VIDEOS=/path/to/videos
FP_DETECTIONS=/path/to/detections
GPU=0  # choose GPU index

export CUDA_VISIBLE_DEVICES=${GPU}; vf pipe \
       open -i ${FP_VIDEOS} -e mp4 --slice 0 1 \
       resize -w 960 -f original \
       detect -m retinaface -n retinaface_0 \
       detect -m retinaface -r 90 -n retinaface_90 \
       detect -m retinaface -r 270 -n retinaface_270 \
       merge --to retinaface  \
       save_data -o ${FP_DETECTIONS}/retinaface.json

export CUDA_VISIBLE_DEVICES=${GPU}; vf pipe \
       open -i ${FP_VIDEOS} -e mp4 --slice 0 1 \
       resize -w 960 -f original \
       detect -m yoloface -n yoloface_0 \
       detect -m yoloface -r 90 -n yoloface_90 \
       detect -m yoloface -r 270 -n yoloface_270 \
       merge --to yoloface \
       save_data -o ${FP_DETECTIONS}/yoloface.json

export CUDA_VISIBLE_DEVICES=${GPU}; vf pipe \
       open -i ${FP_VIDEOS} -e mp4 --slice 0 1 \
       resize -w 960 -f original \
       detect -m yoloface -n yoloface_0 \
       detect -m yoloface -r 90 -n yoloface_90 \
       detect -m yoloface -r 270 -n yoloface_270 \
       merge --to yoloface \
       save_data -o ${FP_DETECTIONS}/yoloface.json
```

If GPU RAM is limited, run each command separately
```
#!/bin/bash
# save this to detect.sh and run "bash detect.sh"

FP_VIDEOS=/path/to/videos
FP_METADATA=/path/to/metadata
DETECTORS=(yoloface retinaface yoloface)
ROTS=(0 90 270)

for DETECTOR in ${DETECTORS}; do
  for ROT in ${ROTS}; do
    vf pipe \
    open -i ${FP_VIDEO} -e mp4  \
    resize -w 960 -f original \
    detect -m ${DETECTOR} -n ${DETECTOR}_${ROT} -t 0.6 \
    save_data -o ${FP_METADATA}/${DETECTOR}_${ROT}.json
  done
done
```

On a remote computer create a tmux session to monitor and detach
```
tmux new -s detect
conda activate vframe
# run above script
# detach: ctl+b then d
# reattach: tmux a -t detect
```

Then merge all JSON files
```
vf dev merge-detections \
  -i [json1] \
  -i [json2] \
  -i [json3] \
  -o ${FP_METADATA}/merged.json
```

Then run NMS on the merged detections
```
vf dev merge-detections-nms \
  -i ${FP_METADATA}/merged.json \
  -o ${FP_METADATA}/merged_nms.json
```

Then blur the videos uses pre-computed detections
```
FP_OUT=/path/to/videos_blurred/
vf pipe \
  open -i ${FP}/merged.json \
  redact \
  draw -C 255 255 255 \
  save-video -o ${FP}
```

To be implemented:
- reattach audio track

## Tips

Detecting Small Faces

By default, the detectors use the image size settings in the ModelZoo YAML configuration files. These are general settings and should be tailored to your media. Supply a `--width` or `--height` argument to increase or decrease the input size. Increasing the input size helps detect smaller faces but is more computationally intensive. Decreasing the input size detects less smaller faces, but is faster.

```
# Increase size (slows down, but detects more small faces)
vf pipe 
  open -i ../data/media/input/samples/faces.jpg \
  detect -m yoloface --width 960 \
  redact \
  display

# Decrease size (speeds up, but detects less small faces)
vf pipe \
  open -i ../data/media/input/samples/faces.jpg \
  detect -m yoloface --width 448 \
  redact \
  display
```



## Credits

Development of the VFRAME face blurring tools was supported by NL Net Privacy Enhancing Technologies during 2019 - 2020.

Research and development: Adam Harvey, VFRAME