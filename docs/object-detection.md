# Object Detection

Use object detection with custom models to detect, quantity, and filter images and vidoes.

## Detection Processing Scripts

Detection using the `pipe` command. To start a detection processing pipe add the `detect` command and specify a model. For example:
```
# open --> detect --> draw --> display
vf pipe open -i /path/to/video.mp4 detect -m rbk250 draw display
```

To save the output as JSON formatted data add the `save-json` command. For example:
```
# open --> detect --> save-json
vf pipe open -i /path/to/video.mp4 detect -m rbk250 save-json -o ~/Downloads/detections.json
```

## Advanced Detection Processing



## Summary Scripts

A utility script `summarize-json` is helpful to summarize the output when working with large collections of media. Be sure to include the labels you want to summarize. For example:
```
vf utils summarize-json -i ~/Downloads/detections.json --label rbk250
```
