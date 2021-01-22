*This module is still under development. Code is subject to major changes.*

# CVAT Guide

Use CVAT for annotating videos and images. Use VFRAME to generate annotations for CVAT.

## Setup

- Install Docker Compose
- clone <https://github.com/opencv/cvat> to `vframe/3rdparty/`
- Edit `docker-compose-override.yml` to include a shared directory
- Start docker and navigate to <http://localhost:8080/>

```
- CLI `docker exec -it cvat bash -ic "python3 utils/cli/cli.py --auth ${CVAT_USERNAME}:${CVAT_PASSWORD} ls"`
```

## Setup

First, setup a shared local directory for CVAT Docker to access. Edit `docker-compose-override.yml` and include paths for you local system.

Edit override yaml
```
# docker-compose-override.yml
version: '3.3'

services:
  cvat:
    environment:
      CVAT_SHARE_URL: "Mounted from /mnt/share host directory"
    volumes:
      - cvat_share:/home/django/share:rw  # ensure rw (not ro)

volumes:
  cvat_share:
    driver_opts:
      type: none
      device: /path/to/my/videos  # <--- Edit "/path/to/my/videos/"
      o: bind
```

Start
```
docker-compose -f docker-compose.yml -f docker-compose-override.yml up -d
```

Stop
```
docker-compose down
```

Create superuser
```
docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'
```

Change password
```
docker exec -it cvat bash -ic 'python3 ~/manage.py changepassword myusername'
```

Update CVAT and rebuild
```
cd vframe/3rdparty/cvat
git pull
docker-compose down
docker-compose up -d --no-deps --build
```

Edit and source username and password then source `vframe/.env`
```
# CVAT
CVAT_USERNAME=myusername
CVAT_PASSWORD=mypassword
```

Create labels.json and save to your shared directory
```
[
  {
    "name": "myobject",
    "attributes": []
  }
]
```

Add video and labels
```
docker exec -it cvat bash -ic \
  'python3 utils/cli/cli.py \
  --auth ${CVAT_USERNAME}:${CVAT_PASSWORD} \
  create "new task" \
  --labels /home/django/share/_data/labels.json local /home/django/share/_data/myfile.mp4
```

Delete tasks
```
docker exec -it cvat bash -ic \
  'python3 utils/cli/cli.py \
  --auth ${CVAT_USERNAME}:${CVAT_PASSWORD} \
  delete 17 18 19 20'
```

Export task annotations
```
# Example: task 1
docker exec -it cvat bash -ic \
  "python3 utils/cli/cli.py \
  --auth ${CVAT_USERNAME}:${CVAT_PASSWORD} \
  dump --format 1 /home/django/share/_data/1.zip"
```

Upload task annotations
```
# Example: task 1
docker exec -it cvat bash -ic \
  "python3 utils/cli/cli.py \
  --auth ${CVAT_USERNAME}:${CVAT_PASSWORD} \
  upload 1 /home/django/share/_data/1.xml"
```


## Generate Annotations and Convert to CVAT

Preparing videos and images for CVAT annotation:

First rename your files as their sha256
```
# Renames a directory of videos or images
./cli.py dev sha256 -i myfolder

# Verify that you want to rename everything, then run again
./cli.py dev sha256 -i myfolder --confirm
```


(under development) Process video into JSON detections
```
# set variable to video filename
export SHA256=1a0b4feb75ccec99e44a44120734d75ab1a235c274d0577a191d631297cccbc5

# Run detector and export to JSON
./cli.py pipe open -i ${SHA256}.mp4 detect -m yolov3_coco save_data -o ${SHA256}.json
```

(under development) Merge Tracks
```
# add incrementing ID to each detection
# Test, and auto-increment if no ID?
# useful for counting too?
# add interpolation option for smoothing gaps between detections with look back and ahead [-----,+++++]
./cli.py dev merge_tracks -i ${SHA256}.json -o ${SHA256}_merged.json
```

(under development) Convert JSON detections into CVAT XML
```
# Converts detections to JSON. Run reid first to merge tracks
./cli.py cvat to_xml -i ${SHA256}.json -o ${SHA256}.xml
```

(under development) Import video to CVAT and set name
```
# Create a new task
./cli.py cvat create -i ${SHA256}.mp4 --name ${SHA256}
```

(under development) Import XML to CVAT using filename to get task ID
```
# Import XML annotations to task
./cli.py cvat import --task 1 -i ${SHA256}.xml
```

(under development) Export annotations to zip
```
# Exports task to zip with XML annotations
./cli.py cvat export --task 1 -o ${SHA256}.zip

# Exports, unzips, then deletes zip
./cli.py cvat export --task 1 -o ${SHA256}.zip --unzip --cleanup
```

(under development) Delete task
```
# Deletes task by task ID
./cli.py cvat delete --task 1

# Deletes task by fileanme
./cli.py cvat delete --name ${SHA256}
```

List tasks
```
# Lists all tasks
./cli.py cvat ls

# List and write to CSV
./cli.py cvat ls -o output.csv
```

Find file
```
# Find if filename (title) exist in tasks
./cli.py cvat find --name ${SHA256}
```

## Automatic Annotation

For others that see this to look at the docker logs you can run

docker logs nuclio-nuclio-tf-faster-rcnn-inception-v2-coco-gpu

and use docker ps -a to find your nuclio function if you try a different model.

Solution was to set maxWorkers to 1 in the function.yaml file


## User Guide for Annotating Images and Video

- [CVAT user guide](https://github.com/opencv/cvat/blob/cb114b52869598db083bb553bc4baf42abbb0585/cvat/apps/documentation/user_guide.md#shortcuts)

## Dev Notes

- set `outside="1"` to end a track?
- merge id?

CVAT video annotation XML format
```
<!-- annotation -->
<box frame="194" outside="0" occluded="0" keyframe="1" xtl="127.69" ytl="73.02" xbr="174.37" ybr="134.66"></box>

<!-- marks the end of annotation, not used for training -->
<box frame="195" outside="1" occluded="0" keyframe="1" xtl="127.69" ytl="73.02" xbr="174.37" ybr="134.66"></box>
```
