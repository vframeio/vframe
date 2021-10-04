*This module is still under development. Code is subject to major changes.*

# Model Zoo

The ModelZoo currently uses only object detection models trained using YOLOV5. Single frame inference is possible using the ONNX or PyTorch model on GPU/CPU. Batch frame inference is possible only using the PyTorch models.


## ModelZoo Utility Scripts

List and download models
```
# list modelzoo commands
./cli.py modelzoo

# list models
./cli.py modelzoo list

# download all models
./cli.py modelzoo download -m yoloface

# download all models
./cli.py modelzoo download --all

```

Test models
```
# benchmark model fps
./cli.py modelzoo benchmark -m coco

# benchmark model fps to csv
./cli.py modelzoo benchmark -m coco -o ~/Downloads/benchmark.csv

# benchmark multiple models to csv
./cli.py modelzoo benchmark \
    -m coco-sm \
    -m coco-md \
    -m coco-lg \
    -m coco-xl \
    -o ~/Downloads/benchmark.csv
```


![](assets/modelzoo/benchmark.png)*Example bar plot comparing models*

Plot benchmark
```
# plot
# file is saved to ~/Downloads/benchmark.png
/cli.py modelzoo plot -i ~/Downloads/benchmark.csv
```


