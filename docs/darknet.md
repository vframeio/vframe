# YOLO Darknet

## Installation

```
# Clone AlexyAB's version of Darknet
git clone https://github.com/AlexeyAB/darknet/

# Set paths to CUDA (skip if using docker)
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
# export PATH=/usr/local/cuda/bin:$PATH

# Edit Makfile
GPU=1
CUDNN=1
CUDNN_HALF=0
OPENCV=0
AVX=1
OPENMP=1

# Edit GPU architecture. Uncomment line for your GPU
ARCH= -gencode arch=compute_61 for GTX1080Ti

# Build
make
# make clean
```


**CFG-Parameters in the `[net]` section:**

1. `[net]` section
    * `batch=1` - number of samples (images, letters, ...) which will be precossed in one batch
    * `subdivisions=1` - number of mini_batches in one batch, size `mini_batch = batch/subdivisions`, so GPU processes `mini_batch` samples at once, and the weights will be updated for `batch` samples (1 iteration processes `batch` images)
    * `width=416` - network size (width), so every image will be resized to the network size during Training and Detection
    * `height=416` - network size (height), so every image will be resized to the network size during Training and Detection
    * `channels=3` - network size (channels), so every image will be converted to this number of channels during Training and Detection
    * `inputs=256` - network size (inputs) is used for non-image data: letters, prices, any custom data
    * `max_chart_loss=20` - max value of Loss in the image `chart.png` 

**For training only**

* Contrastive loss:

    * `contrastive=1` - use Supervised contrastive loss for training Classifier (should be used with `[contrastive]` layer)
    * `unsupervised=1` - use Unsupervised contrastive loss for training Classifier on images without labels (should be used with `contrastive=1` parameter and with `[contrastive]` layer)

* Data augmentation:

    * `angle=0` - randomly rotates images during training (classification only)
    * `saturation = 1.5` - randomly changes saturation of images during training
    * `exposure = 1.5` - randomly changes exposure (brightness) during training
    * `hue=.1` - randomly changes hue (color) during training https://en.wikipedia.org/wiki/HSL_and_HSV
    * `blur=1` - blur will be applied randomly in 50% of the time: if `1` - will be blured background except objects with `blur_kernel=31`, if `>1` - will be blured whole image with `blur_kernel=blur` (only for detection and if OpenCV is used)
    * `min_crop=224` - minimum size of randomly cropped image (classification only)
    * `max_crop=448` - maximum size of randomly cropped image (classification only)
    * `aspect=.75` - aspect ration can be changed during croping from `0.75` - to `1/0.75` (classification only)
    * `letter_box=1` - keeps aspect ratio of loaded images during training (detection training only, but to use it during detection-inference - use flag `-letter_box` at the end of detection command)
    * `cutmix=1` - use CutMix data augmentation (for Classifier only, not for Detector)
    * `mosaic=1` - use Mosaic data augmentation (4 images in one)
    * `mosaic_bound=1` - limits the size of objects when `mosaic=1` is used (does not allow bounding boxes to leave the borders of their images when Mosaic-data-augmentation is used)
    * data augmentation in the last `[yolo]`-layer
        * `jitter=0.3` - randomly changes size of image and its aspect ratio from x`(1 - 2*jitter)` to x`(1 + 2*jitter)`
        * `random=1` - randomly resizes network size after each 10 batches (iterations) from `/1.4` to `x1.4` with keeping initial aspect ratio of network size
    * `adversarial_lr=1.0` - Changes all detected objects to make it unlike themselves from neural network point of view. The neural network do an adversarial attack on itself
    * `attention=1` - shows points of attention during training
    * `gaussian_noise=1` - add gaussian noise

* Optimization:

    * `momentum=0.9` - accumulation of movement, how much the history affects the further change of weights (optimizer)
    * `decay=0.0005` - a weaker updating of the weights for typical features, it eliminates dysbalance in dataset (optimizer) http://cs231n.github.io/neural-networks-3/
    * `learning_rate=0.001` - initial learning rate for training
    * `burn_in=1000` - initial burn_in will be processed for the first 1000 iterations, `current_learning rate = learning_rate * pow(iterations / burn_in, power) = 0.001 * pow(iterations/1000, 4)` where is `power=4` by default
    * `max_batches = 500200` - the training will be processed for this number of iterations (batches)
    * `policy=steps` - policy for changing learning rate: `constant (by default), sgdr, steps, step, sig, exp, poly, random` (f.e., if `policy=random` - then current learning rate will be changed in this way `= learning_rate * pow(rand_uniform(0,1), power)`)
    * `power=4` - if `policy=poly` - the learning rate will be `= learning_rate * pow(1 - current_iteration / max_batches, power)`
    * `sgdr_cycle=1000` - if `policy=sgdr` - the initial number of iterations in cosine-cycle
    * `sgdr_mult=2` - if `policy=sgdr` - multiplier for cosine-cycle https://towardsdatascience.com/https-medium-com-reina-wang-tw-stochastic-gradient-descent-with-restarts-5f511975163
    * `steps=8000,9000,12000` - if `policy=steps` - at these numbers of iterations the learning rate will be multiplied by `scales` factor
    * `scales=.1,.1,.1` - if `policy=steps` - f.e. if `steps=8000,9000,12000`, `scales=.1,.1,.1` and the current iteration number is `10000` then `current_learning_rate = learning_rate * scales[0] * scales[1] = 0.001 * 0.1 * 0.1 = 0.00001`
    * `label_smooth_eps=0.1` - use label smoothing for training Classifier
    * `focal_loss=` modify the loss function to handle class imbalance. `counters_per_class` is more effective, but can try both. set `focal_loss=1` under each of the 3 `[yolo]` layers to enable
    * `counters_per_class=` set `counters_per_class=100, 2000, 300, ...` under each of the 3 `[yolo]` layers

## CLI flags

```
int dont_show = find_arg(argc, argv, "-dont_show");
int benchmark = find_arg(argc, argv, "-benchmark");
int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
//if (benchmark_layers) benchmark = 1;
if (benchmark) dont_show = 1;
int show = find_arg(argc, argv, "-show");
int letter_box = find_arg(argc, argv, "-letter_box");
int calc_map = find_arg(argc, argv, "-map");
int map_points = find_int_arg(argc, argv, "-points", 0);
check_mistakes = find_arg(argc, argv, "-check_mistakes");
int show_imgs = find_arg(argc, argv, "-show_imgs");
int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
int json_port = find_int_arg(argc, argv, "-json_port", -1);
char *http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
char *outfile = find_char_arg(argc, argv, "-out", 0);
char prefix = find_char_arg(argc, argv, "-prefix", 0);
float thresh = find_float_arg(argc, argv, "-thresh", .25); // 0.24
float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5); // 0.5 for mAP
float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
int cam_index = find_int_arg(argc, argv, "-c", 0);
int frame_skip = find_int_arg(argc, argv, "-s", 0);
int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
int width = find_int_arg(argc, argv, "-width", -1);
int height = find_int_arg(argc, argv, "-height", -1);
// extended output in test mode (output of rect bound coords)
// and for recall mode (extended output table-like format with results for best_class fit)
int ext_output = find_arg(argc, argv, "-ext_output");
int save_labels = find_arg(argc, argv, "-save_labels");
char chart_path = find_char_arg(argc, argv, "-chart", 0);
```

## General tips

- Ensure that all objects are labeled. Unlabeled objects are scored negatively
- Dataset should include objects with varying scales, resolution, lighting, angles, backgrounds and include about 2,000 different images for each class
- Use negative samples (images that do not contain any of classes) to improve results. these are included by adding empty .txt files. Use as many negative as positive samples.
- For training for small objects set `layers = -1, 11` instead of <https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L720> and set `stride=4` instead of <https://github.com/AlexeyAB/darknet/blob/6390a5a2ab61a0bdf6f1a9a6b4a739c16b36e0d7/cfg/yolov3.cfg#L717>


## Start, stop, resume training:
    - After 1.000 iterations use multi-gpu: `bash run_train_resume.sh`
    - Can stop training when average loss < 0.6

## Evaluate
```
./darknet detector map meta.data yolo4.cfg backup/yolov4_last.weights
```

