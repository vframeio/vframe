# NVIDIA


CUDA is used for accelerated GPU inference with the OpenCV DNN module. It is highly recommended to install CUDA and cuDNN if you plan to process large media datasets. The prebuilt Python packages from https://github.com/skvark/opencv-python unfortunately can not include the CUDA dependencies due to NVIDIA licensing issues ([see issue 71](https://github.com/skvark/opencv-python/issues/71)). To enable GPU acceleration, OpenCV needs to be built from source.

**This is the most challenging part of installation.**

VFRAME provides a helper\* script to generate the CMake variables for your local build:
```
# After installing CUDA and cuDNN, run:
./cli.py dev cmake -i ../data_store/cmake_configs/opencv_cmake_options.yaml -o ../3rdparty/opencv/build/build.sh
cd ../3rdparty/opencv/build/
./build.sh


```
There's a small catch, you'll need to first pip install opencv, then uninstall it after your create your build file. This is only temporary and will be addressed later. Also verify that the new build file has the right 


## Install CUDA and cuDNN

Install the NVIDIA driver, followed by the latest CUDA, and matching cuDNN.

- Add the NVIDIA repo, update, and then `apt search nvidia-driver-*`
- Find the latest version (compatible with your GPU) and `apt install nvidia-driver-450`
- Find the latest CUDA from https://developer.nvidia.com/cuda-downloads and follow [installatio guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
	- CUDA archives https://developer.nvidia.com/cuda-toolkit-archive
- Sign up for NVIDA dveloper program to download 

During installations may need to udpate your paths when using multiple versions of CUDA

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

## Install CUDA

Option 1

```
# Cuda 11.0
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run
sudo sh cuda_11.0.2_450.51.05_linux.run
# To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.0/bin
```

```
# Cuda 10.2
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
# select Continue and skip installing the driver. Only 
```

Option 2
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

## Install cuDNN

Download "cuDNN Library for Linux [x86]" from https://developer.nvidia.com/rdp/cudnn-download

```
# July 16, 2020: cudnn 8.0 for cuda 11.0
tar -zxf cudnn-11.0-linux-x64-v8.0.1.13.tgz
sudo cp cuda/include/* /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```

```
# 2019: cudnn 7.6.5 for cuda 10.2
tar -zxf cudnn-10.2-linux-x64-v7.6.5.32.tgz
cd cudnn-10.2-linux-x64-v7.6.5.32
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
