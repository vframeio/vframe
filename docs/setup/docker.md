# Docker 

#### Useful Docker commands

- list names of running containers: `docker ps`
- log int to docker: `docker exec -ti -u root container_name bash`
- reload daemon: `sudo systemctl daemon-reload`
- restart docker `sudo systemctl restart docker`

#### Change Image Storage Location

- Using many docker images can use several hundred GBs of store. It's often useful to move this off your statup disk
- Edit `sudo nano /etc/docker/daemon.json`
- Add
```
{
  "data-root": "/path/to/new/docker"
}
```
- stop docker `sudo systemctl stop docker`
- check docker has stopped `ps aux | grep -i docker | grep -v grep`
- copy data to new location `sudo rsync -axPS /var/lib/docker/ /path/to/new/docker`
- `sudo rsync -axPS /var/lib/docker/ /media/ubuntu/disk_name/data_store/docker_images` to copy to your new disk


#### Permissions

- permissions still not solved, but here are useful tips: <http://www.carlboettiger.info/2014/10/21/docker-and-user-permissions-crazyness.html>


#### Installation with Docker (CPU or GPU version)

```
# Check docker.com for latest installation instructions
# Install Docker on your Ubuntu 18 server or local machine
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce
sudo systemctl status docker
sudo usermod -aG docker ${USER}
# log out/in
su - ${USER}
# confirm id/group
id -nG
docker run hello-world
```

Ensure [NVIDIA Docker](# ensure NVIDIA Docker is installed: https://github.com/NVIDIA/nvidia-docker) is installed if you want to use OpenCV DNN CUDA acceleration.

```
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

```
# replace cpu with gpu for the GPU version (e.g. bash rebuild_gpu.sh)
cd vframe_faceless/docker/base
bash rebuild_cpu.sh
cd ../process
bash rebuild_cpu.sh
bash start_cpu.sh
```

## Build

VFRAME provides Docker configurations with GPUs acceleration. To enable GPU build the Docker.gpu file in `docker/process/`. See docs/docker.md for installation notes.

```
# Build Docker images
# replace cpu with gpu for the GPU version (e.g. bash rebuild_gpu.sh)
cd vframe_faceless/docker/base
bash rebuild_cpu.sh
cd ../process
bash rebuild_cpu.sh
bash start_cpu.sh
```