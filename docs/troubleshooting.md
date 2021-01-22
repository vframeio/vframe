
# Installation Troubleshooting

## macOS Installation

Installation on macOS requires additional steps, which may change slightly with each macOS version. The following steps are provided for macOS Mojave.

Install miniconda:
- follow the instructions on <https://docs.conda.io/en/latest/miniconda.html> and install miniconda using the Python 3.8 pkg (using GUI) or bash script (using Terminal)
- if installed correct you can type `conda` in terminal and it will display a list of commands
- then type `conda init zsh` to initialize conda for your shell environment
- if there is a conda not found error this means conda did not install correctly, you can fix this by adding conda to your path
  + `export PATH=/Users/YourUserName/miniconda3/bin:$PATH`
  + then retype `conda` and hit enter

Install the VFRAME conda env:
- run `conda env create -f environment-ox.yml`
- if this fails you may need to install xcode command line tools
- run `xcode-select --install` to install them


If the conda env installation fails:
- libomp may be missing. it is required for onnxrtuntime
- `brew install libomp` to fix

Install brew
- if brew is missing, install it using the directions on <https://brew.sh>
- then try installing libomp again


## ModelZoo Issues

Models not downloading
- check to make sure your network is not blocking the download request
- if you are on a restricted environment they may block requests to the model files