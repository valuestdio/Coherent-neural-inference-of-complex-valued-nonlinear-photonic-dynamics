# Environment and installation

## Reference environment

These values were read from the working `wcy` environment on 2026-10-16.

| Component | Version |
| --- | --- |
| Platform | Windows x64 |
| Python | 3.12.13 |
| PyTorch | 2.5.1 |
| PyTorch CUDA runtime | 12.1 |
| cuDNN reported by PyTorch | 9.1.0 |
| GPU | NVIDIA GeForce RTX 4090 |
| NVIDIA driver | 610.88 |

The original environment uses the Conda build `pytorch-2.5.1-py3.12_cuda12.1_cudnn9_0`
and `pytorch-cuda=12.1`. `requirements.txt` selects the corresponding pip release
`torch==2.5.1+cu121`; it is not an export of the entire Conda environment.
Matching the PyTorch/CUDA versions does not guarantee identical low-level binaries
or bitwise-identical results between Conda and pip builds.

## Install

Create a separate environment, then run these commands from the project directory:

```powershell
conda create -n cnp python=3.12.13 -y
conda activate cnp
python -m pip install -r requirements.txt
```

The requirements file adds the official PyTorch CUDA 12.1 wheel index and pins
the CUDA-specific PyTorch version. The matching runtime libraries are distributed
with the wheel or installed through its dependencies; this project does not build
custom CUDA extensions, so a separate CUDA Toolkit compiler installation is not
required for ordinary training or inference. A compatible NVIDIA driver is still
required for GPU execution. The requirements file does not install a GPU driver.

Official PyTorch 2.5.1 CUDA 12.1 installation instructions:
[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/#v251).

Only packages imported by the current source files are listed. Standard-library
modules and local modules such as `tool`, `layer`, and `model` are not pip packages.
The current code does not import `torchvision`, `torchaudio`, `scipy`, or `mat73`,
so these are not included. `psutil` and `GPUtil` provide resource monitoring.

## Verify GPU support

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('cuDNN:', torch.backends.cudnn.version())"
```

The installed wheel should report `2.5.1+cu121` and CUDA `12.1`. On a compatible
GPU machine, `CUDA available` should be `True`.

The CUDA version displayed by `nvidia-smi` describes the driver's supported CUDA
version; it does not identify the CUDA runtime used by PyTorch. In the reference
environment, `nvidia-smi` reports CUDA UMD 13.3 while `torch.version.cuda` is 12.1.

The pinned direct-package versions were checked against the existing environment.
A fresh installation of this requirements file has not been performed.
