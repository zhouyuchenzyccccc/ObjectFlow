# Setup Instructions

This guide is organized in layers so you can install only what you need:

1. Base ObjectFlow (required)
2. LIBERO simulation and dataset regeneration (optional)
3. Point-cloud backbones and point-tracking extensions (optional)
4. Flash Attention training acceleration (optional, Linux recommended)

Recommended versions for reproducibility:

* Python `3.10.x`
* PyTorch `2.2.0`

## 1) Create Environment

```bash
conda create -n objectflow python=3.10 -y
conda activate objectflow
python -m pip install --upgrade pip setuptools wheel
```

## 2) Install PyTorch First

Install a PyTorch build that matches your CUDA/CPU setup:

```bash
# Example only. Pick the correct command for your machine:
# https://pytorch.org/get-started/locally/
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
```

## 3) Install ObjectFlow Core

```bash
git clone https://github.com/zhouyuchenzyccccc/ObjectFlow.git
cd ObjectFlow
pip install -e .
```

## 4) Optional: LIBERO Simulation + Dataset Regeneration

Use this if you run `run_libero_eval.py` or `regenerate_libero_dataset.py`.

```bash
# Install ObjectFlow optional LIBERO dependencies
pip install -e ".[libero]"

# Install LIBERO repo itself
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO

# Keep support packages aligned with project file
cd ObjectFlow
pip install -r experiments/robot/libero/libero_requirements.txt
```

## 5) Optional: Point-Cloud Extensions

Use this if you enable point backbones that require `pytorch3d` and `diffusion_policy`.

     
```bash
        export PYTHONPATH=$PYTHONPATH:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/LIBERO 
        export PYTHONPATH=$PYTHONPATH:/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/diffusion_policy

        export LIBERO_ROOT=/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/zhouyuchen/LIBERO/libero
        export LIBERO_BDDL_PATH=$LIBERO_ROOT/bddl_files
        export LIBERO_ASSETS_PATH=$LIBERO_ROOT/assets
        export LIBERO_DATASETS_PATH=/inspire/hdd/project/wuliqifa/public/dataset/libero/datasets
```

```bash
pip install -e ".[point]"
conda install -c conda-forge gcc
```

Notes:

```bash
cd ..
git clone https://github.com/facebookresearch/pytorch3d
cd pytorch3d 
python -m pip install -U pip setuptools wheel ninja
pip install -e . --no-build-isolation
```

* `pytorch3d` wheels are platform/CUDA specific. If `pip install -e ".[point]"` fails, install a matching `pytorch3d` wheel manually, then rerun.
* On Windows, `pytorch3d` installation may require extra steps or WSL/Linux.

## 6) Optional: Flash Attention 2 (Training Speed)

Install only if you need Flash Attention training acceleration.

```bash
pip install packaging ninja
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

Notes:

* Flash Attention build is best supported on Linux + NVIDIA CUDA toolchain.
* If install fails, clear the build cache and retry:

```bash
pip cache remove flash_attn
```

## 7) Quick Import Checks

Run these checks after installation to catch missing packages early:

```bash
python -c "import torch; print('torch', torch.__version__)"
python -c "import transformers, tensorflow; print('core ok')"
python -c "import h5py, mujoco, robosuite; print('libero deps ok')"
python -c "import pytorch3d; print('pytorch3d ok')"
```

If you do not use LIBERO or point-cloud modules, you can skip the related checks.