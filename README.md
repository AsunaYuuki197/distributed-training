# Distributed Training

This repository documents my learning journey into the core concepts behind modern distributed training systems used for large-scale deep learning and large language models (LLMs).

The focus is on understanding how these systems work internally rather than simply learning how to use them.

## Overview

Modern LLM training relies on distributed systems to scale computation and memory across multiple GPUs and nodes. To achieve this, various forms of parallelism are employed, including data parallelism (DDP), model parallelism (FSDP), pipeline parallelism, and more advanced techniques such as tensor parallelism.

## Collectives

Distributed training requires **communication** to exchange data between GPUs or nodes. Communication can be either **point-to-point** (one sender and one receiver) or **collective** (multiple participants). Common collective operations include **broadcast**, **all-reduce**, **all-gather**, and **reduce-scatter**, which are used to synchronize parameters, gradients, and optimizer states across workers.

A popular communication library is [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html), which provides high-performance collective operations for CUDA-based systems. Other communication backends are described in the [PyTorch Distributed Backend Documentation](https://docs.pytorch.org/docs/2.12/distributed.html#backends).

For more details on collective communication operations, see [Collectives](collectives).


## Distributed Data Parallel (DDP)

Soon

## Fully Sharded Data Parallel (FSDP)

Soon

## DeepSpeed ZeRO

Soon

## Megatron

Soon

## Code

I used [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) with the Ubuntu-24.04 distribution to run all the code, so I recommend using a Linux environment or WSL for the best compatibility and the fewest errors.

### Installation

1. Install Miniconda to provide a Python installation without requiring a system-wide Python installation (If you already have Python installed, you can jump to step 2 in Setup).

   * Linux: https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install
   * Windows: https://www.anaconda.com/docs/getting-started/miniconda/install/windows-install

2. Create a Conda environment with Python 3.12:

```bash
conda create -n dist_train python=3.12
```

## Setup

1. Activate the Conda environment:

```bash
conda activate dist_train
```

2. Install `uv`:

```bash
pip install uv
```

3. Create and activate the project virtual environment:

```bash
uv venv

# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat
```

4. Install project dependencies:

```bash
uv pip install -e .
```
