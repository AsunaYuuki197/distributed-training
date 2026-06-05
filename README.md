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
