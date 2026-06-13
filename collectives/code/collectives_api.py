import os
import torch
import torch.distributed as dist

# Recommended Read:
#   https://docs.pytorch.org/docs/2.12/distributed.html#point-to-point-communication
#   https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-streams
#   https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams
#   https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.Work
# Main Read:
#   https://docs.pytorch.org/docs/2.12/distributed.html#synchronous-and-asynchronous-collective-operations
#   https://docs.pytorch.org/docs/2.12/distributed.html#collective-functions

# Note:
#   below code is tested on gloo backend, and cuda backend, therefore note is
#   also based on my observation when execute code on those backend. Thus,
#   may not reflex all other backend, and environment


def sync_async_explain():
    """
    "CUDA stream is linear execution that belongs to a specific device
    Operations inside each stream are serialized in the order they are created,
    but operations from different streams can execute concurrently
    in any relative order"

    aync_op = False
    synchronous means it will wait until all things get done,
    and next step can safely use the result.
    However, for CUDA, it is only safe for the same `CUDA stream`
    The reason is CUDA operations asynchronouss, means if there is
    another `CUDA stream` operate on same data with current stream
    the output may non-deterministically if not add synchronous method (check
    https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-streams)

    aync_op = True
    Asynchronous means it wont wait until all things get done,
    next step still executed, and let the things do in background.
    Thus, if modified the things before get done may lead to unexpected.
    Can use is_completed() to check if things get done, wait() to stop execute
    until things completed.
    Similar to synchronous, is_completed() returns True just gaurantees
    the default stream can safely use the results.
    wait() also to block currently active stream until operation completed
    but not the CPU, means another stream can be created and affect the result.

    e.g from pytorch:

    # Code runs on each rank.
    dist.init_process_group("nccl", rank=rank, world_size=2)
    output = torch.tensor([rank]).cuda(rank)
    s = torch.cuda.Stream()
    handle = dist.all_reduce(output, async_op=True)
    # Wait ensures the operation is enqueued, but not necessarily complete.
    handle.wait()
    # Using result on non-default stream.
    with torch.cuda.stream(s):
        s.wait_stream(torch.cuda.default_stream())
        output.add_(100)
    if rank == 0:
        # if the explicit call to wait_stream was omitted,
        # the output below will be non-deterministically 1 or 101,
        # depending on whether the allreduce overwrote
        # the value after the add completed.
        print(output)

    """


def barrier(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Explain barrier API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.barrier

    Args:
    - rank (optional)
    - world_size (optional)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    device = None

    if platform == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

    # example works
    print("Example works (loop 10000)\n")
    for i in range(10000):
        pass

    print(f"Rank {rank} finished\n")

    # Some ranks will process faster or slower than others
    # However, the next step need all ranks finished their works
    # Barrier is for that, blocks the processes, until all ranks enters it
    # But it is quite waste of resource, when one rank wait for other ranks
    # While it can execute some code that not required all ranks first
    # async supports this case
    handler = dist.barrier(
        # choose group process to synchoronize all ranks, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,

        # list of device id, for NCCL this is important
        # cuz barrier in this backend implemented by using
        # all_reduce of 1-element tensor, and need device_id
        # for allocating this tensor
        device_ids=[device.index] if platform == "cuda" else None
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means some ranks not completed "
              "and async is used, need wait\n")
        handler.wait()

    print("If you see this line after all finished print out first, \n"
          "means barrier is set \n"
          "or works haven't express the diff across ranks "
          "(try increase the loop). "
          "Also try to comment barrier\n")

    # Below is code example for monitored barrier
    """
    from datetime import timedelta

    if rank == 1:
        for i in range(1000000000):
            pass

    # monitored barrier supported for gloo backend
    # monitored barrier means timeout for ranks to enter this function
    dist.monitored_barrier(
        # choose group process to synchoronize all ranks, None is default group
        group=None,

        timeout=timedelta(seconds=1),

        # False then whenever one rank cannot enter this function, throw error
        # True then throw error contains all ranks cannot enter this function
        wait_all_ranks=False,
    )
    """


def broadcast(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Testing broadcast API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.broadcast

    Args:
    - rank (required)
    - world_size (required)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    device = None

    if platform == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

    # send tensor
    tensor = torch.tensor(123) if rank == 0 else torch.tensor(1)

    tensor = tensor.cuda() if platform == "cuda" else tensor

    print(f"Tensor at {rank}: {tensor}\n")

    handler = dist.broadcast(
        # tensor is data sent if src, recv place if dest
        # send tensor must has equal elements as recv tensor
        # Tensor type
        tensor=tensor,

        # src rank contains this data
        src=0,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,

    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means broadcast not completed "
              "and async is used, need wait\n")
        print(f"Check current Tensor at {rank}: {tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Tensor at {rank} after broadcast: {tensor}\n")

    # send pickleable object, but has serious performance and scability
    #   asymmetric pickle/unpickle time
    #   inefficient tensor communication
    #   unexpected tensor devices (pickle tensor on one cuda, unpickle
    #   get another tensor on that cuda)
    # rcm read:
    #   https://docs.pytorch.org/docs/2.12/distributed.html#object-collectives
    obj = [1, "2", {3: "Hello"}] if rank == 0 else [None, None, None]

    print(f"Object at {rank}: {obj}\n")

    dist.broadcast_object_list(
        object_list=obj,
        src=0,

        # default is None
        # Not None for obj serializing, tensor converting and moving to device
        # Need for NCCL-based process group
        device=device
    )

    print(f"Object at {rank} after broadcast: {obj}\n")


def all_reduce(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Testing all_reduce API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.all_reduce

    Args:
    - rank (required)
    - world_size (required)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    torch.cuda.set_device(rank) if platform == "cuda" else None

    # send tensor
    tensor = torch.tensor([1, 2, 3])
    tensor = tensor.cuda() if platform == "cuda" else tensor

    print(f"Tensor at {rank}: {tensor}\n")

    handler = dist.all_reduce(
        # tensor is data sent also recv place
        # Tensor type
        tensor=tensor,

        # Reduction operation
        # https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.ReduceOp
        op=dist.ReduceOp.SUM,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means all_reduce not completed "
              "and async is used, need wait\n")
        print(f"Check current Tensor at {rank}: {tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Tensor at {rank} after all_reduce: {tensor}\n")


def reduce(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Testing reduce API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.reduce

    Args:
    - rank (required)
    - world_size (required)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    torch.cuda.set_device(rank) if platform == "cuda" else None

    # send tensor
    tensor = torch.tensor([1, 2, 3])
    tensor = tensor.cuda() if platform == "cuda" else tensor

    print(f"Tensor at {rank}: {tensor}\n")

    handler = dist.reduce(
        # tensor is data sent also recv place
        # Tensor type
        tensor=tensor,

        # destination rank to store result
        dst=1,

        # Reduction operation
        # https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.ReduceOp
        op=dist.ReduceOp.SUM,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means reduce not completed "
              "and async is used, need wait\n")
        print(f"Check current Tensor at {rank}: {tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Tensor at {rank} after reduce: {tensor}\n")

    # You may seen tensor at rank 0 is quite weird
    # The reason is because the implement of reduce collective in backend
    # e.g: gloo backend do the reduce-scatter (for world-size chunks) and
    # gather to desc rank
    # https://github.com/pytorch/gloo/blob/main/gloo/reduce.cc
    #
    # e.g: gloo backend, reduce to rank 1, tensor at rank 0:
    # tensor = [1,2,3] => after reduce => [2,4,3]
    #   Rank0: [1,2] [3]; Rank1: [1,2] [3]
    #   Reduce-Scatter
    #   Rank0: [2,4] [3]; Rank1: [1,2] [6]
    #   Gather chunk to Rank1
    #   Rank0: [2,4] [3]; Rank1: [2,4] [6]
    #
    # tensor = [1,2,3,4] => after reduce => [2,4,3,4]
    # tensor = [1,2,3,4,5] => after reduce => [2,4,6,8,5]
    #
    # While NCCL backend preserves tensor at rank 0
    # tensor = [1,2,3] => after reduce => [1,2,3]
    # Check:
    #   https://github.com/NVIDIA/nccl/blob/49839dfd/src/nccl.h.in#L435-L448


def all_gather(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Testing all_gather API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.all_gather

    Args:
    - rank (required)
    - world_size (required)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    device = None

    if platform == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

    # send tensor (k-dimens)
    tensor = torch.tensor([1, 2, 3])
    tensor = tensor.cuda() if platform == "cuda" else tensor

    # recv tensor (world_size x k-dimens)
    # e.g:
    # world_size = 2, tensor size (1,3)
    # => recv size (2,tensor) with tensor has size (1,3)
    r_tensor = [torch.zeros(3, dtype=torch.int64)
                for _ in range(world_size)]
    r_tensor = [_.cuda() if platform == "cuda" else _ for _ in r_tensor]

    print(f"Tensor at {rank}: {tensor}\n")
    print(f"Receive Tensor at {rank}: {r_tensor}\n")

    # Can send complex and uneven sized tensors.
    # However, it must same size for gloo backend
    handler = dist.all_gather(
        # tensor list is recv tensor output
        # type is "list of tensor"
        tensor_list=r_tensor,

        # tensor to broadcast from current process
        tensor=tensor,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means all_gather not completed "
              "and async is used, need wait\n")
        print(f"Check current Recv Tensor at {rank}: {r_tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Recv Tensor at {rank} after all_gather: {r_tensor}\n")

    dist.barrier(
        device_ids=[device.index] if platform == "cuda" else None
    )

    # All gather into single tensor
    # receive tensor size ((world_size x 1st dimens size) x (k-1) dimens)
    # e.g:
    # world_size = 2, tensor size (1,3)
    # => recv tensor size ((2x1), 3) = (2, 3)
    r_tensor = torch.zeros(3*world_size, dtype=torch.int64)
    r_tensor = r_tensor.cuda() if platform == "cuda" else r_tensor
    print("="*10, "\nAll gather into tensor")
    print(f"Receive Tensor at {rank}: {r_tensor}\n")

    handler = dist.all_gather_into_tensor(
        output_tensor=r_tensor,

        # must have same size across rank
        input_tensor=tensor,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means all_gather_into_tensor not completed "
              "and async is used, need wait\n")
        print(f"Check current Recv Tensor at {rank}: {r_tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Recv Tensor at {rank} after all_gather_into_tensor: {r_tensor}\n")

    dist.barrier(
        device_ids=[device.index] if platform == "cuda" else None
    )

    # Send object
    obj = [1, "2", {3: "Hello"}]
    r_obj = [None]*world_size

    print("="*10, "\nAll gather Object\n")
    print(f"Object at {rank}: {obj}\n")
    print(f"Recv Object at {rank}: {r_obj}\n")

    dist.all_gather_object(
        object_list=r_obj,
        obj=obj
    )
    print(f"Recv Object at {rank} after all_gather: {r_obj}\n")


def gather(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Testing gather API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.gather

    Args:
    - rank (required)
    - world_size (required)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    device = None

    if platform == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

    # send tensor (k-dimens)
    tensor = torch.tensor([1, 2, 3])
    tensor = tensor.cuda() if platform == "cuda" else tensor

    # recv tensor (world_size x k-dimens)
    # e.g:
    # world_size = 2, tensor size (1,3)
    # => recv size (2,tensor) with tensor has size (1,3)
    r_tensor = [torch.zeros(3, dtype=torch.int64)
                for _ in range(world_size)]
    r_tensor = [_.cuda() if platform == "cuda" else _ for _ in r_tensor]

    print(f"Tensor at {rank}: {tensor}\n")
    print(f"Receive Tensor at {rank}: {r_tensor}\n")

    handler = dist.gather(
        # gather list is recv tensor output
        # type is "list of tensor"
        # must specific only for dst rank
        gather_list=r_tensor if rank == 1 else None,

        # tensor to broadcast from current process
        # must same size
        tensor=tensor,

        # destination rank to store the gather result
        dst=1,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means gather not completed "
              "and async is used, need wait\n")
        print(f"Check current Recv Tensor at {rank}: {r_tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Recv Tensor at {rank} after gather to rank 1: {r_tensor}\n")

    dist.barrier(
        device_ids=[device.index] if platform == "cuda" else None
    )

    # Send object
    obj = [1, "2", {3: "Hello"}]
    r_obj = [None]*world_size

    print("="*10, "\nGather Object\n")
    print(f"Object at {rank}: {obj}\n")
    print(f"Recv Object at {rank}: {r_obj}\n")

    dist.gather_object(
        object_gather_list=r_obj if rank == 1 else None,
        obj=obj,
        dst=1
    )
    print(f"Recv Object at {rank} after gather to rank 1: {r_obj}\n")


def scatter(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Testing scatter API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.scatter

    Args:
    - rank (required)
    - world_size (required)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    device = None

    if platform == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

    # recv tensor
    tensor = torch.zeros(3, dtype=torch.int64)
    tensor = tensor.cuda() if platform == "cuda" else tensor

    # with gloo and nccl backend, scatter_list size = world_size
    # also the size of each element must match others and the recv tensor
    scatter_list = None
    if rank == 0:
        scatter_list = [torch.tensor([1, 2, 3]) for _ in range(world_size)]
        scatter_list = [_.cuda() if platform == "cuda" else _
                        for _ in scatter_list]

    print(f"Recv Tensor at {rank}: {tensor}\n")
    print(f"Scatter list at {rank}: {scatter_list}\n")

    handler = dist.scatter(
        # scatter list is send tensor list
        # type is "list of tensor"
        # must specific only for src rank
        scatter_list=scatter_list,

        # tensor is recv tensor output
        tensor=tensor,

        # source rank that store the scatter list
        src=0,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means scatter not completed "
              "and async is used, need wait\n")
        print(f"Check current Recv Tensor at {rank}: {tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Recv Tensor at {rank} after scatter: {tensor}\n")

    dist.barrier(
        device_ids=[device.index] if platform == "cuda" else None
    )

    # Scatter object list
    # obj_list must same size of world size
    obj_list = [1, {2: "Hello"}] if rank == 0 else None
    r_obj = [None]

    print("="*10, "\nScatter Object list\n")
    print(f"Scatter Object list at {rank}: {obj_list}\n")
    print(f"Recv Object at {rank}: {r_obj}\n")

    dist.scatter_object_list(
        # must list, and 1st element will store object scattered
        scatter_object_output_list=r_obj,
        scatter_object_input_list=obj_list,
        src=0,
    )
    print(f"Recv Object at {rank} after scatter: {r_obj}\n")


def reduce_scatter(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Testing reduce_scatter API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.reduce_scatter

    Args:
    - rank (required)
    - world_size (required)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    device = None

    if platform == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

    # send tensor list, size = world_size
    # each element can be different size
    # must match the recv tensor
    tensors = [torch.tensor([9, 8]), torch.tensor([7, 6, 5])]
    if rank == 0:
        tensors = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    tensors = [_.cuda() if platform == "cuda" else _ for _ in tensors]

    # receive tensor
    r_tensor = torch.zeros(3, dtype=torch.int64)
    if rank == 0:
        r_tensor = torch.zeros(2, dtype=torch.int64)
    r_tensor = r_tensor.cuda() if platform == "cuda" else r_tensor

    print(f"Send tensor list at {rank}: {tensors}\n")
    print(f"Recv tensor at {rank}: {r_tensor}\n")

    handler = dist.reduce_scatter(
        # input_list is send/reduce-scatter tensor list
        # type is "list of tensor"
        input_list=tensors,

        # output is recv tensor output
        output=r_tensor,

        # reduction operation
        op=dist.ReduceOp.SUM,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means reduce_scatter not completed "
              "and async is used, need wait\n")
        print(f"Check current Recv Tensor at {rank}: {r_tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Recv Tensor at {rank} after reduce_scatter: {r_tensor}\n")

    dist.barrier(
        device_ids=[device.index] if platform == "cuda" else None
    )

    # Send tensor size = receive tensor size x world_size
    # e.g:
    # recv tensor size = (2,), world_size = 2
    # => send tensor size: (2x2, )
    # => send tensor = [1, 2, 3, 4] (example)
    #
    # recv tensor size = (1,2), world_size = 2
    # => send tensor size: (1,2)x2 => (2x1, 2)
    # => only 1st dimens times world_size
    # => send tensor = [[1, 2], [3, 4]] (example)
    # note:
    #   above is for gloo backend that I tested
    #
    # For NCCL backend, it is quite weird but very flexible when
    # it doesn't care how many nested dimens send tensor, recv tensor
    # Everything still work if the number of element of send tensor equals
    # the number of element of recv tensor times world_size
    #
    # e.g: world_size = 2
    # send tensor at rank 0: [[3, 4], [5, 6]] with size (2,2)
    # send tensor at rank 1: [[[1, 2], [3, 4]]] with size (1,2,2)
    # recv tensor: [[[[0, 0]]]] with size (1,1,1,2)
    #
    # They still work cuz send tensor eles = 4 = 2 * recv tensor eles
    # after reduce_scatter
    # recv tensor at rank 0: [[[[3+1, 4+2]]]] = [[[[4, 6]]]]
    # recv tensor at rank 1: [[[[5+3, 6+4]]]] = [[[[8, 10]]]]
    #
    # The reason is this line
    #   if (inputTensor.numel() != outputTensor.numel() * size_) {
    #       C10_THROW_ERROR(...);
    #   }
    # in
    # https://github.com/pytorch/pytorch/blob/v2.12.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L5171
    # numel() func just gives the number of elements in the tensor
    # therefore when inputTensor.numel() == outputTensor.numel() * size_
    # everything will fine
    #
    # Try this code:
    # `
    # tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    # if rank == 0:
    #     tensor = torch.tensor([[3, 4], [5, 6], [6, 7], [8, 9]])
    # tensor = tensor.cuda() if platform == "cuda" else tensor
    # # Receive tensor must same size across ranks
    # r_tensor = torch.zeros((2, 2), dtype=torch.int64)
    # r_tensor = r_tensor.cuda() if platform == "cuda" else r_tensor
    # `

    tensor = torch.tensor([[1, 2], [3, 4]])

    if rank == 0:
        tensor = torch.tensor([[3, 4], [5, 6]])

    tensor = tensor.cuda() if platform == "cuda" else tensor

    # Receive tensor must same size across ranks
    r_tensor = torch.zeros((1, 2), dtype=torch.int64)
    r_tensor = r_tensor.cuda() if platform == "cuda" else r_tensor

    print("="*10, "\nReduce_Scatter Tensor\n")
    print(f"Send Tensor at {rank}: {tensor}\n")
    print(f"Recv Tensor at {rank}: {r_tensor}\n")

    handler = dist.reduce_scatter_tensor(
        input=tensor,
        output=r_tensor,

        # reduction operation
        op=dist.ReduceOp.SUM,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means reduce_scatter not completed "
              "and async is used, need wait\n")
        print(f"Check current Recv Tensor at {rank}: {r_tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Recv Tensor at {rank} after reduce_scatter: {r_tensor}\n")


def all_to_all(rank: int, world_size: int, platform: str, async_op: bool):
    """
    Testing all_to_all API \n
    https://docs.pytorch.org/docs/2.12/distributed.html#torch.distributed.all_to_all_single

    Args:
    - rank (required)
    - world_size (required)
    - platform (required): "cuda" or "cpu"
    - async_op (required)

    Return
    - str
    """

    if platform not in ["cuda", "cpu"]:
        return "Not implement"

    dist.init_process_group(
        backend='gloo' if platform == "cpu" else "nccl"
    )

    device = None

    if platform == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

    # send tensor
    # default tensors divide equally by world_size
    # means r_tensor size = tensors size / world_size
    # or equal to sum of input split sizes
    # e.g: input_split_sizes on rank 0 is [2,1]
    # => tensors = 3
    tensors = torch.tensor([8, 10, 7, 9])
    if rank == 0:
        tensors = torch.tensor([0, 2, 4, 6, 3, 5])
    tensors = tensors.cuda() if platform == "cuda" else tensors

    # receive tensor
    # default r_tensor divide equally by world_size
    # or equal to sum of output split sizes
    # e.g: output_split_sizes on rank 0 is [2,1]
    # => r_tensors = 3
    r_tensor = torch.zeros(4, dtype=torch.int64)
    if rank == 0:
        r_tensor = torch.zeros(6, dtype=torch.int64)
    r_tensor = r_tensor.cuda() if platform == "cuda" else r_tensor

    print(f"Send tensor at {rank}: {tensors}\n")
    print(f"Recv tensor at {rank}: {r_tensor}\n")

    # note: In torch docs I read rn said
    # "all_to_all_single is experimental and subject to change"
    handler = dist.all_to_all_single(
        # input is send tensor that gonna scatter across ranks
        # type is tensor
        input=tensors,

        # output is recv tensor output
        output=r_tensor,

        # scatter different nums of ele for each rank
        # can be different for each rank
        # type is list, split by dim 0
        # e.g: [2, 1] means rank 0 takes 2 elements, rank 1 takes 1
        #
        # if not set, the input size or sender size
        # must divide equally by world_size
        input_split_sizes=[4, 2] if rank == 0 else [2, 2],

        # how many elements that each rank takes from all ranks
        # can be different for each rank
        # type is list, split by dim 0
        # e.g: [2, 1] means takes 2 elements from rank0 and 1 from rank1
        #
        # if not set, the output size or recv size
        # must divide equally by world_size
        # recv eles must same as send eles
        output_split_sizes=[4, 2] if rank == 0 else [2, 2],

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means all_to_all_single not completed "
              "and async is used, need wait\n")
        print(f"Check current Recv Tensor at {rank}: {r_tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Recv Tensor at {rank} after all_to_all_single: {r_tensor}\n")

    dist.barrier(
        device_ids=[device.index] if platform == "cuda" else None
    )

    # Send tensor size = world_size
    # For gloo backend, must same size on each dimens
    # means cannot do [[1,2], [3,4,5]] for tensors
    #
    # For nccl backend, complex size supported
    # e.g:
    # send at rank 0: [[1,2], [3,4,5]]
    # send at rank 1: [[1,2], [3,4,5]]
    # recv at rank 0 size: (2,2)
    # recv at rank 1 size: (2,3)
    # or
    # send at rank 0: [[3,4,5], [1,2]]
    # send at rank 1: [[1,2], [3,4,5]]
    # recv at rank 0 size: (2,3) + (2,2) = [[3,4,5], [1,2]]
    # recv at rank 1 size: (2,2) + (2,3) = [[1,2], [3,4,5]]
    tensors = list(torch.tensor([[8, 10], [7, 9]]))
    if rank == 0:
        tensors = list(torch.tensor([[4, 6], [3, 5]]))
    tensors = [_.cuda() if platform == "cuda" else _ for _ in tensors]

    # Receive tensor size = world_size
    r_tensor = list(torch.zeros((2, 2), dtype=torch.int64))
    r_tensor = [_.cuda() if platform == "cuda" else _ for _ in r_tensor]

    print("="*10, "\nAnother All_to_All API but divide equally\n")
    print(f"Send Tensor at {rank}: {tensors}\n")
    print(f"Recv Tensor at {rank}: {r_tensor}\n")

    # note: In torch docs I read rn said
    # "all_to_all is experimental and subject to change"
    handler = dist.all_to_all(
        # type "list of tensor"
        input_tensor_list=tensors,

        # type "list of tensor"
        output_tensor_list=r_tensor,

        # choose group process to send data, None is default group
        group=None,

        # sync or async op, if async return async handler,
        # and none otherwise
        async_op=async_op,
    )

    if not (handler is None or handler.is_completed()):
        print("You see this line means all_to_all not completed "
              "and async is used, need wait\n")
        print(f"Check current Recv Tensor at {rank}: {r_tensor}\n")
        handler.wait()
        print("After wait") if rank == 0 else ""

    print(f"Recv Tensor at {rank} after all_to_all: {r_tensor}\n")


rank = int(os.environ["RANK"])
collective_choice = int(os.environ["COLLECTIVE"])

# To run this, using torchrun for easyly peer discovery and env init
# `
# COLLECTIVE=1 torchrun --nproc-per-node=2 collectives_api.py
# COLLECTIVE=2 torchrun --nproc-per-node=2 collectives_api.py
# COLLECTIVE=3 torchrun --nproc-per-node=2 collectives_api.py
# COLLECTIVE=4 torchrun --nproc-per-node=2 collectives_api.py
# COLLECTIVE=5 torchrun --nproc-per-node=2 collectives_api.py
# COLLECTIVE=6 torchrun --nproc-per-node=2 collectives_api.py
# COLLECTIVE=7 torchrun --nproc-per-node=2 collectives_api.py
# COLLECTIVE=8 torchrun --nproc-per-node=2 collectives_api.py
# COLLECTIVE=9 torchrun --nproc-per-node=2 collectives_api.py
# `
match collective_choice:
    case 1:  # broadcast
        # broadcast(rank=rank, world_size=2, platform="cpu", async_op=False)
        broadcast(rank=rank, world_size=2, platform="cpu", async_op=True)

        # broadcast(rank=rank, world_size=2, platform="cuda", async_op=False)
        # broadcast(rank=rank, world_size=2, platform="cuda", async_op=True)
    case 2:  # all_reduce
        # all_reduce(rank=rank, world_size=2, platform="cpu", async_op=False)
        all_reduce(rank=rank, world_size=2, platform="cpu", async_op=True)

        # all_reduce(rank=rank, world_size=2, platform="cuda", async_op=False)
        # all_reduce(rank=rank, world_size=2, platform="cuda", async_op=True)
    case 3:  # reduce
        # reduce(rank=rank, world_size=2, platform="cpu", async_op=False)
        reduce(rank=rank, world_size=2, platform="cpu", async_op=True)

        # reduce(rank=rank, world_size=2, platform="cuda", async_op=False)
        # reduce(rank=rank, world_size=2, platform="cuda", async_op=True)
    case 4:  # all_gather
        # all_gather(rank=rank, world_size=2, platform="cpu", async_op=False)
        all_gather(rank=rank, world_size=2, platform="cpu", async_op=True)

        # all_gather(rank=rank, world_size=2, platform="cuda", async_op=False)
        # all_gather(rank=rank, world_size=2, platform="cuda", async_op=True)
    case 5:  # gather
        # gather(rank=rank, world_size=2, platform="cpu", async_op=False)
        gather(rank=rank, world_size=2, platform="cpu", async_op=True)

        # gather(rank=rank, world_size=2, platform="cuda", async_op=False)
        # gather(rank=rank, world_size=2, platform="cuda", async_op=True)
    case 6:  # scatter
        # scatter(rank=rank, world_size=2, platform="cpu", async_op=False)
        scatter(rank=rank, world_size=2, platform="cpu", async_op=True)

        # scatter(rank=rank, world_size=2, platform="cuda", async_op=False)
        # scatter(rank=rank, world_size=2, platform="cuda", async_op=True)
    case 7:  # reduce_scatter
        # reduce_scatter(rank=rank, world_size=2,
        #                platform="cpu", async_op=False)
        reduce_scatter(rank=rank, world_size=2,
                       platform="cpu", async_op=True)

        # reduce_scatter(rank=rank, world_size=2,
        #                platform="cuda", async_op=False)
        # reduce_scatter(rank=rank, world_size=2,
        #                platform="cuda", async_op=True)
    case 8:  # all-to-all
        # all_to_all(rank=rank, world_size=2, platform="cpu", async_op=False)
        all_to_all(rank=rank, world_size=2, platform="cpu", async_op=True)

        # all_to_all(rank=rank, world_size=2, platform="cuda", async_op=False)
        # all_to_all(rank=rank, world_size=2, platform="cuda", async_op=True)
    case 9:  # barrier
        # barrier(rank=rank, world_size=2, platform="cpu", async_op=False)
        barrier(rank=rank, world_size=2, platform="cpu", async_op=True)

        # barrier(rank=rank, world_size=2, platform="cuda", async_op=False)
        # barrier(rank=rank, world_size=2, platform="cuda", async_op=True)
    case _:
        pass

dist.destroy_process_group()
