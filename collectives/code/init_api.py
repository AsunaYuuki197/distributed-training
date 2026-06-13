import os
from datetime import timedelta
import torch.distributed as dist

# Recommended Read:
#   New process group:
#       https://docs.pytorch.org/docs/2.12/distributed.html#groups
#   Device Mesh (Higher level abstraction that manages process groups):
#       https://docs.pytorch.org/docs/2.12/distributed.html#devicemesh
#   Torchrun: https://docs.pytorch.org/docs/2.12/distributed.elastic.html
# Main Read:
#   https://docs.pytorch.org/docs/2.12/distributed.html#initialization
#   https://docs.pytorch.org/docs/2.12/distributed.html#distributed-key-value-store
#   https://docs.pytorch.org/docs/2.12/distributed.html#post-initialization
print("Distributed package available:", dist.is_available())


def explain_api():

    call = True
    if call:
        # why not `if true`, it gonna make the line below transparent in vscode
        return 'Dont call this :))'

    # Create distributed process group means all processes discorver each other
    # and establish communication channels
    dist.init_process_group(
        # communication backend, gloo for cpu, nccl for cuda gpu
        # no backend is specified, gloo, nccl created
        # combine e.g:
        # "<device_type>:<backend_name>,<device_type>:<backend_name>"
        backend='gloo',

        # URL specifying how to initialize the process group
        # default is env:// means it takes all value from args in torchrun cmd
        # another is TCP init, File init, detail at:
        # https://docs.pytorch.org/docs/2.12/distributed.html#tcp-initialization
        init_method=None,

        # set timeout for operations on process group like collectives
        # e.g, one rank crashes, others rank waiting data from that rank
        # will stop after time set => crash process
        # default nccl = 10 mins, others = 30 mins
        # nccl async means cpu continue without wait for collectives done
        # => subsequent compute is not reliable if crashed
        # TORCH_NCCL_BLOCKING_WAIT is set, cpu wait until timeout or success
        timeout=None,

        # num of proc
        world_size=-1,

        # unique ID of process or rank [0, nproc)
        rank=-1,

        # key/value store for all worker access to
        # exchange connection/address information
        # e.g:
        # rank0 127.0.0.10
        # rank1 127.0.0.20
        # rank2 127.0.0.30
        store=None,

        # additional options need, support nccl config
        pg_options=None,

        # specific device id that this process work on
        device_id=None,
    )

    print("Initialize process group: ", dist.is_initialized())
    print(f"This is world size: {dist.get_world_size()}")
    print(f"This is backend: {dist.get_backend()}")


def init_by_1st_way(store: str, rank: int, world_size: int):
    """
    Initialize the distributed process group by first way using store.\n
    This is for experiment, everything hardcode\n
    More details at:\n
    https://docs.pytorch.org/docs/2.12/distributed.html#distributed-key-value-store\n

    args:
    - store (required): TCPStore, FileStore, HashStore
    - rank (required)
    - world_size (required)

    return:
    - str
    """

    match store:

        # tcp-based store, server store save data
        # client stores connect server store through TCP
        case 'TCPStore':
            store = dist.TCPStore(
                host_name='127.0.0.1',
                port=1234,
                world_size=2,

                # which rank owns ServerStore
                # note that this rank also have ClientStore
                is_master=(rank == 0),

                # Timeout for init and methods like set, get
                timeout=timedelta(seconds=300),

                # default pytorch create socket for store
                # but if you already have another socket init
                # use this to pass socket file descriptor
                # in linux, everything represent by fd
                # e.g: socket -> fd 42, stdout -> fd 1
                # check by print(sock.fileno())
                # Reason:
                # - avoid port is used by bind port to 0
                # - port choose by another sys
                master_listen_fd=None
            )

            dist.init_process_group(
                backend='gloo',
                rank=rank,
                world_size=world_size,
                store=store
            )

            print(f"Hello from {dist.get_rank()}")

            dist.destroy_process_group()

            return f'Finished init using {store}'

        # use file to store key-value
        case 'FileStore':
            store = dist.FileStore(
                file_name="/tmp/filestore",
                world_size=world_size
            )

            dist.init_process_group(
                backend='gloo',
                rank=rank,
                world_size=world_size,
                store=store
            )

            print(f"Hello from {dist.get_rank()}")

            dist.destroy_process_group()

            print("Use `rm -f /tmp/filestore` to clear FileStore")

            return f'Finished init using {store}.'

        # thread-safe store based on hashmap
        # it can be used for threads in process
        # not across process so HashStore for
        # only process own memory, not for init_process_group
        case 'HashStore':
            store = dist.HashStore()

            try:
                dist.init_process_group(
                    backend='gloo',
                    rank=rank,
                    world_size=world_size,
                    store=store,
                    timeout=timedelta(minutes=1)
                )
            except Exception as e:
                print(e)

            return f'Finished init using {store}'
        case _:
            return "Unknown"


def init_by_2nd_way(init_method: str, rank: int, world_size: int):
    """
    Initialize the distributed process group by second way using URL.\n
    This is for experiment, everything hardcode\n
    More details at:\n
    https://docs.pytorch.org/docs/2.12/distributed.html#tcp-initialization\n
    https://docs.pytorch.org/docs/2.12/distributed.html#shared-file-system-initialization\n
    https://docs.pytorch.org/docs/2.12/distributed.html#environment-variable-initialization\n

    args:
    - init_method (required): TCP, file, env
    - rank (required)
    - world_size (required)

    return:
    - str
    """

    match init_method:

        # similar to TCPStore but hide the creation, master rank
        # just give url: `tcp://host:port` for init process group
        case 'TCP':

            dist.init_process_group(
                backend='gloo',
                init_method='tcp://127.0.0.1:1234',
                rank=rank,
                world_size=world_size,
            )

            print(f"Hello from {dist.get_rank()}")

            dist.destroy_process_group()

            return f'Finished init using {init_method}'

        # use file, but file must shared between all processes
        # and nodes (NFS /mnt/)
        # note that this file after created will be deleted
        # when process end, however must manually delete it for sure
        case 'file':

            dist.init_process_group(
                backend='gloo',

                # set init_method='file://<path-to-file>'
                init_method='file:///tmp/sharedfile',
                rank=rank,
                world_size=world_size,
            )

            print(f"Hello from {dist.get_rank()}")

            dist.destroy_process_group()

            print("Use `rm -f /tmp/sharedfile` to clear file")

            return f'Finished init using {init_method}.'

        # using environment provided, url set to 'env://', or no need to set
        case 'env':
            dist.init_process_group(
                backend='gloo',
                init_method='env://',
                rank=rank,
                world_size=world_size,
            )
            print(f"Hello from {dist.get_rank()}")

            return f'Finished init using {init_method}'
        case _:
            return "Unknown"


rank = int(os.environ["RANK"])

init_method = int(os.environ["METHOD"])

# To run this way, you must run one terminal for each process
# e.g:
# Terminal 1
# METHOD=1 RANK=0 python init_api.py
#
# Terminal 2
# METHOD=1 RANK=1 python init_api.py
# =================================================
# Actually for TCP you can use `torchrun`
# but specific --rdzv-endpoint same as TCP host:port
# Note: `--rdzv-endpoint` belongs to the torchrun rendezvous layer
# (worker discovery), while `init_method` belongs to the process-group layer
# (communication setup).
# They are different layers but can share the same TCPStore.
#
# e.g
# METHOD=2 torchrun --nnodes=1 --nproc-per-node=2 \
# --rdzv-endpoint=127.0.0.1:1234 init_api.py
#
# note that METHOD=1 dont not belong to torchrun args, it is used
# for chosing init_method in this file
if init_method == 1:
    print(init_by_1st_way(store='TCPStore', rank=rank, world_size=2))
    # print(init_by_1st_way(store='FileStore', rank=rank, world_size=2))
    # print(init_by_1st_way(store='HashStore', rank=rank, world_size=2))

elif init_method == 2:
    # method 2 using 'TCP' and 'file' still need one terminal for each process

    # print(init_by_2nd_way(init_method='TCP', rank=rank, world_size=2))
    # print(init_by_2nd_way(init_method='file', rank=rank, world_size=2))

    # some attributes need for this method to init process group include:
    # - MASTER_PORT: free port on machine 0
    # - MASTER_ADDR: address of rank 0 node
    # - WORLD_SIZE
    # - RANK
    # However, with torchrun (recommended way), we just need
    # e.g:
    # METHOD=2 torchrun --nnodes=1 --nproc-per-node=2 init_api.py
    #
    # it will create all that attributes, also automate discover process
    # without need one terminal for each process
    #
    # Note: MASTER_PORT, MASTER_ADDR is different concept with --rdzv-endpoint
    print(init_by_2nd_way(init_method='env', rank=rank, world_size=2))
