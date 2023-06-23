from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
import logging
from datetime import datetime
import os
import yaml
import random
import torch
import numpy as np
from gpustat import GPUStatCollection
import sys
import time
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta

def get_free_gpu(mode="memory", memory_need=10000) -> list:
    r"""Get free gpu according to mode (process-free or memory-free).
    Args:
        mode (str, optional): memory-free or process-free. Defaults to "memory".
        memory_need (int): The memory you need, used if mode=='memory'. Defaults to 10000.
    Returns:
        list: free gpu ids sorting by free memory
    """
    assert mode in ["memory", "process"], "mode must be 'memory' or 'process'"
    if mode == "memory":
        assert memory_need is not None, \
            "'memory_need' if None, 'memory' mode must give the free memory you want to apply for"
        memory_need = int(memory_need)
        assert memory_need > 0, "'memory_need' you want must be positive"
    gpu_stats = GPUStatCollection.new_query()
    gpu_free_id_list = []

    for idx, gpu_stat in enumerate(gpu_stats):
        if gpu_check_condition(gpu_stat, mode, memory_need):
            gpu_free_id_list.append([idx, gpu_stat.memory_free])
            print("gpu[{}]: {}MB".format(idx, gpu_stat.memory_free))

    if gpu_free_id_list:
        gpu_free_id_list = sorted(gpu_free_id_list,
                                  key=lambda x: x[1],
                                  reverse=True)
        gpu_free_id_list = [i[0] for i in gpu_free_id_list]
    return gpu_free_id_list


def gpu_check_condition(gpu_stat, mode, memory_need) -> bool:
    r"""Check gpu is free or not.
    Args:
        gpu_stat (gpustat.core): gpustat to check
        mode (str): memory-free or process-free.
        memory_need (int): The memory you need, used if mode=='memory'
    Returns:
        bool: gpu is free or not
    """
    if mode == "memory":
        return gpu_stat.memory_free > memory_need
    elif mode == "process":
        for process in gpu_stat.processes:
            if process["command"] == "python":
                return False
        return True
    else:
        return False

def get_device(gpu_check_list, memory_need=10000):
    # check device
    target_gpus = list(filter(lambda x: x is not None, gpu_check_list))
    # assert torch.cuda.device_count() >= len(target_gpus), 'do you set the gpus in config correctly?'
    flag = None

    while True:
        # Get the gpu ids which have more than 10000MB memory
        free_gpu_ids = get_free_gpu('memory', memory_need)
        if len(free_gpu_ids) < 1:
            if flag is None:
                print("No GPU available now. Wait or Exit? y/n")
                flag = input()
                if flag.strip() == 'y':
                    continue
                else:
                    device = torch.device('cpu')
                    print("Using device %s as main device" % device)
                    break
            time.sleep(60)
        else:
            free_target_gpu = list(set(free_gpu_ids) & set(target_gpus))
            if len(free_target_gpu) == 0:
                gpuid = free_gpu_ids[0]
                print(f"no target GPU is not available")
            else:
                gpuid = free_target_gpu[0]

            device = torch.device('cuda:'+str(gpuid))
            print("Using device %s as main device" % device)
            break

    return device

def get_config_easydict(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
  """

  Parallel map using joblib.

  Parameters
  ----------
  pickleable_fn : callable
      Function to map over data.
  data : iterable
      Data over which we want to parallelize the function call.
  n_jobs : int, optional
      The maximum number of concurrently running jobs. By default, it is one less than
      the number of CPUs.
  verbose: int, optional
      The verbosity level. If nonzero, the function prints the progress messages.
      The frequency of the messages increases with the verbosity level. If above 10,
      it reports all iterations. If above 50, it sends the output to stdout.
  kwargs
      Additional arguments for :attr:`pickleable_fn`.

  Returns
  -------
  list
      The i-th element of the list corresponds to the output of applying
      :attr:`pickleable_fn` to :attr:`data[i]`.
  """
  if n_jobs is None:
    n_jobs = cpu_count() - 1

  results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
  )

  return results

def get_names_from_txt(txt_path):
    with open(txt_path,'r') as f:
        lines = f.read().strip().split('\n')
    return lines

def get_logger(run_dir, rank=0):
    """
    Set the logger
    """
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    logfile_name = os.path.join(run_dir, 'log.txt')

    fmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(fmt, filedatefmt)
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(fmt, sdatefmt)

    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(sformatter)
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh])
    return logging.getLogger()

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file,'rb') as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    #torch.use_deterministic_algorithms(True)

def ddp_setup(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',timeout=timedelta(seconds=7200))

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {} local rank {}): {}".format(
            args.rank, args.local_rank, "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method="env://", world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def ddp_cleanup():
    dist.destroy_process_group()

def get_run_dir(config):
    run_dir = f'{config.root_dir}/{config.data.dataset_name}' \
              f'_{config.target}_model{config.model.model_type}_gtype{config.data.prot_graph_type}' \
              f'_lcut{config.data.ligcut}_pcut{config.data.protcut}_icut{config.data.intercut}_ccut{config.data.chaincut}' \
              f'_pgmn{config.data.prot_max_neighbors}_lgmn{config.data.lig_max_neighbors}' \
              f'_igmn{config.data.inter_min_neighbors}_igmn{config.data.inter_max_neighbors}' \
              f'_test2{config.data.test_2}'
    return run_dir
