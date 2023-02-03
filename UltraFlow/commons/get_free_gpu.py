import torch
from gpustat import GPUStatCollection
import time
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

def get_device(target_gpu_idx, memory_need=10000):
    # check device
    # assert torch.cuda.device_count() >= len(target_gpus), 'do you set the gpus in config correctly?'
    flag = None

    while True:
        # Get the gpu ids which have more than 10000MB memory
        free_gpu_ids = get_free_gpu('memory', memory_need)
        if len(free_gpu_ids) < 1:
            if flag is None:
                print("No GPU available now. sleeping 60s ....")
            time.sleep(6)
        else:

            gpuid = list(set(free_gpu_ids) & set(target_gpu_idx))[0]

            device = torch.device('cuda:'+str(gpuid))
            print("Using device %s as main device" % device)
            break

    return device

if __name__ == '__main__':
    target_gpu_idx = [0,1,2,3,4,5,6,7,8]
    device = get_device(target_gpu_idx)
    print(device)