import argparse
import os
from MBP.models.sbap import *
from MBP import commons, runner
from datetime import datetime
from time import time
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='./workdir/finetune/pdbbind')
    args = parser.parse_args()

    # get config
    config_path = os.path.join(args.work_dir, 'affinity_default.yaml')
    config = commons.get_config_easydict(config_path)

    # get device
    config.train.device = commons.get_device(config.train.gpus, config.train.gpu_memory_need)

    # set random seed
    commons.set_seed(config.seed)

    # load data
    load_start = time()
    test_data, generalize_csar_data = commons.get_test_dataset(config)
    print('load data time:{:.2f}s'.format(time() - load_start))

    # set feats dim
    config.model.lig_node_dim, config.model.lig_edge_dim = test_data.lig_node_dim, test_data.lig_edge_dim
    config.model.pro_node_dim, config.model.pro_edge_dim = test_data.pro_node_dim, test_data.pro_edge_dim
    config.model.inter_edge_dim = test_data.inter_edge_dim

    model = globals()[config.model.model_type + '_MTL'](config).to(config.train.device)

    RMSEs, MAEs, Pearsons, Spearmans, SDs = [], [], [], [], []
    test_RMSEs, test_MAEs, test_Pearsons, test_Spearmans, test_SDs = [], [], [], [], []

    for i in range(config.train.finetune_times):
        checkpoint = os.path.join(args.work_dir, f'checkpointbest_valid_{i}')
        state = torch.load(checkpoint, map_location=config.train.device)
        model.load_state_dict(state["model"])

        RMSE, MAE, SD, Pearson = runner.reproduce_runner.reproduce_result(config, test_data, model, config.train.device)
        test_RMSE, test_MAE, test_SD, test_Pearson = runner.reproduce_runner.reproduce_result(config, generalize_csar_data, model, config.train.device)

        RMSEs.append(RMSE)
        MAEs.append(MAE)
        Pearsons.append(Pearson)
        SDs.append(SD)

        test_RMSEs.append(test_RMSE)
        test_MAEs.append(test_MAE)
        test_Pearsons.append(test_Pearson)
        test_SDs.append(test_SD)

        print(f'PDBbind best metic, RMSE: {RMSE}, MAE: {MAE}, Pearson: {Pearson}, SD: {SD}')
        print(f'CSAR best metic, RMSE: {test_RMSE}, MAE: {test_MAE}, Pearson: {test_Pearson}, SD: {test_SD}')

    print(f'RMSE mean:{np.mean(RMSEs)}, std:{np.std(RMSEs)}')
    print(f'MAE mean:{np.mean(MAEs)}, std:{np.std(MAEs)}')
    print(f'Pearson mean:{np.mean(Pearsons)}, std:{np.std(Pearsons)}')
    print(f'SD mean:{np.mean(SDs)}, std:{np.std(SDs)}')

    print(f'CSAR RMSE mean:{np.mean(test_RMSEs)}, std:{np.std(test_RMSEs)}')
    print(f'CSAR MAE mean:{np.mean(test_MAEs)}, std:{np.std(test_MAEs)}')
    print(f'CSAR Pearson mean:{np.mean(test_Pearsons)}, std:{np.std(test_Pearsons)}')
    print(f'CSAR SD mean:{np.mean(test_SDs)}, std:{np.std(test_SDs)}')