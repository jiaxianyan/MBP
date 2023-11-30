import argparse
import os
from MBP.models.sbap import *
from MBP import commons, runner
from datetime import datetime
from time import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/affinity/affinity_default.yaml',help='path of dataset')
    args = parser.parse_args()

    # get config
    config = commons.get_config_easydict(args.config_path)

    # get device
    config.train.device = commons.get_device(config.train.gpus, config.train.gpu_memory_need)

    # set random seed
    commons.set_seed(config.seed)

    # set run_dir
    now = str(datetime.now()).replace(" ", "_").split(".")[0]
    run_dir = commons.get_run_dir(config)
    run_dir_now = os.path.join(run_dir,now)
    config.train.save_path = run_dir_now

    # get logger
    config.logger = commons.get_logger(run_dir_now)

    # load data
    load_start = time()
    train_data, val_data, test_data = commons.get_dataset(config)
    print('load data time:{:.2f}s'.format(time() - load_start))

    # set feats dim
    config.model.lig_node_dim, config.model.lig_edge_dim = train_data.lig_node_dim, train_data.lig_edge_dim
    config.model.pro_node_dim, config.model.pro_edge_dim = train_data.pro_node_dim, train_data.pro_edge_dim
    config.model.inter_edge_dim = train_data.inter_edge_dim

    # get model
    model = globals()[config.model.model_type+'_MTL'](config).to(config.train.device)

    # get optimizer
    optimizer = commons.get_optimizer(config.train.optimizer, model)

    # get scheduler
    scheduler = commons.get_scheduler(config.train.scheduler, optimizer)

    # get runner
    solver = runner.asrp_runner.DefaultRunner(train_data, val_data, test_data, model, optimizer, scheduler, config)

    # load checkpoint
    if config.train.resume_train:
        solver.load(os.path.join(run_dir, config.test.now), epoch=config.train.resume_epoch, load_optimizer=False, load_scheduler=False)

    # get logger
    config.logger = commons.get_logger(run_dir_now)

    # save config file to run dir
    cmd = f'cp {args.config_path} {run_dir_now}'
    os.system(cmd)

    # train
    solver.train()


