import argparse
import os
from UltraFlow.models.sbap import *
from UltraFlow import commons, runner
from datetime import datetime
from time import time
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/affinity/affinity_default.yaml')
    args = parser.parse_args()

    grid_search_batch_size = [128, 64, 32, 16]
    grid_search_learning_rate = [1e-3, 3e-4, 1e-4, 3e-5]

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

    now = config.test.now
    run_dir = commons.get_run_dir(config)
    run_dir = os.path.join(run_dir,now)
    config.train.pretrain_model_save_path = run_dir

    # get logger
    config.logger = commons.get_logger(run_dir_now)

    # load data
    load_start = time()
    train_data, val_data, test_data, generalize_csar_data = commons.get_finetune_dataset(config)
    print('load data time:{:.2f}s'.format(time() - load_start))

    # set feats dim
    config.model.lig_node_dim, config.model.lig_edge_dim = test_data.lig_node_dim, test_data.lig_edge_dim
    config.model.pro_node_dim, config.model.pro_edge_dim = test_data.pro_node_dim, test_data.pro_edge_dim
    config.model.inter_edge_dim = test_data.inter_edge_dim

    # ====================================Grid search =======================================
    best_rmse = 1000
    best_bs, best_lr = -1, -1
    config.train.save = False
    for bs in grid_search_batch_size:
        for lr in grid_search_learning_rate:
            print(f'grid search batch size: {bs}, lr: {lr}')
            config.train.optimizer.lr = lr
            config.train.batch_size = bs

            if config.train.multi_task:
                model = globals()[config.model.model_type + '_MTL'](config).to(config.train.device)
            else:
                model = globals()[config.model.model_type](config).to(config.train.device)

            # get optimizer
            optimizer = commons.get_optimizer(config.train.optimizer, model)

            # get scheduler
            scheduler = commons.get_scheduler(config.train.scheduler, optimizer)

            # get runner
            solver = runner.finetune_runner.DefaultRunner(train_data, val_data, test_data, generalize_csar_data, model, optimizer, scheduler, config)

            # load pre-trained checkpoint
            if config.train.use_pretrain_model:
                solver.load(config.train.pretrain_model_save_path, epoch=config.test.epoch, load_optimizer=False, load_scheduler=False)

            # get logger
            config.logger = commons.get_logger(run_dir_now)

            # save config file to run dir
            cmd = f'cp {args.config_path} {run_dir_now}'
            os.system(cmd)

            # train
            _, _, _, _ = solver.train(repeat_index=f'bs_{bs}_lr_{lr}')

            # rmse on validation set
            if solver.best_matric < best_rmse:
                best_lr = lr
                best_bs = bs
                best_rmse = solver.best_matric

            print(f'current best lr: {best_lr}, best batch size: {best_bs},  best RMSE: {best_rmse}')

    print('using lr: {}, batch size: {} as final setting')
    config.train.optimizer.lr = lr
    config.train.batch_size = bs

    RMSEs, MAEs, Pearsons, Spearmans = [], [], [], []
    test_RMSEs, test_MAEs, test_Pearsons, test_Spearmans = [], [], [], []
    config.train.save = True
    for i in range(config.train.finetune_times):
        # get model
        if config.train.multi_task:
            model = globals()[config.model.model_type + '_MTL'](config).to(config.train.device)
        else:
            model = globals()[config.model.model_type](config).to(config.train.device)

        # get optimizer
        optimizer = commons.get_optimizer(config.train.optimizer, model)

        # get scheduler
        scheduler = commons.get_scheduler(config.train.scheduler, optimizer)

        # get runner
        solver = runner.finetune_runner.DefaultRunner(train_data, val_data, test_data, generalize_csar_data, model, optimizer, scheduler, config)

        # load pre-trained checkpoint
        if config.train.use_pretrain_model:
            solver.load(config.train.pretrain_model_save_path, epoch=config.test.epoch, load_optimizer=False, load_scheduler=False)

        # get logger
        config.logger = commons.get_logger(run_dir_now)

        # save config file to run dir
        cmd = f'cp {args.config_path} {run_dir_now}'
        os.system(cmd)

        # train
        RMSE, MAE, Pearson, Spearman = solver.train(repeat_index=i)

        # csar_test
        if not config.train.multi_task:
            test_RMSE, test_MAE, test_Pearson, test_Spearman = solver.evaluate('csar', verbose=1)
        elif config.train.multi_task == 'IC50KdKi':
            test_RMSE, test_MAE, test_Pearson, test_Spearman = solver.evaluate_mtl('csar', verbose=1)
        elif config.train.multi_task == 'IC50K':
            test_RMSE, test_MAE, test_Pearson, test_Spearman = solver.evaluate_mtl_v2('csar', verbose=1)

        RMSEs.append(RMSE)
        MAEs.append(MAE)
        Pearsons.append(Pearson)
        Spearmans.append(Spearman)

        test_RMSEs.append(test_RMSE)
        test_MAEs.append(test_MAE)
        test_Pearsons.append(test_Pearson)
        test_Spearmans.append(test_Spearman)


    print(f'RMSE mean:{np.mean(RMSEs)}, std:{np.std(RMSEs)}')
    print(f'MAE mean:{np.mean(MAEs)}, std:{np.std(MAEs)}')
    print(f'Pearson mean:{np.mean(Pearsons)}, std:{np.std(Pearsons)}')
    print(f'Spearman mean:{np.mean(Spearmans)}, std:{np.std(Spearmans)}')


    print(f'CSRA RMSE mean:{np.mean(test_RMSEs)}, std:{np.std(test_RMSEs)}')
    print(f'CSRA MAE mean:{np.mean(test_MAEs)}, std:{np.std(test_MAEs)}')
    print(f'CSRA Pearson mean:{np.mean(test_Pearsons)}, std:{np.std(test_Pearsons)}')
    print(f'CSRA Spearman mean:{np.mean(test_Spearmans)}, std:{np.std(test_Spearmans)}')
