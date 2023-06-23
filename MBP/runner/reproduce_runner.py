from torch.utils.data import DataLoader
import torch
from MBP import dataset, commons, losses, models
import numpy as np

def trans_device(batch, device):
    return [x if isinstance(x, list) else x.to(device) for x in batch]

@torch.no_grad()
def reproduce_result(config, test_set, model, device):
    """
    Evaluate the model.
    Parameters:
        split (str): split to evaluate. Can be ``train``, ``val`` or ``test``.
    """

    dataloader = DataLoader(test_set, batch_size=config.train.batch_size,
                            shuffle=False, collate_fn=dataset.collate_pdbbind_affinity_multi_task_v2,
                            num_workers=config.train.num_workers)
    y_preds, y_preds_IC50, y_preds_K = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
    y, y_IC50, y_K = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
    model.eval()
    for batch in dataloader:
        if device.type == "cuda":
            batch = trans_device(batch, device)

        (regression_loss_IC50, regression_loss_K), \
        (affinity_pred_IC50, affinity_pred_K), \
        (affinity_IC50, affinity_K) = model(batch, ASRP=False)

        affinity_pred = torch.cat([affinity_pred_IC50, affinity_pred_K], dim=0)
        affinity = torch.cat([affinity_IC50, affinity_K], dim=0)

        y_preds_IC50 = torch.cat([y_preds_IC50, affinity_pred_IC50])
        y_preds_K = torch.cat([y_preds_K, affinity_pred_K])
        y_preds = torch.cat([y_preds, affinity_pred])

        y_IC50 = torch.cat([y_IC50, affinity_IC50])
        y_K = torch.cat([y_K, affinity_K])
        y = torch.cat([y, affinity])

    metics_dict = commons.get_sbap_regression_metric_dict(np.array(y.cpu()), np.array(y_preds.cpu()))
    result_str = commons.get_matric_output_str(metics_dict)

    if len(y_IC50) > 0:
        metics_dict_IC50 = commons.get_sbap_regression_metric_dict(np.array(y_IC50.cpu()), np.array(y_preds_IC50.cpu()))
        result_str_IC50 = commons.get_matric_output_str(metics_dict_IC50)
        result_str_IC50 = f'| IC50 ' + result_str_IC50
        result_str += result_str_IC50

    if len(y_K) > 0:
        metics_dict_K = commons.get_sbap_regression_metric_dict(np.array(y_K.cpu()), np.array(y_preds_K.cpu()))
        result_str_K = commons.get_matric_output_str(metics_dict_K)
        result_str_K = f'| K ' + result_str_K
        result_str += result_str_K

    # print(result_str)
    return metics_dict['RMSE'], metics_dict['MAE'], metics_dict['SD'], metics_dict['Pearson']