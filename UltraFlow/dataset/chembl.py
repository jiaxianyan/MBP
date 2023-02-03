import os
import pickle
import random
import dgl
import torch
import numpy as np
import pandas as pd
from math import log
from UltraFlow import commons, layers
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import GetMolFrags, AllChem
from collections import Counter
import random

def check_smiles_valid(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    mw = Chem.RWMol(m)
    frags = GetMolFrags(mw, asMols=True)
    frag_size = [len(frag.GetAtoms()) for frag in frags]
    if len(frag_size) == 1 and frag_size[0] > 5:
        return True

    return False

def split_pretrain_assay(df_, assay_limit=50, assay_num=100):
    assay_have_more_than_1_data = []
    c = Counter(df_.assay)
    for k in c.keys():
        if c[k] > 1:
            assay_have_more_than_1_data.append(k)
    random.shuffle(assay_have_more_than_1_data)

    valid_assay, valid_assay_data_num = [], []
    for a in assay_have_more_than_1_data:
        if c[a] <= assay_limit:
            valid_assay.append(a)
            valid_assay_data_num.append(c[a])
        if len(valid_assay) >= assay_num:
            break
    print(sum(valid_assay_data_num))
    print(sum([n * n for n in valid_assay_data_num]))
    train_assay = list(set(assay_have_more_than_1_data) - set(valid_assay))

    return train_assay, valid_assay

def load_chembl_smina_assay_specific_rank_pretrain(config):
    task_target = config.target
    dataset_name = config.data.dataset_name
    prot_graph_type = config.data.prot_graph_type
    ligcut = config.data.ligcut
    protcut = config.data.protcut
    intercut = config.data.intercut
    chaincut = config.data.chaincut
    lig_max_neighbors = config.data.lig_max_neighbors
    prot_max_neighbors = config.data.prot_max_neighbors
    inter_min_neighbors = config.data.inter_min_neighbors
    inter_max_neighbors = config.data.inter_max_neighbors
    lig_type = config.data.lig_type
    n_jobs = config.data.n_jobs
    affinity_relation = config.data.affinity_relation
    assay_des_type = config.data.assay_des_type
    test_2 = config.data.test_2

    affinity_type = ''
    affinity_type_num = 0
    affinity_type_list = []
    if config.data.use_ic50:
        affinity_type += 'IC50'
        affinity_type_num += 1
        affinity_type_list.append('IC50')
    if config.data.use_ki:
        affinity_type += 'Ki'
        affinity_type_num += 1
        affinity_type_list.append('Ki')
    if config.data.use_kd:
        affinity_type += 'Kd'
        affinity_type_num += 1
        affinity_type_list.append('Kd')

    # affinity_type = config.data.affinity_type
    multitask = config.train.multi_task

    processed_dir = f'{config.base_path}/{config.data.dataset_path}/processed/' \
                    f'{dataset_name}_{affinity_type}_{affinity_relation}' \
                    f'_{lig_type}_{task_target}_gtype{prot_graph_type}' \
                    f'_lcut{ligcut}_pcut{protcut}_icut{intercut}' \
                    f'_lgmn{lig_max_neighbors}_pgmn{prot_max_neighbors}' \
                    f'_igmn{inter_min_neighbors}_igmn{inter_max_neighbors}'

    if test_2:
        processed_dir += '_test2'

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    dataset_path = f'{config.base_path}/{config.data.dataset_path}/{config.data.dataset_name}'

    if affinity_relation == 'all':
        affinity_relation_list = ['=', '>']
    else:
        affinity_relation_list = [affinity_relation]

    if not os.path.exists(f'{processed_dir}/multi_graphs.pkl'):
        uniprot_names = os.listdir(dataset_path)
        if test_2:
            uniprot_names = uniprot_names[:5] + ['P13922']

        if not os.path.exists(f'{processed_dir}/ligand_representations.pkl'):
            valid_ligand_flag_list = []
            uniprot_df_list = []
            for name in tqdm(uniprot_names, desc='get ligand to be processed'):
                uniprot_valid_ligand_flag = []
                uniprot_df = pd.read_csv(os.path.join(dataset_path, name, f'{name}_filter.csv'))
                uniprot_smina_ligands = os.listdir(os.path.join(dataset_path, name, 'ligand_smina_poses'))
                uniprot_valid_index = [int(f.split('_')[0]) for f in uniprot_smina_ligands]
                for index, (at, ar, a, sm) in enumerate(zip(uniprot_df['STANDARD_TYPE'].values,
                                                            uniprot_df['STANDARD_RELATION'].values,
                                                            uniprot_df['STANDARD_VALUE (nM)'].values,
                                                            uniprot_df['SMILES'].values)):
                    if (at in affinity_type_list) and (ar in affinity_relation_list) and (float(a) > 0.0) \
                            and (index in uniprot_valid_index) and check_smiles_valid(sm):
                        uniprot_valid_ligand_flag.append(True)
                    else:
                        uniprot_valid_ligand_flag.append(False)

                valid_ligand_flag_list.append(uniprot_valid_ligand_flag)
                uniprot_df_select = uniprot_df[uniprot_valid_ligand_flag]
                uniprot_df_select.to_csv(
                    os.path.join(dataset_path, name, f'{name}_filter_{affinity_type}_{affinity_relation}.csv'))
                uniprot_df_list.append(uniprot_df_select)

            df_info = pd.concat(uniprot_df_list)
            df_info.to_csv(
                f'{processed_dir}/smina_filter_{affinity_type}_{affinity_relation}.csv')

            ligand_representations = commons.pmap_multi(commons.read_ligands_chembl_smina,
                                                        zip(uniprot_names, valid_ligand_flag_list), dataset_path=dataset_path,
                                                        ligcut=ligcut, lig_type=lig_type,
                                                        n_jobs=n_jobs, desc='read ligands')

            with open(f'{processed_dir}/ligand_representations.pkl','wb') as f:
                pickle.dump(ligand_representations,f)
        else:
            with open(f'{processed_dir}/ligand_representations.pkl','rb') as f:
                ligand_representations = pickle.load(f)
            df_info = pd.read_csv(f'{processed_dir}/smina_filter_{affinity_type}_{affinity_relation}.csv')

        if not os.path.exists(f'{processed_dir}/protein_representations.pkl'):
            protein_representations = commons.pmap_multi(commons.read_proteins,
                                                        zip(uniprot_names), dataset_path=dataset_path,
                                                        protcut=protcut, prot_graph_type=prot_graph_type,
                                                        n_jobs=n_jobs, desc='read proteins')

            with open(f'{processed_dir}/protein_representations.pkl','wb') as f:
                pickle.dump(protein_representations,f)
        else:
            with open(f'{processed_dir}/protein_representations.pkl','rb') as f:
                protein_representations = pickle.load(f)

        valid_lig_coords_list, valid_lig_features_list,\
        valid_lig_edges_list, valid_lig_node_type_list, valid_index_list = map(list, zip(*ligand_representations))
        lig_coords, lig_features, lig_edges, lig_node_type = [], [], [], []
        for valid_lig_coords, valid_lig_features, valid_lig_edges, valid_lig_node_type in zip(
                valid_lig_coords_list, valid_lig_features_list, valid_lig_edges_list, valid_lig_node_type_list):
            lig_coords.extend(valid_lig_coords)
            lig_features.extend(valid_lig_features)
            lig_edges.extend(valid_lig_edges)
            lig_node_type.extend(valid_lig_node_type)

        prot_coords, prot_features, prot_edges, prot_node_type, \
        sec_features, alpha_c_coords, c_coords, n_coords = map(list, zip(*protein_representations))

        lig_graphs = commons.pmap_multi(commons.get_lig_graph_equibind,
                                        zip(lig_coords, lig_features, lig_node_type),
                                        max_neighbors=lig_max_neighbors, cutoff=ligcut,
                                        n_jobs=n_jobs, desc='Get ligand graphs')

        prot_graphs = commons.pmap_multi(commons.get_prot_alpha_c_graph_equibind,
                                         zip(prot_coords, prot_features, prot_node_type, sec_features, alpha_c_coords, c_coords, n_coords),
                                         n_jobs=n_jobs, cutoff=protcut, max_neighbor=prot_max_neighbors,
                                         desc='Get protein alpha carbon graphs')

        prot_coords_list, graph_prot_index = [], []
        index = 0
        for prot_coord, valid_index in zip(prot_coords, valid_index_list):
            prot_coords_list.extend([prot_coord] * len(valid_index))
            graph_prot_index.extend([index] * len(valid_index))
            index += 1

        inter_graphs = commons.pmap_multi(commons.get_interact_graph_knn, zip(lig_coords, prot_coords_list),
                                          n_jobs=n_jobs, cutoff=intercut,
                                          max_neighbor=inter_max_neighbors,min_neighbor=inter_min_neighbors,
                                          desc='Get interaction graphs')

        processed_data = (lig_graphs, prot_graphs, inter_graphs, df_info, graph_prot_index)
        with open(f'{processed_dir}/multi_graphs.pkl','wb') as f:
            pickle.dump(processed_data,f)

    with open(f'{processed_dir}/multi_graphs.pkl', 'rb') as f:
        train_Dataset = pickle.load(f)

    lig_graphs = train_Dataset[0]
    prot_graphs = train_Dataset[1]
    inter_graphs = train_Dataset[2]
    df_info = train_Dataset[3]
    graph_prot_index = train_Dataset[4]

    train_assay_d = {}
    if assay_des_type is not None:
        with open(f'{config.base_path}/{config.data.dataset_path}/total_assay_descriptor_{assay_des_type}.pkl','rb') as f:
            train_assay_des = pickle.load(f)

        for (assay_id, assay_des) in train_assay_des:
            train_assay_d[assay_id] = torch.from_numpy(assay_des)

    if len(df_info.columns) == 13:
        df_info.columns = [str(f) for f in df_info.columns[:4]] + ['assay'] + [str(f) for f in df_info.columns[5:]]
    elif len(df_info.columns) == 12:
        df_info.columns = [str(f) for f in df_info.columns[:3]] + ['assay'] + [str(f) for f in df_info.columns[4:]]

    labels = [-log(float(x) * 1e-9, 10) for x in df_info['STANDARD_VALUE (nM)'].values]
    labels = torch.tensor(labels, dtype=torch.float)

    if not os.path.exists(f'{processed_dir}/chembl_smina_asrp_valid'):
        train_assay, valid_assay = split_pretrain_assay(df_info)
        with open(f'{processed_dir}/chembl_smina_asrp_train', 'w') as f:
            f.write('\n'.join([str(a) for a in train_assay]))
        with open(f'{processed_dir}/chembl_smina_asrp_valid', 'w') as f:
            f.write('\n'.join([str(a) for a in valid_assay]))
    else:
        with open(f'{processed_dir}/chembl_smina_asrp_train', 'r') as f:
            train_assay = [int(a) for a in f.read().strip().split('\n')]
        with open(f'{processed_dir}/chembl_smina_asrp_valid', 'r') as f:
            valid_assay = [int(a) for a in f.read().strip().split('\n')]

    if config.train.pretrain_sampling_method == 'pairwise_v1':
        train_data = chem_assay_specific_rank_pretrain_pairwise_v1(lig_graphs, prot_graphs, inter_graphs, labels,
                                                                   graph_prot_index, df_info, train_assay,
                                                                   config.data.test_2, assay_des_type, train_assay_d,
                                                                   multitask)
        valid_data = chem_assay_specific_rank_valid(lig_graphs, prot_graphs, inter_graphs, labels,
                                                    graph_prot_index, df_info, valid_assay,
                                                    config.data.test_2, assay_des_type, train_assay_d,
                                                    multitask)

    elif config.train.pretrain_sampling_method == 'pointwise':
        train_data = chemb_assay_specific_rank_pretrain_pointwise(lig_graphs, prot_graphs, inter_graphs, labels,
                                                                  graph_prot_index, df_info, train_assay,
                                                                  config.data.test_2, assay_des_type, train_assay_d,
                                                                  multitask)
        valid_data = chem_assay_specific_rank_valid(lig_graphs, prot_graphs, inter_graphs, labels,
                                                    graph_prot_index, df_info, valid_assay,
                                                    config.data.test_2, None, train_assay_d, multitask)

    return train_data, valid_data

class chem_assay_specific_rank_pretrain_pairwise_v1():
    def __init__(self, lig_graphs, prot_graphs, inter_graphs, labels, graph_prot_index,
                 df, assays, test_2, assay_des_type, assay_d=None, multi_task=False):
        index_flag = df['assay'].isin(assays)
        self.assays = assays
        self.assay_des_type = assay_des_type
        self.assay_d = assay_d
        self.multi_task = multi_task
        self.df = df[index_flag]
        print(f'len of dataset: {len(self.df)}')
        assay_to_index = defaultdict(list)
        for idx, a in enumerate(self.df['assay'].values):
            assay_to_index[a].append(idx)
        self.assay_to_index = assay_to_index

        self.IC50_flag = (df['STANDARD_TYPE'] == 'IC50').values.tolist()
        self.Kd_flag = (df['STANDARD_TYPE'] == 'Kd').values.tolist()
        self.Ki_flag = (df['STANDARD_TYPE'] == 'Ki').values.tolist()

        K_flag = []
        for kd, ki in zip(self.Kd_flag, self.Ki_flag):
            if kd or ki:
                K_flag.append(True)
            else:
                K_flag.append(False)
        self.K_flag = K_flag

        print(f'num of IC50: {sum(self.IC50_flag)}')
        print(f'num of Kd: {sum(self.Kd_flag)}')
        print(f'num of Ki: {sum(self.Ki_flag)}')
        print(f'num of K: {sum(self.K_flag)}')

        lig_graphs_select, prot_graphs_select, inter_graphs_select, labels_select, graph_prot_index_select = [], [], [], [], []
        for index, flag in enumerate(index_flag):
            if flag:
                lig_graphs_select.append(lig_graphs[index])
                inter_graphs_select.append(inter_graphs[index])
                labels_select.append(labels[index])
                graph_prot_index_select.append(graph_prot_index[index])

        self.lig_graphs = lig_graphs_select
        self.prot_graphs = prot_graphs
        self.inter_graphs = inter_graphs_select
        self.labels = labels_select
        self.graph_prot_index = graph_prot_index_select
        self.test_2 = test_2
        self._load_node_feats_dim()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        lig_graph = deepcopy(self.lig_graphs[item])
        prot_graph_index = self.graph_prot_index[item]
        prot_graph = deepcopy(self.prot_graphs[prot_graph_index])
        inter_graph = deepcopy(self.inter_graphs[item])
        label = deepcopy(self.labels[item])

        IC50_f1 = deepcopy(self.IC50_flag[item])
        Kd_f1 = deepcopy(self.Kd_flag[item])
        Ki_f1 = deepcopy(self.Ki_flag[item])
        K_f1 = deepcopy(self.K_flag[item])


        inter_d = inter_graph.edata['d']
        squared_distance = inter_d ** 2
        all_sigmas_dist = [1.5 ** x for x in range(15)]
        prot_square_distance_scale = 10.0
        x_rel_mag = torch.cat([torch.exp(-(squared_distance / prot_square_distance_scale) / sigma) for sigma in
                               all_sigmas_dist], dim=-1)
        inter_graph.edata['e'] = x_rel_mag

        lig_graph_2, prot_graph_2, inter_graph_2, label_2, item_2,\
        IC50_f2, Kd_f2, Ki_f2, K_f2 = self.sample_pair_wise_item(item)


        assay_id = self.df['assay'].values[item]
        assay_des = self.assay_d[assay_id]

        if not self.multi_task:
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0),\
                   lig_graph_2, prot_graph_2, inter_graph_2, label_2, item_2, assay_des.unsqueeze(dim=0)
        elif self.multi_task == 'IC50KdKi':
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0),\
                   IC50_f1, Kd_f1, Ki_f1, \
                   lig_graph_2, prot_graph_2, inter_graph_2, label_2, item_2, assay_des.unsqueeze(dim=0),\
                   IC50_f2, Kd_f2, Ki_f2
        elif self.multi_task == 'IC50K':
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0),\
                   IC50_f1, K_f1, \
                   lig_graph_2, prot_graph_2, inter_graph_2, label_2, item_2, assay_des.unsqueeze(dim=0),\
                   IC50_f2, K_f2

    def sample_pair_wise_item(self, item):
        assay_id = self.df['assay'].values[item]
        global_indexs = deepcopy(self.assay_to_index[assay_id])

        random.shuffle(global_indexs)
        sample_flag = False
        for sample_item in global_indexs:
            if (sample_item != item) and (
                    (self.IC50_flag[sample_item] == self.IC50_flag[item]) and (self.Kd_flag[sample_item] == self.Kd_flag[item]) and (self.Ki_flag[sample_item] == self.Ki_flag[item]) ):
                sample_flag = True
                break

        if not sample_flag:
            sample_item = item

        lig_graph2 = deepcopy(self.lig_graphs[sample_item])
        prot_graph_index2 = self.graph_prot_index[sample_item]
        prot_graph2 = deepcopy(self.prot_graphs[prot_graph_index2])
        inter_graph2 = deepcopy(self.inter_graphs[sample_item])
        label2 = deepcopy(self.labels[sample_item])

        IC50_f2 = deepcopy(self.IC50_flag[sample_item])
        Kd_f2 = deepcopy(self.Kd_flag[sample_item])
        Ki_f2 = deepcopy(self.Ki_flag[sample_item])
        K_f2 = deepcopy(self.K_flag[sample_item])


        inter_d = inter_graph2.edata['d']
        squared_distance = inter_d ** 2
        all_sigmas_dist = [1.5 ** x for x in range(15)]
        prot_square_distance_scale = 10.0
        x_rel_mag = torch.cat([torch.exp(-(squared_distance / prot_square_distance_scale) / sigma) for sigma in
                               all_sigmas_dist], dim=-1)
        inter_graph2.edata['e'] = x_rel_mag

        return lig_graph2, prot_graph2, inter_graph2, label2, sample_item, IC50_f2, Kd_f2, Ki_f2, K_f2

    def _load_node_feats_dim(self):
        self.lig_node_dim = self.lig_graphs[0].ndata['h'].shape[1]
        self.lig_edge_dim = self.lig_graphs[0].edata['e'].shape[1]
        self.pro_node_dim = self.prot_graphs[0].ndata['h'].shape[1]
        self.pro_edge_dim = self.prot_graphs[0].edata['e'].shape[1]
        self.inter_edge_dim = 15

# class chem_assay_specific_rank_pretrain_pairwise_v2(chem_assay_specific_rank_pretrain_pairwise_v1):
#     def __init__(self, lig_graphs, prot_graphs, inter_graphs, labels,
#                  graph_prot_index, df, assays,
#                  test_2, batch_size, drop_last, assay_des_type, assay_d=None):
#         super(chem_assay_specific_rank_pretrain_pairwise_v2, self).__init__(lig_graphs, prot_graphs, inter_graphs, labels, graph_prot_index, df, assays, test_2, assay_d)
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.assay_d = assay_d
#         self.assay_des_type = assay_des_type
#
#         length = len(self.lig_graphs) // self.batch_size
#         b = len(self.lig_graphs) % self.batch_size
#         if (b > 1) or (b == 1 and not self.drop_last):
#             length += 1
#         self.length = length
#         print(f'num of data :{len(self.lig_graphs)}')
#         print(f'dataset length :{self.length}')
#
#     def __len__(self):
#
#         return self.length
#
#     def getitem(self, item):
#         lig_graph = deepcopy(self.lig_graphs[item])
#         prot_graph_index = self.graph_prot_index[item]
#         prot_graph = deepcopy(self.prot_graphs[prot_graph_index])
#         inter_graph = deepcopy(self.inter_graphs[item])
#         label = deepcopy(self.labels[item])
#
#         inter_d = inter_graph.edata['d']
#         squared_distance = inter_d ** 2
#         all_sigmas_dist = [1.5 ** x for x in range(15)]
#         prot_square_distance_scale = 10.0
#         x_rel_mag = torch.cat([torch.exp(-(squared_distance / prot_square_distance_scale) / sigma) for sigma in
#                                all_sigmas_dist], dim=-1)
#         inter_graph.edata['e'] = x_rel_mag
#
#         return lig_graph, prot_graph, inter_graph, label, item,
#
#     def get_pair_index(self, batch_assay_index):
#         pair_a_index, pair_b_index = [], []
#         assay_num_list = []
#
#         previous_assay = batch_assay_index[0]
#         assay_num = 1
#         for a in batch_assay_index[1:]:
#             if previous_assay == a:
#                 assay_num += 1
#             else:
#                 assay_num_list.append(assay_num)
#                 previous_assay = a
#                 assay_num = 1
#         assay_num_list.append(assay_num)
#
#         cur_index = 0
#         for i in assay_num_list:
#             for j in range(i):
#                 pair_a_index.extend([j + cur_index] * (i - 1))
#                 pair_b_index.extend([k + cur_index for k in range(i) if j != k])
#             cur_index += i
#
#         return pair_a_index + pair_b_index
#
#     def new_epoch_shuffle(self):
#         random.shuffle(self.assays)
#         sample_random_index = []
#         sample_assay_index = []
#         for a in self.assays:
#             global_indexs = self.assay_to_index[a]
#             random.shuffle(global_indexs)
#             sample_random_index.extend(global_indexs)
#             sample_assay_index.extend([a] * len(global_indexs))
#
#         self.sample_random_index = sample_random_index
#         self.sample_assay_index = sample_assay_index
#
#     def __getitem__(self, item):
#
#         start = item * self.batch_size
#         end = (item + 1) * self.batch_size
#         batch_global_index = self.sample_random_index[start:end]
#         batch_assay_index = self.sample_assay_index[start:end]
#         if len(batch_global_index) == 0:
#             print(f'index num 0, item: {item}')
#             print(f'sample random index num :{len(self.sample_random_index)}')
#             print(f'dataset length :{self.length}')
#
#         g_ligs, g_prots, g_inters, labels, items = [], [], [], [], []
#         for idx in batch_global_index:
#             lig_graph, prot_graph, inter_graph, label, item = self.getitem(idx)
#             g_ligs.append(lig_graph)
#             g_prots.append(prot_graph)
#             g_inters.append(inter_graph)
#             labels.append(label)
#             items.append(item)
#
#         return dgl.batch(g_ligs),\
#                dgl.batch(g_prots),\
#                dgl.batch(g_inters),\
#                torch.unsqueeze(torch.stack(labels, dim=0),dim=-1),\
#                list(items),\
#                self.get_pair_index(batch_assay_index)

class chem_assay_specific_rank_valid():
    def __init__(self, lig_graphs, prot_graphs, inter_graphs, labels,
                 graph_prot_index, df, assays, test_2, assay_des_type, assay_d=None, multi_task=False):
        index_flag = df['assay'].isin(assays)
        self.df = df[index_flag]

        self.assays = self.df['assay'].unique()
        self.assay_des_type = assay_des_type
        self.assay_d = assay_d
        self.multi_task = multi_task
        num_of_assay = len(self.df['assay'].unique())
        print(f'num of assay: {num_of_assay}')

        assay_to_index = defaultdict(list)
        for idx, a in enumerate(self.df['assay'].values):
            assay_to_index[a].append(idx)
        self.assay_to_index = assay_to_index

        self.IC50_flag = (df['STANDARD_TYPE'] == 'IC50').values.tolist()
        self.Kd_flag = (df['STANDARD_TYPE'] == 'Kd').values.tolist()
        self.Ki_flag = (df['STANDARD_TYPE'] == 'Ki').values.tolist()

        lig_graphs_select, prot_graphs_select, inter_graphs_select, labels_select, graph_prot_index_select = [], [], [], [], []
        for index, flag in enumerate(index_flag):
            if flag:
                lig_graphs_select.append(lig_graphs[index])
                inter_graphs_select.append(inter_graphs[index])
                labels_select.append(labels[index])
                graph_prot_index_select.append(graph_prot_index[index])

        self.lig_graphs = lig_graphs_select
        self.prot_graphs = prot_graphs
        self.inter_graphs = inter_graphs_select
        self.labels = labels_select
        self.graph_prot_index = graph_prot_index_select
        self.test_2 = test_2

    def __len__(self):
        return len(self.df['assay'].unique())

    def getitem(self, item):
        lig_graph = deepcopy(self.lig_graphs[item])
        prot_graph_index = self.graph_prot_index[item]
        prot_graph = deepcopy(self.prot_graphs[prot_graph_index])
        inter_graph = deepcopy(self.inter_graphs[item])
        label = deepcopy(self.labels[item])

        inter_d = inter_graph.edata['d']
        squared_distance = inter_d ** 2
        all_sigmas_dist = [1.5 ** x for x in range(15)]
        prot_square_distance_scale = 10.0
        x_rel_mag = torch.cat([torch.exp(-(squared_distance / prot_square_distance_scale) / sigma) for sigma in
                               all_sigmas_dist], dim=-1)
        inter_graph.edata['e'] = x_rel_mag

        IC50_f = deepcopy(self.IC50_flag[item])
        Kd_f = deepcopy(self.Kd_flag[item])
        Ki_f = deepcopy(self.Ki_flag[item])

        return lig_graph, prot_graph, inter_graph, label, item, IC50_f, Kd_f, Ki_f

    def __getitem__(self, item):
        assay_id = self.assays[item]
        assay_des = self.assay_d[assay_id]
        global_indexs = self.assay_to_index[assay_id]
        g_ligs, g_prots, g_inters, labels, items, asssy_dess = [], [], [], [], [], []
        IC50_f_list, Kd_f_list, Ki_f_list = [], [], []
        for idx in global_indexs:
            lig_graph, prot_graph, inter_graph, label, item, IC50_f, Kd_f, Ki_f = self.getitem(idx)
            g_ligs.append(lig_graph)
            g_prots.append(prot_graph)
            g_inters.append(inter_graph)
            labels.append(label)
            items.append(item)
            asssy_dess.append(assay_des.unsqueeze(dim=0))
            IC50_f_list.append(IC50_f)
            Kd_f_list.append(Kd_f)
            Ki_f_list.append(Ki_f)

        if not self.multi_task:
            return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
                   torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
                   torch.cat(asssy_dess, dim=0)
        else:
            return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
                   torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
                   torch.cat(asssy_dess, dim=0), IC50_f_list, Kd_f_list, Ki_f_list

class chemb_assay_specific_rank_pretrain_pointwise():
    def __init__(self, lig_graphs, prot_graphs, inter_graphs, labels, graph_prot_index, df, assays,
                 test_2, assay_des_type, assay_d=None, multi_task=False):
        index_flag = df['assay'].isin(assays)
        self.assays = assays

        self.assay_des_type = assay_des_type
        self.assay_d = assay_d
        self.multi_task = multi_task

        self.df = df[index_flag]
        assay_to_index = defaultdict(list)
        for idx, a in enumerate(self.df['assay'].values):
            assay_to_index[a].append(idx)
        self.assay_to_index = assay_to_index

        self.IC50_flag = (df['STANDARD_TYPE'] == 'IC50').values.tolist()
        self.Kd_flag = (df['STANDARD_TYPE'] == 'Kd').values.tolist()
        self.Ki_flag = (df['STANDARD_TYPE'] == 'Ki').values.tolist()

        K_flag = []
        for kd, ki in zip(self.Kd_flag, self.Ki_flag):
            if kd or ki:
                K_flag.append(True)
            else:
                K_flag.append(False)
        self.K_flag = K_flag

        print(f'num of IC50: {sum(self.IC50_flag)}')
        print(f'num of Kd: {sum(self.Kd_flag)}')
        print(f'num of Ki: {sum(self.Ki_flag)}')
        print(f'num of K: {sum(self.K_flag)}')

        lig_graphs_select, prot_graphs_select, inter_graphs_select, labels_select, graph_prot_index_select = [], [], [], [], []
        for index, flag in enumerate(index_flag):
            if flag:
                lig_graphs_select.append(lig_graphs[index])
                inter_graphs_select.append(inter_graphs[index])
                labels_select.append(labels[index])
                graph_prot_index_select.append(graph_prot_index[index])

        self.lig_graphs = lig_graphs_select
        self.prot_graphs = prot_graphs
        self.inter_graphs = inter_graphs_select
        self.labels = labels_select
        self.graph_prot_index = graph_prot_index_select
        self._load_node_feats_dim()

    def __len__(self):
        return len(self.lig_graphs)

    def __getitem__(self, item):
        lig_graph = deepcopy(self.lig_graphs[item])
        prot_graph_index = self.graph_prot_index[item]
        prot_graph = deepcopy(self.prot_graphs[prot_graph_index])
        inter_graph = deepcopy(self.inter_graphs[item])
        label = deepcopy(self.labels[item])

        IC50_f = deepcopy(self.IC50_flag[item])
        Kd_f = deepcopy(self.Kd_flag[item])
        Ki_f = deepcopy(self.Ki_flag[item])
        K_f = deepcopy(self.K_flag[item])

        assay_id = self.df['assay'].values[item]
        assay_des = self.assay_d[assay_id]

        inter_d = inter_graph.edata['d']
        squared_distance = inter_d ** 2
        all_sigmas_dist = [1.5 ** x for x in range(15)]
        prot_square_distance_scale = 10.0
        x_rel_mag = torch.cat([torch.exp(-(squared_distance / prot_square_distance_scale) / sigma) for sigma in
                               all_sigmas_dist], dim=-1)
        inter_graph.edata['e'] = x_rel_mag

        if not self.multi_task:
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0)
        elif self.multi_task == 'IC50KdKi':
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0), IC50_f, Kd_f, Ki_f
        elif self.multi_task == 'IC50K':
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0), IC50_f, K_f

    def _load_node_feats_dim(self):
        self.lig_node_dim = self.lig_graphs[0].ndata['h'].shape[1]
        self.lig_edge_dim = self.lig_graphs[0].edata['e'].shape[1]
        self.pro_node_dim = self.prot_graphs[0].ndata['h'].shape[1]
        self.pro_edge_dim = self.prot_graphs[0].edata['e'].shape[1]
        self.inter_edge_dim = 15

class pdbbind_finetune():
    def __init__(self, complex_names_path, dataset_name, labels_path, config):
        self.config = config
        self.task_target = config.target
        self.complex_names_path = os.path.join(config.base_path, complex_names_path)
        self.complex_labels_path = labels_path
        self.dataset_name = dataset_name
        self.prot_graph_type = config.data.prot_graph_type
        self.ligcut = config.data.ligcut
        self.protcut = config.data.protcut
        self.intercut = config.data.intercut
        self.chaincut = config.data.chaincut
        self.lig_max_neighbors = config.data.lig_max_neighbors
        self.prot_max_neighbors = config.data.prot_max_neighbors
        self.inter_min_neighbors = config.data.inter_min_neighbors
        self.inter_max_neighbors = config.data.inter_max_neighbors

        self.lig_type = config.data.lig_type

        self.ranking_loss = config.train.ranking_loss

        self.test_100 = config.data.test_100
        self.multi_task = config.train.multi_task

        self.device = config.train.device
        self.n_jobs = config.data.n_jobs
        self.dataset_path = f'{config.base_path}/{config.data.dataset_path}/{dataset_name}'
        self.processed_dir = f'{config.base_path}/{config.data.dataset_path}/processed/' \
                             f'{os.path.basename(complex_names_path)}' \
                             f'_{self.lig_type}_{self.task_target}_gtype{self.prot_graph_type}' \
                             f'_lcut{self.ligcut}_pcut{self.protcut}_icut{self.intercut}' \
                             f'_lgmn{self.lig_max_neighbors}_pgmn{self.prot_max_neighbors}' \
                             f'_igmn{self.inter_min_neighbors}_igmn{self.inter_max_neighbors}' \
                             f'_test2{self.test_100}'
        self._load()
        self._load_node_feats_dim()
        self._load_affinity_type()

    def __len__(self):
        return len(self.Dataset[0])

    def __getitem__(self, item):
        lig_graph = deepcopy(self.lig_graphs[item])
        prot_graph = deepcopy(self.prot_graphs[item])
        inter_graph = deepcopy(self.inter_graphs[item])
        label = deepcopy(self.labels[item])

        fintune_assay_id = -1
        assay_des = deepcopy(self.assay_d[fintune_assay_id])

        IC50_f = deepcopy(self.IC50_flag[item])
        Kd_f = deepcopy(self.Kd_flag[item])
        Ki_f = deepcopy(self.Ki_flag[item])
        K_f = deepcopy(self.K_flag[item])

        if not self.multi_task:
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0)
        elif self.multi_task == 'IC50KdKi':
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0), IC50_f, Kd_f, Ki_f
        elif self.multi_task == 'IC50K':
            return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0), IC50_f, K_f

    def _load_affinity_type(self):
        names = commons.get_names_from_txt(self.complex_names_path)
        with open(os.path.join(self.config.base_path, self.complex_labels_path), 'rb') as f:
            lines = f.read().decode().strip().split('\n')[6:]
        res = {}
        for line in lines:
            temp = line.split()
            name, score = temp[0], float(temp[3])
            affinity_type = temp[4]
            res[name] = (score, affinity_type)
        IC50_f, Kd_f, Ki_f, K_f = [], [], [], []
        for name in names:
            try:
                score, affinity_type = res[name]
                if 'IC50' in affinity_type:
                    IC50_f.append(True)
                    Kd_f.append(False)
                    Ki_f.append(False)
                    K_f.append(False)

                elif 'Kd' in affinity_type:
                    Kd_f.append(True)
                    K_f.append(True)
                    IC50_f.append(False)
                    Ki_f.append(False)

                elif 'Ki' in affinity_type:
                    Ki_f.append(True)
                    K_f.append(True)
                    IC50_f.append(False)
                    Kd_f.append(False)
            except:
                K_f.append(True)
                Ki_f.append(True)
                IC50_f.append(False)
                Kd_f.append(False)

        self.IC50_flag = IC50_f
        self.Kd_flag = Kd_f
        self.Ki_flag = Ki_f
        self.K_flag = K_f

        print(f'num of IC50: {sum(self.IC50_flag)}')
        print(f'num of Kd: {sum(self.Kd_flag)}')
        print(f'num of Ki: {sum(self.Ki_flag)}')
        print(f'num of K: {sum(self.K_flag)}')

    def _load_node_feats_dim(self):
        self.lig_node_dim = self.lig_graphs[0].ndata['h'].shape[1]
        self.lig_edge_dim = self.lig_graphs[0].edata['e'].shape[1]
        self.pro_node_dim = self.prot_graphs[0].ndata['h'].shape[1]
        self.pro_edge_dim = self.prot_graphs[0].edata['e'].shape[1]
        self.inter_edge_dim = 15

    def _load(self):
        if not os.path.exists(f'{self.processed_dir}/multi_graphs.pkl'):
            self._process()
        with open(f'{self.processed_dir}/multi_graphs.pkl','rb') as f:
            self.Dataset = pickle.load(f)

        train_assay_d = {}
        with open(f'{self.config.base_path}/{self.config.data.dataset_path}/total_assay_descriptor_{self.config.data.assay_des_type}.pkl','rb') as f:
            train_assay_des = pickle.load(f)

        for (assay_id, assay_des) in train_assay_des:
            train_assay_d[assay_id] = torch.from_numpy(assay_des)

        self.lig_graphs = self.Dataset[0]
        self.prot_graphs = self.Dataset[1]
        self.inter_graphs = self.Dataset[2]
        self.labels = torch.tensor(self.Dataset[3], dtype=torch.float)
        self.assay_d = train_assay_d

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        names = commons.get_names_from_txt(self.complex_names_path)
        if self.test_100:
            names = names[:100]
        self.names = names
        if self.dataset_name == 'csar_test':
            labels = commons.get_labels_from_names_csar(os.path.join(self.config.base_path, self.complex_labels_path), names)
        else:
            labels = commons.get_labels_from_names(os.path.join(self.config.base_path, self.complex_labels_path),names)

        if not os.path.exists(f'{self.processed_dir}/molecular_representations.pkl'):
            molecular_representations = commons.pmap_multi(commons.read_molecules,zip(names,[self.dataset_path]*len(names)),
                                                           prot_graph_type=self.prot_graph_type,ligcut=self.ligcut,protcut=self.protcut, lig_type=self.lig_type,
                                                           LAS_mask=False, n_jobs=self.n_jobs,desc='read molecules')
            with open(f'{self.processed_dir}/molecular_representations.pkl','wb') as f:
                pickle.dump(molecular_representations,f)
        else:
            with open(f'{self.processed_dir}/molecular_representations.pkl','rb') as f:
                molecular_representations = pickle.load(f)

        lig_coords, lig_features, lig_edges, lig_node_type, lig_rdkit_coords, \
        prot_coords, prot_features, prot_edges, prot_node_type,\
        sec_features, alpha_c_coords, c_coords, n_coords,\
        _, _ = map(list, zip(*molecular_representations))


        if self.prot_graph_type.startswith('atom'):
            lig_graphs = commons.pmap_multi(commons.get_lig_graph,
                                            zip(lig_coords, lig_features, lig_edges, lig_node_type),
                                            cutoff=self.ligcut,
                                            n_jobs=self.n_jobs, desc='Get ligand graphs')
            prot_graphs = commons.pmap_multi(commons.get_prot_atom_graph,
                                             zip(prot_coords, prot_features, prot_edges, prot_node_type),
                                             cutoff=self.protcut,
                                             n_jobs=self.n_jobs, desc='Get protein atom graphs')
        else:
            lig_graphs = commons.pmap_multi(commons.get_lig_graph_equibind,
                                            zip(lig_coords, lig_features, lig_node_type),
                                            max_neighbors=self.lig_max_neighbors, cutoff=self.ligcut,
                                            n_jobs=self.n_jobs, desc='Get ligand graphs')
            prot_graphs = commons.pmap_multi(commons.get_prot_alpha_c_graph_equibind,
                                             zip(prot_coords, prot_features, prot_node_type, sec_features, alpha_c_coords, c_coords, n_coords),
                                             n_jobs=self.n_jobs, cutoff=self.protcut, max_neighbor=self.prot_max_neighbors,
                                             desc='Get protein alpha carbon graphs')

        inter_graphs = commons.pmap_multi(commons.get_interact_graph_knn_v2, zip(lig_coords, prot_coords),
                                          n_jobs=self.n_jobs, cutoff=self.intercut,
                                          max_neighbor=self.inter_max_neighbors,min_neighbor=self.inter_min_neighbors,
                                          desc='Get interaction graphs')

        processed_data = (lig_graphs, prot_graphs, inter_graphs, labels)
        with open(f'{self.processed_dir}/multi_graphs.pkl','wb') as f:
            pickle.dump(processed_data,f)


# def collate_affinity(batch):
#     g_ligs, g_prots, g_inters, labels, items = list(zip(*batch))
#     return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters), torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items)

def collate_pdbbind_affinity(batch):
    g_ligs, g_prots, g_inters, labels, items, des_list = list(zip(*batch))
    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
           torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
           torch.cat(des_list, dim=0)

def collate_pdbbind_affinity_multi_task(batch):
    g_ligs, g_prots, g_inters, labels, items, des_list, IC50_f_list, Kd_f_list, Ki_f_list = list(zip(*batch))
    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
           torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
           torch.cat(des_list, dim=0), list(IC50_f_list), list(Kd_f_list), list(Ki_f_list)

def collate_pdbbind_affinity_multi_task_v2(batch):
    g_ligs, g_prots, g_inters, labels, items, des_list, IC50_f_list, K_f_list = list(zip(*batch))
    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
           torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
           torch.cat(des_list, dim=0), list(IC50_f_list), list(K_f_list)

def collate_affinity_pair_wise(batch):
    g_ligs1, g_prots1, g_inters1, labels1, items1, des1_list,\
    g_ligs2, g_prots2, g_inters2, labels2, items2, des2_list = list(zip(*batch))
    g_ligs = g_ligs1 + g_ligs2
    g_prots = g_prots1 + g_prots2
    g_inters = g_inters1 + g_inters2
    labels =  labels1 + labels2
    items = items1 + items2
    des_list = des1_list + des2_list
    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
           torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
           torch.cat(des_list, dim=0)

def collate_affinity_pair_wise_multi_task(batch):
    g_ligs1, g_prots1, g_inters1, labels1, items1, des1_list, IC50_f_list1, Kd_f_list1, Ki_f_list1, \
    g_ligs2, g_prots2, g_inters2, labels2, items2, des2_list, IC50_f_list2, Kd_f_list2, Ki_f_list2 = list(zip(*batch))
    g_ligs = g_ligs1 + g_ligs2
    g_prots = g_prots1 + g_prots2
    g_inters = g_inters1 + g_inters2
    labels =  labels1 + labels2
    items = items1 + items2
    des_list = des1_list + des2_list
    IC50_f_list = IC50_f_list1 + IC50_f_list2
    Kd_f_list = Kd_f_list1 + Kd_f_list2
    Ki_f_list = Ki_f_list1 + Ki_f_list2

    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
           torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
           torch.cat(des_list, dim=0), list(IC50_f_list), list(Kd_f_list), list(Ki_f_list)


def collate_affinity_pair_wise_multi_task_v2(batch):
    g_ligs1, g_prots1, g_inters1, labels1, items1, des1_list, IC50_f_list1, K_f_list1, \
    g_ligs2, g_prots2, g_inters2, labels2, items2, des2_list, IC50_f_list2, K_f_list2 = list(zip(*batch))
    g_ligs = g_ligs1 + g_ligs2
    g_prots = g_prots1 + g_prots2
    g_inters = g_inters1 + g_inters2
    labels =  labels1 + labels2
    items = items1 + items2
    des_list = des1_list + des2_list
    IC50_f_list = IC50_f_list1 + IC50_f_list2
    K_f_list = K_f_list1 + K_f_list2

    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
           torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
           torch.cat(des_list, dim=0), list(IC50_f_list), list(K_f_list)
