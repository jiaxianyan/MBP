import os
import pickle
import dgl
import torch
from MBP import commons, layers
from copy import deepcopy
from collections import defaultdict
import random
from math import log
import numpy as np

class ChEMBLDock():
    def __init__(self, ligand_representations, prot_graphs, prot_coords, graph_prot_index, df,
                 assays, test_2, assay_des_type, assay_d=None, multi_task=False,
                 ligcut=5.0, protcut=8.0, intercut=12.0, lig_max_neighbors=None, prot_max_neighbors=10,
                 inter_min_neighbors=None, inter_max_neighbors=None, add_chemical_bond_feats=True,
                 use_mean_node_features=False, poses_pred_affinities=None, pose_select_rules=None,
                 confidence_threshold=0.3):
        self.ligand_representations = ligand_representations
        self.prot_graphs = prot_graphs
        self.prot_coords = prot_coords

        self.labels = torch.tensor([-log(float(x) * 1e-9, 10) for x in df['STANDARD_VALUE (nM)'].values], dtype=torch.float)
        self.graph_prot_index = graph_prot_index
        self.df = df

        assay_to_index = defaultdict(list)
        for idx, a in enumerate(df['ASSAY_ID'].values):
            assay_to_index[a].append(idx)

        IC50_flag = (df['STANDARD_TYPE'] == 'IC50').values.tolist()
        Kd_flag = (df['STANDARD_TYPE'] == 'Kd').values.tolist()
        Ki_flag = (df['STANDARD_TYPE'] == 'Ki').values.tolist()

        K_flag = []
        for kd, ki in zip(Kd_flag, Ki_flag):
            if kd or ki:
                K_flag.append(True)
            else:
                K_flag.append(False)
        K_flag = K_flag

        print(f'num of IC50: {sum(IC50_flag)}')
        print(f'num of Kd: {sum(Kd_flag)}')
        print(f'num of Ki: {sum(Ki_flag)}')
        print(f'num of K: {sum(K_flag)}')

        self.assay_to_index = assay_to_index
        self.IC50_flag = IC50_flag
        self.Kd_flag = Kd_flag
        self.Ki_flag = Ki_flag
        self.K_flag = K_flag

        self.assays = assays
        self.test_2 = test_2
        self.assay_des_type = assay_des_type
        self.assay_d = assay_d
        self.multi_task = multi_task

        self.ligcut = ligcut
        self.protcut = protcut
        self.intercut = intercut
        self.lig_max_neighbors = lig_max_neighbors
        self.prot_max_neighbors = prot_max_neighbors
        self.inter_min_neighbors = inter_min_neighbors
        self.inter_max_neighbors = inter_max_neighbors

        self.add_chemical_bond_feats = add_chemical_bond_feats
        self.use_mean_node_features = use_mean_node_features

        self.poses_pred_affinities = poses_pred_affinities
        self.pose_select_rules = pose_select_rules
        self.confidence_threshold = confidence_threshold

        self._load_node_feats_dim()

        print(f'num of data in dataset: {len(self.df)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        item_pw = self.get_pair_wise_item(item)
        lig_graph, prot_graph, inter_graph, label, item, IC50_f, Kd_f, Ki_f, K_f = self.get_item_data(item)
        lig_graph_2, prot_graph_2, inter_graph_2, label_2, item_2, IC50_f2, Kd_f2, Ki_f2, K_f2 = self.get_item_data(item_pw)
        assay_des = self.get_item_assay_emb(item)

        return lig_graph, prot_graph, inter_graph, label, item, assay_des.unsqueeze(dim=0), IC50_f, K_f, \
               lig_graph_2, prot_graph_2, inter_graph_2, label_2, item_2, assay_des.unsqueeze(dim=0), IC50_f2, K_f2

    def get_item_assay_emb(self, item):
        assay_id = self.df['ASSAY_ID'].values[item]

        if self.assay_d is not None:
            assay_des = self.assay_d[assay_id]
        else:
            assay_des = torch.zeros(0)

        return assay_des

    def get_item_data(self, item):
        label = deepcopy(self.labels[item])

        lig_coords, lig_features, lig_edges, lig_node_types = self.ligand_representations[item]
        if isinstance(lig_coords, list):
            conf_index = random.randint(0, len(lig_coords) - 1)
            lig_coords = lig_coords[conf_index]

        lig_graph = commons.get_lig_graph_equibind(lig_coords, lig_features, lig_edges, lig_node_types,
                                                   max_neighbors=self.lig_max_neighbors, cutoff=self.ligcut)

        if self.add_chemical_bond_feats:
            lig_graph.edata['e'] = torch.cat([lig_graph.edata['e'], lig_graph.edata['bond_type']], dim=-1)

        prot_graph_index = self.graph_prot_index[item]
        prot_graph = deepcopy(self.prot_graphs[prot_graph_index])

        if self.use_mean_node_features:
            lig_graph.ndata['h'] = torch.cat([lig_graph.ndata['h'], lig_graph.ndata['mu_r_norm']], dim=-1)
            prot_graph.ndata['h'] = torch.cat([prot_graph.ndata['h'],prot_graph.ndata['mu_r_norm']], dim=-1)

        inter_graph = commons.get_interact_graph_knn(lig_coords, self.prot_coords[prot_graph_index],
                                                     cutoff=self.intercut, max_neighbor=self.inter_max_neighbors,
                                                     min_neighbor=self.inter_min_neighbors)
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
        K_f = deepcopy(self.K_flag[item])

        return lig_graph, prot_graph, inter_graph, label, item, IC50_f, Kd_f, Ki_f, K_f

    def get_pair_wise_item(self, item):
        assay_id = self.df['ASSAY_ID'].values[item]
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

        return sample_item

    def _load_node_feats_dim(self):
        lig_graph, prot_graph, inter_graph, label, item, IC50_f, Kd_f, Ki_f, K_f = self.get_item_data(0)
        self.lig_node_dim = lig_graph.ndata['h'].shape[1]
        self.lig_edge_dim = lig_graph.edata['e'].shape[1]
        self.pro_node_dim = prot_graph.ndata['h'].shape[1]
        self.pro_edge_dim = prot_graph.edata['e'].shape[1]
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


        self.use_mean_node_features = config.data.use_mean_node_features
        self.add_chemical_bond_feats = config.data.add_chemical_bond_feats
        if config.data.add_chemical_bond_feats:
            self.processed_dir += '_bf'
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

        if self.add_chemical_bond_feats:
            lig_graph.edata['e'] = torch.cat([lig_graph.edata['e'], lig_graph.edata['bond_type']], dim=-1)

        if self.use_mean_node_features:
            lig_graph.ndata['h'] = torch.cat([lig_graph.ndata['h'], lig_graph.ndata['mu_r_norm']], dim=-1)
            prot_graph.ndata['h'] = torch.cat([prot_graph.ndata['h'],prot_graph.ndata['mu_r_norm']], dim=-1)

        fintune_assay_id = -1

        if self.assay_d is not None:
            assay_des = deepcopy(self.assay_d[fintune_assay_id])
        else:
            assay_des = torch.zeros(0)

        # assay_des = deepcopy(self.assay_d[fintune_assay_id])

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
        if self.add_chemical_bond_feats:
            self.lig_edge_dim += self.lig_graphs[0].edata['bond_type'].shape[1]

        self.pro_node_dim = self.prot_graphs[0].ndata['h'].shape[1]
        self.pro_edge_dim = self.prot_graphs[0].edata['e'].shape[1]
        self.inter_edge_dim = 15

        if self.use_mean_node_features:
            self.lig_node_dim += 5
            self.pro_node_dim += 5

    def _load(self):
        if not os.path.exists(f'{self.processed_dir}/multi_graphs.pkl'):
            self._process()
        with open(f'{self.processed_dir}/multi_graphs.pkl','rb') as f:
            self.Dataset = pickle.load(f)

        if self.config.train.pretrain_use_assay_description:
            train_assay_d = {}
            with open(f'{self.config.base_path}/{self.config.data.dataset_path}/total_assay_descriptor_{self.config.data.assay_des_type}.pkl','rb') as f:
                train_assay_des = pickle.load(f)
            for (assay_id, assay_des) in train_assay_des:
                train_assay_d[assay_id] = torch.from_numpy(assay_des)
        else:
            train_assay_d = None

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


        lig_graphs = commons.pmap_multi(commons.get_lig_graph_equibind,
                                        zip(lig_coords, lig_features, lig_edges, lig_node_type),
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
            pickle.dump(processed_data, f)

def collate_finetune(batch):
    g_ligs, g_prots, g_inters, labels, items, des_list, IC50_f_list, K_f_list = list(zip(*batch))
    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
           torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
           torch.cat(des_list, dim=0), list(IC50_f_list), list(K_f_list)

def collate_pretrain(batch):
    g_ligs1, g_prots1, g_inters1, labels1, items1, des1_list, IC50_f_list1, K_f_list1, \
    g_ligs2, g_prots2, g_inters2, labels2, items2, des2_list, IC50_f_list2, K_f_list2 = list(zip(*batch))
    g_ligs = g_ligs1 + g_ligs2
    g_prots = g_prots1 + g_prots2
    g_inters = g_inters1 + g_inters2
    labels = labels1 + labels2
    items = items1 + items2
    des_list = des1_list + des2_list
    IC50_f_list = IC50_f_list1 + IC50_f_list2
    K_f_list = K_f_list1 + K_f_list2

    return dgl.batch(g_ligs), dgl.batch(g_prots), dgl.batch(g_inters),\
           torch.unsqueeze(torch.stack(labels, dim=0), dim=-1), list(items),\
           torch.cat(des_list, dim=0), list(IC50_f_list), list(K_f_list)
