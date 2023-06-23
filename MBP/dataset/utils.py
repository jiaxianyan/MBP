from rdkit import Chem
from rdkit.Chem import GetMolFrags, AllChem
from collections import Counter
import numpy as np
import pandas as pd
from math import log
from tqdm import tqdm
import random
import os
from MBP import commons, layers
import pickle
import dgl
import torch
from .chembl import ChEMBL_Dock_PointWise, ChEMBL_Dock_PairWise, ChEMBL_Dock_Valid, ChEMBL_Dock_PointWise_v2, ChEMBL_Dock_PairWise_v2, ChEMBL_Dock_Valid_v2
from collections import defaultdict
import torch.distributed as dist

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
    c = Counter(df_.ASSAY_ID)
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

def process_chembl(config):
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
    top_N = config.data.top_N
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

    processed_dir = f'{config.base_path}/{config.data.dataset_path}/processed/' \
                    f'{dataset_name}_{affinity_type}_{affinity_relation}' \
                    f'_{lig_type}_{task_target}_gtype{prot_graph_type}' \
                    f'_lcut{ligcut}_pcut{protcut}_icut{intercut}' \
                    f'_lgmn{lig_max_neighbors}_pgmn{prot_max_neighbors}' \
                    f'_igmn{inter_min_neighbors}_igmn{inter_max_neighbors}'

    if top_N > 1:
        processed_dir += f'_top_{top_N}'

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
                uniprot_valid_index = [int(f.split('_')[0]) for f in uniprot_smina_ligands if f.endswith('_1.sdf')]
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

            if top_N > 1:
                ligand_representations = commons.pmap_multi(commons.read_ligands_chembl_smina_multi_pose,
                                                            zip(uniprot_names, valid_ligand_flag_list), dataset_path=dataset_path,
                                                            ligcut=ligcut, lig_type=lig_type, top_N=top_N,
                                                            n_jobs=n_jobs, desc='read multi-pose ligands')

            else:
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

        if top_N > 1:
            lig_graphs = commons.pmap_multi(commons.get_lig_multi_pose_graph_equibind,
                                                  zip(lig_coords, lig_features, lig_node_type),
                                                  max_neighbors=lig_max_neighbors, cutoff=ligcut,
                                                  n_jobs=n_jobs, desc='Get multi-pose ligand graphs')
        else:
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

        if top_N > 1:
            inter_graphs = commons.pmap_multi(commons.get_interact_multi_pose_graph_knn,
                                                    zip(lig_coords, prot_coords_list),
                                                    n_jobs=n_jobs, cutoff=intercut,
                                                    max_neighbor=inter_max_neighbors, min_neighbor=inter_min_neighbors,
                                                    desc='Get multi-pose interaction graphs')
        else:
            inter_graphs = commons.pmap_multi(commons.get_interact_graph_knn, zip(lig_coords, prot_coords_list),
                                              n_jobs=n_jobs, cutoff=intercut,
                                              max_neighbor=inter_max_neighbors,min_neighbor=inter_min_neighbors,
                                              desc='Get interaction graphs')

        processed_data = (lig_graphs, prot_graphs, inter_graphs, df_info, graph_prot_index)
        with open(f'{processed_dir}/multi_graphs.pkl','wb') as f:
            pickle.dump(processed_data,f)

    return processed_dir

def process_chembl_v2(config):
    # do not prepare graph
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
    top_N = config.data.top_N
    test_2 = config.data.test_2
    add_chemical_bond_feats = config.data.add_chemical_bond_feats
    docking_type = config.data.docking_type

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

    processed_dir = f'{config.base_path}/{config.data.dataset_path}/processed/' \
                    f'{dataset_name}_{affinity_type}_{affinity_relation}' \
                    f'_{lig_type}_{task_target}_gtype{prot_graph_type}' \
                    f'_lcut{ligcut}_pcut{protcut}_icut{intercut}' \
                    f'_lgmn{lig_max_neighbors}_pgmn{prot_max_neighbors}' \
                    f'_igmn{inter_min_neighbors}_igmn{inter_max_neighbors}'

    if top_N > 1:
        processed_dir += f'_top_{top_N}'

    if add_chemical_bond_feats:
        processed_dir += '_bf'

    if docking_type == 'blind':
        processed_dir +='_bl'
    elif docking_type == 'all':
        processed_dir +='_all'
        assert top_N == 9

    if test_2:
        processed_dir += '_test2'

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    dataset_path = f'{config.base_path}/{config.data.dataset_path}/{config.data.dataset_name}'

    if affinity_relation == 'all':
        affinity_relation_list = ['=', '>']
    else:
        affinity_relation_list = [affinity_relation]

    if not os.path.exists(f'{processed_dir}/simplied_processed_data.pkl'):
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
                uniprot_valid_index = [int(f.split('_')[0]) for f in uniprot_smina_ligands if f.endswith('_1.sdf')]
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

            if top_N > 1:
                ligand_representations = commons.pmap_multi(commons.read_ligands_chembl_smina_multi_pose,
                                                            zip(uniprot_names, valid_ligand_flag_list), dataset_path=dataset_path,
                                                            ligcut=ligcut, lig_type=lig_type, top_N=top_N, docking_type=docking_type,
                                                            n_jobs=n_jobs, desc='read multi-pose ligands')

            else:
                ligand_representations = commons.pmap_multi(commons.read_ligands_chembl_smina,
                                                            zip(uniprot_names, valid_ligand_flag_list), dataset_path=dataset_path,
                                                            ligcut=ligcut, lig_type=lig_type, docking_type=docking_type,
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

        prot_coords, prot_features, prot_edges, prot_node_type, \
        sec_features, alpha_c_coords, c_coords, n_coords = map(list, zip(*protein_representations))

        prot_graphs = commons.pmap_multi(commons.get_prot_alpha_c_graph_equibind,
                                         zip(prot_coords, prot_features, prot_node_type, sec_features, alpha_c_coords, c_coords, n_coords),
                                         n_jobs=n_jobs, cutoff=protcut, max_neighbor=prot_max_neighbors,
                                         desc='Get protein alpha carbon graphs')

        graph_prot_index = []
        index = 0
        for valid_index in valid_index_list:
            graph_prot_index.extend([index] * len(valid_index))
            index += 1

        processed_data = (prot_graphs, prot_coords, df_info, graph_prot_index)
        with open(f'{processed_dir}/simplied_processed_data.pkl','wb') as f:
            pickle.dump(processed_data,f)

    return processed_dir

def load_assay_des(config):
    if not config.train.pretrain_use_assay_description:
        return None

    assay_des_type = config.data.assay_des_type
    train_assay_d = {}
    if assay_des_type is not None:
        with open(f'{config.base_path}/{config.data.dataset_path}/total_assay_descriptor_{assay_des_type}.pkl','rb') as f:
            train_assay_des = pickle.load(f)

        for (assay_id, assay_des) in train_assay_des:
            train_assay_d[assay_id] = torch.from_numpy(assay_des)

    return train_assay_d

def split_assays(df_info, processed_dir):
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

    return train_assay, valid_assay

def SelectData(lig_graphs, prot_graphs, inter_graphs, graph_prot_index, df_total, assay_select):
    index_flag = df_total['ASSAY_ID'].isin(assay_select)
    df_select = df_total[index_flag]

    lig_graphs_select, inter_graphs_select, graph_prot_index_select = [], [], []
    for index, flag in enumerate(index_flag):
        if flag:
            lig_graphs_select.append(lig_graphs[index])
            inter_graphs_select.append(inter_graphs[index])
            graph_prot_index_select.append(graph_prot_index[index])

    return lig_graphs_select, prot_graphs, inter_graphs_select, graph_prot_index_select, df_select

def SelectData_v2(ligand_representations, graph_prot_index, df_total, poses_affinities, assay_select):
    index_flag = df_total['ASSAY_ID'].isin(assay_select)
    df_select = df_total[index_flag]

    lig_coords_list, lig_features_list, \
    lig_edges_list, lig_node_type_list, index_list = map(list, zip(*ligand_representations))
    lig_coords, lig_features, lig_edges, lig_node_type = [], [], [], []

    for valid_lig_coords, valid_lig_features, valid_lig_edges, valid_lig_node_type in zip(
            lig_coords_list, lig_features_list, lig_edges_list, lig_node_type_list):
        lig_coords.extend(valid_lig_coords)
        lig_features.extend(valid_lig_features)
        lig_edges.extend(valid_lig_edges)
        lig_node_type.extend(valid_lig_node_type)

    print(f'num of total dataset: {len(lig_coords)}')

    ligand_representations_select, graph_prot_index_select, poses_affinities_select = [], [], []
    for index, flag in enumerate(index_flag):
        if flag:
            ligand_representations_select.append((lig_coords[index], lig_features[index], lig_edges[index], lig_node_type[index]))
            graph_prot_index_select.append(graph_prot_index[index])
            poses_affinities_select.append(poses_affinities[index])

    return ligand_representations_select, graph_prot_index_select, df_select, poses_affinities_select

def load_ChEMBL_Dock(config):
    # filter dataset and generate graphs
    processed_dir = process_chembl(config)

    # load assay descriptor embedding
    train_assay_d = load_assay_des(config)

    # load generated graphs
    with open(f'{processed_dir}/multi_graphs.pkl', 'rb') as f:
        Dataset = pickle.load(f)
    lig_graphs, prot_graphs, inter_graphs, df_info, graph_prot_index = Dataset

    # split assays
    train_assay, valid_assay = split_assays(df_info, processed_dir)

    # get dataset
    lig_graphs_valid, prot_graphs_valid, inter_graphs_valid, graph_prot_index_valid, df_valid = \
        SelectData(lig_graphs, prot_graphs, inter_graphs, graph_prot_index, df_info, valid_assay)

    valid_data = ChEMBL_Dock_Valid(lig_graphs_valid, prot_graphs_valid, inter_graphs_valid, graph_prot_index_valid,
                                   df_valid, valid_assay, config.data.test_2, config.data.assay_des_type, train_assay_d,
                                   config.train.multi_task)

    lig_graphs_train, prot_graphs_train, inter_graphs_train, graph_prot_index_train, df_train = \
        SelectData(lig_graphs, prot_graphs, inter_graphs, graph_prot_index, df_info, train_assay)

    if config.train.pretrain_sampling_method == 'pairwise_v1':
        train_data = ChEMBL_Dock_PairWise(lig_graphs_train, prot_graphs_train, inter_graphs_train,
                                          graph_prot_index_train, df_train, train_assay, config.data.test_2,
                                          config.data.assay_des_type, train_assay_d, config.train.multi_task)

    elif config.train.pretrain_sampling_method == 'pointwise':
        train_data = ChEMBL_Dock_PointWise(lig_graphs_train, prot_graphs_train, inter_graphs_train,
                                           graph_prot_index_train, df_train, train_assay, config.data.test_2,
                                           config.data.assay_des_type, train_assay_d, config.train.multi_task)

    return train_data, valid_data

def load_ChEMBL_Dock_v2(config):
    # do not prepare graph in advance
    # filter dataset and generate graphs
    processed_dir = process_chembl_v2(config)

    # load assay descriptor embedding
    train_assay_d = load_assay_des(config)

    # load representations
    with open(f'{processed_dir}/ligand_representations.pkl', 'rb') as f:
        ligand_representations = pickle.load(f)

    with open(f'{processed_dir}/simplied_processed_data.pkl', 'rb') as f:
        prot_graphs, prot_coords, df_info, graph_prot_index = pickle.load(f)

    if config.data.top_N > 1 and config.data.poses_select_rules not in ['random', None]:
        with open(f'{processed_dir}/poses_pred_affinities_pretrain.pkl', 'rb') as f:
            poses_affinities = pickle.load(f)
    else:
        poses_affinities = [None] * len(df_info)

    # split assays
    train_assay, valid_assay = split_assays(df_info, processed_dir)

    ligand_representations_train, graph_prot_index_train, df_train, poses_affinities_train = \
        SelectData_v2(ligand_representations, graph_prot_index, df_info, poses_affinities, train_assay)

    if config.train.pretrain_sampling_method == 'pairwise_v1':
        train_data = ChEMBL_Dock_PairWise_v2(ligand_representations_train, prot_graphs, prot_coords,
                                             graph_prot_index_train, df_train, train_assay, config.data.test_2,
                                             config.data.assay_des_type, train_assay_d, config.train.multi_task,
                                             config.data.ligcut, config.data.protcut, config.data.intercut,
                                             config.data.lig_max_neighbors, config.data.prot_max_neighbors,
                                             config.data.inter_min_neighbors, config.data.inter_max_neighbors,
                                             config.data.add_chemical_bond_feats, config.data.use_mean_node_features,
                                             poses_affinities_train, config.data.confidence_threshold)

    # elif config.train.pretrain_sampling_method == 'pointwise':
    #     train_data = ChEMBL_Dock_PointWise_v2(ligand_representations_train, prot_graphs, prot_coords,
    #                                           graph_prot_index_train, df_train, train_assay, config.data.test_2,
    #                                           config.data.assay_des_type, train_assay_d, config.train.multi_task,
    #                                           config.data.ligcut, config.data.protcut, config.data.intercut,
    #                                           config.data.lig_max_neighbors, config.data.prot_max_neighbors,
    #                                           config.data.inter_min_neighbors, config.data.inter_max_neighbors,
    #                                           config.data.add_chemical_bond_feats, config.data.use_mean_node_features)

    return train_data, None


def ddp_dataset_partite(lig_graphs, prot_graphs, inter_graphs, graph_prot_index, df_total, ddp_world_size, processed_dir):

    total_index = list(range(len(lig_graphs)))
    average_num = len(total_index) // ddp_world_size

    for rank in range(ddp_world_size):
        if not os.path.exists(f'{processed_dir}/multi_graphs_partition_{rank}.pkl'):
            partition_data_index = total_index[rank * average_num: (rank + 1) * average_num]
            df_index_flag = [False] * len(total_index)
            lig_graphs_select, prot_graphs_select, inter_graphs_select, graph_prot_index_select = [], [], [], []
            for idx in partition_data_index:
                lig_graphs_select.append(lig_graphs[idx])
                inter_graphs_select.append(inter_graphs[idx])
                graph_prot_index_select.append(graph_prot_index[idx])
                df_index_flag[idx] = True

            df_partition = df_total[df_index_flag]

            data_partition = (lig_graphs_select, prot_graphs, inter_graphs_select, df_partition, graph_prot_index_select)

            with open(f'{processed_dir}/multi_graphs_partition_{rank}.pkl','wb') as f:
                pickle.dump(data_partition, f)

def check_partition_finish(ddp_world_size, processed_dir):
    for rank in range(ddp_world_size):
        if not os.path.exists(f'{processed_dir}/multi_graphs_partition_{rank}.pkl'):
            return False

    return True

def load_memoryefficient_ChEMBL_Dock(config):
    # filter dataset and generate graphs
    processed_dir = process_chembl(config)

    # load assay descriptor embedding
    train_assay_d = load_assay_des(config)

    # data partition
    if dist.get_rank() == 0 and not check_partition_finish(dist.get_world_size(), processed_dir):
        # load full data
        with open(f'{processed_dir}/multi_graphs.pkl', 'rb') as f:
            Full_Dataset = pickle.load(f)
        lig_graphs, prot_graphs, inter_graphs, df_info, graph_prot_index = Full_Dataset

        # split train/valid set
        train_assay, valid_assay = split_assays(df_info, processed_dir)

        # select valid set
        lig_graphs_valid, prot_graphs_valid, inter_graphs_valid, graph_prot_index_valid, df_valid = \
            SelectData(lig_graphs, prot_graphs, inter_graphs, graph_prot_index, df_info, valid_assay)

        # save valid set
        with open(f'{processed_dir}/multi_graphs_partition_valid.pkl','wb') as f:
            pickle.dump((lig_graphs_valid, prot_graphs_valid, inter_graphs_valid, graph_prot_index_valid, df_valid), f)

        # select train set
        lig_graphs_train, prot_graphs_train, inter_graphs_train, graph_prot_index_train, df_train = \
            SelectData(lig_graphs, prot_graphs, inter_graphs, graph_prot_index, df_info, train_assay)

        # partite and save train data
        ddp_dataset_partite(lig_graphs_train, prot_graphs_train, inter_graphs_train, graph_prot_index_train, df_train,
                            dist.get_world_size(), processed_dir)

        del Full_Dataset

    dist.barrier()

    # load train/valid split result
    train_assay, valid_assay = split_assays(None, processed_dir)

    # load valid set
    with open(f'{processed_dir}/multi_graphs_partition_valid.pkl', 'rb') as f:
        Valid_Dataset = pickle.load(f)

    lig_graphs_valid, prot_graphs_valid, inter_graphs_valid, graph_prot_index_valid, df_valid = Valid_Dataset
    valid_data = ChEMBL_Dock_Valid(lig_graphs_valid, prot_graphs_valid, inter_graphs_valid, graph_prot_index_valid,
                                   df_valid, valid_assay, config.data.test_2, config.data.assay_des_type, train_assay_d,
                                   config.train.multi_task)

    # load partition train set
    with open(f'{processed_dir}/multi_graphs_partition_{dist.get_rank()}.pkl', 'rb') as f:
        Partition_Dataset = pickle.load(f)

    lig_graphs_train, prot_graphs_train, inter_graphs_train, df_train, graph_prot_index_train = Partition_Dataset
    if config.train.pretrain_sampling_method == 'pairwise_v1':
        train_data = ChEMBL_Dock_PairWise(lig_graphs_train, prot_graphs_train, inter_graphs_train,
                                          graph_prot_index_train, df_train, train_assay, config.data.test_2,
                                          config.data.assay_des_type, train_assay_d, config.train.multi_task)

    elif config.train.pretrain_sampling_method == 'pointwise':
        train_data = ChEMBL_Dock_PointWise(lig_graphs_train, prot_graphs_train, inter_graphs_train,
                                           graph_prot_index_train, df_train, train_assay, config.data.test_2,
                                           config.data.assay_des_type, train_assay_d, config.train.multi_task)

    return train_data, valid_data