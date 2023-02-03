import os
from collections import defaultdict
import numpy as np
import torch
from openbabel import pybel
from statistics import stdev
from time import time
from .utils import pmap_multi
import pandas as pd
from tqdm import tqdm

MGLTols_PYTHON = '/apdcephfs/private_jiaxianyan/dock/mgltools_x86_64Linux2_1.5.7/bin/python2.7'
Prepare_Ligand = '/apdcephfs/private_jiaxianyan/dock/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'
Prepare_Receptor = '/apdcephfs/private_jiaxianyan/dock/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py'
SMINA = '/apdcephfs/private_jiaxianyan/dock/smina'

def read_matric(matric_file_path):
    with open(matric_file_path) as f:
        lines = f.read().strip().split('\n')
    rmsd, centroid = float(lines[0].split(':')[1]), float(lines[1].split(':')[1])
    return rmsd, centroid

def mol2_add_atom_index_to_atom_name(mol2_file_path):
    MOL_list = [x for x in open(mol2_file_path, 'r')]
    idx = [i for i, x in enumerate(MOL_list) if x.startswith('@')]
    block = MOL_list[idx[1] + 1:idx[2]]
    block = [x.split() for x in block]

    block_new = []
    atom_count = defaultdict(int)
    for i in block:
        at = i[5].strip().split('.')[0]
        if 'H' not in at:
            atom_count[at] += 1
            count = atom_count[at]
            at_new = at + str(count)
            at_new = at_new.rjust(4)
            block_new.append([i[0], at_new] + i[2:])
        else:
            block_new.append(i)

    block_new = ['\t'.join(x) + '\n' for x in block_new]
    MOL_list_new = MOL_list[:idx[1] + 1] + block_new + MOL_list[idx[2]:]
    with open(mol2_file_path, 'w') as f:
        for line in MOL_list_new:
            f.write(line)
    return

def prepare_dock_file(pdb_name, config):
    visualize_dir = os.path.join(config.train.save_path, 'visualize_dir')
    post_align_sdf = os.path.join(visualize_dir, f'{pdb_name}_post_align_{config.train.align_method}.sdf')
    post_align_mol2 = os.path.join(visualize_dir, f'{pdb_name}_post_align_{config.train.align_method}.mol2')
    post_align_pdbqt = os.path.join(visualize_dir, f'{pdb_name}_post_align_{config.train.align_method}.pdbqt')

    # mgltools preprocess
    cmd = f'cd {visualize_dir}'
    cmd += f' && obabel -i sdf {post_align_sdf} -o mol2 -O {post_align_mol2}'

    if not os.path.exists(post_align_mol2):
        os.system(cmd)
        mol2_add_atom_index_to_atom_name(post_align_mol2)

    cmd = f'cd {visualize_dir}'
    cmd += f' && {MGLTols_PYTHON} {Prepare_Ligand} -l {post_align_mol2}'

    if not os.path.exists(post_align_pdbqt):
        os.system(cmd)
    # cmd = f'obabel -i mol2 {post_align_mol2} -o pdbqt -O {post_align_pdbqt}'
    # os.system(cmd)

    return

def get_mol2_atom_name(mol2_file_path):
    MOL_list = [x for x in open(mol2_file_path, 'r')]
    idx = [i for i, x in enumerate(MOL_list) if x.startswith('@')]
    block = MOL_list[idx[1] + 1:idx[2]]
    block = [x.split() for x in block]

    atom_names = []

    for i in block:
        at = i[1].strip()
        atom_names.append(at)
    return atom_names

def align_dock_name_and_target_name(dock_lig_atom_names, target_lig_atom_names):
    dock_lig_atom_index_in_target_lig = []
    target_atom_name_dict = {}
    for index, atom_name in enumerate(target_lig_atom_names):
        try:
            assert atom_name not in target_atom_name_dict.keys()
        except:
            raise ValueError(atom_name,'has appeared before')
        target_atom_name_dict[atom_name] = index

    dock_lig_atom_name_appears_dict = defaultdict(int)
    for atom_name in dock_lig_atom_names:
        try:
            assert atom_name not in dock_lig_atom_name_appears_dict.keys()
        except:
            raise ValueError(atom_name,'has appeared before')
        dock_lig_atom_name_appears_dict[atom_name] += 1
        try:
            dock_lig_atom_index_in_target_lig.append(target_atom_name_dict[atom_name])
        except:
            dock_lig_atom_index_in_target_lig.append(target_atom_name_dict[atom_name+'1'])

    return dock_lig_atom_index_in_target_lig

def smina_dock_result_rmsd(pdb_name, config):
    # target path
    target_lig_mol2 = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_ligand.mol2')

    # get target coords
    target_m_lig = next(pybel.readfile('mol2', target_lig_mol2))
    target_lig_coords = [atom.coords for atom in target_m_lig if atom.atomicnum > 1]
    target_lig_coords = np.array(target_lig_coords, dtype=np.float32)  # np.array, [n, 3]
    target_lig_center = target_lig_coords.mean(axis=0)  # np.array, [3]

    # get target atom names
    visualize_dir = os.path.join(config.train.save_path, 'visualize_dir')
    lig_init_mol2 = os.path.join(visualize_dir, f'{pdb_name}_post_align_{config.train.align_method}.mol2')
    target_atom_name_reference_lig = next(pybel.readfile('mol2', lig_init_mol2))
    target_lig_atom_names = get_mol2_atom_name(lig_init_mol2)
    target_lig_atom_names_no_h = [atom_name for atom, atom_name in zip(target_atom_name_reference_lig, target_lig_atom_names) if atom.atomicnum > 1]

    # get init coords
    coords_before_minimized = [atom.coords for atom in target_atom_name_reference_lig if atom.atomicnum > 1]
    coords_before_minimized = np.array(coords_before_minimized, dtype=np.float32)  # np.array, [n, 3]

    # get smina minimized coords
    dock_lig_mol2_path = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_post_align_{config.train.align_method}_docked.mol2')
    dock_m_lig = next(pybel.readfile('mol2', dock_lig_mol2_path))
    dock_lig_coords = [atom.coords for atom in dock_m_lig if atom.atomicnum > 1]
    dock_lig_coords = np.array(dock_lig_coords, dtype=np.float32) # np.array, [n, 3]
    dock_lig_center = dock_lig_coords.mean(axis=0)  # np.array, [3]

    # get atom names
    dock_lig_atom_names = get_mol2_atom_name(dock_lig_mol2_path)
    dock_lig_atom_names_no_h = [atom_name for atom, atom_name in zip(dock_m_lig, dock_lig_atom_names) if atom.atomicnum > 1]
    dock_lig_atom_index_in_target_lig = align_dock_name_and_target_name(dock_lig_atom_names_no_h, target_lig_atom_names_no_h)

    dock_lig_coords_target_align = np.zeros([len(dock_lig_atom_index_in_target_lig),3], dtype=np.float32)
    for atom_coords, atom_index_in_target_lig in zip(dock_lig_coords, dock_lig_atom_index_in_target_lig):
        dock_lig_coords_target_align[atom_index_in_target_lig] = atom_coords

    # rmsd
    error_lig_coords = dock_lig_coords_target_align - target_lig_coords
    rmsd = np.sqrt((error_lig_coords ** 2).sum(axis=1, keepdims=True).mean(axis=0))

    # centroid
    error_center_coords = dock_lig_center - target_lig_center
    centorid_d = np.sqrt( (error_center_coords ** 2).sum() )

    # get rmsd after minimized
    error_lig_coords_after_minimized = dock_lig_coords_target_align - coords_before_minimized
    rmsd_after_minimized = np.sqrt((error_lig_coords_after_minimized ** 2).sum(axis=1, keepdims=True).mean(axis=0))

    return float(rmsd), float(centorid_d), float(rmsd_after_minimized)

def get_matric_dict(rmsds, centroids):
    rmsd_mean = sum(rmsds)/len(rmsds)
    centroid_mean = sum(centroids) / len(centroids)
    rmsd_std = stdev(rmsds)
    centroid_std = stdev(centroids)

    # rmsd < 2
    count = torch.tensor(rmsds) < 2.0
    rmsd_less_than_2 = 100 * count.sum().item() / len(count)

    # rmsd < 2
    count = torch.tensor(rmsds) < 5.0
    rmsd_less_than_5 = 100 * count.sum().item() / len(count)

    # centorid < 2
    count = torch.tensor(centroids) < 2.0
    centroid_less_than_2 = 100 * count.sum().item() / len(count)

    # centorid < 5
    count = torch.tensor(centroids) < 5.0
    centroid_less_than_5 = 100 * count.sum().item() / len(count)

    metrics_dict = {'rmsd mean': rmsd_mean, 'rmsd std': rmsd_std, 'centroid mean': centroid_mean, 'centroid std': centroid_std,
                         'rmsd less than 2': rmsd_less_than_2, 'rmsd less than 5':rmsd_less_than_5,
                         'centroid less than 2': centroid_less_than_2, 'centroid less than 5': centroid_less_than_5}
    return metrics_dict

def run_smina_dock(pdb_name ,config):

    r_pdbqt = os.path.join(config.test_set.dataset_path, pdb_name, f'{pdb_name}_protein_processed.pdbqt')
    l_pdbqt = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_post_align_{config.train.align_method}.pdbqt')
    out_mol2 = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_post_align_{config.train.align_method}_docked.mol2')
    log_file = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_post_align_{config.train.align_method}_docked.log')
    cmd = f'{SMINA}' \
          f' --receptor {r_pdbqt}' \
          f' --ligand {l_pdbqt}' \
          f' --out {out_mol2}' \
          f' --log {log_file}' \
          f' --minimize'
    os.system(cmd)

    return

def run_score_only(ligand_file, protein_file, out_log_file):
    cmd = f'{SMINA}' \
          f' --receptor {protein_file}' \
          f' --ligand {ligand_file}' \
          f' --score_only' \
          f' > {out_log_file}'
    os.system(cmd)

    with open(out_log_file, 'r') as f:
        lines = f.read().strip().split('\n')
    affinity_score = float(lines[21].split()[1])

    return affinity_score

def run_smina_score_after_predict(config):
    pdb_name_list = config.test_set.names
    smina_score_list = []
    for pdb_name in tqdm(pdb_name_list):
        ligand_file = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_pred.sdf')
        protein_file = os.path.join(config.test_set.dataset_path, pdb_name, f'{pdb_name}_protein_processed.pdbqt')
        out_log_file = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_pred_smina_score.out')
        smina_score = run_score_only(ligand_file, protein_file, out_log_file)
        smina_score_list.append(smina_score)

    result_d = {'pdb_name':pdb_name_list, 'smina_score':smina_score_list}
    pd.DataFrame(result_d).to_csv(os.path.join(config.train.save_path, 'visualize_dir', 'pred_smina_score.csv'))
    return

def run_smina_minimize_after_predict(config):
    minimize_time = 0

    pdb_name_list = config.test_set.names

    # pmap_multi(prepare_dock_file, zip(pdb_name_list, [config] * len(pdb_name_list)),
    #            n_jobs=8, desc='mgltools preparing ...')

    rmsds_post_dock, centroids_post_dock = [], []
    rmsds_post, centroids_post = [], []
    rmsds, centroids = [], []

    rmsds_after_minimized = {'pdb_name':[], 'rmsd':[]}

    valid_pdb_name = []
    error_list = []
    # for pdb_name in tqdm(pdb_name_list):
    #     try:
    #         minimize_begin_time = time()
    #         run_smina_dock(pdb_name, config)
    #         minimize_time += time() - minimize_begin_time
    #         rmsd_post_dock, centroid_post_dock, rmsd_after_minimized = smina_dock_result_rmsd(pdb_name, config)
    #         rmsds_post_dock.append(rmsd_post_dock)
    #         centroids_post_dock.append(centroid_post_dock)
    #
    #         rmsds_after_minimized['pdb_name'].append(pdb_name)
    #         rmsds_after_minimized['rmsd'].append(rmsd_after_minimized)
    #         print(f'{pdb_name} smina minimized, rmsd: {rmsd_post_dock}, centroid: {centroid_post_dock}')
    #
    #         text_matics = 'rmsd:{}\ncentroid_d:{}\n'.format(rmsd_post_dock, centroid_post_dock)
    #         post_dock_matric_path = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_matrics_post_{config.train.align_method}_dock.txt')
    #         with open(post_dock_matric_path, 'w') as f:
    #             f.write(text_matics)
    #
    #         # read matrics
    #         post_matric_path = os.path.join(config.train.save_path, 'visualize_dir',
    #                                              f'{pdb_name}_matrics_post_{config.train.align_method}.txt')
    #
    #         matric_path = os.path.join(config.train.save_path, 'visualize_dir',
    #                                              f'{pdb_name}_matrics.txt')
    #         rmsd_post, centroid_post = read_matric(post_matric_path)
    #         rmsds_post.append(rmsd_post)
    #         centroids_post.append(centroid_post)
    #
    #         rmsd, centroid = read_matric(matric_path)
    #         rmsds.append(rmsd)
    #         centroids.append(centroid)
    #         valid_pdb_name.append(pdb_name)
    #
    #     except:
    #         print(f'{pdb_name} dock error!')
    #         error_list.append(pdb_name)
    #
    dock_score_analysis(pdb_name_list, config)

    pd.DataFrame(rmsds_after_minimized).to_csv(os.path.join(config.train.save_path, 'visualize_dir', f'rmsd_after_smina_minimzed.csv'))

    matric_dict_post_dock = get_matric_dict(rmsds_post_dock, centroids_post_dock)
    matric_dict_post = get_matric_dict(rmsds_post, centroids_post)
    matric_dict = get_matric_dict(rmsds, centroids)

    matric_dict_post_dock_d = {'pdb_name': valid_pdb_name, 'rmsd': rmsds_post_dock, 'centroid': centroids_post_dock}
    pd.DataFrame(matric_dict_post_dock_d).to_csv(
        os.path.join(config.train.save_path, 'visualize_dir', 'matric_distribution_after_minimized.csv'))

    matric_str = ''
    for key in matric_dict_post_dock.keys():
        if key.endswith('mean') or key.endswith('std'):
            matric_str += '| post dock {}: {:.4f} '.format(key, matric_dict_post_dock[key])
        else:
            matric_str += '| post dock {}: {:.4f}% '.format(key, matric_dict_post_dock[key])

    for key in matric_dict_post.keys():
        if key.endswith('mean') or key.endswith('std'):
            matric_str += '| post {}: {:.4f} '.format(key, matric_dict_post[key])
        else:
            matric_str += '| post {}: {:.4f}% '.format(key, matric_dict_post[key])

    for key in matric_dict.keys():
        if key.endswith('mean') or key.endswith('std'):
            matric_str += '| {}: {:.4f} '.format(key, matric_dict[key])
        else:
            matric_str += '| {}: {:.4f}% '.format(key, matric_dict[key])

    print(f'smina minimize results ========================')
    print(matric_str)
    print(f'pdb name error list ==========================')
    print('\t'.join(error_list))
    print(f'smina minimize time: {minimize_time}')

    return

def get_dock_score(log_path):
    with open(log_path, 'r') as f:
        lines = f.read().strip().split('\n')

    affinity_score = float(lines[20].split()[1])

    return affinity_score

def dock_score_analysis(pdb_name_list, config):
    dock_score_d = {'name':[], 'score':[]}
    error_num = 0
    for pdb_name in tqdm(pdb_name_list):
        log_path = os.path.join(config.train.save_path, 'visualize_dir', f'{pdb_name}_post_align_{config.train.align_method}_docked.log')
        try:
            affinity_score = get_dock_score(log_path)
        except:
            affinity_score = None
        dock_score_d['name'].append(pdb_name)
        dock_score_d['score'].append(affinity_score)
    print('error num,', error_num)
    pd.DataFrame(dock_score_d).to_csv(os.path.join(config.train.save_path, 'visualize_dir', f'post_align_{config.train.align_method}_smina_minimize_score.csv'))


def structure2score(score_type):
    try:
        assert score_type in ['vina', 'smina', 'rfscore', 'ign', 'nnscore']
    except:
        raise ValueError(f'{score_type} if not supported scoring function type')



    return