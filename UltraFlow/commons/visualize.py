import os
import pandas as pd
import torch
from prody import writePDB
from rdkit import Chem as Chem
from rdkit.Chem.rdchem import BondType as BT
from openbabel import openbabel, pybel
from io import BytesIO
from .process_mols import read_molecules_crossdock, read_molecules, read_rdmol
from .geomop import canonical_protein_ligand_orientation
from collections import defaultdict
import numpy as np

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}

def simply_modify_coords(pred_coords,file_path,file_type='mol2',pos_no=None,file_label='pred'):
    with open(file_path,'r') as f:
        lines = f.read().strip().split('\n')
    index = 0
    while index < len(lines):
        if '@<TRIPOS>ATOM' in lines[index]:
            break
        index += 1
    for coord in pred_coords:
        index += 1
        new_x = '{:.4f}'.format(coord[0]).rjust(10, ' ')
        new_y = '{:.4f}'.format(coord[1]).rjust(10, ' ')
        new_z = '{:.4f}'.format(coord[2]).rjust(10, ' ')
        new_coord_str = new_x + new_y + new_z
        lines[index] = lines[index][:16] + new_coord_str + lines[index][46:]

    if pos_no is not None:
        with open('{}_{}_{}.{}'.format(os.path.join(os.path.dirname(file_path),os.path.basename(file_path).split('.')[0]), file_label, pos_no, file_type),'w') as f:
            f.write('\n'.join(lines))
    else:
        with open('{}_{}.{}'.format(os.path.join(os.path.dirname(file_path),os.path.basename(file_path).split('.')[0]), file_label, file_type),'w') as f:
            f.write('\n'.join(lines))

def set_new_coords_for_protein_atom(m_prot, new_coords):
    for index,atom in enumerate(m_prot):
        atom.setCoords(new_coords[index])
    return m_prot

def save_ligand_file(m_lig, output_path, file_type='mol2'):

    return

def save_protein_file(m_prot, output_path, file_type='pdb'):
    if file_type=='pdb':
        writePDB(output_path, m_prot)
    return

def generated_to_xyz(data):
    ptable = Chem.GetPeriodicTable()
    num_atoms, atom_type, atom_coords = data
    xyz = "%d\n\n" % (num_atoms, )
    for i in range(num_atoms):
        symb = ptable.GetElementSymbol(int(atom_type[i]))
        x, y, z = atom_coords[i].clone().cpu().tolist()
        xyz += "%s %.8f %.8f %.8f\n" % (symb, x, y, z)
    return xyz

def generated_to_sdf(data):
    xyz = generated_to_xyz(data)
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")

    mol = openbabel.OBMol()
    obConversion.ReadString(mol, xyz)
    sdf = obConversion.WriteString(mol)
    return sdf

def sdf_to_rdmol(sdf):
    stream = BytesIO(sdf.encode())
    suppl = Chem.ForwardSDMolSupplier(stream)
    for mol in suppl:
        return mol
    return None

def generated_to_rdmol(data):
    sdf = generated_to_sdf(data)
    return sdf_to_rdmol(sdf)

def generated_to_rdmol_trajectory(trajectory):
    sdf_trajectory = ''
    for data in trajectory:
        sdf_trajectory += generated_to_sdf(data)
    return sdf_trajectory

def filter_rd_mol(rdmol):
    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]

    # 3-3 ring intersection
    for i, ring_a in enumerate(rings):
        if len(ring_a) != 3:continue
        for j, ring_b in enumerate(rings):
            if i <= j: continue
            inter = ring_a.intersection(ring_b)
            if (len(ring_b) == 3) and (len(inter) > 0):
                return False
    return True


def save_sdf_mol(rdmol, save_path, suffix='test'):
    writer = Chem.SDWriter(os.path.join(save_path, 'visualize_dir', f'{suffix}.sdf'))
    writer.SetKekulize(False)
    try:
        writer.write(rdmol, confId=0)
    except:
        writer.close()
        return False
    writer.close()
    return True

def sdf_string_save_sdf_file(sdf_string, save_path, suffix='test'):
    with open(os.path.join(save_path, 'visualize_dir', f'{suffix}.sdf'), 'w') as f:
        f.write(sdf_string)
    return

def visualize_generate_full_trajectory(trajectory, index, dataset, save_path, move_truth=True,
                                       name_suffix='pred_trajectory', canonical_oritentaion=True):
    if dataset.dataset_name in ['crossdock2020', 'crossdock2020_test']:
        lig_path = index
        lig_path_split = lig_path.split('/')
        lig_dir, lig_base = lig_path_split[0], lig_path_split[1]
        prot_path = os.path.join(lig_dir, lig_base[:10] + '.pdb')

        if not os.path.exists(os.path.join(save_path, 'visualize_dir', lig_dir)):
            os.makedirs(os.path.join(save_path, 'visualize_dir', lig_dir))

        name = index[:-4]

        assert prot_path.endswith('_rec.pdb')
        molecular_representation = read_molecules_crossdock(lig_path, prot_path, dataset.ligcut, dataset.protcut,
                                                            dataset.lig_type, dataset.prot_graph_type,
                                                            dataset.dataset_path, dataset.chaincut)

        lig_path_direct = os.path.join(dataset.dataset_path, lig_path)
        prot_path_direct = os.path.join(dataset.dataset_path, prot_path)


    elif dataset.dataset_name in ['pdbbind2020', 'pdbbind2016']:
        name = index
        molecular_representation = read_molecules(index, dataset.dataset_path, dataset.prot_graph_type,
                                                  dataset.ligcut, dataset.protcut, dataset.lig_type,
                                                  init_type=None, chain_cut=dataset.chaincut)

        lig_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_ligand.mol2')
        if os.path.exists(os.path.join(dataset.dataset_path, name, f'{name}_protein_processed.pdb')):
            prot_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_protein_processed.pdb')
        else:
            prot_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_protein.pdb')

    lig_coords, _, _, lig_node_type, _, prot_coords, _, _, _, _, _, _, _, _, _ = molecular_representation

    if dataset.canonical_oritentaion and canonical_oritentaion:
        new_trajectory = []
        _, _, _, _, _, rotation, translation = canonical_protein_ligand_orientation(lig_coords, prot_coords)
        for coords in trajectory:
            new_trajectory.append((coords @ rotation.T) - translation)
        trajectory = new_trajectory

    trajectory_data = []
    num_atoms = len(coords)
    for coords in trajectory:
        data = (num_atoms, lig_node_type, coords)
        trajectory_data.append(data)
    sdf_file_string = generated_to_rdmol_trajectory(trajectory_data)

    if name_suffix is None:
        sdf_string_save_sdf_file(sdf_file_string, save_path, suffix=name)
    else:
        sdf_string_save_sdf_file(sdf_file_string, save_path, suffix=f'{name}_{name_suffix}')

    if move_truth:
        output_path = os.path.join(save_path, 'visualize_dir')
        cmd = f'cp {prot_path_direct} {output_path}'
        cmd += f'&& cp {lig_path_direct} {output_path}'
        os.system(cmd)

    return

def visualize_generated_coordinates(coords, index, dataset, save_path, move_truth=True, name_suffix=None, canonical_oritentaion=True):

    if dataset.dataset_name in ['crossdock2020', 'crossdock2020_test']:
        lig_path = index
        lig_path_split = lig_path.split('/')
        lig_dir, lig_base = lig_path_split[0], lig_path_split[1]
        prot_path = os.path.join(lig_dir, lig_base[:10]+'.pdb')

        if not os.path.exists(os.path.join(save_path, 'visualize_dir', lig_dir)):
            os.makedirs(os.path.join(save_path, 'visualize_dir', lig_dir))

        name = index[:-4]

        assert prot_path.endswith('_rec.pdb')
        molecular_representation = read_molecules_crossdock(lig_path, prot_path, dataset.ligcut, dataset.protcut,
                                                            dataset.lig_type, dataset.prot_graph_type, dataset.dataset_path, dataset.chaincut)

        lig_path_direct = os.path.join(dataset.dataset_path, lig_path)
        prot_path_direct = os.path.join(dataset.dataset_path, prot_path)


    elif dataset.dataset_name in ['pdbbind2020','pdbbind2016']:
        name = index
        molecular_representation = read_molecules(index, dataset.dataset_path, dataset.prot_graph_type,
                                                  dataset.ligcut, dataset.protcut, dataset.lig_type,
                                                  init_type=None, chain_cut=dataset.chaincut)

        lig_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_ligand.mol2')
        if os.path.exists(os.path.join(dataset.dataset_path, name, f'{name}_protein_processed.pdb')):
            prot_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_protein_processed.pdb')
        else:
            prot_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_protein.pdb')

    lig_coords, _, _, lig_node_type, _, prot_coords, _, _, _, _, _, _, _, _, _ = molecular_representation

    if dataset.canonical_oritentaion and canonical_oritentaion:
        _, _ , _, _, _, rotation, translation = canonical_protein_ligand_orientation(lig_coords, prot_coords)
        coords = (coords @ rotation.T) - translation

    num_atoms = len(coords)

    data = (num_atoms, lig_node_type, coords)
    sdf_string = generated_to_sdf(data)

    sdf_path = os.path.join(save_path, 'visualize_dir', f'{name}_{name_suffix}.sdf')
    with open(sdf_path, 'w') as f:
        f.write(sdf_string)

    if move_truth:
        lig_path_direct_sdf = os.path.join(dataset.dataset_path, name, f'{name}_ligand.sdf')
        output_path = os.path.join(save_path, 'visualize_dir')
        cmd = f'cp {prot_path_direct} {output_path}'
        cmd += f' && cp {lig_path_direct} {output_path}'
        cmd += f' && cp {lig_path_direct_sdf} {output_path}'
        os.system(cmd)

def visualize_predicted_pocket(binding_site_flag, index, dataset, save_path, move_truth=True, name_suffix=None, canonical_oritentaion=True):
    if not os.path.exists(os.path.join(save_path, 'visualize_dir')):
        os.makedirs(os.path.join(save_path, 'visualize_dir'))

    if dataset.dataset_name in ['crossdock2020', 'crossdock2020_test']:
        lig_path = index
        lig_path_split = lig_path.split('/')
        lig_dir, lig_base = lig_path_split[0], lig_path_split[1]
        prot_path = os.path.join(lig_dir, lig_base[:10]+'.pdb')

        if not os.path.exists(os.path.join(save_path, 'visualize_dir', lig_dir)):
            os.makedirs(os.path.join(save_path, 'visualize_dir', lig_dir))

        name = index[:-4]

        assert prot_path.endswith('_rec.pdb')
        molecular_representation = read_molecules_crossdock(lig_path, prot_path, dataset.ligcut, dataset.protcut,
                                                            dataset.lig_type, dataset.prot_graph_type, dataset.dataset_path, dataset.chaincut)

        lig_path_direct = os.path.join(dataset.dataset_path, lig_path)
        prot_path_direct = os.path.join(dataset.dataset_path, prot_path)

    elif dataset.dataset_name in ['pdbbind2020','pdbbind2016']:
        name = index
        molecular_representation = read_molecules(index, dataset.dataset_path, dataset.prot_graph_type,
                                                  dataset.ligcut, dataset.protcut, dataset.lig_type,
                                                  init_type=None, chain_cut=dataset.chaincut)

        lig_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_ligand.mol2')
        if os.path.exists(os.path.join(dataset.dataset_path, name, f'{name}_protein_processed.pdb')):
            prot_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_protein_processed.pdb')
        else:
            prot_path_direct = os.path.join(dataset.dataset_path, name, f'{name}_protein.pdb')

    lig_coords, _, _, lig_node_type, _, prot_coords, _, _, _, _, _, _, _, _, _ = molecular_representation

    coords = torch.from_numpy(prot_coords[binding_site_flag.cpu()])

    num_atoms = len(coords)

    data = (num_atoms, [6] * num_atoms, coords)
    sdf_string = generated_to_sdf(data)

    sdf_path = os.path.join(save_path, 'visualize_dir', f'{name}_{name_suffix}.sdf')
    with open(sdf_path, 'w') as f:
        f.write(sdf_string)

    if move_truth:
        output_path = os.path.join(save_path, 'visualize_dir')
        cmd = f'cp {prot_path_direct} {output_path}'
        cmd += f'&& cp {lig_path_direct} {output_path}'
        os.system(cmd)

def visualize_predicted_link_map(pred_prob, true_prob, pdb_name, dataset, save_path):
    """
    :param pred_prob: [N,M], torch.tensor
    :param true_prob: [N,M], torch.tensor
    :param pdb_name: string
    :param dataset:
    :param save_path:
    :return:
    """
    if not os.path.exists(os.path.join(save_path, 'visualize_dir')):
        os.makedirs(os.path.join(save_path, 'visualize_dir'))

    pd.DataFrame(pred_prob.tolist()).to_csv(os.path.join(save_path, 'visualize_dir', f'{pdb_name}_link_map_pred.csv'))
    pd.DataFrame(true_prob.tolist()).to_csv(os.path.join(save_path, 'visualize_dir', f'{pdb_name}_link_map_true.csv'))

def visualize_edge_coef_map(feats_coef, coords_coef, pdb_name, dataset, save_path, layer_index):
    if not os.path.exists(os.path.join(save_path, 'visualize_dir')):
        os.makedirs(os.path.join(save_path, 'visualize_dir'))

    pd.DataFrame(feats_coef.tolist()).to_csv(os.path.join(save_path, 'visualize_dir', f'{pdb_name}_feats_coef_layer_{layer_index}.csv'))
    pd.DataFrame(coords_coef.tolist()).to_csv(os.path.join(save_path, 'visualize_dir', f'{pdb_name}_coords_coef_layer_{layer_index}.csv'))

def collect_bond_dists(index, dataset, save_path, name_suffix='pred'):
    """
    Collect the lengths for each type of chemical bond in given valid molecular geometries.
    Args:
        mol_dicts (dict): A python dict where the key is the number of atoms, and the value indexed by that key is another python dict storing the atomic
            number matrix (indexed by the key '_atomic_numbers') and the coordinate tensor (indexed by the key '_positions') of all generated molecular geometries with that atom number.
        valid_list (list): the list of bool values indicating whether each molecular geometry is chemically valid. Note that only the bond lengths of
            valid molecular geometries will be collected.
        con_mat_list (list): the list of bond order matrices.

    :rtype: :class:`dict` a python dict where the key is the bond type, and the value indexed by that key is the list of all bond lengths of that bond.
    """
    name = index
    bonds_dist = []

    lig_path_mol2 = os.path.join(dataset.dataset_path, name, f'{name}_ligand.mol2')
    lig_path_sdf = os.path.join(dataset.dataset_path, name, f'{name}_ligand.sdf')
    rdmol = read_rdmol(lig_path_sdf, sanitize=True, remove_hs=True)
    if rdmol == None:  # read mol2 file if sdf file cannot be sanitized
        rdmol = read_rdmol(lig_path_mol2, sanitize=True, remove_hs=True)
    gd_atom_coords = rdmol.GetConformer().GetPositions()

    pred_sdf_path = os.path.join(save_path, 'visualize_dir', f'{name}_{name_suffix}.sdf')
    pred_m_lig = next(pybel.readfile('sdf', pred_sdf_path))
    pred_atom_coords = np.array([atom.coords for atom in pred_m_lig], dtype=np.float32)
    assert len(pred_atom_coords) == len(gd_atom_coords)

    init_sdf_path = os.path.join(save_path, 'visualize_dir', f'{name}_init.sdf')
    inti_m_lig = next(pybel.readfile('sdf', init_sdf_path))
    init_atom_coords = np.array([atom.coords for atom in inti_m_lig], dtype=np.float32)
    assert len(init_atom_coords) == len(gd_atom_coords)

    for bond in rdmol.GetBonds():
        start_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        start_idx, end_idx = start_atom.GetIdx(), end_atom.GetIdx()
        if start_idx < end_idx:
            continue
        start_atom_type, end_atom_type = start_atom.GetAtomicNum(), end_atom.GetAtomicNum()
        bond_type = BOND_TYPES[bond.GetBondType()]

        gd_bond_dist = np.linalg.norm(gd_atom_coords[start_idx] - gd_atom_coords[end_idx])
        pred_bond_dist = np.linalg.norm(pred_atom_coords[start_idx] - pred_atom_coords[end_idx])
        init_bond_dist = np.linalg.norm(init_atom_coords[start_idx] - init_atom_coords[end_idx])

        z1, z2 = min(start_atom_type, end_atom_type), max(start_atom_type, end_atom_type)
        bonds_dist.append((z1, z2, bond_type, gd_bond_dist, pred_bond_dist, init_bond_dist))

    return bonds_dist
