import torch
import rdkit.Chem as Chem
import numpy as np
import copy
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D
from scipy.optimize import differential_evolution
from .process_mols import read_rdmol
import os
import math
from openbabel import pybel
from tqdm import tqdm

def get_d_from_pos(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1) # (num_edge)

def kabsch(coords_A, coords_B, debug=True, device=None):
    # rotate and translate coords_A to coords_B pos
    coords_A_mean = coords_A.mean(dim=0, keepdim=True)  # (1,3)
    coords_B_mean = coords_B.mean(dim=0, keepdim=True)  # (1,3)

    # A = (coords_A - coords_A_mean).transpose(0, 1) @ (coords_B - coords_B_mean)
    A = (coords_A).transpose(0, 1) @ (coords_B )
    if torch.isnan(A).any():
        print('A Nan encountered')
    assert not torch.isnan(A).any()

    if torch.isinf(A).any():
        print('inf encountered')
    assert not torch.isinf(A).any()

    U, S, Vt = torch.linalg.svd(A)
    num_it = 0
    while torch.min(S) < 1e-3 or torch.min(
            torch.abs((S ** 2).view(1, 3) - (S ** 2).view(3, 1) + torch.eye(3).to(device))) < 1e-2:
        if debug: print('S inside loop ', num_it, ' is ', S, ' and A = ', A)
        A = A + torch.rand(3, 3).to(device) * torch.eye(3).to(device)
        U, S, Vt = torch.linalg.svd(A)
        num_it += 1
        if num_it > 10: raise Exception('SVD was consitantly unstable')

    corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=device))
    rotation = (U @ corr_mat) @ Vt

    translation = coords_B_mean - torch.t(rotation @ coords_A_mean.t())  # (1,3)

    # new_coords = (rotation @ coords_A.t()).t() + translation

    return rotation, translation

def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

def align_molecule_a_according_molecule_b(molecule_a_path, molecule_b_path, device=None, save=False, kabsch_no_h=True):
    m_a = Chem.MolFromMol2File(molecule_a_path, sanitize=False, removeHs=False)
    m_b = Chem.MolFromMol2File(molecule_b_path, sanitize=False, removeHs=False)
    pos_a = torch.tensor(m_a.GetConformer().GetPositions())
    pos_b = torch.tensor(m_b.GetConformer().GetPositions())
    m_a_no_h = Chem.RemoveHs(m_a)
    m_b_no_h = Chem.RemoveHs(m_b)
    pos_a_no_h = torch.tensor(m_a_no_h.GetConformer().GetPositions())
    pos_b_no_h = torch.tensor(m_b_no_h.GetConformer().GetPositions())

    if kabsch_no_h:
        rotation, translation = kabsch(pos_a_no_h, pos_b_no_h, device=device)
    else:
        rotation, translation = kabsch(pos_a, pos_b, device=device)
    pos_a_new = (rotation @ pos_a.t()).t() + translation
    # print(np.sqrt(np.sum((pos_a.numpy() - pos_b.numpy()) ** 2,axis=1).mean()))
    # print(np.sqrt(np.sum((pos_a_new.numpy() - pos_b.numpy()) ** 2, axis=1).mean()))

    return pos_a_new, rotation, translation

def get_principle_axes(xyz,scale_factor=20,pdb_name=None):
    #create coordinates array
    coord = np.array(xyz, float)
    # compute geometric center
    center = np.mean(coord, 0)
    # print("Coordinates of the geometric center:\n", center)
    # center with geometric center
    coord = coord - center
    # compute principal axis matrix
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)
    #--------------------------------------------------------------------------
    # order eigen values (and eigen vectors)
    #
    # axis1 is the principal axis with the biggest eigen value (eval1)
    # axis2 is the principal axis with the second biggest eigen value (eval2)
    # axis3 is the principal axis with the smallest eigen value (eval3)
    #--------------------------------------------------------------------------
    order = np.argsort(e_values)
    eval3, eval2, eval1 = e_values[order]
    axis3, axis2, axis1 = e_vectors[:, order].transpose()

    return np.array([axis1, axis2, axis3]), center

def get_rotation_and_translation(xyz):
    protein_principle_axes_system, system_center = get_principle_axes(xyz)
    rotation = protein_principle_axes_system.T
    translation = -system_center
    return rotation, translation

def canonical_protein_ligand_orientation(lig_coords, prot_coords):
    rotation, translation = get_rotation_and_translation(prot_coords)
    lig_canoical_truth_coords = (lig_coords + translation) @ rotation
    prot_canonical_truth_coords = (prot_coords + translation) @ rotation
    rotation_lig, translation_lig = get_rotation_and_translation(lig_coords)
    lig_canonical_init_coords = (lig_coords + translation_lig) @ rotation_lig

    return lig_coords, lig_canoical_truth_coords, lig_canonical_init_coords, \
           prot_coords, prot_canonical_truth_coords,\
           rotation, translation

def canonical_single_molecule_orientation(m_coords):
    rotation, translation = get_rotation_and_translation(m_coords)
    canonical_init_coords = (m_coords + translation) @ rotation
    return canonical_init_coords

# Clockwise dihedral2 from https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def GetDihedralFromPointCloud(Z, atom_idx):
    p = Z[list(atom_idx)]
    b = p[:-1] - p[1:]
    b[0] *= -1 #########################
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2( y, x ))

def A_transpose_matrix(alpha):
    return np.array([[np.cos(np.radians(alpha)), np.sin(np.radians(alpha))],
                     [-np.sin(np.radians(alpha)), np.cos(np.radians(alpha))]], dtype=np.double)

def S_vec(alpha):
    return np.array([[np.cos(np.radians(alpha))],
                     [np.sin(np.radians(alpha))]], dtype=np.double)

def get_dihedral_vonMises(mol, conf, atom_idx, Z):
    Z = np.array(Z)
    v = np.zeros((2,1))
    iAtom = mol.GetAtomWithIdx(atom_idx[1])
    jAtom = mol.GetAtomWithIdx(atom_idx[2])
    k_0 = atom_idx[0]
    i = atom_idx[1]
    j = atom_idx[2]
    l_0 = atom_idx[3]
    for b1 in iAtom.GetBonds():
        k = b1.GetOtherAtomIdx(i)
        if k == j:
            continue
        for b2 in jAtom.GetBonds():
            l = b2.GetOtherAtomIdx(j)
            if l == i:
                continue
            assert k != l
            s_star = S_vec(GetDihedralFromPointCloud(Z, (k, i, j, l)))
            a_mat = A_transpose_matrix(GetDihedral(conf, (k, i, j, k_0)) + GetDihedral(conf, (l_0, i, j, l)))
            v = v + np.matmul(a_mat, s_star)
    v = v / np.linalg.norm(v)
    v = v.reshape(-1)
    return np.degrees(np.arctan2(v[1], v[0]))

def distance_loss_function(epoch, y_pred, x, protein_nodes_xyz, compound_pair_dis_constraint, LAS_distance_constraint_mask=None, mode=0):
    dis = torch.cdist(x, protein_nodes_xyz)
    dis_clamp = torch.clamp(dis, max=10)
    if mode == 0:
        interaction_loss = ((dis_clamp - y_pred).abs()).sum()
    elif mode == 1:
        interaction_loss = ((dis_clamp - y_pred)**2).sum()
    elif mode == 2:
        # probably not a good choice. x^0.5 has infinite gradient at x=0. added 1e-5 for numerical stability.
        interaction_loss = (((dis_clamp - y_pred).abs() + 1e-5)**0.5).sum()
    config_dis = torch.cdist(x, x)
    if LAS_distance_constraint_mask is not None:
        configuration_loss = 1 * (((config_dis-compound_pair_dis_constraint).abs())[LAS_distance_constraint_mask]).sum()
        # basic exlcuded-volume. the distance between compound atoms should be at least 1.22Å
        configuration_loss += 2 * ((1.22 - config_dis).relu()).sum()
    else:
        configuration_loss = 1 * ((config_dis-compound_pair_dis_constraint).abs()).sum()
    # if epoch < 500:
    #     loss = interaction_loss
    # else:
    #     loss = 1 * (interaction_loss + 5e-3 * (epoch - 500) * configuration_loss)
    loss = 1 * (interaction_loss + 5e-3 * (epoch + 200) * configuration_loss)
    return loss, (interaction_loss.item(), configuration_loss.item())


def distance_optimize_compound_coords(coords, y_pred, protein_nodes_xyz,
                        compound_pair_dis_constraint,total_epoch=1000, loss_function=distance_loss_function, LAS_distance_constraint_mask=None, mode=0, show_progress=False):
    # random initialization. center at the protein center.
    c_pred = protein_nodes_xyz.mean(axis=0)
    x = coords
    x.requires_grad = True
    optimizer = torch.optim.Adam([x], lr=0.1)
    loss_list = []
    #     optimizer = torch.optim.LBFGS([x], lr=0.01)
    if show_progress:
        it = tqdm(range(total_epoch))
    else:
        it = range(total_epoch)
    for epoch in it:
        optimizer.zero_grad()
        loss, (interaction_loss, configuration_loss) = loss_function(epoch, y_pred, x, protein_nodes_xyz,
                                                                     compound_pair_dis_constraint,
                                                                     LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                                                     mode=mode)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        # break
    return x, loss_list

def tankbind_gen(lig_pred_coords, lig_init_coords, prot_coords, LAS_mask, device='cpu', mode=0):

    pred_prot_lig_inter_distance = torch.cdist(lig_pred_coords, prot_coords)
    init_lig_intra_distance = torch.cdist(lig_init_coords, lig_init_coords)
    try:
        x, loss_list = distance_optimize_compound_coords(lig_pred_coords.to('cpu'),
                                                         pred_prot_lig_inter_distance.to('cpu'),
                                                         prot_coords.to('cpu'),
                                                         init_lig_intra_distance.to('cpu'),
                                                         LAS_distance_constraint_mask=LAS_mask.bool(),
                                                         mode=mode, show_progress=False)
    except:
        print('error')

    return x

def kabsch_align(lig_pred_coords, name, save_path, dataset_path, device='cpu'):
    rdkit_init_lig_path_sdf = os.path.join(save_path, 'visualize_dir', f'{name}_init.sdf')
    openbabel_init_m_lig = next(pybel.readfile('sdf', rdkit_init_lig_path_sdf))
    rdkit_init_coords = [atom.coords for atom in openbabel_init_m_lig]
    rdkit_init_coords = np.array(rdkit_init_coords, dtype=np.float32)  # np.array, [n, 3]

    coords_pred = lig_pred_coords.detach().cpu().numpy()

    R, t = rigid_transform_Kabsch_3D(rdkit_init_coords.T, coords_pred.T)
    coords_pred_optimized = (R @ (rdkit_init_coords).T).T + t.squeeze()

    opt_ligCoords = torch.tensor(coords_pred_optimized, device=device)
    return opt_ligCoords

def equibind_align(lig_pred_coords, name, save_path, dataset_path, device='cpu'):
    lig_path_mol2 = os.path.join(dataset_path, name, f'{name}_ligand.mol2')
    lig_path_sdf = os.path.join(dataset_path, name, f'{name}_ligand.sdf')
    m_lig = read_rdmol(lig_path_sdf, sanitize=True, remove_hs=True)
    if m_lig == None:  # read mol2 file if sdf file cannot be sanitized
        m_lig = read_rdmol(lig_path_mol2, sanitize=True, remove_hs=True)

    # load rdkit mol
    lig_path_sdf_error = os.path.join(save_path, 'visualize_dir', f'{name}_init')
    pred_lig_path_sdf_error = os.path.join(save_path, 'visualize_dir', f'{name}_pred')
    pred_lig_path_sdf_true = os.path.join(save_path, 'visualize_dir', f'{name}_pred.sdf')

    rdkit_init_lig_path_sdf = os.path.join(save_path, 'visualize_dir', f'{name}_init.sdf')

    if not os.path.exists(rdkit_init_lig_path_sdf):
        cmd = f'mv {lig_path_sdf_error} {rdkit_init_lig_path_sdf}'
        os.system(cmd)
    if not os.path.exists(pred_lig_path_sdf_true):
        cmd = f'mv {pred_lig_path_sdf_error} {pred_lig_path_sdf_true}'
        os.system(cmd)

    openbabel_init_m_lig = next(pybel.readfile('sdf', rdkit_init_lig_path_sdf))
    rdkit_init_coords = [atom.coords for atom in openbabel_init_m_lig]
    rdkit_init_coords = np.array(rdkit_init_coords, dtype=np.float32)  # np.array, [n, 3]
    # rdkit_init_m_lig = read_rdmol(rdkit_init_lig_path_sdf,  sanitize=True, remove_hs=True)
    # rdkit_init_coords = rdkit_init_m_lig.GetConformer().GetPositions()

    rdkit_init_lig = copy.deepcopy(m_lig)
    conf = rdkit_init_lig.GetConformer()
    for i in range(rdkit_init_lig.GetNumAtoms()):
        x, y, z = rdkit_init_coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    coords_pred = lig_pred_coords.detach().cpu().numpy()
    Z_pt_cloud = coords_pred
    rotable_bonds = get_torsions([rdkit_init_lig])
    new_dihedrals = np.zeros(len(rotable_bonds))

    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(rdkit_init_lig, rdkit_init_lig.GetConformer(), r, Z_pt_cloud)
    optimized_mol = apply_changes_equibind(rdkit_init_lig, new_dihedrals, rotable_bonds)

    coords_pred_optimized = optimized_mol.GetConformer().GetPositions()
    R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_pred.T)
    coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()

    opt_ligCoords = torch.tensor(coords_pred_optimized, device=device)
    return opt_ligCoords

def dock_compound(lig_pred_coords, prot_coords, name, save_path,
                  popsize=150, maxiter=500, seed=None, mutation=(0.5, 1),
                  recombination=0.8, device='cpu', torsion_num_cut=20):
    if seed:
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

    # load rdkit mol
    lig_path_init_sdf = os.path.join(save_path, 'visualize_dir', f'{name}_init.sdf')
    openbabel_m_lig_init = next(pybel.readfile('sdf', lig_path_init_sdf))
    rdkit_init_coords = [atom.coords for atom in openbabel_m_lig_init]

    lig_path_true_sdf = os.path.join(save_path, 'visualize_dir', f'{name}_ligand.sdf')
    lig_path_true_mol2 = os.path.join(save_path, 'visualize_dir', f'{name}_ligand.mol2')
    m_lig = read_rdmol(lig_path_true_sdf, sanitize=True, remove_hs=True)
    if m_lig == None:  # read mol2 file if sdf file cannot be sanitized
        m_lig = read_rdmol(lig_path_true_mol2, sanitize=True, remove_hs=True)

    atom_num = len(m_lig.GetConformer().GetPositions())
    if len(rdkit_init_coords) != atom_num:
        rdkit_init_coords = [atom.coords for atom in openbabel_m_lig_init if atom.atomicnum > 1]
        lig_pred_coords_no_h_list = [atom_coords for atom,atom_coords in zip(openbabel_m_lig_init, lig_pred_coords.tolist()) if atom.atomicnum > 1]
        lig_pred_coords = torch.tensor(lig_pred_coords_no_h_list, device=device)

    rdkit_init_coords = np.array(rdkit_init_coords, dtype=np.float32)  # np.array, [n, 3]
    print(f'{name} init coords shape: {rdkit_init_coords.shape}')
    print(f'{name} true coords shape: {m_lig.GetConformer().GetPositions().shape}')

    rdkit_init_lig = copy.deepcopy(m_lig)
    conf = rdkit_init_lig.GetConformer()
    for i in range(rdkit_init_lig.GetNumAtoms()):
        x, y, z = rdkit_init_coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    # move m_lig to pred_coords center
    pred_coords_center = lig_pred_coords.cpu().numpy().mean(axis=0)
    init_coords_center = rdkit_init_lig.GetConformer().GetPositions().mean(axis=0)
    # print(f'{name} pred coords shape: {lig_pred_coords.shape}')

    center_rel_vecs = pred_coords_center - init_coords_center
    values = np.concatenate([np.array([0,0,0]),center_rel_vecs])
    rdMolTransforms.TransformConformer(rdkit_init_lig.GetConformer(), GetTransformationMatrix(values))

    # Set optimization function
    opt = optimze_conformation(mol=rdkit_init_lig, target_coords=lig_pred_coords, device=device,
                               n_particles=1, seed=seed)
    if len(opt.rotable_bonds) > torsion_num_cut:
        return lig_pred_coords

    # Define bounds for optimization
    max_bound = np.concatenate([[np.pi] * 3, prot_coords.cpu().max(0)[0].numpy(), [np.pi] * len(opt.rotable_bonds)], axis=0)
    min_bound = np.concatenate([[-np.pi] * 3, prot_coords.cpu().min(0)[0].numpy(), [-np.pi] * len(opt.rotable_bonds)], axis=0)
    bounds = (min_bound, max_bound)

    # Optimize conformations
    result = differential_evolution(opt.score_conformation, list(zip(bounds[0], bounds[1])), maxiter=maxiter,
                                    popsize=int(np.ceil(popsize / (len(opt.rotable_bonds) + 6))),
                                    mutation=mutation, recombination=recombination, disp=False, seed=seed)

    # Get optimized molecule
    starting_mol = opt.mol
    opt_mol = apply_changes(starting_mol, result['x'], opt.rotable_bonds)
    opt_ligCoords = torch.tensor(opt_mol.GetConformer().GetPositions(), device=device)

    return opt_ligCoords

class optimze_conformation():
    def __init__(self, mol, target_coords, n_particles, save_molecules=False, device='cpu',
                 seed=None):
        super(optimze_conformation, self).__init__()
        if seed:
            np.random.seed(seed)

        self.targetCoords = torch.stack([target_coords for _ in range(n_particles)]).double()
        self.n_particles = n_particles
        self.rotable_bonds = get_torsions([mol])
        self.save_molecules = save_molecules
        self.mol = mol
        self.device = device

    def score_conformation(self, values):
        """
        Parameters
        ----------
        values : numpy.ndarray
            set of inputs of shape :code:`(n_particles, dimensions)`
        Returns
        -------
        numpy.ndarray
            computed cost of size :code:`(n_particles, )`
        """
        if len(values.shape) < 2: values = np.expand_dims(values, axis=0)
        mols = [copy.copy(self.mol) for _ in range(self.n_particles)]

        # Apply changes to molecules
        # apply rotations
        [SetDihedral(mols[m].GetConformer(), self.rotable_bonds[r], values[m, 6 + r]) for r in
         range(len(self.rotable_bonds)) for m in range(self.n_particles)]

        # apply transformation matrix
        [rdMolTransforms.TransformConformer(mols[m].GetConformer(), GetTransformationMatrix(values[m, :6])) for m in
         range(self.n_particles)]

        # Calcualte distances between ligand conformation and pred ligand conformation
        ligCoords_list = [torch.tensor(m.GetConformer().GetPositions(), device=self.device) for m in mols]  # [n_mols, N, 3]
        ligCoords = torch.stack(ligCoords_list).double() # [n_mols, N, 3]

        ligCoords_error = ligCoords - self.targetCoords  # [n_mols, N, 3]
        ligCoords_rmsd = (ligCoords_error ** 2).sum(dim=-1).mean(dim=-1).sqrt().min().cpu().numpy()

        del ligCoords_error, ligCoords, ligCoords_list, mols

        return ligCoords_rmsd

def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.copy(mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[6 + r]) for r in range(len(rotable_bonds))]

    # apply transformation matrix
    rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol

def apply_changes_equibind(mol, values, rotable_bonds):
    opt_mol = copy.deepcopy(mol)
    #     opt_mol = add_rdkit_conformer(opt_mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]

    #     # apply transformation matrix
    #     rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol

def get_torsions(mol_list):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                            or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                        continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def GetTransformationMatrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    transMat = np.array([[np.cos(z) * np.cos(y), (np.cos(z) * np.sin(y) * np.sin(x)) - (np.sin(z) * np.cos(x)),
                          (np.cos(z) * np.sin(y) * np.cos(x)) + (np.sin(z) * np.sin(x)), disp_x],
                         [np.sin(z) * np.cos(y), (np.sin(z) * np.sin(y) * np.sin(x)) + (np.cos(z) * np.cos(x)),
                          (np.sin(z) * np.sin(y) * np.cos(x)) - (np.cos(z) * np.sin(x)), disp_y],
                         [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x), disp_z],
                         [0, 0, 0, 1]], dtype=np.double)
    return transMat

