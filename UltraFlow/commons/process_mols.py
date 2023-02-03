import os
import math
from openbabel import pybel
from openbabel import openbabel
import dgl
import pickle
import numpy as np
import torch
import scipy.spatial as spatial
from functools import partial
from prody import *
from rdkit import Chem as Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from Bio.PDB import get_surface, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy.special import softmax
from scipy.spatial.transform import Rotation
import pandas as pd

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}

graph_type_filename = {'atom_pocket':'valid_pocket.pdb',
                       'atom_complete':'valid_chains.pdb'}
ResDict = {'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,
           'GLN':5,'GLU':6,'GLY':7,'HIS':8,'ILE':9,
          'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,
          'SER':15,'THR':16,'TRP':17,'TYR':18,'VAL':19}
SSEDict = {'H':0,'B':1,'E':2,'G':3,'I':4,'T':5,'S':6,' ':7}
SSEType,UNKOWN_RES = 8,20

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 1)  # number of scalar features
rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 2)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 2)

dbcg_prot_residue_feature_dims = [[21],0]

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def lig_atom_featurizer_rdmol(mol):
    ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])

    return torch.tensor(atom_features_list)

def CusBondFeaturizer(bond):
    return [int(bond.GetBondOrder()) ,int(bond.IsAromatic()), int(bond.IsInRing())]

class Featurizer():
    """Calcaulates atomic features for molecules. Features can encode atom type,
    native pybel properties or any property defined with SMARTS patterns

    Attributes
    ----------
    FEATURE_NAMES: list of strings
        Labels for features (in the same order as features)
    NUM_ATOM_CLASSES: int
        Number of atom codes
    ATOM_CODES: dict
        Dictionary mapping atomic numbers to codes
    NAMED_PROPS: list of string
        Names of atomic properties to retrieve from pybel.Atom object
    CALLABLES: list of callables
        Callables used to calculcate custom atomic properties
    SMARTS: list of SMARTS strings
        SMARTS patterns defining additional atomic properties
    """

    def __init__(self, atom_codes=None, atom_labels=None,
                 named_properties=None, save_molecule_codes=True,
                 custom_properties=None, smarts_properties=None,
                 smarts_labels=None):

        """Creates Featurizer with specified types of features. Elements of a
        feature vector will be in a following order: atom type encoding
        (defined by atom_codes), Pybel atomic properties (defined by
        named_properties), molecule code (if present), custom atomic properties
        (defined `custom_properties`), and additional properties defined with
        SMARTS (defined with `smarts_properties`).

        Parameters
        ----------
        atom_codes: dict, optional
            Dictionary mapping atomic numbers to codes. It will be used for
            one-hot encoging therefore if n different types are used, codes
            shpuld be from 0 to n-1. Multiple atoms can have the same code,
            e.g. you can use {6: 0, 7: 1, 8: 1} to encode carbons with [1, 0]
            and nitrogens and oxygens with [0, 1] vectors. If not provided,
            default encoding is used.
        atom_labels: list of strings, optional
            Labels for atoms codes. It should have the same length as the
            number of used codes, e.g. for `atom_codes={6: 0, 7: 1, 8: 1}` you
            should provide something like ['C', 'O or N']. If not specified
            labels 'atom0', 'atom1' etc are used. If `atom_codes` is not
            specified this argument is ignored.
        named_properties: list of strings, optional
            Names of atomic properties to retrieve from pybel.Atom object. If
            not specified ['hyb', 'heavyvalence', 'heterovalence',
            'partialcharge'] is used.
        save_molecule_codes: bool, optional (default True)
            If set to True, there will be an additional feature to save
            molecule code. It is usefeul when saving molecular complex in a
            single array.
        custom_properties: list of callables, optional
            Custom functions to calculate atomic properties. Each element of
            this list should be a callable that takes pybel.Atom object and
            returns a float. If callable has `__name__` property it is used as
            feature label. Otherwise labels 'func<i>' etc are used, where i is
            the index in `custom_properties` list.
        smarts_properties: list of strings, optional
            Additional atomic properties defined with SMARTS patterns. These
            patterns should match a single atom. If not specified, deafult
            patterns are used.
        smarts_labels: list of strings, optional
            Labels for properties defined with SMARTS. Should have the same
            length as `smarts_properties`. If not specified labels 'smarts0',
            'smarts1' etc are used. If `smarts_properties` is not specified
            this argument is ignored.
        """

        # Remember namse of all features in the correct order
        self.FEATURE_NAMES = []

        if atom_codes is not None:
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values())
            for i in range(len(codes)):
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes)
            self.ATOM_CODES = atom_codes
            if atom_labels is not None:
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: '
                                     '%s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
            self.FEATURE_NAMES += atom_labels
        else:
            self.ATOM_CODES = {}

            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))

            # List of tuples (atomic_num, class_name) with atom types to encode.
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]

            for code, (atom, name) in enumerate(atom_classes):
                if type(atom) is list:
                    #
                    for a in atom:
                        self.ATOM_CODES[a] = code
                else:
                    self.ATOM_CODES[atom] = code
                self.FEATURE_NAMES.append(name)

            self.NUM_ATOM_CLASSES = len(atom_classes)

        if named_properties is not None:
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom)
                             if not prop.startswith('__')]
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError(
                        'named_properties must be in pybel.Atom attributes,'
                        ' %s was given at position %s' % (prop_id, prop)
                    )
            self.NAMED_PROPS = named_properties
        else:
            # pybel.Atom properties to save
            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS

        if not isinstance(save_molecule_codes, bool):
            raise TypeError('save_molecule_codes should be bool, got %s '
                            'instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes
        if save_molecule_codes:
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode')

        self.CALLABLES = []
        if custom_properties is not None:
            for i, func in enumerate(custom_properties):
                if not callable(func):
                    raise TypeError('custom_properties should be list of'
                                    ' callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name)

        if smarts_properties is None:
            # SMARTS definition for other properties
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring']
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None:
            if len(smarts_labels) != len(self.SMARTS):
                raise ValueError('Incorrect number of SMARTS labels: %s'
                                 ' instead of %s'
                                 % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int
            Atomic number

        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_features(self, molecule, molcode=None):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []
        features = []
        heavy_atoms = []

        for i, atom in enumerate(molecule):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))

        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1))))
        features = np.hstack([features,
                              self.find_smarts(molecule)[heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        return coords, features

    def get_features_CSAR(self, molecule, protein_idxs, ligand_idxs, molcode=None):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords,protein_coords,ligand_coords = [],[],[]
        features,protein_features,ligand_features = [],[],[]
        heavy_atoms,protein_heavy_atoms,ligand_heavy_atoms = [],[],[]

        for i, atom in enumerate(molecule):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            index = i
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))
                if index in protein_idxs:
                    protein_heavy_atoms.append(i)
                    protein_coords.append(atom.coords)
                    protein_features.append(np.concatenate((
                        self.encode_num(atom.atomicnum),
                        [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                        [func(atom) for func in self.CALLABLES],
                    )))
                elif index in ligand_idxs:
                    ligand_heavy_atoms.append(i)
                    ligand_coords.append(atom.coords)
                    ligand_features.append(np.concatenate((
                        self.encode_num(atom.atomicnum),
                        [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                        [func(atom) for func in self.CALLABLES],
                    )))

        coords,protein_coords,ligand_coords = np.array(coords, dtype=np.float32),\
                                              np.array(protein_coords, dtype=np.float32),\
                                              np.array(ligand_coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1))))
        features = np.hstack([features,
                              self.find_smarts(molecule)[heavy_atoms]])
        protein_features = np.hstack([protein_features,
                              self.find_smarts(molecule)[protein_heavy_atoms]])
        ligand_features = np.hstack([ligand_features,
                              self.find_smarts(molecule)[ligand_heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        return coords, features, protein_coords, protein_features, ligand_coords, ligand_features

    def to_pickle(self, fname='featurizer.pkl'):
        """Save featurizer in a given file. Featurizer can be restored with
        `from_pickle` method.

        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """

        # patterns can't be pickled, we need to temporarily remove them
        patterns = self.__PATTERNS[:]
        del self.__PATTERNS
        try:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self.__PATTERNS = patterns[:]

    @staticmethod
    def from_pickle(fname):
        """Load pickled featurizer from a given file

        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurizer

        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'rb') as f:
            featurizer = pickle.load(f)
        featurizer.compile_smarts()
        return featurizer

featurizer = Featurizer(save_molecule_codes=False)

def get_labels_from_names(lables_path,names):
    with open(lables_path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')[6:]
    res = {}
    for line in lines:
        temp = line.split()
        name, score = temp[0], float(temp[3])
        res[name] = score
    labels = []
    for name in names:
        labels.append(res[name])
    return labels

def get_labels_from_names_csar(lables_path,names):
    with open(lables_path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')[1:]
    res = {}
    for line in lines:
        temp = [x.strip() for x in line.split(',')]
        name, score = temp[1], float(temp[2])
        res[name] = score
    labels = []
    for name in names:
        labels.append(res[name])
    return labels

def get_lig_coords_ground_truth_from_names(lables_path,names):

    return

def lig_atom_type_obmol(obmol):
    AtomIndex = [atom.atomicnum for atom in obmol if atom.atomicnum > 1]
    return torch.tensor(AtomIndex,dtype=torch.int64)

def lig_atom_type_rdmol(rdmol):
    AtomIndex = [atom.GetAtomicNum() for atom in rdmol.GetAtoms()]
    return torch.tensor(AtomIndex,dtype=torch.int64)

def get_bonded_edges_obmol(pocket):
    edge_l = []
    idx_map = [-1]*(len(pocket.atoms)+1)
    idx_new = 0
    for atom in pocket:
        edges = []
        a1_sym = atom.atomicnum
        a1 = atom.idx
        if a1_sym == 1:
            continue
        idx_map[a1] = idx_new
        idx_new += 1
        for natom in openbabel.OBAtomAtomIter(atom.OBAtom):
            if natom.GetAtomicNum() == 1:
                continue
            a2 = natom.GetIdx()
            bond = openbabel.OBAtom.GetBond(natom,atom.OBAtom)
            bond_type = CusBondFeaturizer(bond)
            edges.append((a1,a2,bond_type))
        edge_l += edges
    edge_l_new = []
    for a1,a2,t in edge_l:
        a1_, a2_ = idx_map[a1], idx_map[a2]
        assert((a1_!=-1)&(a2_!=-1))
        edge_l_new.append((a1_,a2_,t))
    return edge_l_new

def get_bonded_edges_rdmol(rdmol):
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]
    return zip(row,col,edge_type)

def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_

def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

def bond_feature(g):
    src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
    src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
    neighbors_ls = []
    for i, src_node in enumerate(src_nodes):
        tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
        neighbors = g.predecessors(src_node).tolist()
        neighbors.remove(dst_nodes[i])
        tmp.extend(neighbors)
        neighbors_ls.append(tmp)
    D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
    D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
    # D3_info_th = torch.cat([g.edata['e'], D3_info_th], dim=-1)
    return D3_info_th

def read_molecules_crossdock(lig_path, prot_path, ligcut, protcut, lig_type, prot_graph_type, dataset_path, chain_cut=5.0):
    lig_path = os.path.join(dataset_path, lig_path)
    prot_path = os.path.join(dataset_path, prot_path)
    if lig_type=='openbabel':
        m_lig = next(pybel.readfile('sdf', lig_path))
        lig_coords, lig_features = featurizer.get_features(m_lig)
        lig_edges = get_bonded_edges_obmol(m_lig) if ligcut is None else None
        lig_node_type = lig_atom_type_obmol(m_lig)
    elif lig_type=='rdkit':
        m_lig = read_rdmol(lig_path, sanitize=True, remove_hs=True)
        try:
            assert m_lig is not None
        except:
            raise ValueError(f'sanitize error : {lig_path}')

        conf = m_lig.GetConformer()
        lig_coords, lig_features = conf.GetPositions(), lig_atom_featurizer_rdmol(m_lig)
        lig_edges = get_bonded_edges_rdmol(m_lig) if ligcut is None else None
        lig_node_type = lig_atom_type_rdmol(m_lig)

    prot_complex = parsePDB(prot_path)
    prot_structure_no_water = prot_complex.select('protein')
    if chain_cut is not None:
        prot_valid_chains = prot_structure_no_water.select(f'same chain as within {chain_cut} of ligand', ligand=lig_coords)
    else:
        prot_valid_chains = prot_structure_no_water

    prot_valid_pocket = prot_structure_no_water.select('same residue as within 12 of ligand', ligand=lig_coords)
    prot_alpha_c = prot_valid_chains.select('calpha')
    prot_pocket_alpha_c = prot_valid_pocket.select('calpha')
    alpha_c_sec_features = None
    prot_pocket_alpha_c_sec_features = None
    alpha_c_coords, c_coords, n_coords, complete_residues = [], [], [], [] # complete_residue means a residue has alpha_c,beta_c,and ,N

    if prot_graph_type.startswith('atom'):
        m_prot = prot_valid_pocket if prot_graph_type.endswith('pocket') else prot_valid_chains
        sec_features = None
        prot_coords, prot_features = featurizer.get_features(m_prot)
        prot_edges = get_bonded_edges_obmol(m_prot) if protcut is None else None
        prot_node_type = lig_atom_type_obmol(m_prot)

    elif prot_graph_type.startswith('residue'):
        alpha_c_sec_features = None
        prot_pocket_alpha_c_sec_features = None
        m_prot = prot_pocket_alpha_c if prot_graph_type.endswith('pocket') else prot_alpha_c
        m_prot_complete = prot_valid_pocket if prot_graph_type.endswith('pocket') else prot_valid_chains
        sec_features = prot_pocket_alpha_c_sec_features if prot_graph_type.endswith('pocket') else alpha_c_sec_features

        prot_coords, prot_features = prot_alpha_c_featurizer(m_prot)
        prot_node_type = prot_residue_type(m_prot)
        prot_edges = None
        hv = m_prot_complete.getHierView()
        for chain in hv:
            for i, residue in enumerate(chain):
                alpha_c_coord, c_coord, n_coord = None, None, None
                for atom in residue:
                    if atom.getName() == 'CA':
                        alpha_c_coord = atom.getCoords()
                    if atom.getName() == 'C':
                        c_coord = atom.getCoords()
                    if atom.getName() == 'N':
                        n_coord = atom.getCoords()

                if alpha_c_coord is not None and c_coord is not None and n_coord is not None:
                    alpha_c_coords.append(alpha_c_coord)
                    c_coords.append(c_coord)
                    n_coords.append(n_coord)
                    complete_residues.append(True)
                else:
                    complete_residues.append(False)
        assert len(complete_residues) == len(prot_coords)

        prot_coords = prot_coords[complete_residues]
        prot_features = prot_features[complete_residues]
        prot_node_type = prot_node_type[complete_residues]
        if sec_features is not None:
            sec_features = sec_features[complete_residues]
            assert len(sec_features) == len(prot_coords)

        assert len(alpha_c_coords) == len(prot_coords)
        assert len(c_coords) == len(prot_coords)
        assert len(n_coords) == len(prot_coords)

    else:
        raise ValueError("error prot_graph_type")

    return lig_coords, lig_features, lig_edges, lig_node_type, \
           prot_coords, prot_features, prot_edges, prot_node_type, sec_features,\
           np.array(alpha_c_coords), np.array(c_coords), np.array(n_coords)

def read_ligands_chembl_smina(name, valid_ligand_index, dataset_path, ligcut, lig_type='openbabel'):
    valid_lig_coords_list, valid_lig_features_list, valid_lig_edges_list, valid_lig_node_type_list, valid_index_list = [], [], [], [], []

    for index, valid in enumerate(valid_ligand_index):
        lig_path_sdf = os.path.join(dataset_path, name, 'ligand_smina_poses', f'{index}_1.sdf')
        if valid:
            if lig_type == 'openbabel':
                m_lig = next(pybel.readfile('sdf', lig_path_sdf))
                lig_coords, lig_features = featurizer.get_features(m_lig)
                lig_edges = get_bonded_edges_obmol(m_lig) if ligcut is None else None
                lig_node_type = lig_atom_type_obmol(m_lig)

                valid_lig_coords_list.append(lig_coords)
                valid_lig_features_list.append(lig_features)
                valid_lig_edges_list.append(lig_edges)
                valid_lig_node_type_list.append(lig_node_type)
                valid_index_list.append(index)
            elif lig_type == 'rdkit':
                m_lig = read_rdmol(lig_path_sdf)
                conf = m_lig.GetConformer()

                lig_coords, lig_features = conf.GetPositions(), lig_atom_featurizer_rdmol(m_lig)
                lig_edges = get_bonded_edges_rdmol(m_lig) if ligcut is None else None
                lig_node_type = lig_atom_type_rdmol(m_lig)

                valid_lig_coords_list.append(lig_coords)
                valid_lig_features_list.append(lig_features)
                valid_lig_edges_list.append(lig_edges)
                valid_lig_node_type_list.append(lig_node_type)
                valid_index_list.append(index)

    return valid_lig_coords_list, valid_lig_features_list, valid_lig_edges_list, valid_lig_node_type_list, valid_index_list

def read_ligands(name, dataset_path, ligcut, lig_type='openbabel'):
    #########################Read Ligand########################################################
    lig_path_sdf = os.path.join(dataset_path, name, 'visualize_dir', 'total_vs.sdf')
    valid_lig_coords_list, valid_lig_features_list, valid_lig_edges_list, valid_lig_node_type_list, valid_index_list = [], [], [], [], []
    if lig_type == 'openbabel':
        m_ligs = pybel.readfile('sdf', lig_path_sdf)
        for index, m_lig in enumerate(m_ligs):
            try:
                lig_coords, lig_features = featurizer.get_features(m_lig)
                lig_edges = get_bonded_edges_obmol(m_lig) if ligcut is None else None
                lig_node_type = lig_atom_type_obmol(m_lig)

                valid_lig_coords_list.append(lig_coords)
                valid_lig_features_list.append(lig_features)
                valid_lig_edges_list.append(lig_edges)
                valid_lig_node_type_list.append(lig_node_type)
                valid_index_list.append(index)
            except:
                print(f'{index} error')
    elif lig_type == 'rdkit':
        supplier = Chem.SDMolSupplier(lig_path_sdf, sanitize=True, removeHs=False)
        for index, m_lig in enumerate(supplier):
            try:
                conf = m_lig.GetConformer()
                lig_coords, lig_features = conf.GetPositions(), lig_atom_featurizer_rdmol(m_lig)
                lig_edges = get_bonded_edges_rdmol(m_lig) if ligcut is None else None
                lig_node_type = lig_atom_type_rdmol(m_lig)

                valid_lig_coords_list.append(lig_coords)
                valid_lig_features_list.append(lig_features)
                valid_lig_edges_list.append(lig_edges)
                valid_lig_node_type_list.append(lig_node_type)
                valid_index_list.append(index)
            except:
                print(f'{index} error')

    return valid_lig_coords_list, valid_lig_features_list, valid_lig_edges_list, valid_lig_node_type_list, valid_index_list

def read_proteins(name, dataset_path, prot_graph_type, protcut):
    #########################Read Protein########################################################
    prot_valid_chains = parsePDB(os.path.join(dataset_path, name, f'{name}_valid_chains.pdb'))
    prot_alpha_c = prot_valid_chains.select('calpha')
    alpha_c_coords, c_coords, n_coords = [], [], []
    writePDB(os.path.join(dataset_path, name, f'{name}_valid_chains.pdb'), prot_valid_chains)

    if prot_graph_type.startswith('atom'):
        prot_path = os.path.join(dataset_path, name, f'{name}_{graph_type_filename[prot_graph_type]}')
        m_prot = next(pybel.readfile('pdb', prot_path))
        sec_features = None
        prot_coords_valid, prot_features_valid = featurizer.get_features(m_prot)
        prot_edges = get_bonded_edges_obmol(m_prot) if protcut is None else None
        prot_node_type = lig_atom_type_obmol(m_prot)

    elif prot_graph_type.startswith('residue'):
        alpha_c_sec_features = None
        m_prot = prot_alpha_c
        m_prot_complete = prot_valid_chains
        sec_features = alpha_c_sec_features

        prot_coords, prot_features = prot_alpha_c_featurizer(m_prot)
        prot_node_type = prot_residue_type(m_prot)
        prot_edges = None
        hv = m_prot_complete.getHierView()
        index = 0
        valid_index, prot_coords_valid, prot_features_valid = [], [], []
        for chain in hv:
            for i, residue in enumerate(chain):
                alpha_c_coord, c_coord, n_coord = None, None, None
                for atom in residue:
                    if atom.getName() == 'CA':
                        alpha_c_coord = atom.getCoords()

                    if atom.getName() == 'C':
                        c_coord = atom.getCoords()

                    if atom.getName() == 'N':
                        n_coord = atom.getCoords()

                if alpha_c_coord is not None and c_coord is not None and n_coord is not None:
                    alpha_c_coords.append(alpha_c_coord)
                    c_coords.append(c_coord)
                    n_coords.append(n_coord)
                    valid_index.append(index)
                index += 1

        prot_coords_valid = prot_coords[valid_index]
        prot_features_valid = prot_features[valid_index]

    else:
        raise ValueError("error prot_graph_type")

    return prot_coords_valid, prot_features_valid, prot_edges, prot_node_type, sec_features,\
           np.array(alpha_c_coords), np.array(c_coords), np.array(n_coords),\


def read_molecules(name, dataset_path, prot_graph_type, ligcut, protcut, lig_type='openbabel',init_type='redock_init',
                   chain_cut=5.0, p2rank_base=None, binding_site_type='ligand_center', LAS_mask=True,
                   keep_hs_before_rdkit_generate=False, rd_gen_maxIters=200):
    #########################Read Ligand########################################################
    lig_path_mol2 = os.path.join(dataset_path, name, f'{name}_ligand.mol2')
    lig_path_sdf = os.path.join(dataset_path, name, f'{name}_ligand.sdf')
    if lig_type == 'openbabel':
        m_lig = next(pybel.readfile('mol2', lig_path_mol2))
        lig_coords, lig_features = featurizer.get_features(m_lig)
        lig_edges = get_bonded_edges_obmol(m_lig) if ligcut is None else None
        lig_node_type = lig_atom_type_obmol(m_lig)
    elif lig_type == 'rdkit':
        m_lig = read_rdmol(lig_path_sdf, sanitize=True, remove_hs=True)
        if m_lig == None:  # read mol2 file if sdf file cannot be sanitized
            m_lig = read_rdmol(lig_path_mol2, sanitize=True, remove_hs=True)

        conf = m_lig.GetConformer()
        lig_coords, lig_features = conf.GetPositions(), lig_atom_featurizer_rdmol(m_lig)
        lig_edges = get_bonded_edges_rdmol(m_lig) if ligcut is None else None
        lig_node_type = lig_atom_type_rdmol(m_lig)

    #########################Get Ligand Rdkit Init Coordinates###################################
    if init_type == 'rdkit_init':
        rd_lig = read_rdmol(lig_path_sdf, sanitize=True, remove_hs=not keep_hs_before_rdkit_generate)
        if rd_lig == None:  # read mol2 file if sdf file cannot be sanitized
            rd_lig = read_rdmol(lig_path_mol2, sanitize=True, remove_hs=not keep_hs_before_rdkit_generate)
        try:
            lig_init_coords = get_rdkit_coords(rd_lig, sanitize=True, remove_hs=True, maxIters=rd_gen_maxIters)
        except Exception as e:
            lig_init_coords = lig_coords
            with open(f'temp_create_dataset_rdkit_timesplit_no_lig_or_rec_overlap_train_remove_hs_before_generate_{not keep_hs_before_rdkit_generate}.log', 'a') as f:
                f.write('Generating RDKit conformer failed for  \n')
                f.write(name)
                f.write('\n')
                f.write(str(e))
                f.write('\n')
                f.flush()

        assert len(lig_init_coords) == len(lig_coords)

        rdlig_node_type = lig_atom_type_rdmol(rd_lig)

        # remove all h
        if lig_type == 'openbabel':
            lig_init_coords = lig_init_coords[rdlig_node_type != 1]
        try:
            if len(lig_init_coords)!=len(lig_coords):
                raise ValueError('{} {}!={}'.format(name, len(lig_init_coords), len(lig_coords)))
        except ValueError as e:
            print("error raise:", repr(e))
            raise
    elif init_type == 'redock_init':
        lig_init_coords = lig_coords
    elif init_type == 'random_init':
        lig_init_coords = np.random.randn(len(lig_coords),3)
    else:
        lig_init_coords = None

    # random location and orientation
    if lig_init_coords is not None:
        rot_T, rot_b = random_rotation_translation()
        mean_to_remove = lig_init_coords.mean(axis=0, keepdims=True)
        lig_init_coords = (rot_T @ (lig_init_coords - mean_to_remove).T).T + rot_b

    #########################Read Protein########################################################
    if os.path.exists(os.path.join(dataset_path, name, f'{name}_protein_processed.pdb')):
        prot_complex = parsePDB(os.path.join(dataset_path, name, f'{name}_protein_processed.pdb'))
    else:
        prot_complex = parsePDB(os.path.join(dataset_path, name, f'{name}_protein.pdb'))
    prot_structure_no_water = prot_complex.select('protein')
    if chain_cut is not None:
        prot_valid_chains = prot_structure_no_water.select(f'same chain as within {chain_cut} of ligand', ligand=lig_coords)
    else:
        prot_valid_chains = prot_structure_no_water

    prot_valid_pocket = prot_structure_no_water.select('same residue as within 12 of ligand', ligand=lig_coords)
    prot_alpha_c = prot_valid_chains.select('calpha')
    prot_pocket_alpha_c = prot_valid_pocket.select('calpha')
    alpha_c_sec_features = None
    prot_pocket_alpha_c_sec_features = None
    alpha_c_coords, c_coords, n_coords = [], [], []
    writePDB(os.path.join(dataset_path, name, f'{name}_valid_chains.pdb'), prot_valid_chains)
    writePDB(os.path.join(dataset_path, name, f'{name}_valid_pocket.pdb'), prot_valid_pocket)

    if prot_graph_type.startswith('atom'):
        prot_path = os.path.join(dataset_path, name, f'{name}_{graph_type_filename[prot_graph_type]}')
        m_prot = next(pybel.readfile('pdb', prot_path))
        sec_features = None
        prot_coords_valid, prot_features_valid = featurizer.get_features(m_prot)
        prot_edges = get_bonded_edges_obmol(m_prot) if protcut is None else None
        prot_node_type = lig_atom_type_obmol(m_prot)

    elif prot_graph_type.startswith('residue'):
        alpha_c_sec_features = None
        prot_pocket_alpha_c_sec_features = None
        m_prot = prot_pocket_alpha_c if prot_graph_type.endswith('pocket') else prot_alpha_c
        m_prot_complete = prot_valid_pocket if prot_graph_type.endswith('pocket') else prot_valid_chains
        sec_features = prot_pocket_alpha_c_sec_features if prot_graph_type.endswith('pocket') else alpha_c_sec_features

        prot_coords, prot_features = prot_alpha_c_featurizer(m_prot)
        prot_node_type = prot_residue_type(m_prot)
        prot_edges = None
        hv = m_prot_complete.getHierView()
        index = 0
        valid_index, prot_coords_valid, prot_features_valid = [], [], []
        for chain in hv:
            for i, residue in enumerate(chain):
                alpha_c_coord, c_coord, n_coord = None, None, None
                for atom in residue:
                    if atom.getName() == 'CA':
                        alpha_c_coord = atom.getCoords()

                    if atom.getName() == 'C':
                        c_coord = atom.getCoords()

                    if atom.getName() == 'N':
                        n_coord = atom.getCoords()

                if alpha_c_coord is not None and c_coord is not None and n_coord is not None:
                    alpha_c_coords.append(alpha_c_coord)
                    c_coords.append(c_coord)
                    n_coords.append(n_coord)
                    valid_index.append(index)
                index += 1

        prot_coords_valid = prot_coords[valid_index]
        prot_features_valid = prot_features[valid_index]

    else:
        raise ValueError("error prot_graph_type")

    ############################### Read Binding Site ##########################################
    if binding_site_type == 'p2rank':
        p2rank_result_path = os.path.join(p2rank_base, f'{name}_valid_chains.pdb_predictions.csv')
        df = pd.read_csv(p2rank_result_path, usecols= ['   center_x','   center_y','   center_z'])
        possible_binding_sites = df.values
        ligand_center = lig_coords.mean(axis=0)
        if len(possible_binding_sites) == 0:
            binding_site = ligand_center
        else:
            binding_site_index = ((possible_binding_sites - ligand_center) ** 2).sum(axis=1).argmin()
            binding_site = possible_binding_sites[binding_site_index]

    elif binding_site_type == 'ligand_center':
        binding_site = lig_coords.mean(axis=0)

    ############################### Get LAS Mask ##########################################
    if LAS_mask:
        assert lig_type == 'rdkit'
        lig_LAS_mask = get_LAS_distance_constraint_mask(m_lig)
    else:
        lig_LAS_mask = None

    return lig_coords, lig_features, lig_edges, lig_node_type, lig_init_coords, \
           prot_coords_valid, prot_features_valid, prot_edges, prot_node_type, sec_features,\
           np.array(alpha_c_coords), np.array(c_coords), np.array(n_coords),\
           binding_site.reshape(1,-1), lig_LAS_mask

def get_ligand_smiles(name, dataset_path,):
    lig_path_mol2 = os.path.join(dataset_path, name, f'{name}_ligand.mol2')
    lig_path_sdf = os.path.join(dataset_path, name, f'{name}_ligand.sdf')
    m_lig = read_rdmol(lig_path_sdf, sanitize=True, remove_hs=True)
    if m_lig == None:  # read mol2 file if sdf file cannot be sanitized
        m_lig = read_rdmol(lig_path_mol2, sanitize=True, remove_hs=True)

    sm = Chem.MolToSmiles(m_lig)
    m_sm_order = list(m_lig.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])

    sm2m_order = [0] * len(m_sm_order)
    for index, order in enumerate(m_sm_order):
        sm2m_order[order] = index

    return sm, sm2m_order

def get_protein_fasta(name, dataset_path,):
    try:
        prot_valid_chains = parsePDB(os.path.join(dataset_path, name, f'{name}_valid_chains.pdb'))
    except:
        raise ValueError(f'{name} error!')
    hv = prot_valid_chains.getHierView()
    index = 0
    valid_index, prot_coords_valid, prot_features_valid = [], [], []
    alpha_c_coords, c_coords, n_coords = [], [], []
    fasta_list = []
    for chain in hv:
        fasta_list.append(chain.getSequence())
        for i, residue in enumerate(chain):
            alpha_c_coord, c_coord, n_coord = None, None, None
            for atom in residue:
                if atom.getName() == 'CA':
                    alpha_c_coord = atom.getCoords()

                if atom.getName() == 'C':
                    c_coord = atom.getCoords()

                if atom.getName() == 'N':
                    n_coord = atom.getCoords()

            if alpha_c_coord is not None and c_coord is not None and n_coord is not None:
                alpha_c_coords.append(alpha_c_coord)
                c_coords.append(c_coord)
                n_coords.append(n_coord)
                valid_index.append(index)
            index += 1

    return fasta_list, valid_index

def prot_p2rank_feats(p2rank_result_path, p2rank_feats_tpye='zscore', pocket_cut=10):
    df = pd.read_csv(p2rank_result_path)
    df.columns = df.columns.str.strip()
    residue_zscores, residue_pocket_idxs = df[p2rank_feats_tpye].values, df['pocket'].values
    feat_len = pocket_cut + 2
    p2rank_feats = torch.zeros(len(residue_zscores), feat_len)
    for index, residue_zscore in enumerate(residue_zscores):
        pocket_idx = residue_pocket_idxs[index]
        if pocket_idx == 0 :
            p2rank_feats[index, feat_len - 1] = residue_zscore
        elif pocket_idx > pocket_cut :
            p2rank_feats[index, feat_len - 2] = residue_zscore
        else:
            p2rank_feats[index, pocket_idx - 1] = residue_zscore

    return p2rank_feats

def get_p2rank_feats(name, dataset_path, p2rank_base=None, p2rank_feats_tpye='zscore', pocket_cut=10):
    try:
        prot_valid_chains = parsePDB(os.path.join(dataset_path, name, f'{name}_valid_chains.pdb'))
    except:
        raise ValueError(f'{name} error!')
    prot_alpha_c = prot_valid_chains.select('calpha')
    alpha_c_coords, c_coords, n_coords = [], [], []

    p2rank_result_path = os.path.join(p2rank_base, f'{name}_valid_chains.pdb_residues.csv')
    p2rank_features = prot_p2rank_feats(p2rank_result_path, pocket_cut=pocket_cut, p2rank_feats_tpye=p2rank_feats_tpye)
    prot_coords, prot_features = prot_alpha_c_featurizer(prot_alpha_c)

    try:
        assert len(p2rank_features) == len(prot_features)
    except:
        # print(f'p2rank protein number, {len(p2rank_features)}')
        # print(f'prot_features protein number, {len(prot_features)}')
        # raise ValueError(f'p2rank length error, {name}')
        with open('p2rank_feats_error.txt','a') as f:
            f.write(f'{name}\n')
        return torch.zeros(len(prot_features), pocket_cut + 2)

    hv = prot_valid_chains.getHierView()
    index = 0
    valid_index, prot_coords_valid, prot_features_valid = [], [], []
    for chain in hv:
        for i, residue in enumerate(chain):
            alpha_c_coord, c_coord, n_coord = None, None, None
            for atom in residue:
                if atom.getName() == 'CA':
                    alpha_c_coord = atom.getCoords()

                if atom.getName() == 'C':
                    c_coord = atom.getCoords()

                if atom.getName() == 'N':
                    n_coord = atom.getCoords()

            if alpha_c_coord is not None and c_coord is not None and n_coord is not None:
                alpha_c_coords.append(alpha_c_coord)
                c_coords.append(c_coord)
                n_coords.append(n_coord)
                valid_index.append(index)
            index += 1

    p2rank_features_valid = p2rank_features[valid_index]

    return p2rank_features_valid

def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

#adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat

def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask

def get_lig_graph_geodiff(lig_coords, lig_features, lig_node_type, lig_edges):
    g_lig = dgl.DGLGraph()

    num_atoms_lig = len(lig_coords)  # number of ligand atom_level
    g_lig.add_nodes(num_atoms_lig)
    g_lig.ndata['h'] = torch.from_numpy(lig_features) if isinstance(lig_features, np.ndarray) else lig_features
    g_lig.ndata['node_type'] = lig_node_type # schnet\mgcn features

    edges = lig_edges
    src_ls, dst_ls, bond_type = list(zip(*edges))
    src_ls, dst_ls = np.array(src_ls), np.array(dst_ls)

    g_lig.add_edges(src_ls, dst_ls)
    g_lig.ndata['pos'] = torch.tensor(lig_coords, dtype=torch.float)
    g_lig.edata['bond_type'] = torch.tensor(bond_type, dtype=torch.int64)

    return g_lig

def get_lig_graph_equibind(lig_coords, lig_features, lig_node_type, max_neighbors=None, cutoff=5.0):

    num_nodes = lig_coords.shape[0]
    assert lig_coords.shape[1] == 3
    distance = spatial.distance_matrix(lig_coords, lig_coords)

    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            print(
                f'The lig_radius {cutoff} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = lig_coords[src, :] - lig_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)

        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['h'] = torch.from_numpy(lig_features) if isinstance(lig_features, np.ndarray) else lig_features
    graph.ndata['node_type'] = lig_node_type  # schnet\mgcn features
    graph.edata['e'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['pos'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))

    return graph

def get_lig_graph(lig_coords,lig_features, lig_edges, lig_node_type, cutoff=None):
    g_lig = dgl.DGLGraph()

    num_atoms_lig = len(lig_coords)  # number of ligand atom_level
    g_lig.add_nodes(num_atoms_lig)
    g_lig.ndata['h'] = torch.from_numpy(lig_features) if isinstance(lig_features, np.ndarray) else lig_features
    g_lig.ndata['node_type'] = lig_node_type # schnet\mgcn features
    dis_matrix_lig = spatial.distance_matrix(lig_coords, lig_coords)
    if cutoff is None:
        edges = lig_edges
        src_ls, dst_ls, bond_type = list(zip(*edges))
        src_ls, dst_ls = np.array(src_ls), np.array(dst_ls)
    else:
        node_idx = np.where( (dis_matrix_lig < cutoff) & (dis_matrix_lig!=0) ) # no self-loop
        src_ls = node_idx[0]
        dst_ls = node_idx[1]
    g_lig.add_edges(src_ls, dst_ls)
    lig_d = torch.tensor(dis_matrix_lig[src_ls, dst_ls], dtype=torch.float).view(-1, 1)
    g_lig.edata['distance'] = lig_d
    g_lig.edata['e'] = lig_d * 0.1  # g.edata['e'] ~ (n_bond1+n_bond2) * k
    g_lig.ndata['pos'] = torch.tensor(lig_coords, dtype=torch.float)
    D3_info = bond_feature(g_lig)
    g_lig.edata['e'] = torch.cat([g_lig.edata['e'], D3_info], dim=-1)
    g_lig.edata['bond_type'] = torch.tensor(bond_type,dtype=torch.int64)
    # g_lig.ndata.pop('pos')
    assert not torch.any(torch.isnan(D3_info))
    return g_lig

def get_prot_atom_graph(prot_coords, prot_features, prot_edges, prot_node_type, cutoff=None):
    g_prot = dgl.DGLGraph()
    num_atoms_prot = len(prot_coords)
    g_prot.add_nodes(num_atoms_prot)
    g_prot.ndata['h'] = torch.from_numpy(prot_features) if isinstance(prot_features, np.ndarray) else prot_features
    g_prot.ndata['node_type'] = prot_node_type # schnet\mgcn features
    dis_matrix_lig = spatial.distance_matrix(prot_coords, prot_coords)
    if cutoff is None:
        edges = prot_edges
        src_ls, dst_ls, bond_type = list(zip(*edges))
        src_ls, dst_ls = np.array(src_ls), np.array(dst_ls)
    else:
        node_idx = np.where( (dis_matrix_lig < cutoff) & (dis_matrix_lig!=0) ) # no self-loop
        src_ls = node_idx[0]
        dst_ls = node_idx[1]
    g_prot.add_edges(src_ls, dst_ls)
    prot_d = torch.tensor(dis_matrix_lig[src_ls, dst_ls], dtype=torch.float).view(-1, 1)
    g_prot.edata['distance'] = prot_d
    g_prot.edata['e'] = prot_d * 0.1  # g.edata['e'] ~ (n_bond1+n_bond2) * k
    g_prot.ndata['pos'] = torch.tensor(prot_coords, dtype=torch.float)
    D3_info = bond_feature(g_prot)
    g_prot.edata['e'] = torch.cat([g_prot.edata['e'], D3_info], dim=-1)
    g_prot.edata['bond_type'] = torch.tensor(bond_type, dtype=torch.int64)
    # g_prot.ndata.pop('pos')
    assert not torch.any(torch.isnan(D3_info))
    return g_prot

def distance_featurizer(dist_list, divisor) -> torch.Tensor:
    # you want to use a divisor that is close to 4/7 times the average distance that you want to encode
    length_scale_list = [1.5 ** x for x in range(15)]
    center_list = [0. for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)

    transformed_dist = [np.exp(- ((dist_list / divisor) ** 2) / float(length_scale))
                        for length_scale, center in zip(length_scale_list, center_list)]

    transformed_dist = np.array(transformed_dist).T
    transformed_dist = transformed_dist.reshape((num_edge, -1))
    return torch.from_numpy(transformed_dist.astype(np.float32))

def local_coordinate_system_feature(prot_coords, c_alpha_coords, c_coords, n_coords, prot_d, src_ls, dst_ls):
    n_i_list, u_i_list, v_i_list = [], [], []
    for i in range(len(prot_coords)):
        nitrogen = n_coords[i]
        c_alpha = c_alpha_coords[i]
        carbon = c_coords[i]
        u_i = (nitrogen - c_alpha) / np.linalg.norm(nitrogen - c_alpha)
        t_i = (carbon - c_alpha) / np.linalg.norm(carbon - c_alpha)
        n_i = np.cross(u_i, t_i) / np.linalg.norm(np.cross(u_i, t_i))
        v_i = np.cross(n_i, u_i)
        assert (math.fabs(
            np.linalg.norm(v_i) - 1.) < 1e-5), "protein utils protein_to_graph_dips, v_i norm larger than 1"
        n_i_list.append(n_i)
        u_i_list.append(u_i)
        v_i_list.append(v_i)
    n_i_feat, u_i_feat, v_i_feat = np.stack(n_i_list), np.stack(u_i_list), np.stack(v_i_list)

    edge_feat_ori_list = []
    for i in range(len(prot_d)):
        src = src_ls[i]
        dst = dst_ls[i]
        # place n_i, u_i, v_i as lines in a 3x3 basis matrix
        basis_matrix = np.stack((n_i_feat[dst, :], u_i_feat[dst, :], v_i_feat[dst, :]), axis=0)
        p_ij = np.matmul(basis_matrix, c_alpha_coords[src, :] - c_alpha_coords[dst, :])
        q_ij = np.matmul(basis_matrix, n_i_feat[src, :])  # shape (3,)
        k_ij = np.matmul(basis_matrix, u_i_feat[src, :])
        t_ij = np.matmul(basis_matrix, v_i_feat[src, :])
        s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)  # shape (12,)
        edge_feat_ori_list.append(s_ij)
    edge_feat_ori_feat = np.stack(edge_feat_ori_list, axis=0)  # shape (num_edges, 12)
    edge_feat_ori_feat = torch.from_numpy(edge_feat_ori_feat.astype(np.float32))
    c_alpha_edge_feat = torch.cat([distance_featurizer(prot_d, divisor=4), edge_feat_ori_feat],axis=1) # shape (num_edges, 17)
    return c_alpha_edge_feat

def get_prot_alpha_c_graph_equibind(prot_coords, prot_features, prot_node_type, sec_features,
                                    alpha_c_coords, c_coords, n_coords,
                                    max_neighbor=None, cutoff=None):
    try:
        assert len(alpha_c_coords) == len(prot_coords)
        assert len(c_coords) == len(prot_coords)
        assert len(n_coords) == len(prot_coords)
    except:
        raise ValueError(f'{len(alpha_c_coords)} == {len(prot_coords)}, {len(c_coords)} == {len(prot_coords)}, {len(n_coords)} == {len(prot_coords)}')

    g_prot = dgl.DGLGraph()
    num_atoms_prot = len(prot_coords)  # number of pocket atom_level
    g_prot.add_nodes(num_atoms_prot)

    g_prot.ndata['h'] = prot_features
    g_prot.ndata['node_type'] = prot_node_type[:num_atoms_prot]
    distances = spatial.distance_matrix(prot_coords, prot_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_atoms_prot):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            print(
                f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = alpha_c_coords[src, :] - alpha_c_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)

    g_prot.add_edges(src_list, dst_list)

    g_prot.edata['e'] = local_coordinate_system_feature(prot_coords, alpha_c_coords, c_coords, n_coords,
                                                        dist_list, src_list, dst_list)
    residue_representatives_loc_feat = torch.from_numpy(alpha_c_coords.astype(np.float32))
    g_prot.ndata['pos'] = residue_representatives_loc_feat
    g_prot.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return g_prot

def get_prot_alpha_c_graph_ign(prot_coords, prot_features, prot_node_type, sec_features, cutoff=None):
    g_prot = dgl.DGLGraph()
    num_atoms_prot = len(prot_coords)  # number of pocket atom_level
    g_prot.add_nodes(num_atoms_prot)
    g_prot.ndata['h'] = torch.from_numpy(prot_features) if isinstance(prot_features, np.ndarray) else prot_features
    g_prot.ndata['node_type'] = prot_node_type[:num_atoms_prot]
    dis_matrix = spatial.distance_matrix(prot_coords, prot_coords)
    node_idx = np.where((dis_matrix < cutoff) & (dis_matrix != 0)) # no self-loop
    src_ls = node_idx[0]
    dst_ls = node_idx[1]
    g_prot.add_edges(src_ls, dst_ls)
    g_prot.ndata['pos'] = torch.tensor(prot_coords, dtype=torch.float)
    prot_d = torch.tensor(dis_matrix[src_ls, dst_ls], dtype=torch.float).view(-1, 1)
    g_prot.edata['distance'] = prot_d

    # g_prot.edata['e'] = prot_d * 0.1

    # calculate the 3D info for g
    D3_info_th = bond_feature(g_prot)

    g_prot.edata['e'] = torch.cat([D3_info_th, prot_d * 0.1], dim=-1)
    # g_prot.ndata.pop('pos')
    assert not torch.any(torch.isnan(D3_info_th))

    return g_prot

def get_interact_graph_fc(lig_coords,prot_coords,cutoff=None):
    # get fully connected graph
    g_inter = dgl.DGLGraph()
    num_atoms_lig = len(lig_coords)
    num_atoms_prot = len(prot_coords)

    g_inter.add_nodes(num_atoms_lig + num_atoms_prot)
    dis_matrix = spatial.distance_matrix(lig_coords, prot_coords)
    node_idx = np.where(dis_matrix > 0)
    src_ls = np.concatenate([node_idx[0], node_idx[1] + num_atoms_lig])
    dst_ls = np.concatenate([node_idx[1] + num_atoms_lig, node_idx[0]])
    g_inter.add_edges(src_ls, dst_ls)
    # 'd', distance between ligand atom_level and pocket atom_level
    inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]], dis_matrix[node_idx[0], node_idx[1]]])
    inter_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)
    g_inter.edata['e'] = inter_d  # if add_self_loop=ture, need to modify here6+'
    g_inter.ndata['pos'] = torch.cat([torch.tensor(lig_coords, dtype=torch.float),torch.tensor(prot_coords, dtype=torch.float)],dim=0)
    return g_inter

def get_interact_graph_knn(lig_coords,prot_coords,max_neighbor=None,min_neighbor=None,cutoff=None,):
    g_inter = dgl.DGLGraph()
    num_atoms_lig = len(lig_coords)
    num_atoms_prot = len(prot_coords)

    g_inter.add_nodes(num_atoms_lig + num_atoms_prot)
    dis_matrix = spatial.distance_matrix(lig_coords, prot_coords)

    src_list, dst_list, dis_list = [], [], []
    for i in range(num_atoms_lig):
        dst = np.where(dis_matrix[i, :] < cutoff)[0]
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(dis_matrix[i, :]))[:max_neighbor]
        if min_neighbor != None and len(dst) == 0:
            dst = list(np.argsort(dis_matrix[i, :]))[:min_neighbor]

        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend([x + num_atoms_lig for x in dst])
        dis_list.extend(list(dis_matrix[i,dst]))

    for i in range(num_atoms_prot):
        dst = list(np.where(dis_matrix[:, i] < cutoff)[0])

        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(dis_matrix[:, i]))[:max_neighbor]
        if min_neighbor != None and len(dst) == 0:
            dst = list(np.argsort(dis_matrix[:, i]))[:min_neighbor]  # choose second because first is i itself

        src = [i] * len(dst)
        src_list.extend([x + num_atoms_lig for x in src])
        dst_list.extend(dst)
        dis_list.extend(list(dis_matrix[dst, i]))

    src_ls = np.array(src_list)
    dst_ls = np.array(dst_list)
    g_inter.add_edges(src_ls, dst_ls)
    # 'd', distance between ligand atom_level and pocket atom_level
    inter_dis = np.array(dis_list)
    inter_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

    # squared_distance = inter_d ** 2
    # all_sigmas_dist = [1.5 ** x for x in range(15)]
    # prot_square_distance_scale = 10.0
    # x_rel_mag = torch.cat([torch.exp(-(squared_distance / prot_square_distance_scale) / sigma) for sigma in
    #                        all_sigmas_dist], dim=-1)
    # g_inter.edata['e'] = x_rel_mag
    g_inter.edata['d'] = inter_d
    return g_inter

def get_interact_graph_knn_v2(lig_coords,prot_coords,max_neighbor=None,min_neighbor=None,cutoff=None,):
    g_inter = dgl.DGLGraph()
    num_atoms_lig = len(lig_coords)
    num_atoms_prot = len(prot_coords)

    g_inter.add_nodes(num_atoms_lig + num_atoms_prot)
    dis_matrix = spatial.distance_matrix(lig_coords, prot_coords)

    src_list, dst_list, dis_list = [], [], []
    for i in range(num_atoms_lig):
        dst = np.where(dis_matrix[i, :] < cutoff)[0]
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(dis_matrix[i, :]))[:max_neighbor]
        if min_neighbor != None and len(dst) == 0:
            dst = list(np.argsort(dis_matrix[i, :]))[:min_neighbor]

        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend([x + num_atoms_lig for x in dst])
        dis_list.extend(list(dis_matrix[i,dst]))

    for i in range(num_atoms_prot):
        dst = list(np.where(dis_matrix[:, i] < cutoff)[0])

        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(dis_matrix[:, i]))[:max_neighbor]
        if min_neighbor != None and len(dst) == 0:
            dst = list(np.argsort(dis_matrix[:, i]))[:min_neighbor]  # choose second because first is i itself

        src = [i] * len(dst)
        src_list.extend([x + num_atoms_lig for x in src])
        dst_list.extend(dst)
        dis_list.extend(list(dis_matrix[dst, i]))

    src_ls = np.array(src_list)
    dst_ls = np.array(dst_list)
    g_inter.add_edges(src_ls, dst_ls)
    # 'd', distance between ligand atom_level and pocket atom_level
    inter_dis = np.array(dis_list)
    inter_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

    squared_distance = inter_d ** 2
    all_sigmas_dist = [1.5 ** x for x in range(15)]
    prot_square_distance_scale = 10.0
    x_rel_mag = torch.cat([torch.exp(-(squared_distance / prot_square_distance_scale) / sigma) for sigma in
                           all_sigmas_dist], dim=-1)
    g_inter.edata['e'] = x_rel_mag
    g_inter.edata['d'] = inter_d
    return g_inter

def pack_graph_and_labels(lig_graphs, prot_graphs, inter_graphs, labels):
    return lig_graphs, prot_graphs, inter_graphs, labels

SSE_color = {'H':'R','G':'R','I':'R','E':'B','T':'G','S':'W','B':'W',' ':'W'}

def ExcuteDSSP(dataset_path,name,dssp = '/data/jiaxianyan/anaconda3/bin/mkdssp'):
    InPath = os.path.join(dataset_path, name, f'{name}_protein_chains.pdb')
    OutPath = os.path.join(dataset_path, name, f'{name}_protein_chains.pdb.dssp')
    SSPath = os.path.join(dataset_path, name, f'{name}_protein_chains.pdb.SS')
    cmd = '{} -i {} -o {}'.format(dssp,InPath,OutPath)
    os.system(cmd)

def ListToStr(res):
    # res is list
    Res = ''
    for r in res:
        Res += r[0]*r[1]
    return Res

def StrToList(res):
    # res is string
    before, length, Res = 'W',0,[]
    for s in res:
        if s != before:
            Res.append((before, length))
            length = 1
            before = s
        else:
            length += 1
    Res.append((before, length))
    return Res

def AllocateIndex(res,BeginIndex=0):
    # res is list
    str_res = ListToStr(res)
    ClusterIndex,Res = BeginIndex,[]
    for index in range(len(str_res)):
        if index!=0 and str_res[index]!=str_res[index-1]:
            ClusterIndex += 1
        Res.append(str(ClusterIndex)+','+str_res[index])
    return Res,ClusterIndex+1

def SmoothHiearachical(res,threshod=0):
    if threshod==0:
        return res
    # res is list
    Res = [0] * len(res)
    if len(res)==1:
        return res

    Res[0] = (res[1][0], res[0][1])
    Res[-1] = (res[-2][0], res[-1][1])
    for index in range(1,len(res)-1):
        SSE,length = res[index]
        if length<=threshod:
            Res[index] = (Res[index-1][0],length)
        else:
            Res[index] = res[index]
    return Res


def ExtractHiearachical(FilePath):
    with open(FilePath,'r') as f:
        lines = f.read().strip().split('\n')[28:]

    borders, start = [], 0
    for index,line in enumerate(lines):
        if '!' in line:
            borders.append((start,index))
            start = index + 1
    borders.append((start,len(lines)+1))

    SSEIndex,SupplyIndex = 16,18

    ResSSE = ''
    ResSmoothSSE = ''
    NumClusters = 0
    ClusterIndexs = []
    for border in borders:
        SSE,Color = '',''
        OneChain = lines[border[0]:border[1]]
        for index,line in enumerate(OneChain):
            if line[SupplyIndex]=='>' and line[SSEIndex] == ' ':
                SSE += OneChain[index + 1][SSEIndex]
                Color += SSE_color[OneChain[index + 1][SSEIndex]]
            elif line[SupplyIndex]=='<' and line[SSEIndex] ==' ':
                SSE += OneChain[index - 1][SSEIndex]
                Color += SSE_color[OneChain[index - 1][SSEIndex]]
            else:
                SSE += line[SSEIndex]
                Color += SSE_color[line[SSEIndex]]

        ResSSE = ResSSE + SSE
        res = StrToList(SSE)
        Smoothres = SmoothHiearachical(res)
        ClusterIndex,NumCluster = AllocateIndex(Smoothres,NumClusters)
        ResSmoothSSE = ResSmoothSSE + ListToStr(Smoothres)
        NumClusters = NumCluster
        ClusterIndexs.extend(ClusterIndex)

def prot_alpha_c_featurizer(Structure):
    Coords = Structure.getCoords()
    ResNames = Structure.getResnames()
    ResIndex = [ResDict.get(ResName,UNKOWN_RES) for ResName in ResNames]
    ProtFeature = torch.tensor(np.eye(UNKOWN_RES+1)[ResIndex])
    return Coords, ProtFeature

def prot_residue_type(Structure):
    ResNames = Structure.getResnames()
    ResIndex = [ResDict.get(ResName,UNKOWN_RES) for ResName in ResNames]
    return torch.tensor(ResIndex,dtype=torch.int64)

def break_molecules_by_rotatable_bonds(mol):
    patt = Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')
    rotatable_bonds = mol.GetSubstructMatches(patt)
    bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in rotatable_bonds]
    mol_broken = Chem.rdmolops.FragmentOnBonds(mol, bs)
    frags = Chem.rdmolops.GetMolFrags(mol_broken)
    return frags,rotatable_bonds

def get_molecule_mass_center(mol):
    from rdkit.Chem import Descriptors
    numatoms = mol.GetNumAtoms()
    pos = mol.GetConformer().GetPositions()
    atoms = [atom for atom in mol.GetAtoms()]
    # get center of mass
    mass = Descriptors.MolWt(mol)
    center_of_mass = np.array(np.sum(atoms[i].GetMass() * pos[i] for i in range(numatoms))) / mass
    return center_of_mass

def get_frag_geo_center(mol, frag):
    pos = mol.GetConformer().GetPositions()
    center_of_geo = np.array(np.sum(pos[i] for i in frag))/len(frag)
    return center_of_geo

def get_center_frag(mol, frags):
    mass_center = get_molecule_mass_center(mol)
    frags_geo_center = np.array([get_frag_geo_center(mol, frag) for frag in frags])
    center_distance = (frags_geo_center - mass_center) ** 2
    return np.min(center_distance)

def get_frag_neighbors(mol, frags, rotatable_bonds):
    numatoms = mol.GetNumAtoms()
    frag2neighbour,frag2atom,neighbour2bond,atom2bond,bond2frag = {},{},{},{},{}

    for i in range(len(numatoms)):
        atom2bond[i] = []

    for b_index,rb in enumerate(rotatable_bonds):
        x, y = rb
        atom2bond[x].append(b_index)
        atom2bond[y].append(b_index)
        bond2frag[b_index] = []

    for f_index,frag in enumerate(frags):
        frag2neighbour[f_index] = []
        frag2atom[f_index] = []
        neighbour2bond[f_index] = []
        for atom in frag:
            if len(atom2bond[atom]) != 0:
                for bond in atom2bond[atom]:
                    bond2frag[bond].append(frag)
                    x, y = rotatable_bonds[bond]
                    if x == atom:
                        frag2atom[f_index].append(y)
                    else:
                        frag2atom[f_index].append(x)

    for bond in bond2frag.keys():
        x,y = bond2frag[bond]
        frag2neighbour[x].append(y)
        neighbour2bond[x].append(bond)
        frag2neighbour[y].append(x)
        neighbour2bond[y].append(bond)

    return frag2neighbour,frag2atom,neighbour2bond

def bfs(frag_neighbors,neighbour2bond,center_frag_index):
    bfs_rank = [center_frag_index]
    bfs_bond_rank = []
    visit = [0 for i in range(len(frag_neighbors))]
    left,right = 0,1
    while left < right:
        cur_frag_index = bfs_rank[left]
        for index,neighbor_frag_index in enumerate(frag_neighbors[cur_frag_index]):
            if not visit[neighbor_frag_index]:
                visit[neighbor_frag_index] = 1
                bfs_rank.append(neighbor_frag_index)
                bfs_bond_rank.append(neighbour2bond[cur_frag_index][index])
                right += 1
        left += 1
    return bfs_rank

def get_bfs_generate_rank(mol, center_frag_index, frags, rotatable_bonds):
    frag_neighbors,frag2atom,neighbour2bond = get_frag_neighbors(mol, frags, rotatable_bonds)
    bfs_rank = bfs(frag_neighbors,neighbour2bond,center_frag_index)
    assert bfs_rank==len(frags)
    return bfs_rank

def get_molecule_tree_by_rotatable_bonds(molecule_path):
    mol = Chem.MolFromMol2File(molecule_path, sanitize=False, removeHs=False)
    frags, rotatable_bonds = break_molecules_by_rotatable_bonds(mol)
    center_frag_index = get_center_frag(mol, frags)
    moltree = get_bfs_generate_rank(mol, center_frag_index, frags, rotatable_bonds)
    return moltree

def get_rdkit_coords(mol, sanitize=True, remove_hs=True, maxIters=200):
    ps = AllChem.ETKDGv2()
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=maxIters, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=maxIters, confId=0)

    if remove_hs:
        mol = Chem.RemoveHs(mol, sanitize=sanitize)

    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    # return torch.tensor(lig_coords, dtype=torch.float32)
    return lig_coords


def read_rdmol_v2(dataset_path, name):

    lig_path_mol2 = os.path.join(dataset_path, name, f'{name}_ligand.mol2')
    lig_path_sdf = os.path.join(dataset_path, name, f'{name}_ligand.sdf')
    m_lig = read_rdmol(lig_path_sdf, sanitize=True, remove_hs=True)
    if m_lig == None:  # read mol2 file if sdf file cannot be sanitized
        m_lig = read_rdmol(lig_path_mol2, sanitize=True, remove_hs=True)
    return m_lig

def read_rdmol(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol

def random_rotation_translation(translation_distance=5.0):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt( np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_distance)
    t = t * length
    return torch.from_numpy(rotation_matrix.astype(np.float32)), torch.from_numpy(t.astype(np.float32))