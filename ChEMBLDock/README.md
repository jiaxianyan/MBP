# ChEMBL-Dock
## 1.Dataset Introduction
ChEMBL-Dock is a protein-ligand affinity dataset built based on ChEMBL. It consists of protein-ligand binding affinity data from 505,579 experimental measurements in 51,907 bioassays. The dataset includes 2,121 proteins, 276,211 molecules, and 7,963,020 3D binding conformations. (MBP only utilizes a small portion of the data in the paper.)

ChEMBL-Dock can be used for drug design tasks involving protein-ligand interactions, such as protein-ligand binding affinity prediction, molecule generation, and molecular docking.

## 2.Composition and Acquisition of the Dataset
ChEMBL-Dock consists of three parts:	
(MBP only utilizes a portion of the data from the second part, and only the second part is introduced in the article.)
### 2.1 ChEMBL-Dock-InPDBbind-BlindDocking
- a.Proteins in this category can all be found in PDBbind. We used the protein structures provided by PDBbind and performed blind docking using SMINA.
- b.This subset contains affinity data from 391,973 experimental measurements in 38,513 bioassays, involving 1,072 proteins, 222,630 molecules, and 3,506,990 3D binding conformations.
- c.Download link：https://drive.google.com/file/d/1v1DzzpGyniI-q2zU8750SaMNE46Rojal/view?usp=drive_link
### 2.2 ChEMBL-Dock-InPDBbind-SiteSpecificDocking
- a.Proteins in this category can all be found in PDBbind. We utilized the protein structures provided by PDBbind to determine the binding sites and performed site-specific docking using SMINA.
- b.This subset contains affinity data from 391,973 experimental measurements in 38,513 bioassays, involving 1,072 proteins, 222,630 molecules, and 3,439,290 3D binding conformations.
- c.Download link：https://drive.google.com/file/d/1R0Q4M-A03ZWruyMBwpF3kuZVnMSllIY2/view?usp=drive_link
### 2.3 ChEMBL-Dock-NotInPDBbind-BlindDocking
- a.Proteins in this category are not present in PDBbind. We utilized protein structures obtained from PDB and performed blind docking using SMINA.
- b.It includes affinity data from 113,606 experimental measurements in 13,394 bioassays, involving 1,049 proteins, 75,416 molecules, and 1,016,740 binding conformations.
- c.Download link:https://drive.google.com/file/d/1WIxbystjFqC5VElPOMpn5iM1eQxB7qC0/view?usp=drive_link

## 3.ChEMBL-Dock directory format
```
ChEMBL-Dock
|---Uniprot1
|	|----docking_input # docking input structures
|	|	   |---- 0.sdf
|	|	   |---- 1.sdf
|	|	   |---- 2.sdf
|	|	   ...
|	|----docking_output # docking output strucrtures and log files（including SMINA scoring）
|	|	   |---- 0.mol2 
|	|	   |---- 0.log
|	|	   |---- 1.sdf
|	|	   |---- 2.sdf
|	|	   ...
|	|----PDB_valid_chains.pdb # docking target
|	|----PDB_ligand.sdf # Ligands corresponding to the docking targets are provided only if the protein is present in PDBbind.
|	|----PDB_ligand.mol2 # Ligands corresponding to the docking targets are provided only if the protein is present in PDBbind.
|	|----Uniprot1_filter.csv  # Affinity labels and ChEMBL index
|	|----Uniprot1_dock_config.txt # docking config file
|---Uniprot2                               
....

```


