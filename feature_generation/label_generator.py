import numpy as np
from time import time
import gzip
import warnings
import pickle

warnings.filterwarnings("ignore")
from config import DefaultConfig

from Bio.PDB import *

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import SeqUtils

protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1
ppb = PPBuilder()
parser = PDBParser()
configs = DefaultConfig()

THIRD_ATOM = 'O'

def residue_min_distance(res1, res2):
    distance = []
    for atom1 in res1:
        if atom1.get_name() == 'H':
            continue
        for atom2 in res2:
            if atom2.get_name() == 'H':
                continue

            d = abs(atom1 - atom2)
            if d <= 6.0:
                return d
            distance += [d]
    return min(distance)

def get_protein_residues_for_masif (save_dir, protein_name, protein_type):
    complex_name, ligand, receptor = protein_name.split('_')
    if protein_type == 'l':
        chains = ligand 
    else: 
        chains = receptor
    
    residues = []
    for chain in chains:
        path = save_dir + '/pdb_files/' + complex_name + '_' + chain + '.pdb'
        structure = parser.get_structure (complex_name, path)
        model = structure[0]

        chain_residues = model[chain].get_residues()

        for res in chain_residues:
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                continue
            else:
                residues += [res]
    return residues

def get_protein_residues_for_dbd5_dockground (save_dir, protein_name, protein_type, binding_type):
    try:
        path = save_dir + '/pdb_files/' + protein_name + '_' + protein_type + '_' + binding_type + '.pdb'
        structure = parser.get_structure(protein_name, path)
    except:
        path = save_dir + '/pdb_files/' + protein_name.lower() + '_' + protein_type + '_' + binding_type + '.pdb'
        structure = parser.get_structure(protein_name, path)

    model = structure[0] 
    residues = []
    chains = list(model.child_dict.keys())

    for chain_id in chains:
        for res in model[chain_id].get_residues():
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                continue
            else:
                residues += [res]
    return residues

def generate_labels(input_dir, dataset_name, binding_type='u'):

    t0 = time()
    save_dir = input_dir + '/' + dataset_name
    
    with open(save_dir + '/proteins.txt', 'r') as f:
        protein_list = [x.strip() for x in f.readlines()]

    labels = {}

    for idx, protein_name in enumerate(protein_list):
        if dataset_name == 'masif':
            ligand_residues = get_protein_residues_for_masif (save_dir, protein_name, 'l')
            receptor_residues = get_protein_residues_for_masif (save_dir, protein_name, 'r')
        else:
            ligand_residues = get_protein_residues_for_dbd5_dockground (save_dir, protein_name, 'l', binding_type)
            receptor_residues = get_protein_residues_for_dbd5_dockground (save_dir, protein_name, 'r', binding_type)
        
        labels[protein_name] = []

        for i, res1 in enumerate(ligand_residues):
            for j, res2 in enumerate(receptor_residues):
                d = residue_min_distance(res1, res2)
                if d <= 6.0:
                    labels[protein_name] += [[i, j, 1]]
                else:
                    labels[protein_name] += [[i, j, -1]]
        
        x = np.array(labels[protein_name])
        pos = x[x[:, 2] == 1]
        neg = x[x[:, 2] == -1]
        np.random.shuffle(neg)
        neg = neg[:len(pos)*10]
        labels[protein_name] = np.concatenate([pos, neg])
        np.random.shuffle(labels[protein_name])
        
        # saving intermediate steps so we can restart if needed
        if idx > 0 and idx % 100 == 0:
            pickle.dump(labels, gzip.open(save_dir + "/labels_{}.pkl.gz".format(idx), "wb"))

    pickle.dump(labels, gzip.open(save_dir + '/all_labels.pkl.gz', 'wb'))
    print('Total time for label generation:', time() - t0)