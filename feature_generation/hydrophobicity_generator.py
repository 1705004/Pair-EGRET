
import numpy as np
from time import time
import gzip
import warnings
import pickle

warnings.filterwarnings("ignore")

from Bio.PDB import *
from config import DefaultConfig


from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import SeqUtils
import json

configs = DefaultConfig()
parser = PDBParser()
protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1
ppb = PPBuilder()

def get_protein_residue_list_for_masif (save_dir, protein_name, protein_type):
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

def get_protein_residue_list_for_dbd5_dockground (save_dir, protein_name, protein_type, binding_type):
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


def generate_hydrophobicity(input_dir, dataset_name, protein_type='l', binding_type='u'):

    t0 = time()
    save_dir = input_dir + '/' + dataset_name
    
    with open(input_dir + '/hydro.json') as file:
        hydro_dict = json.load(file)
    hydrophobicity_map = {}  # key: idx, value: {'protein_id':<complexName>_<chains>, 'hydrophobicity':hydrophobicity_list}

    with open(save_dir + '/proteins.txt', 'r') as f:
        protein_list = [x.strip() for x in f.readlines()]

    for protein_name in protein_list:
        if dataset_name == 'masif':
            residues = get_protein_residue_list_for_masif (save_dir, protein_name, protein_type)
        else:
            residues = get_protein_residue_list_for_dbd5_dockground (save_dir, protein_name, protein_type, binding_type)
        for res in residues:
            hydrophobicities += [hydro_dict[res.get_resname()]]

        hydrophobicity_map[protein_name] = np.array(hydrophobicities).reshape([-1, 1])
        
    pickle.dump(hydrophobicity_map, gzip.open(save_dir + '/' + protein_type + '_hydrophobicity_map.pkl.gz', 'wb'))
    print('Total time for hydrophobicity generation:', time() - t0)