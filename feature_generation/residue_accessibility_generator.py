
import numpy as np
from time import time
import gzip
import warnings
import pickle

warnings.filterwarnings("ignore")

from Bio.PDB import *
from config import DefaultConfig


from Bio import SeqUtils
from Bio.PDB import ShrakeRupley
from Bio.PDB.ResidueDepth import ResidueDepth


configs = DefaultConfig()
parser = PDBParser()
protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1
ks = list(protein_letters_3to1.keys())
# for k in ks:
#     protein_letters_3to1[k.upper()] = protein_letters_3to1[k]
# protein_letters_3to1["MSE"] = "M"
# protein_letters_3to1["Mse"] = "M"

ppb = PPBuilder()


MaxASA = {
    'A': 113.,
    'R': 241.,
    'N': 158.,
    'D': 151.,  # *
    'C': 140.,
    'E': 183.,  # *
    'Q': 189.,
    'G': 85.,
    'H': 194.,
    'I': 182.,
    'L': 180.,
    'K': 211.,
    'M': 204.,
    'F': 218.,
    'P': 143.,
    'S': 122.,
    'T': 146.,
    'W': 259.,
    'Y': 229.,
    'V': 160.,
}


def get_rasa(residues, model):
    
    size = len(residues)
    sr = ShrakeRupley(n_points=1000)
    sr.compute(model, level="R")
    
    rasas = np.zeros([size])
    
    for i, res in enumerate(residues):
        res_char = protein_letters_3to1[res.get_resname()[0] + res.get_resname()[1:].lower()]
        rasas[i] = residues[i].sasa / MaxASA[res_char]
    
    return rasas


def get_residue_depth(residues, res_chains, model):

    size = len(residues)

    residue_depths = np.zeros([size])
    depths = ResidueDepth(model)

    for i, res in enumerate(residues):
        residue_depths[i] = depths[res_chains[i], res.get_id()][0]

    return residue_depths

def get_protein_residue_acc_for_masif (save_dir, protein_name, protein_type):
    complex_name, ligand, receptor = protein_name.split('_')
    if protein_type == 'l':
        chains = ligand 
    else: 
        chains = receptor
    
    rasas = []
    residue_depths = []
        
    for chain in chains:
        path = save_dir + '/pdb_files/' + complex_name + '_' + chain + '.pdb'
        structure = parser.get_structure (complex_name, path)
        model = structure[0]

        chain_residues = model[chain].get_residues()
        residues = []

        for res in chain_residues:
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                continue
            else:
                residues += [res]
                
        rasas += get_rasa(residues, model)
        residue_depths += get_residue_depth(residues, chain, model)

    return rasas, residue_depths

def get_protein_residue_acc_for_dbd5_dockground (save_dir, protein_name, protein_type, binding_type):
    try:
        path = save_dir + '/pdb_files/' + protein_name + '_' + protein_type + '_' + binding_type + '.pdb'
        structure = parser.get_structure(protein_name, path)
    except:
        path = save_dir + '/pdb_files/' + protein_name.lower() + '_' + protein_type + '_' + binding_type + '.pdb'
        structure = parser.get_structure(protein_name, path)

    model = structure[0] 
    residues = []
    res_chains = []
    chains = list(model.child_dict.keys())

    for chain_id in chains:
        for res in model[chain_id].get_residues():
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                continue
            else:
                residues += [res]
                res_chains += [chain_id]

    rasas = get_rasa(residues, model)
    residue_depths = get_residue_depth(residues, res_chains, model)

    return rasas, residue_depths

def generate_residue_accessibility(input_dir, dataset_name, protein_type='l', binding_type='u'):

    t0 = time()
    save_dir = input_dir + '/' + dataset_name
    
    residue_accessibility = {}  
    with open(save_dir + '/proteins.txt', 'r') as f:
        protein_list = [x.strip() for x in f.readlines()]

    for protein_name in protein_list:

        if dataset_name == 'masif':
            rasas, residue_depths = get_protein_residue_acc_for_masif (save_dir, protein_name, protein_type)
        else:
            rasas, residue_depths = get_protein_residue_acc_for_dbd5_dockground (save_dir, protein_name, protein_type, binding_type)

        residue_accessibility[protein_name] = np.concatenate([rasas.reshape([-1,1]), residue_depths.reshape([-1,1])], axis=1)
        
    pickle.dump(residue_accessibility, gzip.open(save_dir + '/' + protein_type + '_residue_accessibility.pkl.gz', 'wb'))
    print('Total time for relative surface generation:', time() - t0)