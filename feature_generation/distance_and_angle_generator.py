
import numpy as np
from time import time
import gzip
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

from Bio.PDB import *

from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio import SeqUtils

from config import DefaultConfig

configs = DefaultConfig()
protein_letters_3to1 = SeqUtils.IUPACData.protein_letters_3to1
ppb = PPBuilder()
parser = PDBParser()

THIRD_ATOM = 'O'

def residue_distance(res1, res2):
    distance = []
    cnt = 0
    for atom1 in res1:
        for atom2 in res2:
            distance += [abs(atom1 - atom2)]
            cnt += 1
    distance = np.array(distance)
    dist_mean = distance.mean()
    dist_std = distance.std()
    if 'CA' in res1 and 'CA' in res2:
        dist_ca = abs(res1['CA'] - res2['CA'])
    else:
        dist_ca = dist_mean
    return dist_mean, dist_std, dist_ca


def residue_relative_angle(res1, res2):
    if 'CA' in res1 and THIRD_ATOM in res1 and 'C' in res1:
        v1 = res1['CA'].get_vector().get_array()
        v2 = res1[THIRD_ATOM].get_vector().get_array()
        v3 = res1['C'].get_vector().get_array()
        normal1 = np.cross((v2 - v1), (v3 - v1))
    else:
        k = list(res1)
        if len(k) >= 1:
            normal1 = k[0].get_vector().get_array()
        else:
            raise
    normal1 = normal1 / np.linalg.norm(normal1)

    if 'CA' in res2 and THIRD_ATOM in res2 and 'C' in res2:
        v1 = res2['CA'].get_vector().get_array()
        v2 = res2[THIRD_ATOM].get_vector().get_array()
        v3 = res2['C'].get_vector().get_array()
        normal2 = np.cross((v2 - v1), (v3 - v1))
    else:
        k = list(res2)
        if len(k) >= 1:
            normal2 = k[0].get_vector().get_array()
        else:
            raise
    normal2 = normal2 / np.linalg.norm(normal2)

    return np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))
    
def get_dist_and_angle_matrix(residues):
    size = len(residues)
    dist_mat = np.zeros([size, size, 3])
    angle_mat = np.zeros([size, size])
    for i in range(size):
        for j in range(i + 1, size):
            dist_mean, dist_std, dist_ca = residue_distance(residues[i], residues[j])
            angle = residue_relative_angle(residues[i], residues[j])

            dist_mat[i, j, 0] = dist_mean
            dist_mat[i, j, 1] = dist_std
            dist_mat[i, j, 2] = dist_ca

            dist_mat[j, i, 0] = dist_mean
            dist_mat[j, i, 1] = dist_std
            dist_mat[j, i, 2] = dist_ca

            angle_mat[i, j] = angle
            angle_mat[j, i] = angle

    return dist_mat, angle_mat

def get_pep_seq_for_masif (save_dir, protein_name, protein_type):
    complex_name, ligand, receptor = protein_name.split('_')
    if protein_type == 'l':
        chains = ligand 
    else: 
        chains = receptor
    pep_seq_from_res_list = ''
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
                pep_seq_from_res_list += protein_letters_3to1[res_name[0] + res_name[1:].lower()]

    return pep_seq_from_res_list, residues

def get_pep_seq_for_dbd5_dockground (save_dir, protein_name, protein_type, binding_type):
    try:
        path = save_dir + '/pdb_files/' + protein_name + '_' + protein_type + '_' + binding_type + '.pdb'
        structure = parser.get_structure(protein_name, path)
    except:
        path = save_dir + '/pdb_files/' + protein_name.lower() + '_' + protein_type + '_' + binding_type + '.pdb'
        structure = parser.get_structure(protein_name, path)

    model = structure[0] 
    residues = []
    chains = list(model.child_dict.keys())
    pep_seq_from_res_list = ''

    for chain_id in chains:
        for res in model[chain_id].get_residues():
            res_name = res.get_resname()
            if res_name[0] + res_name[1:].lower() not in protein_letters_3to1:
                continue
            else:
                residues += [res]
                pep_seq_from_res_list += protein_letters_3to1[res_name[0] + res_name[1:].lower()]
     
    return pep_seq_from_res_list, residues

def generate_distance_and_angle_matrix(input_dir, dataset_name, protein_type='l', binding_type='u'):
    t0 = time()
    save_dir = input_dir + '/' + dataset_name
    
    dist_matrix_map = {}  # key: complex_name, value: {'protein_id':<complexName>_<chains>, 'dist_matrix':dist_matrix}
    angle_matrix_map = {}  # key: complex_name, value: {'protein_id':<complexName>_<chains>, 'angle_matrix':angle_matrix}

    with open(save_dir + '/proteins.txt', 'r') as f:
        protein_list = [x.strip() for x in f.readlines()]

    if not os.path.exists (save_dir + '/fasta_files'):
        os.mkdir (save_dir + '/fasta_files')

    for idx, protein_name in enumerate(protein_list):
        if dataset_name == 'masif':
            pep_seq_from_res_list, residues = get_pep_seq_for_masif (save_dir, protein_name, protein_type)
        else:
            pep_seq_from_res_list, residues = get_pep_seq_for_dbd5_dockground (save_dir, protein_name, protein_type, binding_type)
            
        print(idx, ':', protein_name, len(pep_seq_from_res_list))
        fasta_string = '>{}\n'.format(protein_name+'_' + protein_type + '_' + binding_type)  +  pep_seq_from_res_list

        with open(save_dir + '/fasta_files/{}.fasta'.format(protein_name+ '_' + protein_type + '_' + binding_type), 'w') as f:
            f.write(fasta_string)

        dist_mat, angle_mat = get_dist_and_angle_matrix(residues[:configs.max_residue_seq_length])
        dist_matrix_map[protein_name] = dist_mat
        angle_matrix_map[protein_name] = angle_mat
        
    pickle.dump(dist_matrix_map, gzip.open(save_dir + '/' + protein_type + '_ppisp_dist_matrix_map.pkl.gz', 'wb'))
    pickle.dump(angle_matrix_map, gzip.open(save_dir + '/' + protein_type +'_ppisp_angle_matrix_map.pkl.gz', 'wb'))
    print('Total time for distance and angle matrix generation:', time() - t0)

