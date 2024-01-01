from time import time

import torch
from transformers import XLNetModel, XLNetTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import gzip
import pickle


def generate_protxlnet_features(input_dir, dataset_name, protein_type='l', binding_type='u'):
    t0 = time()
    save_dir = input_dir + '/' + dataset_name
    
    downloadFolderPath = input_dir + '/ProtXLNet_model/'

    modelFolderPath = downloadFolderPath
    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
    vocabFilePath = os.path.join(modelFolderPath, 'vocab-spiece.model')

    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    if not os.path.exists(modelFilePath):
        tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False)
        xlnet_mem_len = 512
        model = XLNetModel.from_pretrained("Rostlab/prot_xlnet", mem_len=xlnet_mem_len)
        model.save_pretrained(modelFolderPath)
    else:
        tokenizer = XLNetTokenizer(vocabFilePath, do_lower_case=False)
        model = XLNetModel.from_pretrained(modelFolderPath)

    if not os.path.exists(vocabFilePath):
        tokenizer.save_vocabulary(modelFolderPath, 'vocab')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()

    sequences = []

    with open(save_dir + '/proteins.txt', 'r') as f:
        protein_list = [x.strip() for x in f.readlines()]

    for protein in protein_list:
        seq = open(save_dir + '/fasta_files/' + protein + '_' + protein_type + '_' + binding_type + '.fasta','r').readlines()
        for i in range(1, len(seq), 2):
            sequences += [seq[i].strip()]

    sequences_Example = [' '.join(list(seq)) for seq in sequences]
    sequences_Example = [re.sub(r"[-UZOB]", "X", sequence) for sequence in sequences_Example]
    print("Sequence example len=", len(sequences_Example))
    all_protein_features = {}

    for i, seq in enumerate(sequences_Example):
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len - 1]
            features.append(seq_emd)

        all_protein_features[protein_list[i]] = features[0]

    if protein_type == 'l':
        output = 'ProtXLNet_all_Ligand_features'
    else:
        output = 'ProtXLNet_all_Receptor_features'

    pickle.dump({output: all_protein_features}, gzip.open(save_dir + '/' + output + '.pkl.gz', 'wb'))

    print('Total time spent generating ProtXLNet featuers:', time() - t0)
