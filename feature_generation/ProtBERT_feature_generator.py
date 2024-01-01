from time import time

import torch
from transformers import BertModel, BertTokenizer
import re
import os
import requests
from tqdm.auto import tqdm
import gzip
import pickle

def generate_protbert_features(input_dir, dataset_name, protein_type='l', binding_type='u'):
    t0=time()
    save_dir = input_dir + '/' + dataset_name
    
    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'

    downloadFolderPath = input_dir+'/ProtBert_model/'

    modelFolderPath = downloadFolderPath

    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')

    configFilePath = os.path.join(modelFolderPath, 'config.json')

    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    def download_file(url, filename):
      response = requests.get(url, stream=True)
      with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                        total=int(response.headers.get('content-length', 0)),
                        desc=filename) as fout:
          for chunk in response.iter_content(chunk_size=4096):
              fout.write(chunk)

    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)

    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)

    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)

    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
    model = BertModel.from_pretrained(modelFolderPath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model = model.eval()

    sequences = []

    with open(save_dir + '/proteins.txt', 'r') as f:
        protein_list = [x.strip() for x in f.readlines()]

    print("total proteins:", len(protein_list))
    for protein in protein_list:
        seq = open(save_dir+'/fasta_files/'+protein+'_'+protein_type+ + '_' + binding_type + '.fasta', 'r').readlines()
        if len(seq) == 0 or len(seq) == 1:
            print(protein, len(seq))
            print(seq)
        for i in range(1, len(seq), 2):
            sequences += [seq[i].strip()]

    print("sequence length:", len(sequences))
    sequences_Example = [' '.join(list(seq)) for seq in sequences]
    sequences_Example = [re.sub(r"[-UZOB]", "X", sequence) for sequence in sequences_Example]
    print("Sequence example length=", len(sequences_Example))
    all_protein_features = {}
    print(len(sequences_Example))
 

    for i, seq in enumerate(sequences_Example):
        ids = tokenizer.batch_encode_plus([seq], add_special_tokens=True, pad_to_max_length=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)[0]
        embedding = embedding.cpu().numpy()
        features = []
        for seq_num in range(len(embedding)):
        
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            features.append(seq_emd)
        all_protein_features[protein_list[i]] = features[0]

    if protein_type == 'l':
        output = 'ProtTrans_Bert_all_Ligand_features'
    else:
        output = 'ProtTrans_Bert_all_Receptor_features'

    pickle.dump({output: all_protein_features},gzip.open(save_dir+'/' + output + '.pkl.gz','wb'))
    print('Total time spent generating ProtBERT features:', time()-t0)
