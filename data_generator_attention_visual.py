import os
import pickle
import torch as t
import numpy as np
from torch.utils import data
import gzip
from time import time
from config import DefaultConfig
import torch
import dgl
import threading

class dataSet(data.Dataset):
    def __init__(self, root_dir, data_file_prefix='train'):
        super(dataSet, self).__init__()
        
        self.config = DefaultConfig ()
        self.data_file_prefix = data_file_prefix

        with gzip.open (root_dir+'/inputs/'+self.config.dataset_name+'/{}.pkl.gz'.format(data_file_prefix), 'rb') as f:
            self.dataset = pickle.load (f)
            self.protein_name_list = self.dataset[0]
            self.protein_data = self.dataset[1]            

        with gzip.open (root_dir+'/inputs/'+self.config.dataset_name+'/node_and_edge_mean_std.pkl.gz', 'rb') as f:
            mean_and_std = pickle.load (f)

            if self.config.language_model == 'ProtTrans_Bert':
                self.node_feat_mean = mean_and_std['bert_mean']
                self.node_feat_std = mean_and_std['bert_std']
            else:
                self.node_feat_mean = mean_and_std['xlnet_mean']
                self.node_feat_std = mean_and_std['xlnet_std']

            if self.config.num_phychem_features == 1:
                self.phychem_feat_mean = mean_and_std['hydrophobicity_mean']
                self.phychem_feat_std = mean_and_std['hydrophobicity_std']
            else:
                self.phychem_feat_mean = mean_and_std['phychem_mean']
                self.phychem_feat_std = mean_and_std['phychem_std']

            self.residue_accessibility_mean = mean_and_std['residue_accessibility_mean']
            self.residue_accessibility_std = mean_and_std['residue_accessibility_std']

            self.edge_feat_mean = mean_and_std['edge_mean']
            self.edge_feat_std = mean_and_std['edge_std']

        self.max_seq_len = self.config.max_sequence_length
        self.protein_list_len = len (self.protein_name_list)
        
        self.all_graphs_receptor = self.generate_all_graphs (prot='r')
        self.all_graphs_ligand = self.generate_all_graphs (prot='l')
        
        for index in range (len (self.protein_name_list)):
            label = self.protein_data[index]['label']
            
            pos = label[label[:, 2] == 1]
            neg = label[label[:, 2] != 1]

            if self.data_file_prefix.startswith ('train'):
                np.random.shuffle (neg)
                label = np.vstack ([pos, neg[: self.config.pos_neg_ratio * len(pos)]])
                np.random.shuffle (label)
            self.protein_data[index]['label'] = label

    def __getitem__(self, index):
        
        complex_name = self.protein_name_list[index]
        complex_info = {
            'complex_name': complex_name,
            'complex_idx': index,
        }

        _protBert_feature_receptor_ = self.protein_data[index]['r_bert_features'] if self.config.language_model == 'ProtTrans_Bert' else self.protein_data[index]['r_xlnet_features']
        _protBert_feature_receptor_ = _protBert_feature_receptor_[:self.max_seq_len]
        phychem_features_receptor = self.protein_data[index]['r_hydrophobicity'] if self.config.num_phychem_features == 1 else self.protein_data[index]['r_phychem_features']
        phychem_features_receptor = phychem_features_receptor[:self.max_seq_len]
        residue_accessibility_receptor = self.protein_data[index]['r_residue_accessibility'][:self.max_seq_len]
        
        seq_len = _protBert_feature_receptor_.shape[0]
        complex_info['receptor_seq_length'] = seq_len
        if seq_len < self.max_seq_len:
            temp = np.zeros([self.max_seq_len, _protBert_feature_receptor_.shape[1]])
            temp[:seq_len, :] = _protBert_feature_receptor_
            _protBert_feature_receptor_ = temp

            temp = np.zeros([self.max_seq_len, phychem_features_receptor.shape[1]])
            temp[:seq_len, :] = phychem_features_receptor
            phychem_features_receptor = temp

            temp = np.zeros([self.max_seq_len, residue_accessibility_receptor.shape[1]])
            temp[:seq_len, :] = residue_accessibility_receptor
            residue_accessibility_receptor = temp

        _protBert_feature_receptor_ = _protBert_feature_receptor_[np.newaxis, :, :]
        phychem_features_receptor = phychem_features_receptor[np.newaxis, :, :]
        residue_accessibility_receptor = residue_accessibility_receptor[np.newaxis, :, :]
        G_receptor = self.all_graphs_receptor[index]
        
        
        _protBert_feature_ligand_ = self.protein_data[index]['l_bert_features'] if self.config.language_model == 'ProtTrans_Bert' else self.protein_data[index]['l_xlnet_features']
        _protBert_feature_ligand_ = _protBert_feature_ligand_[:self.max_seq_len]
        phychem_features_ligand = self.protein_data[index]['l_hydrophobicity'] if self.config.num_phychem_features == 1 else self.protein_data[index]['l_phychem_features']
        phychem_features_ligand = phychem_features_ligand[:self.max_seq_len]
        residue_accessibility_ligand = self.protein_data[index]['l_residue_accessibility'][:self.max_seq_len]
        
        seq_len = _protBert_feature_ligand_.shape[0]
        complex_info['ligand_seq_length'] = seq_len
        if seq_len < self.max_seq_len:
            temp = np.zeros([self.max_seq_len, _protBert_feature_ligand_.shape[1]])
            temp[:seq_len, :] = _protBert_feature_ligand_
            _protBert_feature_ligand_ = temp

            temp = np.zeros([self.max_seq_len, phychem_features_ligand.shape[1]])
            temp[:seq_len, :] = phychem_features_ligand
            phychem_features_ligand = temp

            temp = np.zeros([self.max_seq_len, residue_accessibility_ligand.shape[1]])
            temp[:seq_len, :] = residue_accessibility_ligand
            residue_accessibility_ligand = temp

        _protBert_feature_ligand_ = _protBert_feature_ligand_[np.newaxis, :, :]
        phychem_features_ligand = phychem_features_ligand[np.newaxis, :, :]
        residue_accessibility_ligand = residue_accessibility_ligand[np.newaxis, :, :]
        G_ligand = self.all_graphs_ligand[index]
        
        if self.config.STANDARDIZE_NODE_FEATURES:
            _protBert_feature_receptor_ = (_protBert_feature_receptor_ - self.node_feat_mean) / self.node_feat_std
            _protBert_feature_ligand_ = (_protBert_feature_ligand_ - self.node_feat_mean) / self.node_feat_std
            phychem_features_receptor = (phychem_features_receptor - self.phychem_feat_mean) / self.phychem_feat_std
            phychem_features_ligand = (phychem_features_ligand - self.phychem_feat_mean) / self.phychem_feat_std
            residue_accessibility_receptor = (residue_accessibility_receptor - self.residue_accessibility_mean) / self.residue_accessibility_std
            residue_accessibility_ligand = (residue_accessibility_ligand - self.residue_accessibility_mean) / self.residue_accessibility_std

        label = self.protein_data[index]['label']
        label[label == -1] = 0

        positives = label[label[:, 2] == 1]
        pos_ligands = set(positives[:, 0])
        pos_receptors = set(positives[:, 1])

        ligand_labels = np.array([x[0] in pos_ligands  for x in label], dtype=int)
        receptor_labels = np.array([x[1] in pos_receptors  for x in label], dtype=int)
        
        complex_info['total_usable_residues'] = label.shape[0]

        return torch.from_numpy(_protBert_feature_receptor_).type(torch.FloatTensor), \
               G_receptor, \
               torch.from_numpy(phychem_features_receptor).type(torch.FloatTensor), \
               torch.from_numpy(residue_accessibility_receptor).type(torch.FloatTensor), \
               torch.from_numpy(_protBert_feature_ligand_).type(torch.FloatTensor), \
               G_ligand, \
               torch.from_numpy(phychem_features_ligand).type(torch.FloatTensor), \
               torch.from_numpy(residue_accessibility_ligand).type(torch.FloatTensor), \
               complex_info, \
               label, \
               ligand_labels, \
               receptor_labels

    def __len__(self):
        return self.protein_list_len

    def generate_all_graphs(self, prot):
        graph_list = {}
        for id_idx in range(self.protein_list_len):
            G = dgl.DGLGraph()
            G.add_nodes(self.max_seq_len)
            
            neighborhood_indices = self.protein_data[id_idx][prot+'_hood_indices'].reshape([-1,self.config.neighbourhood_size-1])
            edge_feat = self.protein_data[id_idx][prot+'_edge']
            
            if self.config.STANDARDIZE_EDGE_FEATURES:
                edge_feat = (edge_feat - self.edge_feat_mean) / self.edge_feat_std  # standardize features

            self.add_edges_custom(G, neighborhood_indices, edge_feat)
            graph_list[id_idx]= G

        return  graph_list

    def add_edges_custom(self, G, neighborhood_indices, edge_features):
        t1 = time()
        size = min(neighborhood_indices.shape[0], self.max_seq_len)

        src = []
        dst = []
        temp_edge_features = []
        for center in range(size):
            
            # keep only the residues within max_seq_len
            mask = neighborhood_indices[center] < self.max_seq_len
            neighbors = neighborhood_indices[center][mask]
            src += neighbors.tolist()
            dst += [center] * len(neighbors)
            
            # keep only the edges within max_seq_len
            temp_edge_features += edge_features[center][mask].tolist()
            
        if len(src) != len(dst):
            print('source (src) and destination (dst) array should have been of the same length: '
                  'len(src)={} and len(dst)={}'.format(len(src), len(dst)))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.from_numpy((np.array(temp_edge_features)).astype(np.float32))


class GraphCollate(object):
    def __init__(self, loader_name='train'):
        self.loader_name = loader_name

    def __call__(self, samples):
        configs = DefaultConfig()

        protbert_data_receptor, graph_batch_receptor, phychem_feat_receptor, residue_accessibility_receptor, \
        protbert_data_ligand, graph_batch_ligand, phychem_feat_ligand, residue_accessibility_ligand, \
        complex_info_batch, label_batch, ligand_label_batch, receptor_label_batch = map(list, zip(*samples))

        protbert_data_receptor = torch.cat(protbert_data_receptor)
        graph_batch_receptor = dgl.batch(graph_batch_receptor)
        phychem_feat_receptor = torch.cat(phychem_feat_receptor)
        residue_accessibility_receptor = torch.cat(residue_accessibility_receptor)

        protbert_data_ligand = torch.cat(protbert_data_ligand)
        graph_batch_ligand = dgl.batch(graph_batch_ligand)
        phychem_feat_ligand = torch.cat(phychem_feat_ligand)
        residue_accessibility_ligand = torch.cat(residue_accessibility_ligand)

        pair_residue_label = np.zeros([0, 3])
        ligand_label = np.zeros(0)
        receptor_labels = np.zeros(0)
        for i, temp in enumerate(label_batch):

            temp[:, :2] = temp[:, :2] + i * configs.max_sequence_length  # padding all proteins to make the lengths equal
            pair_residue_label = np.concatenate([pair_residue_label, temp])
            ligand_label = np.concatenate([ligand_label, ligand_label_batch[i]])
            receptor_labels = np.concatenate([receptor_labels, receptor_label_batch[i]])

        return protbert_data_receptor, graph_batch_receptor, phychem_feat_receptor, residue_accessibility_receptor, \
               protbert_data_ligand, graph_batch_ligand, phychem_feat_ligand, residue_accessibility_ligand, \
               complex_info_batch, torch.from_numpy(pair_residue_label).type(torch.LongTensor), \
               torch.from_numpy(ligand_label).type(torch.LongTensor), torch.from_numpy(receptor_labels).type(torch.LongTensor)

