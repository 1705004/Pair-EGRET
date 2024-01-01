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
    def __init__(self):
        super(dataSet, self).__init__()
        self.config = DefaultConfig()
        self.SIGMA = 18.0

        self.neighbourhood_size = 21
        self.EDGE_FEATURES_ABSOLUTE_VALUED = False
        self.STANDARDIZE_EDGE_FEATURES = True
        self.STANDARDIZE_NODE_FEATURES = True
        save_dir = 'inputs/' + self.config.dataset_name

        with open(save_dir + "/proteins.txt", "r") as file:
            self.protein_name_list = [x.strip() for x in file.readlines()]
        
        self.ligand_name_list = self.protein_name_list
        self.receptor_name_list = self.protein_name_list

        with gzip.open(save_dir + '/l_ppisp_dist_matrix_map.pkl.gz', 'rb') as f:
            self.ligand_dist_matrix = pickle.load(f)

        with gzip.open(save_dir + '/r_ppisp_dist_matrix_map.pkl.gz', 'rb') as f:
            self.receptor_dist_matrix = pickle.load(f)

        with gzip.open(save_dir + '/l_ppisp_angle_matrix_map.pkl.gz', 'rb') as f:
            self.ligand_angle_matrix = pickle.load(f)

        with gzip.open(save_dir + '/r_ppisp_angle_matrix_map.pkl.gz', 'rb') as f:
            self.receptor_angle_matrix = pickle.load(f)

        with gzip.open(save_dir + '/all_labels.pkl.gz', 'rb') as f:
            self.all_labels = pickle.load(f)

        with gzip.open(save_dir + '/ProtTrans_Bert_all_Receptor_features.pkl.gz', 'rb') as f:
            self.receptor_bert_features = pickle.load(f)['ProtTrans_Bert_all_Receptor_features']

        with gzip.open(save_dir + '/ProtTrans_Bert_all_Ligand_features.pkl.gz', 'rb') as f:
            self.ligand_bert_features = pickle.load(f)['ProtTrans_Bert_all_Ligand_features']

        with gzip.open(save_dir + '/ProtXLNet_all_Receptor_features.pkl.gz', 'rb') as f:
            self.receptor_xlnet_features = pickle.load(f)['ProtXLNet_all_Receptor_features']

        with gzip.open(save_dir + '/ProtXLNet_all_Ligand_features.pkl.gz', 'rb') as f:
            self.ligand_xlnet_features = pickle.load(f)['ProtXLNet_all_Ligand_features']

        with gzip.open(save_dir + '/l_phychem_features.pkl.gz', 'rb') as f:
            self.ligand_phychem_features = pickle.load(f)

        with gzip.open(save_dir + '/r_phychem_features.pkl.gz', 'rb') as f:
            self.receptor_phychem_features = pickle.load(f)

        with gzip.open(save_dir + '/l_hydrophobicity_map.pkl.gz', 'rb') as f:
            self.ligand_hydrophobicity = pickle.load(f)

        with gzip.open(save_dir + '/r_hydrophobicity_map.pkl.gz', 'rb') as f:
            self.receptor_hydrophobicity = pickle.load(f)

        with gzip.open(save_dir + '/l_residue_accessibility.pkl.gz', 'rb') as f:
            self.ligand_residue_accessibility = pickle.load(f)

        with gzip.open(save_dir + '/r_residue_accessibility.pkl.gz', 'rb') as f:
            self.receptor_residue_accessibility = pickle.load(f)

        self.graph_data = {}
        self.max_seq_len = self.config.max_sequence_length

        for i in range(len(self.protein_name_list)):
            p = self.protein_name_list[i]
            l = self.ligand_name_list[i]
            r = self.receptor_name_list[i]

            label_mask = (self.all_labels[p][:, 0] >= 0) & (self.all_labels[p][:, 0] < self.max_seq_len) & \
                         (self.all_labels[p][:, 1] >= 0) & (self.all_labels[p][:, 1] < self.max_seq_len)
            new_labels = self.all_labels[p][label_mask]
            pos = new_labels[new_labels[:, 2] == 1]
            if len(pos) == 0:
                continue
            max_seq_labels = self.all_labels[p][label_mask]

            self.graph_data[p] = {
                "l_bert_features": self.ligand_bert_features[l],
                "r_bert_features": self.receptor_bert_features[r],
                "l_xlnet_features": self.ligand_xlnet_features[l],
                "r_xlnet_features": self.receptor_xlnet_features[r],
                "l_hydrophobicity": self.ligand_hydrophobicity[l],
                "r_hydrophobicity": self.receptor_hydrophobicity[r],
                "l_residue_accessibility": self.ligand_residue_accessibility[l],
                "r_residue_accessibility": self.receptor_residue_accessibility[r],
                "l_phychem_features": self.ligand_phychem_features[l],
                "r_phychem_features": self.receptor_phychem_features[r],
                'label': max_seq_labels
            }

        self.protein_name_list = list(self.graph_data.keys())
        self.ligand_name_list = self.protein_name_list
        self.receptor_name_list = self.protein_name_list
        self.protein_list_len = len(self.protein_name_list)

        self.generate_all_graphs(prot='r')
        self.generate_all_graphs(prot='l')
        print('All graphs generated:')

        all_protein = tuple((list(self.graph_data.keys()), list(self.graph_data.values())))
        self.train_proteins = tuple((all_protein[0][:self.config.train_size + self.config.val_size],
                                     all_protein[1][:self.config.train_size + self.config.val_size]))
        self.test_proteins = tuple((all_protein[0][self.config.train_size+self.config.val_size:], all_protein[1][self.config.train_size+self.config.val_size:]))

        pickle.dump(self.train_proteins, gzip.open(save_dir + '/train.pkl.gz', 'wb'))
        pickle.dump(self.test_proteins, gzip.open(save_dir + '/test.pkl.gz', 'wb'))

        print('total train', len(self.train_proteins[0]), 'total test', len(self.test_proteins[0]))

        bert_mean, bert_std = self.generate_node_mean(self.receptor_bert_features, self.ligand_bert_features, 1024)
        xlnet_mean, xlnet_std = self.generate_node_mean(self.receptor_xlnet_features, self.ligand_xlnet_features, 1024)
        phychem_mean, phychem_std = self.generate_node_mean(self.receptor_phychem_features,
                                                            self.ligand_phychem_features, 14)
        hydrophobicity_mean, hydrophobicity_std = self.generate_node_mean(self.receptor_hydrophobicity,
                                                                          self.ligand_hydrophobicity, 1)
        residue_accessibility_mean, residue_accessibility_std = self.generate_node_mean(
            self.receptor_residue_accessibility, self.ligand_residue_accessibility, 2)
        edge_mean, edge_std = self.generate_edge_mean()

        self.mean_std = {
            'bert_mean': bert_mean,
            'bert_std': bert_std,
            'xlnet_mean': xlnet_mean,
            'xlnet_std': xlnet_std,
            'phychem_mean': phychem_mean,
            'phychem_std': phychem_std,
            'hydrophobicity_mean': hydrophobicity_mean,
            'hydrophobicity_std': hydrophobicity_std,
            'residue_accessibility_mean': residue_accessibility_mean,
            'residue_accessibility_std': residue_accessibility_std,
            'edge_mean': edge_mean,
            'edge_std': edge_std
        }
        pickle.dump(self.mean_std, gzip.open(save_dir + '/node_and_edge_mean_std.pkl.gz', 'wb'))

    def __len__(self):
        return self.protein_list_len

    def generate_all_graphs(self, prot):

        for i, protein in enumerate(self.protein_name_list):
            protein = self.protein_name_list[i]
            l = self.ligand_name_list[i]
            r = self.receptor_name_list[i]

            print('Generating graphs for', protein, prot)
            if prot == 'l':
                neighborhood_indices = self.ligand_dist_matrix[l] \
                                           [:self.max_seq_len, :self.max_seq_len, 0].argsort()[:,
                                       1:self.neighbourhood_size]

                self.graph_data[protein][prot + '_hood_indices'] = neighborhood_indices

                if neighborhood_indices.max() > self.max_seq_len - 1 or neighborhood_indices.min() < 0:
                    print(prot + '_neighbourhood_indices value error')
                    print(neighborhood_indices.max(), neighborhood_indices.min())
                    raise

                dist = self.ligand_dist_matrix[l][:self.max_seq_len, :self.max_seq_len, 0]
                angle = self.ligand_angle_matrix[l][:self.max_seq_len, :self.max_seq_len]

                dist = np.array([dist[i, neighborhood_indices[i]] for i in range(dist.shape[0])])
                angle = np.array([angle[i, neighborhood_indices[i]] for i in range(angle.shape[0])])

                # pass through a gaussian function : f(x) = e^(-x^2 / sigma^2)
                # sigma = 18 (from pipcgn)

                dist = np.e ** (-np.square(dist) / self.SIGMA ** 2)
                edge_feat = np.array([dist, angle])
                edge_feat = np.transpose(edge_feat, (1, 2, 0))

                self.graph_data[protein][prot + '_edge'] = edge_feat

            else:
                neighborhood_indices = self.receptor_dist_matrix[r] \
                                           [:self.max_seq_len, :self.max_seq_len, 0].argsort()[:,
                                       1:self.neighbourhood_size]

                self.graph_data[protein][prot + '_hood_indices'] = neighborhood_indices

                if neighborhood_indices.max() > self.max_seq_len - 1 or neighborhood_indices.min() < 0:
                    print(prot + '_neighbourhood_indices value error')
                    print(neighborhood_indices.max(), neighborhood_indices.min())
                    raise

                dist = self.receptor_dist_matrix[r][:self.max_seq_len, :self.max_seq_len, 0]
                angle = self.receptor_angle_matrix[r][:self.max_seq_len, :self.max_seq_len]

                dist = np.array([dist[i, neighborhood_indices[i]] for i in range(dist.shape[0])])
                angle = np.array([angle[i, neighborhood_indices[i]] for i in range(angle.shape[0])])

                # pass through a gaussian function : f(x) = e^(-x^2 / sigma^2)
                # sigma = 18 (from pipcgn)

                dist = np.e ** (-np.square(dist) / self.SIGMA ** 2)
                edge_feat = np.array([dist, angle])
                edge_feat = np.transpose(edge_feat, (1, 2, 0))
                self.graph_data[protein][prot + '_edge'] = edge_feat

        return


    def generate_node_mean(self, receptor_features, ligand_features, dimension=1024):

        n = 0
        mean = np.zeros([dimension])
        std = np.zeros([dimension])
        for k in receptor_features:
            mean += sum(receptor_features[k])
            n += receptor_features[k].shape[0]
        for k in ligand_features:
            mean += sum(ligand_features[k])
            n += ligand_features[k].shape[0]
        mean /= n

        for k in receptor_features:
            temp = np.square(receptor_features[k] - mean)
            std += sum(temp)
        for k in ligand_features:
            temp = np.square(ligand_features[k] - mean)
            std += sum(temp)

        std = np.sqrt(std / n)

        return mean, std

    def generate_edge_mean(self):

        edge_features = self.train_proteins[1] + self.test_proteins[1]
        r = np.vstack([edge_features[k]['r_edge'] for k in range(len(edge_features))])
        l = np.vstack([edge_features[k]['l_edge'] for k in range(len(edge_features))])
        lr = np.vstack([l, r])
        dimension = lr.shape[2]

        mean = np.zeros([dimension])
        std = np.zeros([dimension])

        for i in range(dimension):
            mean[i] = np.mean(lr[:, :, i])
            std[i] = np.std(lr[:, :, i])

        return mean, std


if __name__ == '__main__':
    data = dataSet()