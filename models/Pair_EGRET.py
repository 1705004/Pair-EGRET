import sys
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F

from models import EdgeAggregatedGAT_attention_visual as egret

sys.path.append("../")
from config import DefaultConfig
configs = DefaultConfig()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super (PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout (p=dropout)

        pe = torch.zeros (max_len, d_model)
        position = torch.arange (0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp (torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin (position * div_term)
        pe[:, 1::2] = torch.cos (position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).view([-1, d_model])
        self.register_buffer('pe', pe)

    def forward(self, x, batch_size):
        x = x + self.pe.repeat(batch_size, 1)
        return self.dropout(x)

class EgretPPI(nn.Module):
    def __init__(self):
        super(EgretPPI, self).__init__()
        global configs

        self.conv_encoder = nn.Sequential(
            nn.Conv1d (
                in_channels=configs.num_bert_features + configs.num_phychem_features + configs.num_residue_accessibility_features,
                out_channels=32,
                kernel_size=7, 
                stride=1,
                padding=7 // 2, 
                dilation=1, 
                groups=1,
                bias=True, 
                padding_mode='zeros'
            ),
            nn.LeakyReLU (negative_slope=.01),
            nn.BatchNorm1d (
                num_features=32,
                eps=1e-05, 
                momentum=0.1, 
                affine=True, 
                track_running_stats=True
            ),
            nn.Dropout(.5)
        )
        
        config_dict = egret.config_dict
        config_dict['feat_drop'] = 0.2
        config_dict['edge_feat_drop'] = 0.1
        config_dict['attn_drop'] = 0.2

        self.gat_layers = []
        for _ in range(configs.num_gat_layers):
            self.gat_layers += [
                egret.MultiHeadEGRETLayer(
                    in_dim=32,
                    out_dim=32,
                    edge_dim=2,
                    num_heads=configs.num_heads,
                    use_bias=False,
                    merge='avg',
                    config_dict=config_dict
                )
            ]

    def forward (self, protbert_feature, graph_batch, phychem_data, residue_accessibility_data, max_sequence_length=configs.max_sequence_length):
        merged_features = torch.randn(0)
        if configs.use_bert_data:
            merged_features = torch.cat((merged_features, protbert_feature), axis=2)
        if configs.use_phychem_data:
            merged_features = torch.cat((merged_features, phychem_data), axis=2)
        if configs.use_residue_accessibility_features:
            merged_features = torch.cat((merged_features, residue_accessibility_data), axis=2)

        shapes = merged_features.data.shape
        
        features = merged_features.squeeze(1).permute(0, 2, 1)
        features = self.conv_encoder(features)
        features = features.permute(0, 2, 1).contiguous()
        
        features2 = torch.clone(features)
        for i in range(configs.num_gat_layers):
            features2, head_attn_scores = self.gat_layers[i](graph_batch, features2.view([shapes[0]*max_sequence_length, 32]))
            features2 = features2.view([shapes[0], max_sequence_length, 32])

        features = torch.cat((features2, features), 2).view([shapes[0], max_sequence_length, 32*2])
        return features, head_attn_scores

class PairEgret(nn.Module):
    def __init__(self):
        super(PairEgret, self).__init__()
        global configs
        self.siamese_layer = EgretPPI()
        self.pos_encoder = PositionalEncoding(d_model=64, dropout=0.1, max_len=configs.max_sequence_length)

        self.attn_hidden_size = 64
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.5, batch_first=True)
        self.attention_layer_norm = nn.LayerNorm(64)

        self.dropout = nn.Dropout(configs.dropout)

        self.bn_fc = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.1, affine=True,
                                    track_running_stats=True)
        
        self.pairwise_classifier = nn.Sequential(
            nn.Linear(64*2, 1),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.interface_region_classifier = nn.Sequential(
          nn.Linear(64, 1),
          nn.Sigmoid()
        )

    def forward(self,
                protbert_data_receptor, graph_batch_receptor, phychem_data_receptor, residue_accessibility_data_receptor,
                    protbert_data_ligand, graph_batch_ligand, phychem_data_ligand, residue_accessibility_data_ligand,
                        pair_residue_label):

        features_receptor, head_attn_scores_R = self.siamese_layer (
            protbert_feature=protbert_data_receptor,
            graph_batch=graph_batch_receptor,
            phychem_data=phychem_data_receptor,
            residue_accessibility_data=residue_accessibility_data_receptor
        )
        features_ligand, head_attn_scores_L = self.siamese_layer (
            protbert_feature=protbert_data_ligand,
            graph_batch=graph_batch_ligand,
            phychem_data=phychem_data_ligand,
            residue_accessibility_data=residue_accessibility_data_ligand
        )

        batch_size = features_receptor.shape[0]
        max_seq_len = features_receptor.shape[1]

        features_receptor = features_receptor.view ([-1, 64])
        features_ligand = features_ligand.view ([-1, 64])
        
        features_receptor = self.pos_encoder (features_receptor, batch_size)
        features_ligand = self.pos_encoder (features_ligand, batch_size)

        features_ligand = features_ligand.view ([batch_size, max_seq_len, 64])
        features_receptor = features_receptor.view ([batch_size, max_seq_len, 64])
        
        attended_receptor, head_attn_scores_R = self.attention (features_receptor, features_ligand, features_ligand) 
        attended_ligand, head_attn_scores_L = self.attention (features_ligand, features_receptor, features_receptor) 

        features_receptor = (features_receptor + attended_receptor).view ([-1, 64])
        features_ligand = (features_ligand + attended_ligand).view ([-1, 64])

        features_receptor = self.attention_layer_norm (features_receptor)
        features_ligand = self.attention_layer_norm (features_ligand)

        features_receptor = features_receptor[pair_residue_label[:, 1]] 
        features_ligand = features_ligand[pair_residue_label[:, 0]] 
        
        output_R_L = torch.cat ([features_receptor, features_ligand], axis=1)
        output_L_R = torch.cat ([features_ligand, features_receptor], axis=1)

        output_R_L = self.bn_fc (output_R_L)
        output_L_R = self.bn_fc (output_L_R)

        output_R_L = self.pairwise_classifier (output_R_L).view([-1]) 
        output_L_R = self.pairwise_classifier (output_L_R).view([-1])

        pairwise_output = (output_R_L + output_L_R) / 2.

        interface_output_receptor = self.interface_region_classifier (features_receptor).view([-1])
        interface_output_ligand = self.interface_region_classifier (features_ligand).view([-1])

        return interface_output_receptor, interface_output_ligand, pairwise_output, (head_attn_scores_R, head_attn_scores_L)
