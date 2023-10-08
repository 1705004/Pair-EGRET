import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EGRETLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, use_bias, config_dict=None):
        super(EGRETLayer, self).__init__()
        
        self.apply_attention = True
        self.transform_edge_for_att_calc = False
        self.apply_attention_on_edge = False
        self.aggregate_edge = False 
        self.edge_dependent_attention = False 
        self.self_loop = False
        self.self_node_transform = False and self.self_loop
        self.activation = None 

        if config_dict is not None:
          self.apply_attention = config_dict['apply_attention']
          self.transform_edge_for_att_calc = config_dict['transform_edge_for_att_calc']
          self.apply_attention_on_edge = config_dict['apply_attention_on_edge']
          self.aggregate_edge = config_dict['aggregate_edge']
          self.edge_dependent_attention = config_dict['edge_dependent_attention']
          self.self_loop = config_dict['self_loop']
          self.self_node_transform = config_dict['self_node_transform'] and self.self_loop
          self.activation = config_dict['activation']
          self.feat_drop = nn.Dropout(config_dict['feat_drop'])
          self.attn_drop = nn.Dropout(config_dict['attn_drop'])
          self.edge_feat_drop = nn.Dropout(config_dict['edge_feat_drop'])
          self.use_batch_norm = config_dict['use_batch_norm']

        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.bn_fc = nn.BatchNorm1d(num_features=out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if self.use_batch_norm else nn.Identity()
        if self.edge_dependent_attention:
          self.attn_fc = nn.Linear(2 * out_dim+edge_dim, 1, bias=use_bias)
        else:
          self.attn_fc = nn.Linear(2 * out_dim, 1, bias=use_bias)
        if self.aggregate_edge:
          self.fc_edge = nn.Linear(edge_dim, out_dim, bias=use_bias)
          self.bn_fc_edge = nn.BatchNorm1d(num_features=out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if self.use_batch_norm else nn.Identity()
        if self.self_node_transform:
          self.fc_self = nn.Linear(in_dim, out_dim, bias=use_bias)
          self.bn_fc_self = nn.BatchNorm1d(num_features=out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if self.use_batch_norm else nn.Identity()
        if self.transform_edge_for_att_calc:
          self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=use_bias)
          self.bn_fc_edge_for_att_calc = nn.BatchNorm1d(num_features=edge_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) if self.use_batch_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.aggregate_edge:
          nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        if self.self_node_transform:
          nn.init.xavier_normal_(self.fc_self.weight, gain=gain)
        if self.transform_edge_for_att_calc:
          nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
          
    def edge_attention(self, edges):
        if self.edge_dependent_attention:
          if self.transform_edge_for_att_calc:
            z2 = torch.cat([edges.src['z'], edges.dst['z'], self.bn_fc_edge_for_att_calc(self.fc_edge_for_att_calc(edges.data['ex']))], dim=1)
          else:
            z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
        else:
          z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)

        if self.aggregate_edge:
            ez = self.bn_fc_edge(self.fc_edge(edges.data['ex']))
            return {'e': F.leaky_relu(a, negative_slope=0.2), 'ez': ez}
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        if self.aggregate_edge:
          return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}
        else:
          return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = None
        if not self.apply_attention:
          h = torch.sum(nodes.mailbox['z'], dim=1)
        else:
          alpha = self.attn_drop(F.softmax(nodes.mailbox['e'], dim=1))
          h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        if self.aggregate_edge:
          if self.apply_attention_on_edge:
            h = h + torch.sum(alpha * nodes.mailbox['ez'], dim=1)
          else:
            h = h + torch.sum(nodes.mailbox['ez'], dim=1)
        return {'h': h, 'alpha':alpha}

    def forward(self, g, nfeatures):
        g = g.local_var()
        nfeatures = self.feat_drop(nfeatures)
        g.edata['ex'] = self.edge_feat_drop(g.edata['ex'])
        g.ndata['z'] = self.bn_fc(self.fc(nfeatures))

        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        
        if self.self_loop:
          if self.self_node_transform:
            g.ndata['h'] = g.ndata['h'] + self.bn_fc_self(self.fc_self(nfeatures))
          else:
            g.ndata['h'] = g.ndata['h'] + nfeatures
        
        if self.activation is not None:
          g.ndata['h'] = self.activation(g.ndata['h'])
        
        return g.ndata.pop('h'), g.ndata.pop('alpha')

class MultiHeadEGRETLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, num_heads, use_bias, merge='cat', config_dict=None):
        super(MultiHeadEGRETLayer, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(EGRETLayer(in_dim, out_dim, edge_dim, use_bias, config_dict=config_dict)) 
        self.merge = merge
        self.fc = nn.Linear(num_heads * out_dim, out_dim, bias=use_bias)

    def forward(self, g, h):
        head_outs_all = [attn_head(g, h) for attn_head in self.heads]
        head_outs = []
        head_attn_scores = []
        for x in head_outs_all:
            head_outs += [x[0]]
            head_attn_scores += [x[1].cpu().detach()]
           
        head_outs = torch.cat(head_outs, dim=1)
        if self.merge == 'cat':
            head_outs = self.fc(head_outs)
            return head_outs, head_attn_scores
        else:
            head_outs = head_outs.reshape((head_outs.shape[0],self.num_heads, -1))
            return torch.mean(head_outs, 1), head_attn_scores

config_dict = {
    'use_batch_norm': False,
    'feat_drop': 0.0,
    'attn_drop': 0.0,
    'edge_feat_drop': 0.0,
    'hidden_dim' : 32,
    'out_dim' : 32, 
    'apply_attention' : True,
    'transform_edge_for_att_calc': True, 
    'apply_attention_on_edge': True, 
    'aggregate_edge' : True, 
    'edge_dependent_attention' : True, 
    'self_loop' : False, 
    'self_node_transform' : True, 
    'activation' :None 
}


