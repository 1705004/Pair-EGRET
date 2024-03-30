from datetime import datetime
from time import time
import random
import argparse
import json

t000 = time()

import os
import gzip, pickle

from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn
import torch.utils.data.sampler as sampler

import dgl

from config import DefaultConfig
from models.Pair_EGRET import PairEgret
import data_generator_attention_visual as data_generator
from data_generator_attention_visual import GraphCollate

from utils import basic_metrics, calculate_median_auroc, set_seed
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

configs = DefaultConfig()
THRESHOLD = 0.9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        xavier_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        xavier_normal_(m.weight.data)
    
def train_epoch(model, loader, optimizer, epoch, all_epochs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    global configs
    global THRESHOLD
    
    model.train()
    end = time()

    for batch_idx, batch_data in enumerate(loader):
        protbert_data_receptor, graph_batch_receptor, phychem_data_receptor, residue_accessibility_receptor, \
        protbert_data_ligand, graph_batch_ligand, phychem_data_ligand, residue_accessibility_ligand, \
        complex_info_batch, pair_residue_label, ligand_label_batch, receptor_label_batch = batch_data

        protbert_data_receptor = torch.autograd.Variable(protbert_data_receptor.to(device).float())
        graph_batch_receptor.edata['ex'] = torch.autograd.Variable(graph_batch_receptor.edata['ex'].to(device).float())
        phychem_data_receptor = torch.autograd.Variable(phychem_data_receptor.to(device).float())
        residue_accessibility_receptor = torch.autograd.Variable(residue_accessibility_receptor.to(device).float())

        protbert_data_ligand = torch.autograd.Variable(protbert_data_ligand.to(device).float())
        graph_batch_ligand.edata['ex'] = torch.autograd.Variable(graph_batch_ligand.edata['ex'].to(device).float())
        phychem_data_ligand = torch.autograd.Variable(phychem_data_ligand.to(device).float())
        residue_accessibility_ligand = torch.autograd.Variable(residue_accessibility_ligand.to(device).float())

        pair_residue_label = torch.autograd.Variable(pair_residue_label.to(device).long())
        ligand_label_batch = torch.autograd.Variable(ligand_label_batch.to(device).long())
        receptor_label_batch = torch.autograd.Variable(receptor_label_batch.to(device).long())
        
        receptor_pred, ligand_pred, pairwise_pred, _ = \
            model (
                protbert_data_receptor,
                graph_batch_receptor,
                phychem_data_receptor,
                residue_accessibility_receptor,
                protbert_data_ligand,
                graph_batch_ligand,
                phychem_data_ligand,
                residue_accessibility_ligand,
                pair_residue_label
            )

        receptor_pred = receptor_pred.view(-1)
        ligand_pred = ligand_pred.view(-1)
        pairwise_pred = pairwise_pred.view(-1)
        label = pair_residue_label[:, 2]

        weight = (configs.pos_neg_ratio / (configs.pos_neg_ratio + 1) * label + 1.0 / (configs.pos_neg_ratio + 1) * (
                    1. - label)).float() if configs.WEIGHTED_LOSS else None

        if configs.TRAIN_FOR_PAIRWISE_PROTEINS:
            ce_loss = torch.nn.functional.binary_cross_entropy(pairwise_pred, label.float(),weight=weight, reduction='none').to(device)
            loss = ce_loss

        if configs.TRAIN_FOR_SINGLE_PROTEINS:
            ce_loss = torch.nn.functional.binary_cross_entropy(receptor_pred, label.float(),weight=weight, reduction='none').to(device)
            loss += ce_loss
            
            ce_loss = torch.nn.functional.binary_cross_entropy(ligand_pred, label.float(),weight=weight, reduction='none').to(device)
            loss += ce_loss

        loss = torch.mean(loss)
        losses.update(loss.item(), configs.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time() - end)
        end = time()

        if batch_idx % configs.print_freq == 0:
            res = '\t'.join([
                '\n',
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg)])
            print(res)
            
            label = pair_residue_label[:, 2].data.cpu().numpy()
            for (output_name, output) in [('receptor_pred', receptor_pred), ('ligand_pred', ligand_pred),
                                          ('pairwise_pred', pairwise_pred)]:
                output = output.data.cpu()
                output = output.numpy()
                auroc, auprc, accuracy, f1, p, r, threshold = basic_metrics(output, label)
                res = '\t'.join([
                    'output_name: ' + output_name,
                    'auroc:%.6f' % (auroc),
                    'auprc:%.6f' % (auprc),
                    'accuracy:%.6f' % (accuracy),
                    'f1:%.6f' % (f1),
                    'p:%.6f' % (p),
                    'r:%.6f' % (r),
                    'threshold:%.6f' % (threshold)])
                print(res)

    return batch_time.avg, losses.avg


def eval_epoch(model, loader, save_attention_scores=False):
    global configs
    global THRESHOLD

    results = []
    complex_infos_ = []

    receptor_pred_all = np.zeros([0])
    ligand_pred_all = np.zeros([0])
    pairwise_pred_all = np.zeros([0])
    label_all = np.zeros([0])
    residue_pairs = np.zeros([0, 2])
    ligand_label_all = np.zeros([0])
    receptor_label_all = np.zeros([0])
    
    model.eval()
    end = time()

    ligand_attention_scores = []
    receptor_attention_scores = []

    with torch.no_grad():
        for _, batch_data in enumerate(loader):
            protbert_data_receptor, graph_batch_receptor, phychem_data_receptor, residue_accessibility_receptor, \
            protbert_data_ligand, graph_batch_ligand, phychem_data_ligand, residue_accessibility_ligand, \
            complex_info_batch, pair_residue_label, ligand_label_batch, receptor_label_batch = batch_data

            complex_infos_ += complex_info_batch

            protbert_data_receptor = torch.autograd.Variable(protbert_data_receptor.to(device).float())
            graph_batch_receptor.edata['ex'] = torch.autograd.Variable(graph_batch_receptor.edata['ex'].to(device).float())
            phychem_data_receptor = torch.autograd.Variable(phychem_data_receptor.to(device).float())
            residue_accessibility_receptor = torch.autograd.Variable(residue_accessibility_receptor.to(device).float())
            
            protbert_data_ligand = torch.autograd.Variable(protbert_data_ligand.to(device).float())
            graph_batch_ligand.edata['ex'] = torch.autograd.Variable(graph_batch_ligand.edata['ex'].to(device).float())
            phychem_data_ligand = torch.autograd.Variable(phychem_data_ligand.to(device).float())
            residue_accessibility_ligand = torch.autograd.Variable(residue_accessibility_ligand.to(device).float())
            
            pair_residue_label = torch.autograd.Variable(pair_residue_label.to(device).long())
            ligand_label_batch = torch.autograd.Variable(ligand_label_batch.to(device).long())
            receptor_label_batch = torch.autograd.Variable(receptor_label_batch.to(device).long())

            receptor_pred, ligand_pred, pairwise_pred, head_attn_scores = model(protbert_data_receptor,
                                                                                graph_batch_receptor,
                                                                                phychem_data_receptor,
                                                                                residue_accessibility_receptor,
                                                                                protbert_data_ligand,
                                                                                graph_batch_ligand,
                                                                                phychem_data_ligand,
                                                                                residue_accessibility_ligand,
                                                                                pair_residue_label)
                                                                                
            receptor_pred = receptor_pred.view(-1)
            ligand_pred = ligand_pred.view(-1)
            pairwise_pred = pairwise_pred.view(-1)

            receptor_pred_all = np.concatenate([receptor_pred_all, receptor_pred.data.cpu().numpy()])
            ligand_pred_all = np.concatenate([ligand_pred_all, ligand_pred.data.cpu().numpy()])
            pairwise_pred_all = np.concatenate([pairwise_pred_all, pairwise_pred.data.cpu().numpy()])

            label_all = np.concatenate([label_all, pair_residue_label[:, 2].data.cpu().numpy()])
            residue_pairs = np.concatenate([residue_pairs, pair_residue_label[:, :2].data.cpu().numpy()])
            ligand_label_all = np.concatenate([ligand_label_all, ligand_label_batch.data.cpu().numpy()])
            receptor_label_all = np.concatenate([receptor_label_all, receptor_label_batch.data.cpu().numpy()])

            if save_attention_scores:
                receptor_attention_scores += [head_attn_scores[0].data.cpu().numpy()]
                ligand_attention_scores += [head_attn_scores[1].data.cpu().numpy()]

    weight = torch.from_numpy(configs.pos_neg_ratio/(configs.pos_neg_ratio + 1) * label_all + 1.0/(configs.pos_neg_ratio + 1) * (1. - label_all)) if configs.WEIGHTED_LOSS else None
    
    val_losses = []
    for (output_name, output, true_label, weights) in [('receptor_pred_all', receptor_pred_all, label_all, weight),
                                              ('ligand_pred_all', ligand_pred_all, label_all, weight),
                                              ('pairwise_pred_all', pairwise_pred_all, label_all, weight)]:
        
        
        auroc, auprc, accuracy, f1, p, r, threshold = basic_metrics(output, true_label)
        median_auroc = calculate_median_auroc(output, true_label, complex_infos_)

        ce_loss = torch.nn.functional.binary_cross_entropy(torch.from_numpy(output), torch.from_numpy(true_label), weight=weights, reduction='none')
        val_loss = ce_loss

        val_losses += [val_loss]
        val_loss = torch.mean(val_loss)

        res = '\t'.join([
            'output_name: ' + output_name,
            'median_auroc:%.6f' % (median_auroc),
            'auroc:%.6f' % (auroc),
            'auprc:%.6f' % (auprc),
            'accuracy:%.6f' % (accuracy),
            'f1:%.6f' % (f1),
            'p:%.6f' % (p),
            'r:%.6f' % (r),
            'threshold:%.6f' % (threshold),
            'elapsed time:%.6f' % (time()-end)])
        print(res)

        results += [[output_name, median_auroc, auroc, auprc, accuracy, f1, p, r, val_loss, threshold]]

    if configs.TRAIN_FOR_SINGLE_PROTEINS and configs.TRAIN_FOR_PAIRWISE_PROTEINS:
        total_val_loss = torch.mean(val_losses[0] + val_losses[1] + val_losses[2])
    elif configs.TRAIN_FOR_SINGLE_PROTEINS:
        total_val_loss = torch.mean(val_losses[0] + val_losses[1])
    else:
        total_val_loss = torch.mean(val_losses[2])
    return results, label_all, ligand_pred_all, receptor_pred_all, pairwise_pred_all, complex_infos_, residue_pairs, total_val_loss.item(), receptor_attention_scores, ligand_attention_scores

def run_experiment (root_dir, total_epochs, save_dir, experiment_num, model_file):
    patience_before_dropping_training = configs.patience_before_dropping_training
    Path(save_dir + '/experiment_number_' + str(experiment_num)).mkdir(parents=False, exist_ok=True)
    set_seed()
    
    train_dataSet = data_generator.dataSet(root_dir=root_dir, data_file_prefix='train')
    test_dataSet = data_generator.dataSet(root_dir=root_dir, data_file_prefix='test')

    # generate train, val, test samples
    if configs.regenerate_train_val_split:
        total_samples = np.arange(configs.train_size + configs.val_size)
        train_samples = total_samples[:configs.train_size]
        val_samples = total_samples[configs.train_size:]
        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)
        train_val_samples = {'train_samples': train_samples, 'val_samples': val_samples}
        pickle.dump (train_val_samples, open(save_dir + '/train_and_val_samples_final.pkl', 'wb'))
    else:
        with open(save_dir + '/train_and_val_samples_final.pkl', "rb") as f:
            samples = pickle.load(f)
        train_samples = samples['train_samples']
        val_samples = samples['val_samples']

    train_loader = torch.utils.data.DataLoader (
        train_dataSet,
        sampler=train_samples,
        batch_size=configs.batch_size,
        pin_memory=(torch.cuda.is_available()),
        num_workers=configs.num_workers, 
        drop_last=False, 
        collate_fn=GraphCollate('train'))

    val_loader = torch.utils.data.DataLoader (
        train_dataSet, 
        sampler=val_samples,
        batch_size=configs.batch_size,
        pin_memory=(torch.cuda.is_available()),
        num_workers=configs.num_workers, 
        drop_last=False, 
        collate_fn=GraphCollate('val'))
    test_loader = torch.utils.data.DataLoader (
        test_dataSet,
        batch_size=configs.batch_size,
        shuffle=False,
        pin_memory=(torch.cuda.is_available()),
        num_workers=configs.num_workers, 
        drop_last=False,
        collate_fn=GraphCollate('test'))
    
    model = PairEgret()
    model.apply (weight_init)
    if model_file is not None:
        try:
            pretrained_dict = torch.load (model_file)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update (pretrained_dict)
            model.load_state_dict (model_dict)
        except:
            pass
    model = model.to(device)

    f = open(save_dir+'/experiment_number_'+str(experiment_num) + '/Log_of_validation(per_epoch)_and_test_results.csv', 'w')
    f.write('epoch,output_name,median_auroc,auroc,auprc,accuracy,f1_score,precision,recall,val_loss,train_loss,best_threshold\n')
    
    best_median_auroc = 0.
    best_auprc = 0.
    best_f1 = 0.
    
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
    if configs.USE_STEP_LR:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.3)
    
    train_losses = []
    val_losses = []
    all_median_aurocs = []
    for epoch in range(total_epochs):
        t0 = time()

        # training steps
        _, train_loss = train_epoch(model=model, loader=train_loader,
                                    optimizer=optimizer, epoch=epoch, all_epochs=total_epochs)
        train_losses += [train_loss]

        if configs.USE_STEP_LR:
            scheduler.step()

        # validation steps
        results, label_all, ligand_pred_all, receptor_pred_all,  pairwise_pred_all, complex_infos_, residue_pairs, total_val_loss, _ , _ = eval_epoch(model=model, loader=val_loader)

        updated = False
        val_losses += [total_val_loss]

        for _, (output_name, median_auroc, auroc, auprc, accuracy, f1, p, r, val_loss, threshold) in enumerate(results):
            f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, output_name, median_auroc, auroc, auprc,
                                                              accuracy, f1, p, r,
                                                              val_loss, train_loss, threshold))

            if output_name == 'pairwise_pred_all':
                all_median_aurocs += [median_auroc]
                if auprc > best_auprc:
                    print("new best pairwise_pred's auprc: {}".format(auprc))
                    best_auprc = auprc

                if median_auroc > best_median_auroc:
                    print("new best pairwise_pred's median_auroc: {}".format(median_auroc))
                    best_median_auroc = median_auroc
                    torch.save(model.state_dict(),
                               save_dir + '/experiment_number_' + str(experiment_num) + '/model_{0}.dat'.format(
                                   experiment_num))
                    updated = True
                    patience_before_dropping_training = configs.patience_before_dropping_training

                if f1 > best_f1:
                    print("new best pairwise_pred's f1: {}".format(f1))
                    best_f1 = f1

        if not updated:
            patience_before_dropping_training -= 1
        if patience_before_dropping_training == 0:
            print('Patience limit crossed.')
            break

        f.write('-,-,-,-,-,-,-,-,-,-,-,-\n')
        print('epoch_time:', time() - t0, ', training_losses_avg: ', train_loss)

    # results and plots for training and validation
    print('Training best median auroc: {}, best auprc: {}'.format(best_median_auroc, best_auprc))
    plt.plot(train_losses, label='train_losses')
    plt.plot(val_losses, label='val_losses')
    plt.legend()
    plt.savefig(save_dir + '/experiment_number_'+str(experiment_num)+'/losses.png')
    plt.cla()

    plt.plot(all_median_aurocs, label='pairwise median AUROC')
    plt.legend()
    plt.savefig(save_dir + '/experiment_number_'+str(experiment_num)+'/pairwise_median_AUROC.png')
    plt.cla()
    
    print('\tTraining complete. Testing...')
    t0 = time()

    # testing steps
    pretrained_dict = torch.load(save_dir+'/experiment_number_'+str(experiment_num)+ '/model_{0}.dat'.format(experiment_num))
    
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    results, label_all, ligand_pred_all, receptor_pred_all,  pairwise_pred_all, complex_infos, residue_pairs, total_test_loss, receptor_attention_scores, ligand_attention_scores = eval_epoch(model=model, loader=test_loader, save_attention_scores=True)

    f.write('-,-,-,-,-,-,-,-,-,-,-,-\n')
    pred_out_all = None
    for _, (output_name, median_auroc, auroc, auprc, accuracy, f1, p, r, test_loss, threshold) in enumerate(results):
        f.write('{},{},{},{},{},{},{},{},{},{},{},{}\n'.format('TEST', output_name, median_auroc, auroc, auprc,
                                                          accuracy, f1, p, r,
                                                          test_loss, '-', threshold))
        if output_name == 'pairwise_pred_all':
            pred_out_all = (pairwise_pred_all > threshold).astype(int)
    f.close()

    cm = confusion_matrix(label_all, pred_out_all)
    print(cm)
    
    if configs.PREDICT_INTERFACE_REGION:

        pos_ligands = set()
        pos_receptors = set()

        for i in range(len(label_all)):
            if label_all[i] == 1.0:
                pos_ligands.add(int(residue_pairs[i][0]))
                pos_receptors.add(int(residue_pairs[i][1]))

        pos_ligands = np.array(list(pos_ligands))
        pos_receptors = np.array(list(pos_receptors))

        ligand_binding_site_labels = np.zeros(int(max(residue_pairs[:, 0]))+1)
        receptor_binding_site_labels = np.zeros(int(max(residue_pairs[:, 1]))+1)

        ligand_binding_site_labels[pos_ligands] = 1.0
        receptor_binding_site_labels[pos_receptors] = 1.0

        ligand_binding_site_predictions = {}
        receptor_binding_site_predictions = {}

        for i in range(len(ligand_binding_site_labels)):
            ligand_binding_site_predictions[i] = []
        for i in range(len(receptor_binding_site_labels)):
            receptor_binding_site_predictions[i] = []

        for i in range(len(label_all)):
            l = residue_pairs[i][0]
            r = residue_pairs[i][1]
            ligand_binding_site_predictions[l] += [ligand_pred_all[i]]
            receptor_binding_site_predictions[r] += [receptor_pred_all[i]]

        ligand_binding_pred_max = np.zeros_like(ligand_binding_site_labels)
        ligand_binding_pred_avg = np.zeros_like(ligand_binding_site_labels)

        receptor_binding_pred_max = np.zeros_like(receptor_binding_site_labels)
        receptor_binding_pred_avg = np.zeros_like(receptor_binding_site_labels)

        for i in range(len(ligand_binding_site_labels)):
            ligand_binding_site_predictions[i] = np.array(ligand_binding_site_predictions[i])
            if len(ligand_binding_site_predictions[i]) > 0:
                ligand_binding_pred_max[i] = np.max(ligand_binding_site_predictions[i])
                ligand_binding_pred_avg[i] = np.mean(ligand_binding_site_predictions[i])

        for i in range(len(receptor_binding_site_labels)):
            receptor_binding_site_predictions[i] = np.array(receptor_binding_site_predictions[i])
            if len(receptor_binding_site_predictions[i]) > 0:
                receptor_binding_pred_max[i] = np.max(receptor_binding_site_predictions[i])
                receptor_binding_pred_avg[i] = np.mean(receptor_binding_site_predictions[i])

        # interface region prediction results

        result_pairs = [(ligand_binding_pred_avg, ligand_binding_site_labels, "ligand_binding_sites"),
                        (receptor_binding_pred_avg, receptor_binding_site_labels, "receptor_binding_sites"),
                        (np.concatenate((ligand_binding_pred_avg, receptor_binding_pred_avg)),
                        np.concatenate((ligand_binding_site_labels, receptor_binding_site_labels)), "both_binding_sites")]
        for (outputs, labels, output_name) in result_pairs:
            auroc, auprc, accuracy, f1, p, r, threshold = basic_metrics(outputs, labels)

            res = '\t'.join([
                'output_name: ' + output_name,
                'auroc:%.6f' % (auroc),
                'auprc:%.6f' % (auprc),
                'accuracy:%.6f' % (accuracy),
                'f1:%.6f' % (f1),
                'p:%.6f' % (p),
                'r:%.6f' % (r),
                'threshold:%.6f' % (threshold),
                ])
            print(res)


    if configs.SAVE_TEST_OUTPUTS:
        pickle.dump({"labels": label_all, "ligand_predictions": ligand_pred_all, "receptor_predictions": receptor_pred_all,
                     "pairwise_predictions": pairwise_pred_all, "complex_infos": complex_infos,
                     "residue_pairs": residue_pairs},
                    gzip.open(save_dir + '/experiment_number_' + str(experiment_num) + "/test_output.pkl.gz", "wb"))

        pickle.dump({"receptor_attn_scores": receptor_attention_scores, "ligand_attn_scores": ligand_attention_scores},
                    gzip.open(save_dir + '/experiment_number_' + str(experiment_num) + "/attention_scores.pkl.gz", "wb"))
    
    if not configs.SAVE_MODEL:
        os.remove(save_dir+'/experiment_number_'+str(experiment_num)+'/model_{0}.dat'.format(experiment_num))

    print('Testing time:', time() - t0)
    print('Experiment complete.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_name", help="name of the dataset", default=configs.dataset_name)
    parser.add_argument("--language_model", help="language model used for node features", default=configs.language_model)
    parser.add_argument("--WEIGHTED_LOSS", help="Weighted Loss or not", default=configs.WEIGHTED_LOSS)
    parser.add_argument("--total_epochs", help="number of epochs to train, set 0 for only testing", default=configs.total_epochs)
    parser.add_argument('--patience_before_dropping_training', help='tolerance of performance dropping', default=configs.patience_before_dropping_training)
    parser.add_argument('--num_phychem_features', help='physicochemical features. options: [0, 1, 14]', default=configs.num_phychem_features)
    parser.add_argument('--num_residue_accessibility_features', help='residue depth and RASA. options:[0,1,2]', default=configs.num_residue_accessibility_features)
    parser.add_argument('--num_bert_features', help='language model embedding size', default=configs.num_bert_features)
    parser.add_argument('--num_heads', help='number of heads in gat layer', default=configs.num_heads)
    parser.add_argument('--num_gat_layers', help='number of gat layers', default=configs.num_gat_layers)
    parser.add_argument('--pos_neg_ratio', help='for unbalanced data. Undersampling ratio', default=configs.pos_neg_ratio)
    parser.add_argument('--SAVE_TEST_OUTPUTS', default=configs.SAVE_TEST_OUTPUTS)
    parser.add_argument('--SAVE_MODEL', default=configs.SAVE_MODEL)

    args = parser.parse_args()
    args = vars(args)

    for key, value in args.items():
        setattr(configs, key, value)

    np.random.seed(1)
    root_dir = '.'
    
    save_dir = root_dir + '/saved_models_and_logs/'
    print('Save dir:', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = time()

    model_file = None
    for experiment_num in range(configs.init_experiment_number, configs.total_experiments_num):
        run_experiment(root_dir=root_dir, total_epochs=configs.total_epochs, save_dir=save_dir,
                       experiment_num=experiment_num, model_file=model_file)

    with open(os.path.join(save_dir, 'args.json'), 'w') as file:
        json.dump(args, file)
    finish_time = time()

    print('Experiment ran in {} minutes'.format((finish_time - start_time) / 60))
