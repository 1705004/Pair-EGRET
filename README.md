# Pair-EGRET

This repository contains the implementation of our paper "Pair-EGRET: Enhancing the prediction of protein-protein interaction sites through graph attention networks and protein language models". 

## Datasets
Pair-EGRET was evaluated on three benchmark datasets: DBD5, Dockground, and MASIF. Training and test data used in this experiment can be found [**Here**](https://zenodo.org/records/10449060). Save the *train.pkl.gz* and *test.pkl.gz* files of each dataset inside the corresponding *inputs/{dataset_name}* directory.

Alternatively, you can regenerate train and test features following these steps. 
1. Download the *pdb_files.zip* of a dataset and extract it at the correspoding *inputs/{dataset_name}* folder.
2. Set the dataset_name in *config.py*
3. Run the following commands from the root directory of the repository (This may take a while to complete).
```
    python generate_all_features.py
    python save_all_protein_data.py
```
1. *train.pkl.gz* and *test.pkl.gz* should be saved inside *inputs/{dataset_name}* now.

## Run Experiments
Run the full pipeline by running ```python run_egret.py``` from the root directory of the repository. You can run inference using existing models by setting *total_epochs = 0* in *config.py*.