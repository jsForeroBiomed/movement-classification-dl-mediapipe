# Core
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# GCN
import torch_geometric.nn as geom_nn

# TCN
from pytorch_tcn import TCN

# Métricas y optimización
from sklearn.metrics import classification_report, confusion_matrix
import optuna

def load_data():
    # Configuración de rutas y clases
    DATA_PATH = os.path.join("data")
    actions = np.array(['TouchingChest', 'Hit', 'AleatoryMovement', 'Static'])
    n_people = 25
    vid_length = 16
    rng = np.random.default_rng(seed=42)
    
    #no_vids_pp = 15
    #no_vids = n_people * no_vids_pp
    
    # Configuración de dataset
    label_map = {label: idx for idx, label in enumerate(actions)}
    
    # Partición de datos
    people_ids = np.arange(n_people)
    rng.shuffle(people_ids)

    train_ids = people_ids[:17]     # 68%
    val_ids   = people_ids[17:20]   # 12%
    test_ids  = people_ids[20:]     # 20%
    print(train_ids,val_ids,test_ids)
    
    # Función auxiliar para cargar un conjunto (train/val/test)
    def load_videos(subject_ids, actions, data_path, vid_length, label_map):
        videos, labels = [], []
        for subject in subject_ids:
            for action in actions:
                frames = [
                    np.load(os.path.join(data_path, action, str(subject), f"{i}.npy"))
                    for i in range(vid_length)
                ]
                videos.append(frames)
                labels.append(label_map[action])
        return np.array(videos), np.array(labels)
    
    # Cargar datasets
    vids_training, labels_training = load_videos(train_ids, actions, DATA_PATH, vid_length, label_map)
    vids_val, labels_val = load_videos(val_ids, actions, DATA_PATH, vid_length, label_map)
    vids_test, labels_test = load_videos(test_ids, actions, DATA_PATH, vid_length, label_map)

    return vids_training, labels_training, vids_val, labels_val, vids_test, labels_test