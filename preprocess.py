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
from torch_geometric.loader import DataLoader as GeoDataLoader
import torch.optim as optim

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
    
    # Configuración de dataset
    no_vids_pp = 15
    no_vids = n_people * no_vids_pp
    vid_length = 16
    label_map = {label: idx for idx, label in enumerate(actions)}
    
    # Partición de datos
    no_vid_for_training = 255 # 68% del conjunto total
    no_vid_for_val = 45 # 12% del conjunto total
    no_vid_for_test = 75 # 20% del conjunto total
    
    # Función auxiliar para cargar un conjunto (train/val/test)
    def load_videos(start_idx, end_idx, actions, data_path, vid_length, label_map):
        videos, labels = [], []
        for vid in range(start_idx, end_idx):
            for action in actions:
                frames = [
                    np.load(os.path.join(data_path, action, str(vid), f"{frame_num}.npy"))
                    for frame_num in range(vid_length)
                ]
                videos.append(frames)
                labels.append(label_map[action])
        return videos, labels
    
    # Cargar datasets
    vids_training, labels_training = load_videos(0, no_vid_for_training, actions, DATA_PATH, vid_length, label_map)
    vids_val, labels_val = load_videos(no_vid_for_training, no_vid_for_training + no_vid_for_val, actions, DATA_PATH, vid_length, label_map)
    vids_test, labels_test = load_videos(no_vid_for_training + no_vid_for_val, no_vids, actions, DATA_PATH, vid_length, label_map)

    return vids_training, labels_training, vids_val, labels_val, vids_test, labels_test


class VideoDataset(Dataset):
    def __init__(self, videos, labels):
        self.videos = videos
        self.labels = labels

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        x = torch.tensor(self.videos[idx], dtype=torch.float32)  # (16, 258)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
