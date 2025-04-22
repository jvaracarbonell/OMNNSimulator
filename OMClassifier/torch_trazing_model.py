import torch
from torch.nn import functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchmetrics import Metric
import torchmetrics
from torch.utils.data import random_split, DataLoader
import glob
import h5py
from collections import OrderedDict  # This is the missing import
from pathlib import Path
import random
from tqdm import tqdm
import concurrent.futures
import healpy as hp
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler
import pytorch_lightning as pl
from omsimulator.data.dataloader import BaseDataModule
from omsimulator.model.model_torch import PMTNetwork
import time
import yaml

ckpt_file = "/scratch/tmp/fvaracar/train_omclassfier/model_mse_weight_0.1_wv_global_False/lightning_logs/version_3/checkpoints/epoch=6-step=22244215.ckpt"
cfg_file = "/scratch/tmp/fvaracar/train_omclassfier/first_model.yaml"

with open(cfg_file, 'r') as file:
    model_dict = yaml.safe_load(file)
    
    
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
device = "cpu"

model = PMTNetwork(
        num_conv_layers=model_dict["num_conv_layers"],
        num_filters=model_dict["num_filters"],
        num_lin_layers=model_dict["num_lin_layers"],
        linear_features=model_dict["linear_features"],
        num_global_features=model_dict["num_global_features"],
        mse_weight=model_dict["mse_weight"],
        QE=model_dict["QE"],
        global_features=model_dict["global_features"],
        wv_global = model_dict["wv_global"]
    ).to(device)

checkpoint = torch.load(ckpt_file, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

my_array = np.array([ 4.0000e+02,  3.3010e+01,  6.5313e+01,  2.1805e+02, -4.7131e-02,
         1.9522e-02, -9.9870e-01])

photon_inputs = torch.tensor(my_array, dtype=torch.float32, device=device)
photon_inputs = photon_inputs.unsqueeze(0)  

print(photon_inputs)
print("\n")

with torch.no_grad():
    #for i in range(0, num_photons, batch_size):
    batch = photon_inputs.to(device)  # Move batch to GPU    
    outputs = model(batch)

print(outputs)

print(torch.sum(outputs, -1, keepdim=False))

scripted_model = torch.jit.trace(model, batch)
scripted_model.save("OMNNSim.pt")
