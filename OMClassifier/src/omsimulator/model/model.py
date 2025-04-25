 
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
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import sys

class PMTNetwork(pl.LightningModule):
    def __init__(self, OM: str = "LOM16", num_conv_layers = 3, num_filters = 150, num_lin_layers = 3, linear_features = 150, num_global_features = 5, mse_weight = 0.01, learning_rate=1e-4, wv_global=False, QE=False, global_features=True,QE_measurement="NNVT",conv_map=False, old_norm = False):
        
        super(PMTNetwork, self).__init__()
        if OM == "LOM16":
            pmt_positions, pmt_directions = self.init_lom16()
            pmt_positions = pmt_positions / 230 # normalize same way as input data
        
        # Use or not old wv normalization, i.e. from 300 to 700 nm
        self.old_norm = old_norm
        
        self.pmt_positions = torch.tensor(pmt_positions, dtype=torch.float32)  # Shape: (16, 3) for 16 PMTs with (x, y, z)
        self.pmt_directions = torch.tensor(pmt_directions, dtype=torch.float32)  # Shape: (16, 3) for 16 PMTs' direction vectors
        self.wv_global = False, # Deprecated
        self.global_features = global_features
        if not self.global_features:
            num_global_features = 0
        self.QE = False # Deprecated
        
        if self.QE:
            script_dir = Path(__file__).parent
            # Read QE files and extract values/wavelengths
            if QE_measurement == "NNVT":
                qe_file = "../measurement_values/lom_QE_NNVT.txt"
            else:
                qe_file = "../measurement_values/lom_QE_NNVT.txt"
            qe_path = script_dir / qe_file
            qe_data = pd.read_csv(qe_path, sep='\s+', header=None)
            self.QE_values = torch.tensor(qe_data.iloc[:, 1].values, dtype=torch.float32)
            self.QE_wavelengths = torch.tensor(qe_data.iloc[:, 0].values, dtype=torch.float32)
        
        # Network layers
        
        self._conv_layers = torch.nn.ModuleList()
        
        for i in range(num_conv_layers):
            if i == 0:
                conv_layer = nn.Conv1d(in_channels=7, out_channels=num_filters, kernel_size=1)
            else:
                conv_layer = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=1)
            
            self._conv_layers.append(conv_layer)
        
        
        # Use this conv_layer for mapping 
        self.conv_map = conv_map
        if self.conv_map:
            self.final_conv_layer = nn.Conv1d(in_channels=num_filters, out_channels=1, kernel_size=1)
            num_filters = 1
        
        self._linear_layers = torch.nn.ModuleList()
        self.num_lin_layers = num_lin_layers
        for i in range(num_lin_layers):
            if i == 0:
                lin_layer = nn.Linear(in_features=num_filters+num_global_features, out_features=linear_features)
            else:
                lin_layer = nn.Linear(in_features=linear_features, out_features=linear_features)
            
            self._linear_layers.append(lin_layer)
        
        if self.global_features:
            self.linear_map = nn.Linear(7,num_global_features)
        
        if num_lin_layers == 0:
            linear_features = 1
        
        self.output_layer = nn.Linear(linear_features * 16 , 17)  # Map to 17 logits (16 PMTs + 1 absorption)

        
        # Define loss function
        self.loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = torch.nn.MSELoss()
        self.mse_weight = mse_weight
        self.learning_rate = learning_rate
    
    def QE_interpolation(self, wavelengths):
        
        self.QE_wavelengths = self.QE_wavelengths.to(wavelengths.device)
        self.QE_values = self.QE_values.to(wavelengths.device)

        
        # Find indices of the closest larger and smaller elements
        try:
            indices = torch.searchsorted(self.QE_wavelengths, wavelengths.contiguous(), right=True)
        except:
            indices = torch.searchsorted(self.QE_wavelengths, wavelengths, side="right")
        # Ensure indices are within bounds
        indices_lower = torch.clamp(indices - 1, min=0)
        indices_upper = torch.clamp(indices, max=len(self.QE_wavelengths) - 1)
        
        min_wavelength = self.QE_wavelengths[indices_lower]
        max_wavelength = self.QE_wavelengths[indices_upper]
        min_QE = self.QE_values[indices_lower]
        max_QE = self.QE_values[indices_upper]
        
        denominator = (max_wavelength - min_wavelength).clamp(min=1e-6)  
        return_QE = min_QE + (wavelengths - min_wavelength) * ((max_QE - min_QE) / denominator)
        
        return return_QE

    
    def init_lom16(self):
        # Positions in mm
        pmt_positions = np.array([[59.36840117, 59.36840117, 130.77414122],
                            [-59.36840117, 59.36840117, 130.77414122],
                            [-59.36840117, -59.36840117, 130.77414122],
                            [59.36840117, -59.36840117, 130.77414122],
                            [101.68377658, -0.00000000, 60.14489021],
                            [0.00000000, 101.68377658, 60.14489021],
                            [-101.68377658, 0.00000000, 60.14489021],
                            [-0.00000000, -101.68377658, 60.14489021],
                            [101.68377658, -0.00000000, -60.14489021],
                            [0.00000000, 101.68377658, -60.14489021],
                            [-101.68377658, 0.00000000, -60.14489021],
                            [0.00000000, -101.68377658, -60.14489021],
                            [59.36840117, 59.36840117, -130.77414122],
                            [-59.36840117, 59.36840117, -130.77414122],
                            [-59.36840117, -59.36840117, -130.77414122],
                            [59.36840117, -59.36840117, -130.77414122]])
        
        # Direciton in cartesian coordinates
        pmt_directions = np.array([[0.41562694, 0.41562694, 0.80901699],
                                [-0.41562694, 0.41562694, 0.80901699],
                                [-0.41562694, -0.41562694, 0.80901699],
                                [0.41562694, -0.41562694, 0.80901699],
                                [0.88294759, 0.00000000, 0.46947156],
                                [0.00000000, 0.88294759, 0.46947156],
                                [-0.88294759, 0.00000000, 0.46947156],
                                [0.00000000, -0.88294759, 0.46947156],
                                [0.88294759, -0.00000000, -0.46947157],
                                [0.00000000, 0.88294759, -0.46947157],
                                [-0.88294759, -0.00000000, -0.46947157],
                                [-0.00000000, -0.88294759, -0.46947157],
                                [0.41562694, 0.41562694, -0.80901699],
                                [-0.41562694, 0.41562694, -0.80901699],
                                [-0.41562694, -0.41562694, -0.80901699],
                                [0.41562694, -0.41562694, -0.80901699]])
        
        return pmt_positions, pmt_directions

    def forward(self, photon_batch):
        # Process photon batch and PMT features (similar to previous implementation)
        photon_batch = photon_batch.float()
        
        wavelength = (photon_batch[:, 0].unsqueeze(-1) - 270) / (700 - 270)
        # Photon properties
        photon_position = photon_batch[:, 1:4] / 230  # photon (x, y, z)
        photon_direction = photon_batch[:, 4:7]  # photon direction (dx, dy, dz)

        # PMT properties
        pmt_positions = self.pmt_positions.to(photon_batch.device)  # (16, 3)
        pmt_directions = self.pmt_directions.to(photon_batch.device)  # (16, 3)

        # Binary label for polar (1) vs equatorial (0) PMTs
        pmt_polar_label = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).to(photon_batch.device)
        pmt_upper_label = torch.tensor([1]*8 + [0]*8, dtype=torch.float32).to(photon_batch.device)

        # 1. Delta z (flip sign for lower PMTs)
        delta_z = pmt_positions[:, 2].unsqueeze(0) - photon_position[:, 2].unsqueeze(1)  # (batch_size, 16)
        delta_z = delta_z * torch.where(pmt_upper_label.unsqueeze(0).bool(), 1, -1)  # Flip sign for lower PMTs

        # 2. Rho in the x-y plane
        photon_xy = photon_position[:, :2]  # photon x-y coordinates
        pmt_xy = pmt_positions[:, :2]  # PMT x-y coordinates
        rho = torch.norm(pmt_xy.unsqueeze(0) - photon_xy.unsqueeze(1), dim=-1)  # (batch_size, 16)

        # 3. Azimuthal difference (Δφ) between photon and each PMT in x-y plane
        photon_phi = torch.atan2(photon_position[:, 1], photon_position[:, 0])  # photon azimuth (φ)
        pmt_phi = torch.atan2(pmt_positions[:, 1], pmt_positions[:, 0])  # PMT azimuth (φ)
        delta_phi = torch.abs(pmt_phi.unsqueeze(0) - photon_phi.unsqueeze(1))  # (batch_size, 16)
        #delta_phi = torch.min(delta_phi, 2 * torch.pi - delta_phi)  # Symmetry in azimuthal angle
        cos_delta_phi_position = torch.cos(delta_phi) # normalize [0-1]

        # 4. Azimutal symmetry of the photon direction
        # Calculate the relative position between photon and PMT in the x-y plane
        photon_relative_xy = photon_position[:, :2].unsqueeze(1) - pmt_positions[:, :2].unsqueeze(0)  # (batch_size, 16, 2)

        # Photon direction in the x-y plane
        photon_direction_xy = photon_direction[:, :2]  # (batch_size, 2)

        # Step 1: Calculate azimuthal angles for photon direction and PMT
        photon_dir_phi_xy = torch.atan2(photon_direction_xy[:, 1], photon_direction_xy[:, 0])  # Photon azimuth in x-y plane
        pmt_phi_xy = torch.atan2(pmt_positions[:, 1], pmt_positions[:, 0])  # PMT azimuth in x-y plane

        # Step 2: Calculate relative azimuthal difference (Δφ) and apply cosine for symmetry
        delta_phi_xy = photon_dir_phi_xy.unsqueeze(1) - pmt_phi_xy.unsqueeze(0)  # (batch_size, 16)
        cos_delta_phi = torch.cos(delta_phi_xy)  # Normalized azimuth difference using cosine

        # Step 3: Convergence/divergence check based on y-component comparison
        step_size = 0.1
        next_y_positions = photon_position[:, 1] + step_size * photon_direction_xy[:, 1]  # New y position after a small step
        is_converging = (torch.abs(next_y_positions.unsqueeze(1) - pmt_positions[:, 1]) < torch.abs(photon_position[:, 1].unsqueeze(1) - pmt_positions[:, 1])).float()

        # 5. Cosine of the zenith angle (normalized)
        # LOM PMTs do not have zenith symmetry, but the two halves are symmetrical. Therefore we flip the sign of the cosine zenith in the lower half.
        cos_zenith_angle = photon_direction[:, 2].unsqueeze(1)  # Cosine of zenith (z-direction)
        cos_zenith_angle = cos_zenith_angle * torch.where(pmt_upper_label.unsqueeze(0).bool(), 1, -1)  # Flip sign for lower PMTs
        
        wavelength_repeated = wavelength.repeat(1, 16)  # Repeat wavelength for each PMT, now (batch_size, 16)
        
        inputs = torch.stack([
            delta_z, 
            wavelength_repeated, 
            cos_zenith_angle, 
            cos_delta_phi_position,
            cos_delta_phi,  # Enforces symmetry for azimuth
            is_converging,
            pmt_polar_label.unsqueeze(0).repeat(photon_batch.size(0), 1)
        ], dim=-1) 
        
        # Reshape, conv1d acts on the second dimension
        x = inputs.permute(0, 2, 1)  # (batch_size, 2, 16)

        for conv_layer in self._conv_layers:
            x = conv_layer(x)
            x = torch.relu(x)
        
        if self.conv_map:
            x = self.final_conv_layer(x)

        x = x.permute(0, 2, 1)  # (batch_size, 16, num_filters)

        # Consider global features that break symmetry
        if self.global_features:
            
            global_inputs = torch.cat([photon_direction,photon_position,wavelength],dim=-1).unsqueeze(1)
            global_inputs = global_inputs.repeat(1,16,1)
            x_global = self.linear_map(global_inputs)  # (batch_size, 16, num_global_features)
    
            x = torch.cat([x,x_global],dim=-1)
        
        if self.num_lin_layers > 0:
            for lin_layer in self._linear_layers:
                x = lin_layer(x)
                x = torch.relu(x)
           
        # Flatten for output
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.output_layer(x) 
        
        if self.QE:
            unmodified_wavelengths = photon_batch[:, 0].unsqueeze(-1).to(photon_batch.device)
            batch_QEs = self.QE_interpolation(unmodified_wavelengths)
            # Incorporate QE info
            x[:,:16]*=batch_QEs
            
       
        x = torch.log_softmax(x, dim=-1) 
        
        # Final output: log-softmax for the KLDivLoss, it is more stable according to documentation
        if not self.training:  
            x = torch.exp(x)
            x = x / x.sum(dim=-1, keepdim=True)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay = 1e-2)
        # StepLR scheduler to reduce learning rate by 3 after each epoch
        scheduler = StepLR(optimizer, step_size=1, gamma=1/3)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, labels)
        outputs_prob = torch.exp(outputs)
        outputs_prob = outputs_prob / outputs_prob.sum(dim=-1, keepdim=True)
        mse_loss = self.mse_loss(outputs_prob ,labels)
        loss = loss + self.mse_weight*mse_loss
        # Log only at the end of each epoch
        #vram_used = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        #sys.stdout.write(f"\rVRAM used after batch {batch_idx}: {vram_used:.2f} MB")
        #sys.stdout.flush()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        
        # Compute primary KLDiv loss and MSE loss
        loss = self.loss_fn(outputs, labels)
        outputs_prob = torch.exp(outputs)
        outputs_prob = outputs_prob / outputs_prob.sum(dim=-1, keepdim=True)
        mse_loss = self.mse_loss(outputs_prob, labels)
        
        total_loss = loss + self.mse_weight * mse_loss
        self.log("val_loss", total_loss)
        
        return loss
