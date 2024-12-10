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

class HDF5Dataset(Dataset):
    def __init__(self, file_list, chunk_size=2, transform=None):
        """
        A dataset that loads a few files at a time (defined by chunk_size).
        This avoids having to index or load data from all files at once.

        file_list: List of HDF5 files.
        chunk_size: How many files to load into memory at once.
        """
        self.file_list = file_list
        self.chunk_size = chunk_size
        self.transform = transform
        self.current_chunk = None
        self.current_chunk_start_index = 0  # Start index of the current chunk
        self.global_index = 0  # Track global index across chunks
        self.local_chunk_index = 0  # Index within the currently loaded chunk
        
        self.total_samples = self._calculate_total_samples()  # Calculate total samples
        self._load_next_chunk()

    def _calculate_total_samples(self):
        """Calculate the total number of samples across all files."""
        total_samples = 0
        for file in tqdm(self.file_list):
            with h5py.File(file, 'r') as h5_file:
                total_samples += len(h5_file['inputs'])  # Assuming 'inputs' dataset exists
        return total_samples

    def _load_next_chunk(self):
        """Load the next chunk of files into memory."""
        # Ensure we don't exceed the number of available files
        if self.current_chunk_start_index >= len(self.file_list):
            self.current_chunk_start_index = 0  # Start over if needed

        self.current_chunk = []  # Clear current chunk
        selected_files = self.file_list[self.current_chunk_start_index:self.current_chunk_start_index + self.chunk_size]
        
        for file in selected_files:
            with h5py.File(file, 'r') as h5_file:
                inputs = h5_file['inputs'][:]  # Load the entire 'inputs' dataset
                labels = h5_file['labels'][:]  # Load the entire 'labels' dataset
                self.current_chunk.append((inputs, labels))

        self.current_chunk_start_index += self.chunk_size
        self.local_chunk_index = 0  # Reset local chunk index

    def __len__(self):
        return self.total_samples

    def _get_chunk_data(self, global_index):
        """Retrieve the sample data for the current chunk based on global index."""
        cumulative_length = 0
        
        # Iterate through the current chunk to find the correct index
        for inputs, labels in self.current_chunk:
            if global_index < cumulative_length + len(inputs):
                relative_index = global_index - cumulative_length
                input_data = inputs[relative_index]
                label_data = labels[relative_index]
                return input_data, label_data
            
            cumulative_length += len(inputs)
        
        return None, None

    def __getitem__(self, global_index):
        """Retrieve a single sample based on the global index."""
        # Calculate total samples in the current chunk
        current_chunk_size = sum([len(inputs) for inputs, _ in self.current_chunk])
        
        if self.local_chunk_index >= current_chunk_size:
            # Load the next chunk if local index exceeds the current chunk size
            self._load_next_chunk()
            self.local_chunk_index = 0
        
        # Retrieve input and label data from the current chunk
        input_data, label_data = self._get_chunk_data(self.local_chunk_index)
        if input_data is None or label_data is None:
            # If no data was found, load a new chunk and retry
            self._load_next_chunk()
            input_data, label_data = self._get_chunk_data(self.local_chunk_index)

        # Apply transformations, if any
        if self.transform:
            input_data = self.transform(input_data)

        # Convert labels to appropriate shape (17 values, last for absorption)
        y_final = np.zeros(17)
        y_final[:16] = label_data
        y_final[16] = 1 - np.sum(label_data)
        label_data = torch.from_numpy(y_final).float()

        self.local_chunk_index += 1
        return torch.from_numpy(input_data).float(), label_data



class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_batch_size= 32, num_workers=8, val_size=0.001, chunk_size=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.chunk_size = chunk_size

        # Find all files in the dataset directory
        p = Path(data_dir)
        self.all_files = sorted(p.glob('**/*.h5'))

    def setup(self, stage=None):
        total_files = len(self.all_files)
        print(f"Total Files: {total_files}")

        # Split files into train/val datasets by files, not by individual samples
        val_file_count = int(total_files * self.val_size)
        train_file_count = total_files - val_file_count

        self.train_files = self.all_files[:train_file_count]
        self.val_files = self.all_files[train_file_count:]

        print(f"Train Files: {len(self.train_files)}, Val Files: {len(self.val_files)}")

    def train_dataloader(self):
        train_dataset = HDF5Dataset(self.train_files, chunk_size=self.chunk_size)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        val_dataset = HDF5Dataset(self.val_files, chunk_size=self.chunk_size)
        return DataLoader(val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers)
    
    
class PMTNetwork(pl.LightningModule):
    def __init__(self, OM: str = "LOM16", num_conv_layers = 3, num_filters = 150, num_lin_layers = 3, linear_features = 150, num_global_features = 5, mse_weight = 0.01 ):
        
        super(PMTNetwork, self).__init__()
        if OM == "LOM16":
            pmt_positions, pmt_directions = self.init_lom16()
            pmt_positions = pmt_positions / 230 # normalize same way as input data
        
        
        self.pmt_positions = torch.tensor(pmt_positions, dtype=torch.float32)  # Shape: (16, 3) for 16 PMTs with (x, y, z)
        self.pmt_directions = torch.tensor(pmt_directions, dtype=torch.float32)  # Shape: (16, 3) for 16 PMTs' direction vectors

        # Network layers
        
        self._conv_layers = torch.nn.ModuleList()
        
        for i in range(num_conv_layers):
            if i == 0:
                conv_layer = nn.Conv1d(in_channels=6, out_channels=num_filters, kernel_size=1)
            else:
                conv_layer = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=1)
            
            self._conv_layers.append(conv_layer)
        
        self._linear_layers = torch.nn.ModuleList()
        
        for i in range(num_lin_layers):
            if i == 0:
                lin_layer = nn.Linear(in_features=num_filters+num_global_features, out_features=linear_features)
            else:
                lin_layer = nn.Linear(in_features=linear_features, out_features=linear_features)
            
            self._linear_layers.append(lin_layer)
        
        self.linear_map = nn.Linear(7,num_global_features)
        self.output_layer = nn.Linear(linear_features * 16 , 17)  # Map to 17 logits (16 PMTs + 1 absorption)

        
        # Define loss function
        self.loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = torch.nn.MSELoss()
        self.mse_weight = mse_weight
        
    
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
        
        # Wavelength
        
        wavelength = (photon_batch[:, 0].unsqueeze(-1) - 300) / (700 - 300)
        
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
        

        # Stack inputs: Δz, ρ, cos(Δφ), cos(θ), polar/equatorial label
        inputs = torch.stack([
            delta_z, 
            #rho, 
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

        x = x.permute(0, 2, 1)  # (batch_size, 16, num_filters)

        # Consider global features that break symmetry
        global_inputs = torch.cat([photon_direction,photon_position,wavelength],dim=-1).unsqueeze(1)
        global_inputs = global_inputs.repeat(1,16,1)
        x_global = self.linear_map(global_inputs)  # (batch_size, 16, num_global_features)
        x = torch.cat([x,x_global],dim=-1)
        
        for lin_layer in self._linear_layers:
            x = lin_layer(x)
            x = torch.relu(x)
        
        # Flatten for output
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.output_layer(x) 
        
        # Final output: log-softmax for the KLDivLoss, it is more stable according to documentation
        if self.training:
            x = torch.log_softmax(x, dim=-1)  # Log-softmax more stable for KL
        else:
            x = torch.softmax(x, dim=-1) # Softmax at inference time
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay = 1e-2)
        # StepLR scheduler to reduce learning rate by 3 after each epoch
        scheduler = StepLR(optimizer, step_size=1, gamma=1/3)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, labels)
        mse_loss = self.mse_loss(torch.exp(outputs),labels)
        loss = loss + self.mse_weight*mse_loss
        # Log only at the end of each epoch
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        inputs, labels = batch  # Assuming batch consists of (inputs, labels)
        outputs = self.forward(inputs)
        # Compute loss
        loss = self.loss_fn(outputs, labels)  # Check that both are of shape [batch_size, 16, 17]
        mse_loss = self.mse_loss(outputs.exp(),labels)
        
        loss = loss + self.mse_weight*mse_loss
        
        # Logging
        self.log("val_loss", loss)
        
        return loss



model = PMTNetwork()

# Use the BaseDataModule to load data
my_dir = BaseDataModule("/scratch/tmp/fvaracar/geant_h5_files/first_run", batch_size=1000, val_batch_size=256, num_workers=8)

checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',  # You can replace 'val_loss' with 'val_accuracy' if you log accuracy
    mode='min',           # Use 'max' if monitoring accuracy instead of loss
    save_top_k=1,         # Save only the best model
    verbose=True
)
# Train the model
# Create the trainer with the checkpoint callback
trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback])
trainer.fit(model, my_dir)