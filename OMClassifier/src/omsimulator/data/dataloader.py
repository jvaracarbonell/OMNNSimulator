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
import pandas as pd

class HDF5Dataset(Dataset):
    def __init__(self, file_list, chunk_size=2, transform=None, QE=False, QE_measurement = "NNVT"):
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
        self.QE = QE # Weigth directly by the QE or not
        self.QE_measurement = QE_measurement
        if self.QE:
            script_dir = Path(__file__).parent
            # Read QE files and extract values/wavelengths
            if QE_measurement == "NNVT":
                qe_file = "../measurement_values/lom_QE_NNVT.txt"
            else:
                qe_file = "../measurement_values/lom_QE_NNVT.txt"
            qe_path = script_dir / qe_file
            qe_data = pd.read_csv(qe_path, sep='\s+', header=None)
            self.QE_values = np.array(qe_data.iloc[:, 1].values)
            self.QE_wavelengths = np.array(qe_data.iloc[:, 0].values)
        

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
    
    def QE_interpolation(self, wavelengths):

        # Find indices of the closest larger and smaller elements
        indices = np.searchsorted(self.QE_wavelengths, wavelengths, side="right")

        # Ensure indices are within bounds
        indices_lower = np.clip(indices - 1, a_min=0, a_max=len(self.QE_wavelengths) - 1)
        indices_upper = np.clip(indices, a_min=0, a_max=len(self.QE_wavelengths) - 1)

        # Retrieve the corresponding wavelengths and QE values
        min_wavelength = self.QE_wavelengths[indices_lower]
        max_wavelength = self.QE_wavelengths[indices_upper]
        min_QE = self.QE_values[indices_lower]
        max_QE = self.QE_values[indices_upper]

        # Avoid division by zero
        denominator = np.clip((max_wavelength - min_wavelength), a_min=1e-6, a_max=None)

        # Perform linear interpolation
        return_QE = min_QE + (wavelengths - min_wavelength) * ((max_QE - min_QE) / denominator)
        return return_QE
        


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
        if self.QE:
            wavelengths = input_data[0]
            QE_values = self.QE_interpolation(wavelengths)
            y_final[:16] = label_data * QE_values
        else:
            y_final[:16] = label_data
        total_prob = np.sum(y_final[:16])
        y_final[16] = max(0, 1 - total_prob)  # Absorption probability
        label_data = torch.from_numpy(y_final).float()

        self.local_chunk_index += 1
        return torch.from_numpy(input_data).float(), label_data



class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, val_batch_size= 32, num_workers=8, val_size=0.001, chunk_size=2,QE=False,QE_measurement="NNVT"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.chunk_size = chunk_size
        self.QE = QE
        self.QE_measurement = QE_measurement

        # Find all files in the dataset directory
        self.all_files = sorted(glob.glob(f'{data_dir}/*/*.h5'))

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
        train_dataset = HDF5Dataset(self.train_files, chunk_size=self.chunk_size, QE=self.QE, QE_measurement=self.QE_measurement)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        val_dataset = HDF5Dataset(self.val_files, chunk_size=self.chunk_size, QE=self.QE, QE_measurement=self.QE_measurement)
        return DataLoader(val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers)