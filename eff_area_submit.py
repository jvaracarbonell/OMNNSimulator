# Standard library imports
import os
import time
import glob
import random
from pathlib import Path
from collections import OrderedDict
import argparse
import concurrent.futures

# Third-party library imports
import numpy as np
import h5py
import healpy as hp
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch and related imports
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler

# Torchmetrics
from torchmetrics import Metric

# Project-specific imports
from omsimulator.data.dataloader import BaseDataModule
from omsimulator.model.model import PMTNetwork

# YAML handling
import yaml

#ckpt_file = "/scratch/tmp/fvaracar/train_omclassfier/model_mse_weight_0.1_wv_global_False_QE_24K_params/lightning_logs/version_30321330/checkpoints/epoch=0-step=3189500.ckpt"
#cfg_file = "/scratch/tmp/fvaracar/train_omclassfier/model_mse_weight_0.1_wv_global_False_QE_24K_params.yaml"

#cfg_file = "/scratch/tmp/fvaracar/train_omclassfier/model_mse_weight_0.1_wv_global_False_QE_133K_params.yaml"
#ckpt_file = "/scratch/tmp/fvaracar/train_omclassfier/model_mse_weight_0.1_wv_global_False_QE_133K_params/lightning_logs/version_30321328/checkpoints/epoch=0-step=3189500.ckpt"


def parse_args():
    parser = argparse.ArgumentParser(description="Effective area calculation using NN")
    parser.add_argument("--cfg", type=str, default="", help="Model's config path")
    parser.add_argument("--ckpt", type=str, default="", help="Model's cpkt path")
    
    return parser.parse_args()



def main():
    # Parse arguments
    args = parse_args()
    ckpt_file = args.ckpt
    cfg_file = args.cfg

    name = cfg_file.split("/")[-1]
    name = name.split(".y")[0]

    with open(cfg_file, 'r') as file:
        model_dict = yaml.safe_load(file)
        
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

    old_norm = False
    if "old_norm" in model_dict.keys():
        old_norm = model_dict["old_norm"]

    try:
        model = PMTNetwork(
                num_conv_layers=model_dict["num_conv_layers"],
                num_filters=model_dict["num_filters"],
                num_lin_layers=model_dict["num_lin_layers"],
                linear_features=model_dict["linear_features"],
                num_global_features=model_dict["num_global_features"],
                mse_weight=model_dict["mse_weight"],
                conv_map = model_dict["conv_map"],
                QE=model_dict["QE"],
                global_features=model_dict["global_features"],
                wv_global = model_dict["wv_global"],
                old_norm = old_norm
                
            ).to(device)
        
    except:
        model = PMTNetwork(
                num_conv_layers=model_dict["num_conv_layers"],
                num_filters=model_dict["num_filters"],
                num_lin_layers=model_dict["num_lin_layers"],
                linear_features=model_dict["linear_features"],
                conv_map = model_dict["conv_map"],
                num_global_features=model_dict["num_global_features"],
                mse_weight=model_dict["mse_weight"],
                QE=model_dict["QE"],
                global_features=model_dict["global_features"],
                wv_global = model_dict["wv_global"]
                
            ).to(device)
        

    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    def generate_disk_projected_points_vectorized(incident_direction, radius, num_points=5000):
        """
        Generates points uniformly on a disk and projects them onto the upper hemisphere, 
        then rotates them to align with the incident direction.

        Parameters:
        - incident_direction: The target direction vector for alignment (as a 3D unit vector).
        - radius: Radius of the sphere.
        - num_points: Number of points to generate.

        Returns:
        - points: Array of generated points on the sphere (shape: num_points x 3).
        """
        # Generate random points within a disk of given radius
        r = np.sqrt(np.random.rand(num_points)) * radius  # Radial distance with sqrt for uniformity
        theta = 2 * np.pi * np.random.rand(num_points)  # Random angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Calculate z for the upper hemisphere
        z = np.sqrt(radius**2 - x**2 - y**2)

        # Stack x, y, z coordinates to create points
        points = np.stack([x, y, z], axis=1)#np.column_stack((x, y, z))

        # Rotate points to align with the incident direction
        points = rotate_points_vectorized(points, incident_direction)

        return points

    def rotate_points_vectorized(points, target_direction):
        """
        Rotates points so that the 'z-axis' aligns with the target incident direction.
        """
        # Original direction aligned with z-axis
        z_axis = np.array([0, 0, 1])
        target_direction = target_direction / np.linalg.norm(target_direction)
        rotation_axis = np.cross(z_axis, target_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm == 0:  # Already aligned
            return points

        rotation_axis /= rotation_axis_norm
        angle = np.arccos(np.dot(z_axis, target_direction))
        
        # Compute rotation matrix using Rodrigues' rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        # Apply rotation matrix to all points
        return points @ R.T

    # Parameters
    radius = 230
    beam_area = np.pi * (radius / 10) ** 2
    num_photons = 1000000
    wavelengths = np.arange(300,600,50)  # Wavelength in nm
    batch_size = int(num_photons/10)
    
    path = f"/scratch/tmp/fvaracar/nn_eff_areas/{name}"
    os.makedirs(path, exist_ok=True)

    for wavelength in wavelengths:
        # Initialize Healpy directions (nside=32)
        nside = 32
        npix = hp.nside2npix(nside)
        x_dir, y_dir, z_dir = hp.pix2vec(nside, np.arange(npix))
        directions = np.vstack([x_dir, y_dir, z_dir]).T  # Combine into (npix, 3) array
        effective_areas = []
        polar_effective_areas = []
        eq_effective_areas = []

        # Generate random photons and compute effective area for each Healpy direction
        for direction in tqdm(directions):  # Iterate over each direction
            
            #print(direction)
            incident_direction = np.array(direction)

            # Generate points on the sphere
            points = generate_disk_projected_points_vectorized(incident_direction, radius, num_photons)
            
            positions = points

            # Scale positions to align with current Healpy direction
            rotation = direction * radius  # Shape (3,)
            norm_positions = positions + rotation  # Apply single direction per iteration
            norm_positions = norm_positions / np.linalg.norm(norm_positions, axis=1, keepdims=True) * radius

            # Generate the photon directions aligned with the Healpy direction, pointing towards the origin
            photon_directions = -incident_direction / np.linalg.norm(incident_direction)  # Normalize and point towards origin
            photon_directions = np.tile(photon_directions, (num_photons, 1))  # Repeat for all photons

            # Convert to torch tensors and move to GPU
            positions_tensor = torch.tensor(points, dtype=torch.float32, device=device)
            directions_tensor = torch.tensor(photon_directions, dtype=torch.float32, device=device)
            wavelength_tensor = torch.full((num_photons, 1), wavelength, dtype=torch.float32, device=device)
            photon_inputs = torch.cat([wavelength_tensor, positions_tensor, directions_tensor], dim=1)

            # Calculate total detection probability in batches
            total_detection = 0.0
            total_polar = 0.0
            total_eq = 0.0
            with torch.no_grad():
                for i in range(0, num_photons, batch_size):
                    batch = photon_inputs[i:i+batch_size].to(device)  # Move batch to GPU
                    start=time.time()
                    outputs = model(batch)
                    end=time.time()
                    #print(f"Time taken for process 1: {end - start} seconds", end="\r")
                    detection_probs = outputs
                    detection_probs = detection_probs / detection_probs.sum(dim=-1, keepdim=True)
                    #Polar and eq
                    polar_det = detection_probs[:,1]
                    eq_det = detection_probs[:,9]
                    detection_probs = detection_probs[:, :16].sum(dim=1)  # Sum of first 16 outputs per photon
                    total_detection += detection_probs.sum().item()
                    total_polar+= polar_det.sum().item()
                    total_eq += eq_det.sum().item()

            # Calculate effective area for this direction
            Ndet = total_detection
            Nemit = num_photons
            effective_area = (Ndet / Nemit) * beam_area
            effective_areas.append(effective_area)
            
            polar_eff_area = (total_polar / Nemit) * beam_area
            polar_effective_areas.append(polar_eff_area)
            
            eq_eff_area = (total_eq / Nemit) * beam_area
            eq_effective_areas.append(eq_eff_area)
        
        

        # Plot results using Healpy
        m = np.array(effective_areas)
        hp.mollview(m, title=f"NN Effective area wv={wavelength} nm, average={round(np.mean(m),2)}")
        hp.graticule()
        plt.savefig(f"{path}/nn_eff_area_{wavelength}.png")
        print(f"Mean effective area is {np.mean(effective_areas)}")
        np.save(f"{path}/eff_area_{wavelength}.npy",m)

        m = np.array(polar_effective_areas)
        hp.mollview(m, title=f"Polar NN Effective area wv={wavelength} nm, average={round(np.mean(m),2)}")
        hp.graticule()
        plt.savefig(f"{path}/polar_nn_eff_area_{wavelength}.png")
        print(f"Mean effective area is {np.mean(effective_areas)}")
        np.save(f"{path}/polar_eff_area_{wavelength}.npy",m)

        m = np.array(eq_effective_areas)
        hp.mollview(m, title=f"Equatorial NN Effective area wv={wavelength} nm, average={round(np.mean(m),2)}")
        hp.graticule()
        plt.savefig(f"{path}/eq_nn_eff_area_{wavelength}.png")
        print(f"Mean effective area is {np.mean(effective_areas)}")
        np.save(f"{path}/eq_eff_area_{wavelength}.npy",m)
        
if __name__ == "__main__":
    main()