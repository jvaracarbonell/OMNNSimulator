import os
import numpy as np
import h5py
import glob

# Source and destination directories
src_dir = '/scratch/tmp/fvaracar/Geant_outputs'
dst_dir = '/scratch/tmp/fvaracar/geant_h5_files/second_run'

# List of .dat files to convert
txt_files = sorted(glob.glob(f'{src_dir}/sim_*/*.dat'))

# Loop through the .dat files and convert each to .h5
for txt_file in txt_files:
    # Get the relative path of the .dat file and split it to preserve the subdirectory structure
    relative_path = os.path.relpath(txt_file, src_dir)
    dir_name = os.path.dirname(relative_path)
    
    # Create the corresponding directory structure in the destination directory
    h5_dir = os.path.join(dst_dir, dir_name)
    os.makedirs(h5_dir, exist_ok=True)
    
    # Get the base file name for the .h5 file
    file_name = os.path.basename(txt_file).replace('.dat', '.h5')
    h5_file_path = os.path.join(h5_dir, file_name)
    
    try:
        # Load the .dat file into a NumPy array (assuming whitespace-separated values)
        data = np.loadtxt(txt_file)
        if len(data)<2500000:
            # Probably something went wrong
            continue
        # Split the data into inputs (first 7 columns) and labels (remaining columns)
        inputs = data[:, :7]  # First 7 columns as inputs
        labels = data[:, 7:]  # The rest are labels
        
        # Check if the labels have the expected shape
        if labels.shape[1] != 16:
            print(f"Skipping file {txt_file} due to unexpected label shape: {labels.shape}")
            continue
        
        # Create the .h5 file in the corresponding subdirectory
        with h5py.File(h5_file_path, 'w') as h5f:
            # Create datasets in the HDF5 file
            h5f.create_dataset('inputs', data=inputs, compression='gzip', compression_opts=4)
            h5f.create_dataset('labels', data=labels, compression='gzip', compression_opts=4)
        
        print(f"Successfully converted {txt_file} to {h5_file_path}")
    
    except Exception as e:
        print(f"Error processing file {txt_file}: {e}")
        continue

print("Conversion to HDF5 completed!")
