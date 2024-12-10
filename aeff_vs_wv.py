import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Example data
# Replace these with your actual NumPy arrays
wavelengths = np.array([300, 350, 400, 450, 500, 550])  # Example wavelengths in nm
NN = []             # Example NN values
Geant = []          # Example Geant values

for wv in wavelengths:
    df = pd.read_csv(f'/scratch/tmp/fvaracar/geant_eff_areas/lom16_eff_area_wavelength_{wv}.dat', delim_whitespace=True, header=None)

    # Retrieve the second-to-last column
    m = df.iloc[:, -2]
    m2 = np.load(f"/scratch/tmp/fvaracar/nn_eff_areas/eff_area_{wv}_old.npy")
    NN.append(np.mean(m2))
    Geant.append(np.mean(m))
    
NN = np.array(NN)
Geant = np.array(Geant)
# Calculate residuals
residuals = Geant / NN

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Top plot: Mean Effective Area
ax1.plot(wavelengths, NN, 'o', label='NN', color='orange')
ax1.plot(wavelengths, Geant, 'o', label='Geant', color='green')
ax1.set_ylabel('Mean Effective Area (cm²)',fontsize=12)
#ax1.set_title('Mean Effective Area and Residuals')
ax1.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
ax1.legend(fontsize=12)

# Bottom plot: Residuals
ax2.plot(wavelengths, residuals, 'o', color='blue')
ax2.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
ax2.axhline(1, linestyle='--', color='gray')
ax2.set_ylabel('Geant/NN',fontsize=12)
ax2.set_xlabel('Wavelength (nm)',fontsize=12)
ax2.set_yticks([round(min(residuals),2),1,round(max(residuals),2)])
ax2.set_ylim(min(residuals)-0.01,max(residuals)+0.01)
# Fine-tune layout
plt.tight_layout()
plt.savefig("/scratch/tmp/fvaracar/OMClassifier/compare_plots/aeff_vs_wv.png")
