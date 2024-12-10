import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.projections.geo import GeoAxes
scale= False
NSIDE = 32
print(
    "Approximate resolution at NSIDE {} is {:.2} deg".format(
        NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
    )
)

nside=32
npix = hp.nside2npix(nside)  # Number of pixels
# Calculate the mean value of the data
class ThetaFormatterShiftPi(GeoAxes.ThetaFormatter):
    """Shifts labelling by pi

    Shifts labelling from -180,180 to 0-360"""
    def __call__(self, x, pos=None):
        if x != 0:
            x *= -1
        if x < 0:
            x += 2*np.pi
        return GeoAxes.ThetaFormatter.__call__(self, x, pos)
    
def plot_mollweide_projection(m, title='Mollweide Projection', show_ticks=True, save_name = "mollview.png"):
    """
    Plot a Mollweide projection of the given data array `m` on a Healpix map.
    
    Parameters:
    m (numpy.ndarray): 1D numpy array containing the data (length corresponding to Healpix npix).
    title (str): Title for the plot.
    show_ticks (bool): Whether to display latitude and longitude ticks on the map.
    
    Returns:
    None: Displays the plot.
    """
    # Calculate the mean value of the data
    mean_value = np.mean(m)

    # Define the resolution (nside) of your Healpix map
    nside = 32  # Healpix resolution
    npix = hp.nside2npix(nside)  # Number of pixels

    # Create the figure with Mollweide projection
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='mollweide')

    # Generate the grid for Mollweide projection (latitude and longitude)
    theta = np.linspace(np.pi, 0, int(npix / 2))  # Zenith angle (theta)
    phi = np.linspace(-np.pi, np.pi, npix)  # Azimuth angle (phi)
    longitude = np.radians(np.linspace(-180, 180, npix))
    latitude = np.radians(np.linspace(-90, 90, int(npix / 2)))

    # Create a meshgrid for theta and phi
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(nside, THETA, PHI)
    grid_map = m[grid_pix]

    # Create a custom colormap with white at the mean value
    cmap = plt.cm.RdBu_r  # Red-Blue colormap
    norm = mcolors.TwoSlopeNorm(vmin=np.min(m), vcenter=mean_value, vmax=np.max(m))  # Normalize around the mean

    # Plot using pcolormesh for smooth color mapping
    image = ax.pcolormesh(longitude[::-1], latitude, grid_map, cmap=cmap, norm=norm, shading='auto', antialiased=False)

    # Add a colorbar with adjusted label spacing
    baroffset = 0.15  # Increase this to move the colorbar further from the plot
    cb = fig.colorbar(image, orientation='horizontal', shrink=.95, pad=baroffset, ticks=[np.min(m), mean_value, np.max(m)])
    cb.set_label('%', fontsize=10)  # Set the colorbar label
    cb.ax.xaxis.labelpad = 15  # Adjust space between label and ticks
    cb.solids.set_edgecolor("face")  # workaround for issue with viewers

    # Set grid and labels
    ax.set_longitude_grid(60)
    ax.xaxis.set_major_formatter(ThetaFormatterShiftPi(60))
    ax.set_longitude_grid_ends(90)

    # Show or hide ticks based on `show_ticks` argument
    if not show_ticks:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.ylabel(r'Zenith $\theta\,\,(^\circ)$')
    plt.xlabel(r'Azimuth $\phi\,\,(^\circ)$')

    # Grid and title
    plt.grid(color='black', alpha=0.5, linestyle='--', linewidth=0.5)
    plt.title(title)

    # Save the figure
    plt.savefig(f'{save_name}', bbox_inches='tight', pad_inches=0.02)

    # Display the plot
    plt.show()
    
    
NPIX = hp.nside2npix(NSIDE)
print(NPIX)
wv = 550
# Load the data into a DataFrame, assuming the file is space-delimited
df = pd.read_csv(f'/scratch/tmp/fvaracar/geant_eff_areas/lom16_eff_area_wavelength_{wv}.dat', delim_whitespace=True, header=None)

# Retrieve the second-to-last column
m = df.iloc[:, -2]
m2 = np.load(f"/scratch/tmp/fvaracar/nn_eff_areas/eff_area_{wv}_old.npy")

m = m - m2 * np.mean(m)/np.mean(m2)  

m = np.array(m)

m_polar = df.iloc[:,4]
m_equtorial = df.iloc[:,12]

plot_mollweide_projection(m,title=r'$(\mathrm{Geant4} - \mathrm{NN} \cdot \frac{\mathrm{mean}(\mathrm{Geant4})}{\mathrm{mean}(\mathrm{NN})}) / \mathrm{Geant4}$', show_ticks=False, save_name = f"/scratch/tmp/fvaracar/OMClassifier/compare_plots/scaled_diff_{wv}.png")

m = df.iloc[:, -2]
m = np.array(m)
plt.figure(figsize=(5,3))
hp.mollview(m, title=fr"Geant4 $\bar{{A}}_{{eff}} = {np.mean(m):.2f}$")
hp.graticule()
plt.savefig(f"/scratch/tmp/fvaracar/OMClassifier/compare_plots/geant4_{wv}.png")

m2 = np.load(f"/scratch/tmp/fvaracar/nn_eff_areas/eff_area_{wv}_old.npy")
plt.figure(figsize=(5,3))
hp.mollview(m2, title=fr"NN $\bar{{A}}_{{eff}} = {np.mean(m2):.2f}$")
hp.graticule()
plt.savefig(f"/scratch/tmp/fvaracar/OMClassifier/compare_plots/nn_{wv}.png")

plt.figure(figsize=(5,3))
m_polar = (np.array(m_polar)/1e6)*np.pi*23**2
hp.mollview(m_polar, title=fr"Geant4 Polar PMT $\bar{{A}}_{{eff}} = {np.mean(m_polar):.2f}$")
hp.graticule()
plt.savefig(f"/scratch/tmp/fvaracar/OMClassifier/compare_plots/geant_{wv}_polar.png")

plt.figure(figsize=(5,3))
m_equtorial = (np.array(m_equtorial)/1e6)*np.pi*23**2
hp.mollview(m_equtorial, title=fr"Geant4 Equatorial PMT $\bar{{A}}_{{eff}} = {np.mean(m_equtorial):.2f}$")
hp.graticule()
plt.savefig(f"/scratch/tmp/fvaracar/OMClassifier/compare_plots/geant_{wv}_eq.png")


plt.figure(figsize=(5,3))
m_polar = np.load(f"/scratch/tmp/fvaracar/nn_eff_areas/polar_eff_area_{wv}_old.npy")
hp.mollview(m_polar, title=fr"NN Polar PMT $\bar{{A}}_{{eff}} = {np.mean(m_polar):.2f}$")
hp.graticule()
plt.savefig(f"/scratch/tmp/fvaracar/OMClassifier/compare_plots/nn_{wv}_polar.png")

plt.figure(figsize=(5,3))
m_equtorial = np.load(f"/scratch/tmp/fvaracar/nn_eff_areas/eq_eff_area_{wv}_old.npy")
hp.mollview(m_equtorial, title=fr"NN Equatorial PMT $\bar{{A}}_{{eff}} = {np.mean(m_equtorial):.2f}$")
hp.graticule()
plt.savefig(f"/scratch/tmp/fvaracar/OMClassifier/compare_plots/nn_{wv}_eq.png")