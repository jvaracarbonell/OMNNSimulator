import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import glob
NSIDE = 32
print(
    "Approximate resolution at NSIDE {} is {:.2} deg".format(
        NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
    )
)
NPIX = hp.nside2npix(NSIDE)
print(NPIX)
QE_dict = {
  270.0: 0.0,
  280.0: 0.00341,
  290.0: 0.00955,
  300.0: 0.04397,
  310.0: 0.12224,
  320.0: 0.2022,
  330.0: 0.25633,
  340.0: 0.28621,
  350.0: 0.30071,
  360.0: 0.31081,
  370.0: 0.30876,
  380.0: 0.30748,
  390.0: 0.30843,
  400.0: 0.30273,
  410.0: 0.2907,
  420.0: 0.28041,
  430.0: 0.26431,
  440.0: 0.25164,
  450.0: 0.23893,
  460.0: 0.21902,
  470.0: 0.1979,
  480.0: 0.18105,
  490.0: 0.16625,
  500.0: 0.15631,
  510.0: 0.1433,
  520.0: 0.1144,
  530.0: 0.08245,
  540.0: 0.06259,
  550.0: 0.04844,
  560.0: 0.03797,
  570.0: 0.03021,
  580.0: 0.02287,
  590.0: 0.01558,
  600.0: 0.0103,
  610.0: 0.00708,
  620.0: 0.00343,
  630.0: 0.00267,
  640.0: 0.00076,
  650.0: 0.0,
  660.0: 0.0,
  670.0: 0.0,
  680.0: 0.0,
  690.0: 0.0,
  700.0: 0.0
}

files = glob.glob("/scratch/tmp/fvaracar/geant_eff_areas/*dat")
for file in files:
    wv = file.split("_")[-1]
    wv = int(wv.split(".")[0])
    # Load the data into a DataFrame, assuming the file is space-delimited
    df = pd.read_csv(f'/scratch/tmp/fvaracar/geant_eff_areas/lom16_eff_area_wavelength_{wv}.dat', delim_whitespace=True, header=None)

    # Retrieve the second-to-last column
    if wv in QE_dict.keys():
        QE = QE_dict[wv]
    else:
        QE=0
    m = df.iloc[:, -2]#*QE
    m_pol = df.iloc[:, -4]/1000000*np.pi*(23/2)**2#*QE
    m_eq = df.iloc[:, -8]/1000000*np.pi*(23/2)**2#*QE

    # Healpy directions with nside=32
    #nside = 32
    #npix = hp.nside2npix(nside)
    #theta, phi = hp.pix2ang(nside, np.arange(npix))  
    #n = []
    # Iterate over each Healpy direction
    #for index, (zenith, azimuth) in enumerate(zip(np.degrees(theta), np.degrees(phi))):
    #    if zenith > 90:
    #        continue
    #    if azimuth > 90:
    #        continue
    #    n.append(m[index])
    #print(f"Mean eff area is: {(np.mean(n))} {wv} nm") 
    

    hp.mollview(m, title=f"Effective area wv={wv} nm, average={round(np.mean(m),2)}")
    hp.graticule()
    plt.savefig(f"/scratch/tmp/fvaracar/geant_eff_areas/plots/geant_eff_area_{wv}_weigth.png")
    hp.mollview(m_pol, title=f"Polar PMT Effective area wv={wv} nm, average={round(np.mean(m_pol),2)}")
    hp.graticule()
    plt.savefig(f"/scratch/tmp/fvaracar/geant_eff_areas/plots/geant_eff_area_{wv}_polar_weigth.png")
    hp.mollview(m_eq, title=f"Eq PMT Effective area wv={wv} nm, average={round(np.mean(m_eq),2)}")
    hp.graticule()
    plt.savefig(f"/scratch/tmp/fvaracar/geant_eff_areas/plots/geant_eff_area_{wv}_eq_weigth.png")
    