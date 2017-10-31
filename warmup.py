import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math
from scipy import ndimage, misc

import astropy.io.fits as fits
import astropy.units as u
from spectral_cube import SpectralCube
from galaxies import Galaxy

from pandas import DataFrame, read_csv
import pandas as pd
import statsmodels.formula.api as smf

# Calculate line width, sigma; and surface density, Sigma.
alpha = 6.7
sigma = I_mom0 / (np.sqrt(2*np.pi * I_max))
Sigma = alpha*I_mom0
# Importing radius of each data point.
gal = Galaxy('NGC1672')
hdr = fits.getheader('ngc1672_co21_12m+7m+tp_mom0.fits')
rad = gal.radius(header=hdr)
rad = (rad * u.Mpc.to(u.kpc)) * u.kpc / u.Mpc           # Converts rad from Mpc to kpc.
# Calculating width of each pixel, in parsecs.
pixsizes_deg = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hdr))     # The size of each pixel, in degrees. Ignore that third dimension; that's pixel size for the speed.
pixsizes = pixsizes_deg[0] * np.pi / 180.                          # Pixel size, in radians.
pcperpixel =  pixsizes*d      					   # Number of parsecs per pixel.
pcperdeg = pcperpixel / pixsizes_deg[0]                            # Number of parsecs per degree.

# Getting beam width, in degrees and parsecs.
beam = hdr['BMAJ']                                      # Beam size, in degrees
beam = beam * pcperdeg                                  # Beam size, in pc

# Import data from .fits files.
I_mom0 = fits.getdata('ngc1672_co21_12m+7m+tp_mom0.fits')
I_mom1 = fits.getdata('ngc1672_co21_12m+7m+tp_mom1.fits')      # Intensity-weighted mean velocity of data, in km/s/pixel.
I_max = fits.getdata('ngc1672_co21_12m+7m+tp_tpeak.fits')
gradv = ndimage.sobel(I_mom1/pcperpixel)              # Sobel gradient of I_mom1; = grad(mean velocity), in km/s/pc
gradvl = ndimage.sobel(I_mom1 * beam / pcperpixel)    # Sobel gradient of I_mom1 * beam width, in km/s



# Plotting sigma versus the mom0 intensity.
plt.legend(loc='lower right')
plt.plot(np.ravel(I_mom0), np.ravel(sigma), 'k.')
plt.xlabel('$I_{mom0}$')
plt.ylabel('$\sigma$')
plt.title('Line Width vs mom0 Intensity')
plt.xscale('log')
plt.yscale('log')

plt.savefig('warmup_linewidth_vs_mom0.png')
plt.clf()

# Plotting sigma vs a Sobel image gradient of the intensity-weighted velocity (mom1) times beam width.
plt.legend(loc='lower right')
plt.plot(np.ravel(gradvl), np.ravel(sigma), 'k.')
plt.xlabel('grad($v_c*l$) (km/s)')
plt.ylabel('$\sigma$')
plt.title('Line Width vs Sobel Gradient of (Mean Velocity * Beam Width)')
plt.xscale('log')
plt.yscale('log')

plt.savefig('warmup_linewidth_vs_gradvl.png')
plt.clf()

# Plotting sigma vs radius.
plt.legend(loc='lower right')
plt.semilogy(np.ravel(rad), np.ravel(sigma), 'k.')
plt.xlabel('$Radius$ $(kpc)$')
plt.ylabel('$\sigma$')
plt.title('Line Width vs Radius')

plt.savefig('warmup_linewidth_vs_radius.png')
plt.clf()




# Create log arrays for sigma, Sigma, sobel(mean velocity), and radius.
logsigma = np.ravel(np.log10(sigma))
logSigma = np.ravel(np.log10(Sigma))
loggradv = np.ravel(np.log10(gradv))
logR = np.ravel(np.log10(rad.value))

# Deleting any values that are not finite
index = np.arange(logsigma.size)
a = np.isfinite(logsigma)==False      # A boolean array that is True where logsigma is nan or inf.
b = np.isfinite(logSigma)==False
c = np.isfinite(loggradv)==False
d = np.isfinite(logR) == False        # A bool. array that is True where logsigma, logSigma, OR logR is nan or inf.
logsigma_clean = np.delete(logsigma, index[a+b+c+d])
logSigma_clean = np.delete(logSigma, index[a+b+c+d])
loggradv_clean = np.delete(loggradv, index[a+b+c+d])
logR_clean = np.delete(logR, index[a+b+c+d])
R_clean = np.delete(rad, index[a+b+c+d])

# Overwriting the main arrays with the cleaned versions
logsigma = logsigma_clean
logSigma = logSigma_clean
loggradv = loggradv_clean
logR = logR_clean
R = R_clean.value



# Creating and saving a dataframe of logsigma, logSigma, loggradv, logR
dataframe = pd.DataFrame(data=zip(np.ravel(logsigma), np.ravel(logSigma), np.ravel(loggradv)), columns = ["$log(\sigma)$", "$log(\Sigma)$", "$log(gradv)$"])
#dataframe = pd.DataFrame(data=zip(np.ravel(logsigma), np.ravel(logSigma), np.ravel(logR)), columns = ["$log(\sigma)$", "$log(\Sigma)$", "$log(R)$"])
#dataframe = pd.DataFrame(data=zip(np.ravel(logsigma), np.ravel(logSigma), np.ravel(R)), columns = ["$log(\sigma)$", "$log(\Sigma)$", "$R$"])
dataframe.to_csv('sigma_Sigma_gradv.csv',index=False,header=False)
#dataframe.to_csv('sigma_Sigma_R.csv',index=False,header=False)

# Reloading this same dataframe, just for kicks
df = pd.read_csv('sigma_Sigma_gradv.csv', names = ['logsigma','logSigma','loggradv'])
#df = pd.read_csv('sigma_Sigma_R.csv', names = ['logsigma','logSigma','logR'])
#df = pd.read_csv('sigma_Sigma_R.csv', names = ['logsigma','logSigma','R'])


# Fitting: logsigma ~ logSigma + logR (or loggradv)
mod = smf.ols(formula='logsigma ~ logSigma + loggradv', data=df)
#mod = smf.ols(formula='logsigma ~ logSigma + R', data=df)
res = mod.fit()
print(res.summary())
