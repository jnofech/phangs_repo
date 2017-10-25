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


# Import data from .fits files.
I_mom0 = fits.getdata('ngc1672_co21_12m+7m+tp_mom0.fits')
I_mom1 = fits.getdata('ngc1672_co21_12m+7m+tp_mom1.fits')      # Intensity-weighted mean velocity of data.
I_max = fits.getdata('ngc1672_co21_12m+7m+tp_tpeak.fits')

sobel = ndimage.sobel(I_mom1)        # Sobel image gradient of I_mom1. Highlights edges.


# Calculate line width, sigma; and surface density, Sigma.
alpha = 6.7
sigma = I_mom0 / (np.sqrt(2*np.pi * I_max))
Sigma = alpha*I_mom0
# Importing radius of each data point.
gal = Galaxy('NGC1672')
hdr = fits.getheader('ngc1672_co21_12m+7m+tp_mom0.fits')
rad = gal.radius(header=hdr)
rad = (rad * u.Mpc.to(u.kpc)) * u.kpc / u.Mpc           # Converts rad from Mpc to kpc.

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

# Plotting sigma vs a Sobel image gradient of the intensity-weighted velocity (mom1).
plt.legend(loc='lower right')
plt.plot(np.ravel(sobel), np.ravel(sigma), 'k.')
plt.xlabel('Sobel Gradient of Intensity-Weighted Mean Velocity')
plt.ylabel('$\sigma$')
plt.title('Line Width vs Sobel Gradient of Intensity-Weighted Mean Velocity')
#plt.xscale('log')
#plt.yscale('log')

plt.savefig('warmup_linewidth_vs_mom1.png')
#plt.clf()

# Plotting sigma vs radius.
plt.legend(loc='lower right')
plt.semilogy(np.ravel(rad), np.ravel(sigma), 'k.')
plt.xlabel('$Radius$ $(kpc)$')
plt.ylabel('$\sigma$')
plt.title('Line Width vs Radius')

plt.savefig('warmup_linewidth_vs_radius.png')
plt.clf()




# Create log arrays for sigma, Sigma, and radius.
logsigma = np.log10(sigma)
logSigma = np.log10(Sigma)
logR = np.log10(rad.value)

# Deleting any values that are not finite
index = np.arange(logsigma.size)
a = np.isfinite(logsigma)==False      # A boolean array that is True where logsigma is nan or inf.
b = np.isfinite(logSigma)==False
c = np.isfinite(logR) == False        # A bool. array that is True where logsigma, logSigma, OR logR is nan or inf.
logsigma_clean = np.delete(logsigma, index[a+b+c])
logSigma_clean = np.delete(logSigma, index[a+b+c])
logR_clean = np.delete(logR, index[a+b+c])

# Overwriting the main arrays with the cleaned versions
logsigma = logsigma_clean
logSigma = logSigma_clean
logR = logR_clean



# Creating and saving a dataframe of logsigma, logSigma, logR
dataframe = pd.DataFrame(data=zip(np.ravel(logsigma), np.ravel(logSigma), np.ravel(logR)), columns = ["$log(\sigma)$", "$log(\Sigma)$", "$log(R)$"])
dataframe.to_csv('sigma_Sigma_R.csv',index=False,header=False)
# Reloading this same dataframe, just for kicks
df = pd.read_csv('sigma_Sigma_R.csv', names = ['logsigma','logSigma','logR'])


# Fitting: logsigma ~ logSigma + logR
mod = smf.ols(formula='logsigma ~ logSigma + logR', data=df)
res = mod.fit()
print(res.summary())
