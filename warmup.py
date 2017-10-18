import numpy as np
import matplotlib.pyplot as plt
import math

import astropy.io.fits as fits
import astropy.units as u
from spectral_cube import SpectralCube
%matplotlib inline

from galaxies import Galaxy


# Import data from .fits files.
I_mom0 = fits.getdata('ngc1672_co21_12m+7m+tp_mom0.fits')
I_max = fits.getdata('ngc1672_co21_12m+7m+tp_tpeak.fits')


# Calculate line width, sigma.
sigma = I_mom0 / (np.sqrt(2*np.pi * I_max))


# Plotting sigma versus the mom0 array.
plt.legend(loc='lower right')
plt.plot(np.ravel(I_mom0), np.ravel(sigma), 'k.')
plt.xlabel('$I_{mom0}$')
plt.ylabel('$\sigma$')
plt.title('Line Width vs mom0 Plot')
plt.xscale('log')
plt.yscale('log')

plt.savefig('warmup_linewidth_vs_mom0.png')
plt.clf()


# Importing radius of each data point.
gal = Galaxy('NGC1672')
hdr = fits.getheader('ngc1672_co21_12m+7m+tp_mom0.fits')
rad = gal.radius(header=hdr)


# Plotting sigma vs radius.
plt.legend(loc='lower right')
plt.semilogy(np.ravel(rad), np.ravel(sigma), 'k.')
plt.xlabel('$Radius$ $(Mpc)$')
plt.ylabel('$\sigma$')
plt.title('Line Width vs Radius')

plt.savefig('warmup_linewidth_vs_radius.png')
plt.clf()
