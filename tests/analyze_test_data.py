__author__ = 'brandonkelly'

import numpy as np
import matplotlib.pyplot as plt
from os import environ
from scipy.misc import comb
import carmcmc

# true values
p = 5  # order of AR polynomial
sigmay = 2.3
qpo_width = np.array([1.0/100.0, 1.0/100.0, 1.0/500.0])
qpo_cent = np.array([1.0/5.0, 1.0/50.0])
ar_roots = carmcmc.get_ar_roots(qpo_width, qpo_cent)
# calculate moving average coefficients under z-transform of Belcher et al. (1994)
kappa = 0.7
ma_coefs = comb(p-1 * np.ones(p), np.arange(p)) / kappa ** np.arange(p)
sigsqr = sigmay ** 2 / carmcmc.carma_variance(1.0, ar_roots, ma_coefs=ma_coefs)

data_dir = environ['HOME'] + '/Projects/carma_pack/tests/data/'
fname = data_dir + 'zcarma5_mcmc.dat'
data = np.genfromtxt(data_dir + 'zcarma5_test.dat')

Zcarma = carmcmc.ZCarmaSample(data[:, 0], data[:, 1], data[:, 2], filename=fname)

Zcarma.assess_fit()

print "True value of log10(kappa) is: ", np.log10(kappa)
plt.hist(Zcarma.get_samples('kappa'), bins=100)
plt.show()

Zcarma.plot_parameter('kappa', doShow=True)

psd_low, psd_high, psd_mid, freq = Zcarma.plot_power_spectrum(percentile=90.0, nsamples=5000, doShow=False)

ar_coef = np.poly(ar_roots)
true_psd = carmcmc.power_spectrum(freq, np.sqrt(sigsqr), ar_coef, ma_coefs=ma_coefs)
plt.loglog(freq, true_psd, 'r', lw=2)
plt.xlabel('Frequency')
plt.ylabel('PSD, ZCARMA(5)')
plt.show()
