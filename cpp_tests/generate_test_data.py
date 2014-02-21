
__author__ = 'bkelly'

import numpy as np
from carmcmc.carma_pack import carma_process, get_ar_roots, carma_variance, power_spectrum
from scipy.misc import comb
from scipy import stats
import matplotlib.pyplot as plt

ax1 = plt.subplot(211)

ny = 1000
sigmay = 2.3  # standard deviation of the time series
yerr = 0.1 * sigmay * np.sqrt(10.0 / np.random.chisquare(10.0, ny))  # standard deviations of measurement errors

# generate time values
dt = 0.1 + np.abs(np.random.standard_cauchy(ny))  # use a Cauchy distribution to simulate long gaps in time series
time = np.cumsum(dt)
time = time - time.min()

###### generate CAR(1) process
tau = 100.0
sigma = sigmay * np.sqrt(2.0 / tau)

car1 = np.empty(ny)
car1[0] = np.random.normal(0.0, sigmay)
for i in xrange(1, ny):
    dt = time[i] - time[i-1]
    rho = np.exp(-dt / tau)
    car1[i] = np.random.normal(rho * car1[i-1], sigmay * np.sqrt(1.0 - rho ** 2))

plt.plot(time, car1)

car1 += np.random.normal(0.0, yerr)  # add measurement errors

plt.plot(time, car1, '.')
plt.ylabel('CAR(1)')

# save the CAR(1) data
car1_data = np.vstack((time, car1, yerr))
# np.savetxt("data/car1_test.dat", car1_data.transpose(), fmt='%10.5f')

###### generate a CAR(5,4) process using the Belcher et al. (1994) notation, i.e., a ZCAR(5) process

plt.subplot(212, sharex=ax1)

p = 5  # order of AR polynomial
qpo_width = np.array([1.0/100.0, 1.0/100.0, 1.0/500.0])
qpo_cent = np.array([1.0/5.0, 1.0/50.0])
ar_roots = get_ar_roots(qpo_width, qpo_cent)

# calculate moving average coefficients under z-transform of Belcher et al. (1994)
dt = time[1:] - time[0:-1]
# kappa = 1.0 / dt.min()
kappa = 0.5
ma_coefs = comb(p-1 * np.ones(p), np.arange(p)) / kappa ** np.arange(p)

sigsqr = sigmay ** 2 / carma_variance(1.0, ar_roots, ma_coefs=ma_coefs)
print sigsqr, ar_roots, ma_coefs
zcar5 = carma_process(time, sigsqr, ar_roots, ma_coefs=ma_coefs)

print np.std(zcar5), np.std(car1), np.sqrt(carma_variance(sigsqr, ar_roots, ma_coefs=ma_coefs))

plt.plot(time, zcar5)

zcar5 += np.random.normal(0.0, yerr)

# save the CAR(5) data
zcarma_data = np.vstack((time, zcar5, yerr))
np.savetxt("data/carma_test.dat", zcarma_data.transpose(), fmt='%10.5f')

plt.plot(time, zcar5, '.')
plt.xlabel("Time")
plt.ylabel("ZCAR(5)")
plt.show()

freq_max = 1.0 / np.min(time[1:] - time[0:-1])
freq_min = 1.0 / (np.max(time) - np.min(time))
freq = np.logspace(np.log10(freq_min), np.log10(freq_max), num=256)
ar_coef = np.poly(ar_roots)
psd = power_spectrum(freq, np.sqrt(sigsqr), ar_coef, ma_coefs=ma_coefs)
plt.loglog(freq, psd, lw=3)
plt.xlabel('Frequency')
plt.ylabel('PSD, ZCAR(5)')
plt.show()

# covar = np.empty((ny, ny))
# for i in xrange(ny):
#     print i
#     for j in xrange(ny):
#         dt = np.abs(time[i] - time[j])
#         covar[i, j] = carma_variance(sigsqr, ar_roots, ma_coefs=ma_coefs, lag=dt)
#         if i == j:
#             covar[i, j] += yerr[i] * yerr[j]
#
# L = np.linalg.cholesky(covar)
# znorm = np.linalg.inv(L).dot(zcarma5)
# print znorm.mean(), znorm.std()
# print stats.anderson(znorm)

