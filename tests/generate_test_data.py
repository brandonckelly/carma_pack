
__author__ = 'bkelly'

import numpy as np
from carmcmc.carma_pack import carp_process, get_ar_roots, carp_variance
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
np.savetxt("data/car1_test.dat", car1_data.transpose(), fmt='%10.5f')

###### generate a CAR(5) process

plt.subplot(212, sharex=ax1)

qpo_width = np.array([1.0/100.0, 1.0/100.0, 1.0/500.0])
qpo_cent = np.array([1.0/5.0, 1.0/50.0])
ar_roots = get_ar_roots(qpo_width, qpo_cent)

sigsqr = sigmay ** 2 / carp_variance(1.0, ar_roots)

car5 = carp_process(time, sigsqr, ar_roots)

plt.plot(time, car5)

car5 += np.random.normal(0.0, yerr)

# save the CAR(5) data
car5_data = np.vstack((time, car5, yerr))
np.savetxt("data/car5_test.dat", car5_data.transpose(), fmt='%10.5f')

plt.plot(time, car5, '.')
plt.xlabel("Time")
plt.ylabel("CAR(5)")
plt.show()

