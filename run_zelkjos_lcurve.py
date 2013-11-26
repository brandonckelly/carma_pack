__author__ = 'brandonkelly'

import numpy as np
import carmcmc as cm
import matplotlib.pyplot as plt
import os

dir = os.environ['HOME'] + '/data/variability/zeljko/'

lc1 = np.genfromtxt(dir + '23417507.txt')
lc2 = np.genfromtxt(dir + 'catalina_lightcurve.txt')
lc3 = np.genfromtxt(dir + 'lc_PTFS1310w.txt')

t = np.hstack((lc1[:, 0], lc2[:, 2], lc3[:, 0] - 2.4e6))
mag = np.hstack((lc1[:, 1], lc2[:, 0] + 0.1, lc3[:, 1] - 0.15))
merr = np.hstack((lc1[:, 2], lc2[:, 1], lc3[:, 2]))

good = np.where(mag > 16.0)[0]
t = t[good]
mag = mag[good]
merr = merr[good]

torder = np.argsort(t)
t = t[torder]
mag = mag[torder]
merr = merr[torder]

plt.errorbar(t, mag, yerr=merr, fmt='.')
plt.show()
plt.close()

model = cm.CarmaModel(t, mag, merr, p=5, q=1)
post = model.run_mcmc(50000, nburnin=25000)