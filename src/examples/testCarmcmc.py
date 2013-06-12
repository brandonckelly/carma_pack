import sys, os
import yamcmcpp 
import carmcmc 
import numpy as np
import matplotlib.pyplot as plt

infile   = sys.argv[1]
x, y, dy = np.loadtxt(sys.argv[1], unpack=True)
#dy     = 0.01 * np.ones(len(x))
#dy     = np.zeros(len(x))

opt = yamcmcpp.MCMCOptions()
opt.setSampleSize(100000)
opt.setThin(1)
opt.setBurnin(10000)
opt.setChains(5)
pModel = 4
nWalkers = 10

trace  = carmcmc.RunEnsembleCarSampler(opt, x, y, dy, pModel, nWalkers)
sample = carmcmc.CarSample(x, y, dy, trace=trace)
#sample.autocorr_timescale(sample._samples["ar_coefs"])
sample.effective_samples("ar_coefs")
#sample.plot_trace("ar_coefs")
#sample.plot_1dpdf("ar_coefs")
#sample.plot_autocorr("ar_coefs")
#sample.plot_2dpdf("log_centroid", 0, "log_centroid", 1)
#sample.plot_2dkde("ar_roots", 1, 2)
for i in range(pModel//2):
    sample.plot_2dpdf("log_centroid", i, "log_width", i)

for i in range(pModel-1):
    sample.plot_2dpdf("ar_roots", i, "ar_roots", i+1)
    sample.plot_2dpdf("ar_coefs", i, "ar_coefs", i+1)

import pdb; pdb.set_trace()

sample.plot_power_spectrum()

kalman_filter(sample.time, sample.y - self.y.mean(), sample.dy, 
              sample._samples["var"][100], sample._samples["ar_roots"][100])

    
kmean, kvar = carmcmc.kalman_filter(sample.time, sample.y - sample.y.mean(), sample.ysig**2,  
                                    sample._samples["var"][100], sample._samples["ar_roots"][100])
chi2        = (sample.y - sample.y.mean() - kmean)**2 / kvar
print -0.5 * np.sum(chi2)
plt.errorbar(sample.time, sample.y, yerr=sample.ysig, fmt="ro")
plt.plot(sample.time, kmean, "k-")
