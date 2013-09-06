import sys, os
import yamcmcpp
import carmcmc
import numpy as np
import matplotlib.pyplot as plt

infile = sys.argv[1]
xv, yv, dyv = np.loadtxt(sys.argv[1], unpack=True)

nSample = 500
nBurnin = 50
nThin   = 1
pModel  = 3
qModel  = 1

carma1  = carmcmc.CarmaMCMC(xv, yv, dyv, 1, nSample, q=0, nburnin=nBurnin, nthin=nThin)
post1   = carma1.RunMCMC()

carmap  = carmcmc.CarmaMCMC(xv, yv, dyv, pModel, nSample, q=0, nburnin=nBurnin, nthin=nThin)
postp   = carmap.RunMCMC()

carmapq = carmcmc.CarmaMCMC(xv, yv, dyv, pModel, nSample, q=qModel, nburnin=nBurnin, nthin=nThin)
postpq  = carmapq.RunMCMC()

# 1-d
post1.plot_1dpdf("mu")
postp.plot_1dpdf("mu")
postpq.plot_1dpdf("mu")

post1.plot_1dpdf("psd_centroid")
postp.plot_1dpdf("psd_centroid")
postpq.plot_1dpdf("psd_centroid")

import pdb; pdb.set_trace()

# plot the 95% probability regions of the power spectrum
post1.plot_power_spectrum(percentile=95.0)
postp.plot_power_spectrum(percentile=95.0)
postpq.plot_power_spectrum(percentile=95.0)

import pdb; pdb.set_trace()

# assess the fit quality
post.assess_fit()

# find out which parameters we can access
print post.parameters

# grab the MCMC samples for the roots of the autoregressive polynomial
ar_roots = post.get_samples('ar_roots')

# grab the MCMC samples for the moving average coefficients
ma_coefs = post.get_samples('ma_coefs')




sampler = carmcmc.run_mcmc(nSample, nBurnin, xv, yv, dyv, pModel, nWalkers, nThin)
sample = carmcmc.CarSample(x, y, dy, sampler)

#sample.autocorr_timescale(sample._samples["ar_coefs"])
sample.posterior_summaries("log_centroid")
#sample.plot_trace("ar_coefs")
#sample.plot_1dpdf("ar_coefs")
#sample.plot_autocorr("ar_coefs")
#sample.plot_2dpdf("log_centroid", 0, "log_centroid", 1)
#sample.plot_2dkde("ar_roots", 1, 2)
for i in range(pModel // 2):
    sample.plot_2dkde("log_centroid", i, "log_width", i, doShow=True)

# for i in range(pModel - 1):
#     sample.plot_2dpdf("ar_roots", i, "ar_roots", i + 1, doShow=True)
#     sample.plot_2dpdf("ar_coefs", i, "ar_coefs", i + 1, doShow=True)

# import pdb;

# pdb.set_trace()

sample.plot_power_spectrum(doShow=True)

print "Calculating interpolated values..."
sample.assess_fit()

print "Calculating forecasted and simulated lightcurve..."
tpredict = np.linspace(580.0, 640.0, 200)
tsim = tpredict
pmean, pvar = sample.predict_lightcurve(tpredict)
ysim = sample.simulate_lightcurve(tsim)
plt.fill_between(tpredict, pmean - np.sqrt(pvar), pmean + np.sqrt(pvar), alpha=0.25)
plt.plot(x, y, 'k.', label='Measured')
plt.plot(tsim, ysim, 'b-', label='Simulated 1')
ysim = sample.simulate_lightcurve(tsim)
plt.plot(tsim, ysim, 'r-', label='Simulated 2')
plt.xlim(tsim.min(), tsim.max())
plt.legend()
plt.show()

# kalman_filter(sample.time, sample.y - self.y.mean(), sample.dy,
#               sample._samples["var"][100], sample._samples["ar_roots"][100])
#
# kmean, kvar = carmcmc.kalman_filter(sample.time, sample.y - sample.y.mean(), sample.ysig ** 2,
#                                     sample._samples["var"][100], sample._samples["ar_roots"][100])
# chi2 = (sample.y - sample.y.mean() - kmean) ** 2 / kvar
# print -0.5 * np.sum(chi2)
# plt.errorbar(sample.time, sample.y, yerr=sample.ysig, fmt="ro")
# plt.plot(sample.time, kmean, "k-")
