import sys, os
import yamcmcpp
import carmcmc
import numpy as np
import matplotlib.pyplot as plt

infile = sys.argv[1]
xv, yv, dyv = np.loadtxt(sys.argv[1], unpack=True)

nSample = 100
nBurnin = 10
nThin   = 1
pModel  = 3
qModel  = 1

carma1  = carmcmc.CarmaMCMC(xv, yv, dyv, 1, nSample, q=0, nburnin=nBurnin, nthin=nThin)
post1   = carma1.RunMCMC()

carmap  = carmcmc.CarmaMCMC(xv, yv, dyv, pModel, nSample, q=0, nburnin=nBurnin, nthin=nThin)
postp   = carmap.RunMCMC()

carmapqo = carmcmc.CarmaMCMC(xv, yv, dyv, pModel, nSample, q=qModel, nburnin=nBurnin, nthin=nThin)
postpqo  = carmapqo.RunMCMC()

carmapqe = carmcmc.CarmaMCMC(xv, yv, dyv, pModel+1, nSample, q=qModel, nburnin=nBurnin, nthin=nThin)
postpqe  = carmapqe.RunMCMC()

for bestfit in ["map", "median", "mean"]:
    post1.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
    postp.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
    postpqo.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
    postpqe.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
import pdb; pdb.set_trace()

# tests of yamcmcpp samplers.py

post1.effective_samples("sigma")
postp.effective_samples("ar_roots")
postpqo.effective_samples("ma_coefs")
postpqe.effective_samples("ar_coefs")

post1.plot_trace("sigma")
postp.plot_trace("sigma")
postpqo.plot_trace("sigma")
postpqe.plot_trace("sigma")

post1.plot_1dpdf("mu")
postp.plot_1dpdf("mu")
postpqo.plot_1dpdf("mu")
postpqe.plot_1dpdf("mu")

post1.plot_1dpdf("psd_centroid")
postp.plot_1dpdf("psd_centroid")
postpqo.plot_1dpdf("psd_width")
postpqe.plot_1dpdf("psd_width")

post1.plot_2dpdf("sigma", "var")
postp.plot_2dpdf("sigma", "var", pindex1=0, pindex2=0)
postpqo.plot_2dpdf("sigma", "var", pindex1=1, pindex2=1)
postpqe.plot_2dpdf("sigma", "var", pindex1=2, pindex2=2)

post1.plot_2dkde("sigma", "var")
postp.plot_2dkde("sigma", "var", pindex1=0, pindex2=0)
postpqo.plot_2dkde("sigma", "var", pindex1=1, pindex2=1)
postpqe.plot_2dkde("sigma", "var", pindex1=2, pindex2=2)

post1.plot_autocorr("psd_centroid")
postp.plot_autocorr("psd_centroid")
postpqo.plot_autocorr("psd_centroid")
postpqe.plot_autocorr("psd_centroid")

post1.plot_parameter("psd_centroid")
postp.plot_parameter("psd_centroid")
postpqo.plot_parameter("psd_centroid")
postpqe.plot_parameter("psd_centroid")

post1.posterior_summaries("psd_width")
postp.posterior_summaries("psd_width")
postpqo.posterior_summaries("psd_width")
postpqe.posterior_summaries("psd_width")

post1.posterior_summaries("sigma")
postp.posterior_summaries("sigma")
postpqo.posterior_summaries("sigma")
postpqe.posterior_summaries("sigma")

# tests of carma_pack carma_pack.py

post1.plot_power_spectrum(percentile=95.0, doShow=False)
postp.plot_power_spectrum(percentile=95.0, doShow=False)
postpqo.plot_power_spectrum(percentile=95.0, doShow=False)
postpqe.plot_power_spectrum(percentile=95.0, doShow=False)

post1.plot_models(nplot=1000, doShow=False)
postp.plot_models(nplot=1000, doShow=False)
postpqo.plot_models(nplot=1000, doShow=False)
postpqe.plot_models(nplot=1000, doShow=False)

for bestfit in ["map", "median", "mean"]:
    post1.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
    postp.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
    postpqo.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
    postpqe.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
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
