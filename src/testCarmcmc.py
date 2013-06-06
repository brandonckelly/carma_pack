import sys, os
import yamcmcpp 
import carmcmc 
import numpy as np
import matplotlib.pyplot as plt

infile = sys.argv[1]
x, y   = np.loadtxt(sys.argv[1], unpack=True)
#dy     = 0.01 * np.ones(len(x))
dy     = np.zeros(len(x))

opt = yamcmcpp.MCMCOptions()
opt.setSampleSize(1000)
opt.setThin(1)
opt.setBurnin(100)
opt.setChains(5)
pModel = 3
nWalkers = 10

trace  = carmcmc.RunEnsembleCarSampler(opt, x, y, dy, pModel, nWalkers)
sample = carmcmc.CarSample(x, y, dy, trace=trace)
#sample.autocorr_timescale(sample._samples["ar_coefs"])
sample.effective_samples("ar_coefs")
#sample.plot_trace("ar_coefs")
#sample.plot_1dpdf("ar_coefs")
sample.plot_autocorr("ar_coefs")
import pdb; pdb.set_trace()
