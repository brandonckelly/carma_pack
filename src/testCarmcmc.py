import sys, os
import yamcmcpp.yamcmcppLib as yamcmcpp
import carmcmc.carmcmcLib as carmcmc
import numpy as np

infile = sys.argv[1]
x, y   = np.loadtxt(sys.argv[1], unpack=True)
dy     = 0.01 * np.ones(len(x))

opt = yamcmcpp.MCMCOptions()
opt.setSampleSize(100000)
opt.setThin(1)
opt.setBurnin(1000)
opt.setChains(5)
opt.setDataFileName("foo1.txt")
opt.setOutFileName("foo2.txt")
pModel = 1
nWalkers = 10

#import pdb; pdb.set_trace()
results = carmcmc.RunEnsembleCarSampler(opt, x, y, dy, pModel, nWalkers)
import pdb; pdb.set_trace()
