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
pModel  = 4
qModel  = 1

# This seems to fail when pModel is odd and qModel = 1, and works when pModel is even and qModel=1
carma_model = carmcmc.CarmaMCMC(xv, yv, dyv, pModel, nSample, q=qModel, nburnin=nBurnin, nthin=nThin)
post = carma_model.RunMCMC()
