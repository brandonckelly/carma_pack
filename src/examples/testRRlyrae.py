import sys, os
import yamcmcpp 
import carmcmc 
import numpy as np
import matplotlib.pyplot as plt

def loadData(infile):
    dtype = np.dtype([("mjd", np.float),
                      ("filt", np.str, 1),
                      ("mag", np.float),
                      ("dmag", np.float)])
    data = np.loadtxt(infile, comments="#", dtype=dtype)
    uIdx = np.where(data["filt"] == "u")
    gIdx = np.where(data["filt"] == "g")
    rIdx = np.where(data["filt"] == "r")
    iIdx = np.where(data["filt"] == "i")
    zIdx = np.where(data["filt"] == "z")

    return (data["mjd"][uIdx], data["mag"][uIdx], data["dmag"][uIdx]), \
        (data["mjd"][gIdx], data["mag"][gIdx], data["dmag"][gIdx]), \
        (data["mjd"][rIdx], data["mag"][rIdx], data["dmag"][rIdx]), \
        (data["mjd"][iIdx], data["mag"][iIdx], data["dmag"][iIdx]), \
        (data["mjd"][zIdx], data["mag"][zIdx], data["dmag"][zIdx]), 


if __name__ == "__main__":
    u, g, r, i, z = loadData("/astro/users/acbecker/SDSS/RRLyrae/CAR/1640797.txt")

    opt = yamcmcpp.MCMCOptions()
    opt.setSampleSize(10000)
    opt.setThin(1)
    opt.setBurnin(1000)
    opt.setChains(5)
    pModel = 4
    nWalkers = 10

    for f in (u, g, r, i , z):
        x, y, dy = f
        trace    = carmcmc.RunEnsembleCarSampler(opt, x, y, dy, pModel, nWalkers)
        sample   = carmcmc.CarSample(x, y, dy, trace=trace)
        import pdb; pdb.set_trace()
