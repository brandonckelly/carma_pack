import sys, os
import yamcmcpp 
import carmcmc 
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

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

def doit(args):
    pModel   = int(args[0])
    x, y, dy = args[1]

    opt = yamcmcpp.MCMCOptions()
    opt.setSampleSize(1000000)
    opt.setThin(1)
    opt.setBurnin(100000)
    opt.setChains(5)
    nWalkers = 10

    return carmcmc.RunEnsembleCarSampler(opt, x, y, dy, pModel, nWalkers)
    

if __name__ == "__main__":
    u, g, r, i, z = loadData("/astro/users/acbecker/SDSS/RRLyrae/CAR/1640797.txt")

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(int, range(multiprocessing.cpu_count())) 

    args = []
    #for f in (u, g, r, i , z):
    for f in (r,):
        for pModel in np.arange(1, 10):
            args.append((pModel, f))
    results = pool.map(doit, args)
    import pdb; pdb.set_trace()
