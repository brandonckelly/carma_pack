import sys, os
import yamcmcpp 
import carmcmc 
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

import pdb; pdb.set_trace()

colors = ["b", "g", "r", "m", "k"]

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

    nSample    = 10000
    nBurnin    = 1000
    nThin      = 1
    nWalkers   = 10

    logpost, params = carmcmc.run_mcmc(nSample, nBurnin, x, y, dy, pModel, nWalkers, nThin)
    if pModel == 1:
        sample = carmcmc.CarSample1(x, y, dy, logpost=logpost, trace=params)
    else:
        sample = carmcmc.CarSample(x, y, dy, logpost=logpost, trace=params)
    
    return sample
    

if __name__ == "__main__":
    u, g, r, i, z = loadData("/astro/users/acbecker/SDSS/RRLyrae/CAR/1640797.txt")

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(int, range(multiprocessing.cpu_count())) 

    args = []
    #for f in (u, g, r, i , z):
    for f in (r,):
        for pModel in np.arange(1, 3):
            args.append((pModel, f))
    results = pool.map(doit, args)
    import pdb; pdb.set_trace()

    fig = plt.figure()
    sp0 = fig.add_subplot(3, 3, 1)
    for i in range(min(9, len(results))):
        sample = results[i]

        if i == 0:
            sp = sp0
        else:
            sp = fig.add_subplot(3, 3, i+1, sharex=sp0, sharey=sp0)

        ps = sample.plot_power_spectrum(color=colors[0], sp=sp)
        sp.set_title("Order %d" % (i+1))
    
    plt.show()
