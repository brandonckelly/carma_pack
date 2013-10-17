__author__ = 'brandonkelly'

import numpy as np
import carmcmc as cm
import matplotlib.pyplot as plt
import multiprocessing as mp
from os import environ
import pickle
from scipy.misc import comb


base_dir = environ['HOME'] + '/Projects/carma_pack/src/paper/plots/'


def run_carma_sampler(args):

    pmodel = args[0]
    qmodel = args[1]
    time, y, ysig = args[2]
    nsamples = 75000

    carma_mcmc = cm.CarmaMCMC(time, y, ysig, pmodel, nsamples, q=qmodel, nburnin=25000)
    carma = carma_mcmc.RunMCMC()

    return carma


def make_sampler_plots(time, y, ysig, pmodels, file_root, pmax=9):

    froot = base_dir + file_root
    data = (time, y, ysig)

    pool = mp.Pool(mp.cpu_count()-1)

    args = []
    for p in xrange(1, pmax + 1):
        args.append((p, data))

    carma_run = pool.map(run_carma_sampler, args)

    dic = []
    pmodels = []
    qmodels = []
    for crun in carma_run:
        dic.append(crun.DIC())
        pmodels.append(crun.p)
        qmodels.append(crun.q)

    pmodels = np.array(pmodels)
    qmodels = np.array(qmodels)
    dic = np.array(dic)

    plt.subplot(111)
    for i in xrange(qmodels.max()+1):
        if i == qmodels.max():
            marker = 's'
        else:
            markers = '-'
        plt.plot(pmodels[qmodels == i], dic[qmodels == i], marker, label='q=' + str(i))
    plt.plot(pmodels[qmodels == qmodels.max()], dic[qmodels == qmodels.max()])
    plt.xlim(0, pmodels.max() + 1)
    plt.xlabel('p')
    plt.ylabel('DIC')
    plt.legend()
    plt.show()
    print "DIC", dic
    plt.savefig(file_root + 'DIC.eps')

    carma = carma_run[np.argmin(dic)]

    print "order of best model is", carma.p

    carma.plot_power_spectrum(percentile=95.0, nsamples=5000)
    plt.savefig(file_root + 'PSD.eps')

    carma.assess_fit()
    plt.savefig(file_root + 'fit_quality.eps')


def do_simulated_regular():

    # first generate some data assuming a CAR(5) process on a uniform grid
    sigmay = 2.3  # dispersion in lightcurve
    p = 5  # order of AR polynomial
    qpo_width = np.array([1.0/100.0, 1.0/100.0, 1.0/500.0])
    qpo_cent = np.array([1.0/5.0, 1.0/50.0])
    ar_roots = cm.get_ar_roots(qpo_width, qpo_cent)
    ma_coefs = np.zeros(p)
    ma_coefs[0] = 1.0
    ma_coefs[1] = 4.5
    ma_coefs[2] = 1.25
    sigsqr = sigmay ** 2 / cm.carma_variance(1.0, ar_roots, ma_coefs=ma_coefs)

    ny = 1028
    time = np.arange(0.0, ny)
    y0 = cm.carma_process(time, sigsqr, ar_roots, ma_coefs=ma_coefs)

    ysig = np.ones(ny) * np.sqrt(1e-2)
    ysig = np.ones(ny) * np.sqrt(1e-6)

    y = y0 + ysig * np.random.standard_normal(ny)

    data = (time, y, ysig)

    froot = base_dir + 'car5_regular_'

    plt.subplot(111)
    plt.plot(time, y0, 'k-')
    plt.plot(time, y, '.')
    plt.xlim(time.min(), time.max())
    plt.xlabel('Time')
    plt.ylabel('CAR(5) Process')
    plt.savefig(froot + 'tseries.eps')

    ar_coef = np.poly(ar_roots)

    print 'Getting maximum-likelihood estimates...'

    carma_model = cm.CarmaModel(time, y, ysig, 10)
    pmax = 5
    MAP = carma_model.choose_order(pmax)

    mcmc_sample = carma_model.run_mcmc()
    carma_sample = cm.CarmaSample(time, y, ysig, mcmc_sample, q=carma_model.q, MAP=MAP)

    #pool = mp.Pool(mp.cpu_count()-1)
    #
    #args = []
    #maxp = 8
    #for p in xrange(1, maxp + 1):
    #    for q in xrange(p):
    #        args.append((p, q, data))
    #
    #print "Running the CARMA MCMC samplers..."
    #
    #carma_run = pool.map(run_carma_sampler, args)
    #
    #dic = []
    #pmodels = []
    #qmodels = []
    #for crun in carma_run:
    #    dic.append(crun.DIC())
    #    pmodels.append(crun.p)
    #    qmodels.append(crun.q)
    #
    #pmodels = np.array(pmodels)
    #qmodels = np.array(qmodels)
    #dic = np.array(dic)

    plt.clf()
    plt.subplot(111)
    for i in xrange(qmodels.max()+1):
        plt.plot(pmodels[qmodels == i], dic[qmodels == i], 's-', label='q=' + str(i), lw=2)
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('DIC')
    plt.xlim(0, pmodels.max() + 1)
    print "DIC", dic
    plt.savefig(froot + 'dic.eps')

    carma = carma_run[np.argmin(dic)]

    print "order of best model is", carma.p, carma.q

    plt.subplot(111)
    pgram, freq = plt.psd(y)
    plt.clf()

    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                      color='SkyBlue', nsamples=5000)
    ax.loglog(freq / 2.0, pgram, 'o', color='DarkOrange')
    psd = cm.power_spectrum(frequencies, np.sqrt(sigsqr), ar_coef, ma_coefs=ma_coefs)
    ax.loglog(frequencies, psd, 'k', lw=2)
    ax.loglog(frequencies, psd_mid, '--b', lw=2)
    noise_level = np.mean(ysig ** 2)
    ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
    ax.set_ylim(bottom=noise_level / 100.0)
    ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')

    plt.savefig(froot + 'psd.eps')

    print 'Assessing the fit quality...'
    carma.assess_fit(doShow=False)
    plt.savefig(froot + 'fit_quality.eps')


def do_simulated_irregular():

    # first generate some data assuming a CAR(5) process on a uniform grid
    sigmay = 2.3  # dispersion in lightcurve
    p = 5  # order of AR polynomial
    mu = 17.0  # mean of time series
    qpo_width = np.array([1.0/100.0, 1.0/300.0, 1.0/200.0])
    qpo_cent = np.array([1.0/5.0, 1.0/25.0])
    ar_roots = cm.get_ar_roots(qpo_width, qpo_cent)
    ma_coefs = np.zeros(p)
    ma_coefs[0] = 1.0
    ma_coefs[1] = 4.5
    ma_coefs[2] = 1.25
    sigsqr = sigmay ** 2 / cm.carma_variance(1.0, ar_roots, ma_coefs=ma_coefs)

    ny = 270
    time = np.empty(ny)
    dt = np.random.uniform(1.0, 3.0, ny)
    time[0:90] = np.cumsum(dt[0:90])
    time[90:2*90] = 180 + time[90-1] + np.cumsum(dt[90:2*90])
    time[2*90:] = 180 + time[2*90-1] + np.cumsum(dt[2*90:])

    y = mu + cm.carma_process(time, sigsqr, ar_roots, ma_coefs=ma_coefs)

    # ysig = np.ones(ny) * y.std() / 5.0
    ysig = np.ones(ny) * 1e-6
    y0 = y.copy()
    y += ysig * np.random.standard_normal(ny)

    data = (time, y, ysig)

    froot = base_dir + 'car5_irregular_'

    plt.subplot(111)
    for i in xrange(3):
        plt.plot(time[90*i:90*(i+1)], y0[90*i:90*(i+1)], 'k', lw=2)
        plt.plot(time[90*i:90*(i+1)], y[90*i:90*(i+1)], 'bo')

    plt.xlim(time.min(), time.max())
    plt.xlabel('Time')
    plt.ylabel('CAR(5) Process')
    plt.savefig(froot + 'tseries.eps')

    ar_coef = np.poly(ar_roots)

    pool = mp.Pool(mp.cpu_count()-1)

    args = []
    maxp = 8
    for p in xrange(1, maxp + 1):
        for q in xrange(p):
            args.append((p, q, data))

    print "Running the CARMA MCMC samplers..."


    # carma_run = run_carma_sampler(args[0])

    carma_run = pool.map(run_carma_sampler, args)

    dic = []
    pmodels = []
    qmodels = []
    for crun in carma_run:
        dic.append(crun.DIC())
        pmodels.append(crun.p)
        qmodels.append(crun.q)

    dic = np.array(dic)
    pmodels = np.array(pmodels)
    qmodels = np.array(qmodels)

    plt.clf()
    plt.subplot(111)
    for i in xrange(qmodels.max()+1):
        plt.plot(pmodels[qmodels == i], dic[qmodels == i], 's-', label='q=' + str(i), lw=2)
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('DIC')
    plt.xlim(0, pmodels.max() + 1)
    print "DIC", dic
    plt.savefig(froot + 'dic.eps')

    carma = carma_run[np.argmin(dic)]

    print "order of best model is", carma.p, carma.q

    plt.clf()
    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                      color='SkyBlue', nsamples=5000)
    psd = cm.power_spectrum(frequencies, np.sqrt(sigsqr), ar_coef, ma_coefs=ma_coefs)
    ax.loglog(frequencies, psd_mid, '--b', lw=2)
    ax.loglog(frequencies, psd, 'k', lw=2)
    noise_level = np.mean(ysig ** 2) * dt.min()
    ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
    ax.set_ylim(bottom=noise_level / 100.0)
    ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')

    plt.savefig(froot + 'psd.eps')

    print 'Assessing the fit quality...'
    carma.assess_fit(doShow=False)
    plt.savefig(froot + 'fit_quality.eps')

    # compute the marginal mean and variance of the predicted values
    nplot = 1028
    time_predict = np.linspace(time.min(), 1.25 * time.max(), nplot)
    time_predict = time_predict[1:]
    predicted_mean, predicted_var = carma.predict_lightcurve(time_predict, bestfit='map')
    predicted_low = predicted_mean - np.sqrt(predicted_var)
    predicted_high = predicted_mean + np.sqrt(predicted_var)

    # plot the time series and the marginal 1-sigma error bands
    plt.clf()
    plt.subplot(111)
    plt.fill_between(time_predict, predicted_low, predicted_high, color='cyan')
    plt.plot(time_predict, predicted_mean, '-b', label='Predicted')
    plt.plot(time[0:90], y0[0:90], 'k', lw=2, label='True')
    plt.plot(time[0:90], y[0:90], 'bo')
    for i in xrange(1, 3):
        plt.plot(time[90*i:90*(i+1)], y0[90*i:90*(i+1)], 'k', lw=2)
        plt.plot(time[90*i:90*(i+1)], y[90*i:90*(i+1)], 'bo')

    plt.xlabel('Time')
    plt.ylabel('CAR(5) Process')
    plt.xlim(time_predict.min(), time_predict.max())
    plt.legend()
    plt.savefig(froot + 'interp.eps')


if __name__ == "__main__":
    do_simulated_regular()
    do_simulated_irregular()