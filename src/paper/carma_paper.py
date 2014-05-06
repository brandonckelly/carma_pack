__author__ = 'brandonkelly'

import numpy as np
import carmcmc as cm
import matplotlib.pyplot as plt
from os import environ
import cPickle
from astropy.io import fits
import multiprocessing
from matplotlib.mlab import detrend_mean


base_dir = environ['HOME'] + '/Projects/carma_pack/src/paper/'
data_dir = base_dir + 'data/'

nthreads = multiprocessing.cpu_count()


def make_sampler_plots(time, y, ysig, pmax, file_root, title, do_mags=False, njobs=-1):

    froot = base_dir + 'plots/' + file_root

    # clean data
    dt = time[1:] - time[0:-1]
    if np.sum(dt <= 0) > 0:
        time = time[dt > 0]
        y = y[dt > 0]
        ysig = ysig[dt > 0]

    good = np.where(np.isfinite(time))[0]
    time = time[good]
    y = y[good]
    ysig = ysig[good]

    good = np.where(np.isfinite(y))[0]
    time = time[good]
    y = y[good]
    ysig = ysig[good]

    good = np.where(np.isfinite(ysig))[0]
    time = time[good]
    y = y[good]
    ysig = ysig[good]

    print 'Getting maximum-likelihood estimates...'

    carma_model = cm.CarmaModel(time, y, ysig)
    MAP, pqlist, AIC_list = carma_model.choose_order(pmax, njobs=njobs)

    # convert lists to a numpy arrays, easier to manipulate
    pqarray = np.array(pqlist)
    pmodels = pqarray[:, 0]
    qmodels = pqarray[:, 1]
    AICc = np.array(AIC_list)

    plt.clf()
    plt.subplot(111)
    for i in xrange(qmodels.max()+1):
        plt.plot(pmodels[qmodels == i], AICc[qmodels == i], 's-', label='q=' + str(i), lw=2)
    plt.legend(loc='best')
    plt.xlabel('p')
    plt.ylabel('AICc(p,q)')
    plt.xlim(0, pmodels.max() + 1)
    plt.title(title)
    plt.savefig(froot + 'aic.eps')
    plt.close()

    # make sure to change these back!!!!
    # carma_model.p = 7
    # carma_model.q = 3

    nsamples = 50000
    carma_sample = carma_model.run_mcmc(nsamples)
    carma_sample.add_mle(MAP)

    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                             color='SkyBlue', nsamples=5000)
    psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
    ax.loglog(frequencies, psd_mle, '--b', lw=2)
    dt = time[1:] - time[0:-1]
    noise_level = 2.0 * np.median(dt) * np.mean(ysig ** 2)
    ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
    ax.set_ylim(bottom=noise_level / 100.0)
    do_s82 = True
    if do_s82:
        ax.annotate("Measurement Noise Level", (3.0e-2, 1e-2))
    else:
        ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
    ax.set_xlabel('Frequency [1 / day]')
    if do_mags:
        ax.set_ylabel('Power Spectral Density [mag$^2$ day]')
    else:
        ax.set_ylabel('Power Spectral Density [flux$^2$ day]')
    plt.title(title)
    plt.savefig(froot + 'psd.eps')

    print 'Assessing the fit quality...'
    fig = carma_sample.assess_fit(doShow=False)
    ax_again = fig.add_subplot(2, 2, 1)
    ax_again.set_title(title)
    if do_mags:
        ylims = ax_again.get_ylim()
        ax_again.set_ylim(ylims[1], ylims[0])
        ax_again.set_ylabel('magnitude')
    else:
        ax_again.set_ylabel('ln Flux')
    plt.savefig(froot + 'fit_quality.eps')

    return carma_sample


def do_simulated_regular():

    # first generate some data assuming a CARMA(5,3) process on a uniform grid
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
    # ysig = np.ones(ny) * np.sqrt(1e-6)

    y = y0 + ysig * np.random.standard_normal(ny)

    froot = base_dir + 'plots/car5_regular_'

    plt.subplot(111)
    plt.plot(time, y0, 'k-')
    plt.plot(time, y, '.')
    plt.xlim(time.min(), time.max())
    plt.xlabel('Time')
    plt.ylabel('CARMA(5,3) Process')
    plt.savefig(froot + 'tseries.eps')

    ar_coef = np.poly(ar_roots)

    print 'Getting maximum-likelihood estimates...'

    carma_model = cm.CarmaModel(time, y, ysig)
    pmax = 7
    MAP, pqlist, AIC_list = carma_model.choose_order(pmax, njobs=-1)

    # convert lists to a numpy arrays, easier to manipulate
    pqarray = np.array(pqlist)
    pmodels = pqarray[:, 0]
    qmodels = pqarray[:, 1]
    AICc = np.array(AIC_list)

    plt.clf()
    plt.subplot(111)
    for i in xrange(qmodels.max()+1):
        plt.plot(pmodels[qmodels == i], AICc[qmodels == i], 's-', label='q=' + str(i), lw=2)
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('AICc(p,q)')
    plt.xlim(0, pmodels.max() + 1)
    plt.savefig(froot + 'aic.eps')
    plt.close()

    nsamples = 50000
    carma_sample = carma_model.run_mcmc(nsamples)
    carma_sample.add_mle(MAP)

    plt.subplot(111)
    pgram, freq = plt.psd(y)
    plt.clf()

    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                             color='SkyBlue', nsamples=5000)
    psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
    ax.loglog(freq / 2.0, pgram, 'o', color='DarkOrange')
    psd = cm.power_spectrum(frequencies, np.sqrt(sigsqr), ar_coef, ma_coefs=ma_coefs)
    ax.loglog(frequencies, psd, 'k', lw=2)
    ax.loglog(frequencies, psd_mle, '--b', lw=2)
    noise_level = 2.0 * np.mean(ysig ** 2)
    ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
    ax.set_ylim(bottom=noise_level / 100.0)
    ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')

    plt.savefig(froot + 'psd.eps')

    print 'Assessing the fit quality...'
    carma_sample.assess_fit(doShow=False)
    plt.savefig(froot + 'fit_quality.eps')

    pfile = open(data_dir + froot + '.pickle', 'wb')
    cPickle.dump(carma_sample, pfile)
    pfile.close()


def do_simulated_irregular():

    # first generate some data assuming a CARMA(5,3) process on a uniform grid
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

    ysig = np.ones(ny) * y.std() / 5.0
    # ysig = np.ones(ny) * 1e-6
    y0 = y.copy()
    y += ysig * np.random.standard_normal(ny)

    froot = base_dir + 'plots/car5_irregular_'

    plt.subplot(111)
    for i in xrange(3):
        plt.plot(time[90*i:90*(i+1)], y0[90*i:90*(i+1)], 'k', lw=2)
        plt.plot(time[90*i:90*(i+1)], y[90*i:90*(i+1)], 'bo')

    plt.xlim(time.min(), time.max())
    plt.xlabel('Time')
    plt.ylabel('CARMA(5,3) Process')
    plt.savefig(froot + 'tseries.eps')

    ar_coef = np.poly(ar_roots)

    print 'Getting maximum-likelihood estimates...'

    carma_model = cm.CarmaModel(time, y, ysig)
    pmax = 7
    MAP, pqlist, AIC_list = carma_model.choose_order(pmax, njobs=1)

    # convert lists to a numpy arrays, easier to manipulate
    pqarray = np.array(pqlist)
    pmodels = pqarray[:, 0]
    qmodels = pqarray[:, 1]
    AICc = np.array(AIC_list)

    plt.clf()
    plt.subplot(111)
    for i in xrange(qmodels.max()+1):
        plt.plot(pmodels[qmodels == i], AICc[qmodels == i], 's-', label='q=' + str(i), lw=2)
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('AICc(p,q)')
    plt.xlim(0, pmodels.max() + 1)
    plt.savefig(froot + 'aic.eps')
    plt.close()

    nsamples = 50000
    carma_sample = carma_model.run_mcmc(nsamples)
    carma_sample.add_mle(MAP)

    plt.clf()
    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                             color='SkyBlue', nsamples=5000)
    psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
    psd = cm.power_spectrum(frequencies, np.sqrt(sigsqr), ar_coef, ma_coefs=ma_coefs)
    ax.loglog(frequencies, psd_mle, '--b', lw=2)
    ax.loglog(frequencies, psd, 'k', lw=2)
    dt = np.median(time[1:] - time[0:-1])
    noise_level = 2.0 * dt * np.mean(ysig ** 2)
    ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
    ax.set_ylim(bottom=noise_level / 100.0)
    ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')

    plt.savefig(froot + 'psd.eps')

    print 'Assessing the fit quality...'
    carma_sample.assess_fit(doShow=False)
    plt.savefig(froot + 'fit_quality.eps')

    # compute the marginal mean and variance of the predicted values
    nplot = 1028
    time_predict = np.linspace(time.min(), 1.25 * time.max(), nplot)
    time_predict = time_predict[1:]
    predicted_mean, predicted_var = carma_sample.predict(time_predict, bestfit='map')
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
    plt.ylabel('CARMA(5,3) Process')
    plt.xlim(time_predict.min(), time_predict.max())
    plt.legend()
    plt.savefig(froot + 'interp.eps')


def do_simulated_irregular_nonstationary():

    # generate first half of lightcurve assuming a CARMA(5,3) process on a uniform grid
    sigmay1 = 2.3  # dispersion in lightcurve
    p = 5  # order of AR polynomial
    qpo_width1 = np.array([1.0/100.0, 1.0/300.0, 1.0/200.0])
    qpo_cent1 = np.array([1.0/5.0, 1.0/25.0])
    ar_roots1 = cm.get_ar_roots(qpo_width1, qpo_cent1)
    ma_coefs1 = np.zeros(p)
    ma_coefs1[0] = 1.0
    ma_coefs1[1] = 4.5
    ma_coefs1[2] = 1.25
    sigsqr1 = sigmay1 ** 2 / cm.carma_variance(1.0, ar_roots1, ma_coefs=ma_coefs1)

    ny = 270
    time = np.zeros(ny)
    dt = np.random.uniform(1.0, 3.0, ny)
    time[0:90] = np.cumsum(dt[0:90])
    time[90:2*90] = 180 + time[90-1] + np.cumsum(dt[90:2*90])
    time[2*90:] = 180 + time[2*90-1] + np.cumsum(dt[2*90:])

    y = cm.carma_process(time, sigsqr1, ar_roots1, ma_coefs=ma_coefs1)

    # first generate some data assuming a CARMA(5,3) process on a uniform grid
    sigmay2 = 4.5  # dispersion in lightcurve
    p = 5  # order of AR polynomial
    qpo_width2 = np.array([1.0/100.0, 1.0/100.0, 1.0/500.0])
    qpo_cent2 = np.array([1.0/5.0, 1.0/50.0])
    ar_roots2 = cm.get_ar_roots(qpo_width2, qpo_cent2)
    ma_coefs2 = np.zeros(p)
    ma_coefs2[0] = 1.0
    ma_coefs2[1] = 4.5
    ma_coefs2[2] = 1.25
    sigsqr2 = sigmay2 ** 2 / cm.carma_variance(1.0, ar_roots2, ma_coefs=ma_coefs2)

    ny = 270
    time2 = np.zeros(ny)
    dt = np.random.uniform(1.0, 3.0, ny)
    time2[0:90] = np.cumsum(dt[0:90])
    time2[90:2*90] = 180 + time2[90-1] + np.cumsum(dt[90:2*90])
    time2[2*90:] = 180 + time2[2*90-1] + np.cumsum(dt[2*90:])

    time = np.append(time, time.max() + 180 + time2)

    y2 = cm.carma_process(time2, sigsqr2, ar_roots2, ma_coefs=ma_coefs2)

    y = np.append(y, y2)

    ysig = np.ones(len(y)) * y.std() / 8.0
    # ysig = np.ones(ny) * 1e-6
    y0 = y.copy()
    y += ysig * np.random.standard_normal(len(y))

    froot = base_dir + 'plots/car5_nonstationary_'

    plt.subplot(111)
    for i in xrange(6):
        plt.plot(time[90*i:90*(i+1)], y0[90*i:90*(i+1)], 'k', lw=2)
        plt.plot(time[90*i:90*(i+1)], y[90*i:90*(i+1)], 'bo')

    plt.xlim(time.min(), time.max())
    plt.xlabel('Time')
    plt.ylabel('Non-Stationary Process')
    plt.savefig(froot + 'tseries.eps')
    plt.show()

    ar_coef1 = np.poly(ar_roots1)
    ar_coef2 = np.poly(ar_roots2)

    print 'Getting maximum-likelihood estimates...'

    carma_model = cm.CarmaModel(time, y, ysig)
    pmax = 7
    MAP, pqlist, AIC_list = carma_model.choose_order(pmax, njobs=1)

    # convert lists to a numpy arrays, easier to manipulate
    pqarray = np.array(pqlist)
    pmodels = pqarray[:, 0]
    qmodels = pqarray[:, 1]
    AICc = np.array(AIC_list)

    plt.clf()
    plt.subplot(111)
    for i in xrange(qmodels.max()+1):
        plt.plot(pmodels[qmodels == i], AICc[qmodels == i], 's-', label='q=' + str(i), lw=2)
    plt.legend()
    plt.xlabel('p')
    plt.ylabel('AICc(p,q)')
    plt.xlim(0, pmodels.max() + 1)
    plt.savefig(froot + 'aic.eps')
    plt.close()

    nsamples = 50000
    carma_sample = carma_model.run_mcmc(nsamples)
    carma_sample.add_mle(MAP)

    cPickle.dump(carma_sample, open(data_dir + 'nonstationary.pickle', 'wb'))

    plt.clf()
    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                             color='SkyBlue', nsamples=5000)
    psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
    psd1 = cm.power_spectrum(frequencies, np.sqrt(sigsqr1), ar_coef1, ma_coefs=ma_coefs1)
    psd2 = cm.power_spectrum(frequencies, np.sqrt(sigsqr2), ar_coef2, ma_coefs=ma_coefs2)
    # ax.loglog(frequencies, psd_mle, '--b', lw=2)
    ax.loglog(frequencies, psd1, 'k', lw=2)
    ax.loglog(frequencies, psd2, '--k', lw=2)
    dt = np.median(time[1:] - time[0:-1])
    noise_level = 2.0 * dt * np.mean(ysig ** 2)
    ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
    ax.set_ylim(bottom=noise_level / 100.0)
    ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power Spectral Density')

    plt.savefig(froot + 'psd.eps')

    print 'Assessing the fit quality...'
    carma_sample.assess_fit(doShow=False)
    plt.savefig(froot + 'fit_quality.eps')

    # compute the marginal mean and variance of the predicted values
    nplot = 1028
    time_predict = np.linspace(time.min(), 1.25 * time.max(), nplot)
    time_predict = time_predict[1:]
    predicted_mean, predicted_var = carma_sample.predict(time_predict, bestfit='map')
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
    plt.ylabel('CARMA(5,3) Process')
    plt.xlim(time_predict.min(), time_predict.max())
    plt.legend()
    plt.savefig(froot + 'interp.eps')


def do_AGN_Stripe82():

    s82_id = '1627677'
    data = np.genfromtxt(data_dir + s82_id)
    # do r-band data
    jdate = data[:, 6]
    rmag = data[:, 7]
    rerr = data[:, 8]

    load_pickle = True
    if load_pickle:
        time = jdate - jdate.min()
        ysig = rerr
        carma_sample = cPickle.load(open(data_dir + s82_id + '.pickle', 'rb'))
        froot = base_dir + 'plots/' + s82_id + '_'

        print 'Assessing the fit quality...'
        fig = carma_sample.assess_fit(doShow=False)
        ax_again = fig.add_subplot(2, 2, 1)
        ax_again.set_title("S82 Quasar, r-band")
        ylims = ax_again.get_ylim()
        ax_again.set_ylim(ylims[1], ylims[0])
        ax_again.set_ylabel('magnitude')
        ax_again.set_xlabel('Time [days]')
        ax_again = fig.add_subplot(2, 2, 2)
        ax_again.set_xlabel('Time [days]')
        ax_again = fig.add_subplot(2, 2, 3)
        ax_again.set_xlabel('Lag')
        ax_again = fig.add_subplot(2, 2, 4)
        ax_again.set_xlabel('Lag')
        plt.savefig(froot + 'fit_quality.eps')
    else:
        carma_sample = make_sampler_plots(jdate - jdate.min(), rmag, rerr, 7, s82_id + '_', 'S82 Quasar, r-band',
                                          do_mags=True)

        pfile = open(data_dir + '1627677.pickle', 'wb')
        cPickle.dump(carma_sample, pfile)
        pfile.close()


def do_AGN_Kepler():

    sname = 'Zw 229-15'
    data = fits.open(data_dir + 'kepler_zw229_Q7.fits')[1].data
    jdate = data['time']
    flux = np.array(data['SAP_FLUX'], dtype=float)
    ferr = np.array(data['SAP_FLUX_ERR'], dtype=float)

    keep = np.where(np.logical_and(np.isfinite(jdate), np.isfinite(flux)))[0]
    jdate = jdate[keep]
    jdate -= jdate.min()
    flux = flux[keep]
    ferr = ferr[keep]

    df = flux[1:] - flux[0:-1]  # remove outliers
    keep = np.where(np.abs(df) < 56.0)
    jdate = jdate[keep]
    flux = flux[keep]
    ferr = ferr[keep]

    load_pickle = True
    if load_pickle:
        carma_sample = cPickle.load(open(data_dir + 'zw229.pickle', 'rb'))
        # rerun MLE
        # carma_model = cm.CarmaModel(jdate, flux, ferr, p=carma_sample.p, q=carma_sample.q)
        # mle = carma_model.get_map(carma_sample.p, carma_sample.q)
        # carma_sample.add_map(mle)
    else:
        carma_sample = make_sampler_plots(jdate, flux, ferr, 7, 'zw229_', sname, njobs=1)

    # transform the flux through end matching
    tflux = flux - flux[0]
    slope = (tflux[-1] - tflux[0]) / (jdate[-1] - jdate[0])
    tflux -= slope * jdate

    plt.subplot(111)
    dt = jdate[1] - jdate[0]
    pgram, freq = plt.psd(tflux, 512, 2.0 / dt, detrend=detrend_mean)
    plt.clf()

    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                             color='SkyBlue', nsamples=5000)
    psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
    ax.loglog(freq / 2.0, pgram, 'o', color='DarkOrange')
    psd_slope = 3.14
    above_noise = np.where(freq / 2.0 < 1.0)[0]
    psd_norm = np.mean(np.log(pgram[above_noise[1:]]) + 3.14 * np.log(freq[above_noise[1:]] / 2.0))
    psd_plaw = np.exp(psd_norm) / (freq[1:] / 2.0) ** psd_slope
    ax.loglog(freq[1:] / 2.0, psd_plaw, '-', lw=2, color='DarkOrange')
    ax.loglog(frequencies, psd_mid, '--b', lw=2)
    noise_level = 2.0 * dt * np.mean(ferr ** 2)
    ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
    ax.set_ylim(bottom=noise_level / 100.0)
    ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
    ax.set_xlabel('Frequency [1 / day]')
    ax.set_ylabel('Power Spectral Density [flux$^2$ day]')
    plt.title(sname)
    plt.savefig(base_dir + 'plots/zw229_psd.eps')

    plt.clf()
    carma_sample.plot_1dpdf('measerr_scale')
    plt.savefig(base_dir + 'plots/zw229_measerr_scale.eps')
    measerr_scale = carma_sample.get_samples('measerr_scale')
    print "95% credibility interval on Kepler measurement error scale parameter:", np.percentile(measerr_scale, 2.5), \
        np.percentile(measerr_scale, 97.5)

    pfile = open(data_dir + 'zw229.pickle', 'wb')
    cPickle.dump(carma_sample, pfile)
    pfile.close()


def do_AGN_Xray():

    sname = 'MCG-6-30-15, X-ray'
    data = np.genfromtxt(data_dir + 'mcg-6-30-15_rxte_xmm.txt')
    jdate = data[:, 0]
    flux = data[:, 1] * np.log(10.0)  # convert to natural logarithm
    ferr = data[:, 2] * np.log(10.0)

    jdate = jdate - jdate.min()
    time = jdate * 86.4e3  # convert to seconds

    dt = time[1:] - time[0:-1]
    rxte = np.where(dt > 50.0)[0]
    dt_rxte = np.median(dt[rxte])
    xmm = np.where(dt < 50.0)[0]
    dt_xmm = 48.0

    carma_sample = make_sampler_plots(time[rxte], flux[rxte], ferr[rxte] / 1e6, 7, 'mcg63015_rxte_', sname, njobs=4)

    measerr_scale = carma_sample.get_samples('measerr_scale')
    print "95% credibility interval on Kepler measurement error scale parameter:", np.percentile(measerr_scale, 2.5), \
        np.percentile(measerr_scale, 97.5)

    pfile = open(data_dir + 'mcg63015.pickle', 'wb')
    cPickle.dump(carma_sample, pfile)
    pfile.close()

    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                             color='SkyBlue', nsamples=5000)
    psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
    ax.loglog(frequencies, psd_mle, '--b', lw=2)
    noise_level_rxte = 2.0 * dt_rxte * np.mean(ferr[rxte] ** 2)
    noise_level_xmm = 2.0 * dt_xmm * np.mean(ferr[xmm] ** 2)
    rxte_frange = np.array([1.0 / time[rxte].max(), 1.0 / dt_rxte])
    xmm_frange = np.array([1.0 / (time[xmm].max() - time[xmm].min()), 1.0 / dt_xmm])
    ax.loglog(rxte_frange, np.ones(2) * noise_level_rxte, color='grey', lw=2)
    ax.loglog(xmm_frange, np.ones(2) * noise_level_xmm, color='grey', lw=2)
    noise_level = np.min([noise_level_rxte, noise_level_xmm])
    ax.set_ylim(bottom=noise_level / 100.0)
    ax.annotate("Measurement Noise Level, RXTE", (2.0 * ax.get_xlim()[0], noise_level_rxte / 2.5))
    ax.annotate("Noise Level, XMM", (xmm_frange[0], noise_level_xmm / 2.5))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power Spectral Density [fraction$^2$ / Hz]')
    plt.title(sname)
    plt.savefig(base_dir + 'plots/mcg63015_psd.eps')


def do_RRLyrae():

    dtype = np.dtype([("mjd", np.float), ("filt", np.str, 1), ("mag", np.float), ("dmag", np.float)])
    data = np.loadtxt(data_dir + 'RRLyrae.txt', comments="#", dtype=dtype)

    # do g-band light curve
    gIdx = np.where(data["filt"] == "g")[0]
    jdate = data['mjd'][gIdx]
    gmag = data['mag'][gIdx]
    gerr = data['dmag'][gIdx]

    load_pickle = True
    if load_pickle:
        time = jdate - jdate.min()
        ysig = gerr
        carma_sample = cPickle.load(open(data_dir + 'RRLyrae.pickle', 'rb'))
        froot = base_dir + 'plots/' + 'RRLyrae_'
        ax = plt.subplot(111)
        print 'Getting bounds on PSD...'
        psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                                 color='SkyBlue', nsamples=5000)
        psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                    ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
        ax.loglog(frequencies, psd_mle, '--b', lw=2)
        dt = time[1:] - time[0:-1]
        noise_level = 2.0 * np.median(dt) * np.mean(ysig ** 2)
        ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
        ax.set_ylim(bottom=noise_level / 100.0)
        ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
        ax.annotate("C", (1.0 / 0.5, 1e3))
        ax.annotate("B", (1.0 / 1.3, 2.0))
        ax.annotate("A", (1.0 / 2.49, 1.0))
        ax.set_xlabel('Frequency [1 / day]')
        ax.set_ylabel('Power Spectral Density [mag$^2$ day]')
        plt.title("RR Lyrae, g-band")
        plt.savefig(froot + 'psd.eps')

        print 'Assessing the fit quality...'
        fig = carma_sample.assess_fit(doShow=False)
        ax_again = fig.add_subplot(2, 2, 1)
        ax_again.set_title("RR Lyrae, g-band")
        ylims = ax_again.get_ylim()
        ax_again.set_ylim(ylims[1], ylims[0])
        ax_again.set_ylabel('magnitude')
        ax_again.set_xlabel('Time [days]')
        ax_again = fig.add_subplot(2, 2, 2)
        ax_again.set_xlabel('Time [days]')
        ax_again = fig.add_subplot(2, 2, 3)
        ax_again.set_xlabel('Lag')
        ax_again = fig.add_subplot(2, 2, 4)
        ax_again.set_xlabel('Lag')
        plt.savefig(froot + 'fit_quality.eps')

    else:
        carma_sample = make_sampler_plots(jdate - jdate.min(), gmag, gerr, 7, 'RRLyrae_', 'RR Lyrae, g-band', do_mags=True,
                                          njobs=1)
        pfile = open(data_dir + 'RRLyrae.pickle', 'wb')
        cPickle.dump(carma_sample, pfile)
        pfile.close()


def do_OGLE_LPV():
    sname = 'LPV, RGB, i-band'
    data = np.genfromtxt(data_dir + 'OGLE-LMC-LPV-00007.dat')
    jdate = data[:, 0]
    imag = data[:, 1]
    ierr = data[:, 2]

    load_pickle = False
    if load_pickle:
        time = jdate
        ysig = ierr
        carma_sample = cPickle.load(open(data_dir + 'ogle_lpv_rgb.pickle', 'rb'))
        froot = base_dir + 'plots/' + 'ogle_lpv_rgb_'
        ax = plt.subplot(111)
        print 'Getting bounds on PSD...'
        psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                                 color='SkyBlue', nsamples=5000)
        psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                    ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
        ax.loglog(frequencies, psd_mle, '--b', lw=2)
        dt = time[1:] - time[0:-1]
        noise_level = 2.0 * np.median(dt) * np.mean(ysig ** 2)
        ax.loglog(frequencies, np.ones(frequencies.size) * noise_level, color='grey', lw=2)
        ax.set_ylim(bottom=noise_level / 100.0)
        ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 2.5))
        ax.annotate("A", (1.0 / 25.0, 2.5e-3))
        ax.annotate("B", (1.0 / 16.0, 2.5e-3))
        ax.set_xlabel('Frequency [1 / day]')
        ax.set_ylabel('Power Spectral Density [mag$^2$ day]')
        plt.title(sname)
        plt.savefig(froot + 'psd.eps')
    else:
        carma_sample = make_sampler_plots(jdate - jdate.min(), imag, ierr, 7, 'ogle_lpv_rgb_', sname, do_mags=True,
                                          njobs=1)
        pfile = open(data_dir + 'ogle_lpv_rgb.pickle', 'wb')
        cPickle.dump(carma_sample, pfile)
        pfile.close()


def do_XRB():
    sname = 'XTE 1550-564'
    data_file = data_dir + 'LC_B_3.35-12.99keV_1div128s_total.fits'
    data = fits.open(data_file)[1].data
    tsecs = data['TIME']
    flux = data['RATE']
    dt = tsecs[1:] - tsecs[:-1]
    gap = np.where(dt > 1)[0]

    tsecs = tsecs[gap[0]+1:gap[1]][:40000]
    flux = flux[gap[0]+1:gap[1]][:40000]

    tsecs0 = tsecs.copy()
    flux0 = flux.copy()

    ndown_sample = 4000
    idx = np.random.permutation(len(flux0))[:ndown_sample]
    idx.sort()
    tsecs = tsecs[idx]
    logflux = np.log(flux[idx])
    ferr = np.sqrt(flux[idx])
    logf_err = ferr / flux[idx]

    # # high-frequency sampling lightcurve
    # high_cutoff = 10000
    # tsecs_high = tsecs[:high_cutoff]
    # logflux_high = np.log(flux[:high_cutoff])
    # ferr_high = np.sqrt(flux[:high_cutoff])
    # logferr_high = ferr_high / flux[:high_cutoff]
    #
    # ndown_sample_high = 1000
    # idx_high = np.random.permutation(len(logflux_high))[:ndown_sample_high]
    # idx_high.sort()
    #
    # # middle-frequency sampling lightcurve
    # tsecs_mid = tsecs[high_cutoff:]
    # logflux_mid = np.log(flux[high_cutoff:])
    # ferr_mid = np.sqrt(flux[high_cutoff:])
    # logf_err_mid = ferr_mid / flux[high_cutoff:]
    # # logf_err = np.sqrt(0.00018002985939372774 / 2.0 / np.median(dt))  # eyeballed from periodogram
    # # logf_err = np.ones(len(tsecs)) * logf_err
    #
    # ndown_sample_mid = 4000 - ndown_sample_high
    # idx_mid = np.random.permutation(len(logflux_mid))[:ndown_sample_mid]
    # idx_mid.sort()
    #
    # tsecs = np.concatenate((tsecs_high[idx_high], tsecs_mid[idx_mid]))
    # logflux = np.concatenate((logflux_high[idx_high], logflux_mid[idx_mid]))
    # logf_err = np.concatenate((logferr_high[idx_high], logf_err_mid[idx_mid]))
    # idx = np.concatenate((idx_high, idx_mid))

    plt.plot(tsecs0, np.log(flux0))
    plt.errorbar(tsecs, logflux, yerr=logf_err)
    print 'Measurement errors are', np.mean(logf_err) / np.std(logflux) * 100, ' % of observed standard deviation.'
    print 'Mean time spacing:', np.mean(tsecs[1:] - tsecs[:-1])
    # print 'Mean time spacing for high-frequency sampling:', np.mean(tsecs_high[idx_high[1:]]-tsecs_high[idx_high[:-1]])
    # print 'Mean time spacing for low-frequency sampling:', np.mean(tsecs_mid[idx_mid[1:]]-tsecs_mid[idx_mid[:-1]])
    plt.show()
    plt.clf()
    plt.plot(tsecs, logflux)
    plt.show()
    plt.hist(logflux, bins=100, normed=True)
    plt.xlabel('log Flux')
    print 'Standard deviation in lightcurve:', np.std(logflux)
    print 'Typical measurement error:', np.mean(logf_err)
    plt.show()
    plt.clf()
    assert np.all(np.isfinite(tsecs))
    assert np.all(np.isfinite(logflux))
    assert np.all(np.isfinite(logf_err))
    dt_idx = tsecs[1:] - tsecs[:-1]
    assert np.all(dt_idx > 0)

    load_pickle = True
    if load_pickle:
        carma_sample = cPickle.load(open(data_dir + 'xte1550_p5q4.pickle', 'rb'))
    else:
        carma_sample = make_sampler_plots(tsecs, logflux, logf_err, 7, 'xte1550_', sname, njobs=7)

    plt.subplot(111)
    pgram, freq = plt.psd(np.log(flux0), 512, 2.0 / np.median(dt), detrend=detrend_mean)
    plt.clf()

    ax = plt.subplot(111)
    print 'Getting bounds on PSD...'
    psd_low, psd_hi, psd_mid, frequencies = carma_sample.plot_power_spectrum(percentile=95.0, sp=ax, doShow=False,
                                                                             color='SkyBlue', nsamples=5000)
    psd_mle = cm.power_spectrum(frequencies, carma_sample.mle['sigma'], carma_sample.mle['ar_coefs'],
                                ma_coefs=np.atleast_1d(carma_sample.mle['ma_coefs']))
    ax.loglog(freq / 2, pgram, 'o', color='DarkOrange')
    nyquist_freq = np.mean(0.5 / dt_idx)
    nyquist_idx = np.where(frequencies <= nyquist_freq)[0]
    ax.loglog(frequencies, psd_mle, '--b', lw=2)
    # noise_level = 2.0 * np.mean(dt_idx) * np.mean(logf_err ** 2)
    noise_level0 = 0.00018002985939372774 / 2.0  # scale the noise level seen in the PSD
    noise_level = noise_level0 * (0.5 / np.median(dt)) / nyquist_freq
    ax.loglog(frequencies[nyquist_idx], np.ones(len(nyquist_idx)) * noise_level, color='grey', lw=2)
    # ax.loglog(frequencies, np.ones(len(frequencies)) * noise_level0)
    ax.set_ylim(bottom=noise_level0 / 10.0)
    ax.annotate("Measurement Noise Level", (3.0 * ax.get_xlim()[0], noise_level / 1.5))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power Spectral Density [fraction$^2$ Hz$^{-1}$]')
    plt.title(sname)

    plt.savefig(base_dir + 'plots/xte1550_psd.eps')

    # plot the standardized residuals and compare with the standard normal
    plt.clf()
    kfilter, mu = carma_sample.makeKalmanFilter('map')
    kfilter.Filter()
    kmean = np.asarray(kfilter.GetMean())
    kvar = np.asarray(kfilter.GetVar())
    standardized_residuals = (carma_sample.y - mu - kmean) / np.sqrt(kvar)
    plt.hist(standardized_residuals, bins=100, normed=True, color='SkyBlue', histtype='stepfilled')
    plt.xlabel('Standardized Residuals')
    plt.ylabel('Probability Distribution')
    xlim = plt.xlim()
    xvalues = np.linspace(xlim[0], xlim[1], num=100)
    expected_pdf = np.exp(-0.5 * xvalues ** 2) / np.sqrt(2.0 * np.pi)
    plt.plot(xvalues, expected_pdf, 'k', lw=3)
    plt.title(sname)
    plt.savefig(base_dir + 'plots/xte1550_resid_dist.eps')

    # plot the autocorrelation function of the residuals and compare with the 95% confidence intervals for white
    # noise
    plt.clf()
    maxlag = 50
    wnoise_upper = 1.96 / np.sqrt(carma_sample.time.size)
    wnoise_lower = -1.96 / np.sqrt(carma_sample.time.size)
    plt.fill_between([0, maxlag], wnoise_upper, wnoise_lower, facecolor='grey')
    lags, acf, not_needed1, not_needed2 = plt.acorr(standardized_residuals, maxlags=maxlag, lw=3)
    plt.xlim(0, maxlag)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('Time Lag')
    plt.ylabel('ACF of Residuals')
    plt.savefig(base_dir + 'plots/xte1550_resid_acf.eps')

    # plot the autocorrelation function of the squared residuals and compare with the 95% confidence intervals for
    # white noise
    plt.clf()
    squared_residuals = standardized_residuals ** 2
    wnoise_upper = 1.96 / np.sqrt(carma_sample.time.size)
    wnoise_lower = -1.96 / np.sqrt(carma_sample.time.size)
    plt.fill_between([0, maxlag], wnoise_upper, wnoise_lower, facecolor='grey')
    lags, acf, not_needed1, not_needed2 = plt.acorr(squared_residuals - squared_residuals.mean(), maxlags=maxlag,
                                                    lw=3)
    plt.xlim(0, maxlag)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('Time Lag')
    plt.ylabel('ACF of Sqrd. Resid.')
    plt.savefig(base_dir + 'plots/xte1550_sqrres_acf.eps')

    if not load_pickle:
        pfile = open(data_dir + 'xte1550_nonoise.pickle', 'wb')
        cPickle.dump(carma_sample, pfile)
        pfile.close()


if __name__ == "__main__":
    # do_simulated_regular()
    # do_simulated_irregular()
    # do_simulated_irregular_nonstationary()
    # do_AGN_Stripe82()
    # do_AGN_Kepler()
    # do_RRLyrae()
    do_OGLE_LPV()
    # do_AGN_Xray()
    # do_XRB()
