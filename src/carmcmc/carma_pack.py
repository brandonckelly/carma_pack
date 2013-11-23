__author__ = 'Brandon C. Kelly'

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.optimize import minimize
import samplers
import multiprocessing
import _carmcmc as carmcmcLib


class CarmaModel(object):
    """
    Class for running the MCMC sampler assuming a CARMA(p,q) model.
    """

    def __init__(self, time, y, ysig, p=1, q=0):
        """
        Constructor for the CarmaMCMC class.

        :param time: The observation times.
        :param y: The measured time series.
        :param ysig: The standard deviation in the measurements errors on the time series.
        :param p: The order of the autoregressive (AR) polynomial.
        :param nsamples: The number of MCMC samples to generate.
        :param q: The order of the moving average polynomial. Default is q = 0. Note that p > q.
        :param doZcarma: If true, then use the z-transformed CAR parameterization.
        :param nwalkers: Number of parallel MCMC chains to run in the parallel tempering algorithm. Default is 1 (no
            tempering) for p = 1 and max(10, p+q) for p > 1.
        :param nburnin: Number of burnin iterations to run. The default is nsamples / 2.
        :param nthin: Thinning interval for the MCMC sampler. Default is 1.
        """
        try:
            p > q
        except ValueError:
            " Order of AR polynomial, p, must be larger than order of MA polynomial, q."

        # convert input to std::vector<double> extension class
        self._time = carmcmcLib.vecD()
        self._time.extend(time)
        self._y = carmcmcLib.vecD()
        self._y.extend(y)
        self._ysig = carmcmcLib.vecD()
        self._ysig.extend(ysig)

        # save parameters
        self.time = time
        self.y = y
        self.ysig = ysig
        self.p = p
        self.q = q

    def run_mcmc(self, nsamples, nburnin=None, nwalkers=None, nthin=1):
        """
        Run the MCMC sampler. This is actually a wrapper that calls the C++ code that runs the MCMC sampler.

        :return: Either a CarmaSample, ZCarmaSample, or CarSample1 object, depending on the values of self.p and
                 self.doZcarma.
        """

        if nwalkers is None:
            nwalkers = max(10, self.p + self.q)

        if nburnin is None:
            nburnin = nsamples / 2

        if self.p == 1:
            # Treat the CAR(1) case separately
            cppSample = carmcmcLib.run_mcmc_car1(nsamples, nburnin, self._time, self._y, self._ysig,
                                                 nthin)
            # run_mcmc_car1 returns a wrapper around the C++ CAR1 class, convert to python object
            sample = CarSample1(self.time, self.y, self.ysig, cppSample)
        else:
            cppSample = carmcmcLib.run_mcmc_carma(nsamples, nburnin, self._time, self._y, self._ysig,
                                                  self.p, self.q, nwalkers, False, nthin)
            # run_mcmc_car returns a wrapper around the C++ CARMA/ZCAR class, convert to a python object
            sample = CarmaSample(self.time, self.y, self.ysig, cppSample, q=self.q)

        return sample

    def get_map(self, p, q, ntrials=100, njobs=1):

        if njobs == -1:
            njobs = multiprocessing.cpu_count()

        args = [(p, q, self.time, self.y, self.ysig)] * ntrials

        if njobs == 1:
            MAPs = map(_get_map_single, args)
        else:
            # use multiple processors
            pool = multiprocessing.Pool(njobs)
            MAPs = pool.map(_get_map_single, args)

        best_MAP = MAPs[0]
        for MAP in MAPs:
            if MAP.fun < best_MAP.fun:  # note that MAP.fun is -loglik since we use scipy.optimize.minimize
                # new MLE found, save this value
                best_MAP = MAP

        print best_MAP.message

        return best_MAP

    def choose_order(self, pmax, qmax=None, pqlist=None, njobs=1, ntrials=100):

        try:
            pmax > 0
        except ValueError:
            "Order of AR polynomial must be at least 1."

        if qmax is None:
            qmax = pmax - 1

        try:
            pmax > qmax
        except ValueError:
            " Order of AR polynomial, p, must be larger than order of MA polynimial, q."

        if pqlist is None:
            pqlist = []
            for p in xrange(1, pmax+1):
                for q in xrange(p):
                    pqlist.append((p, q))

        MAPs = []
        for pq in pqlist:
            MAP = self.get_map(pq[0], pq[1], ntrials=ntrials, njobs=njobs)
            MAPs.append(MAP)

        best_AICc = 1e300
        AICc = []
        best_MAP = MAPs[0]
        print 'p, q, AICc:'
        for MAP, pq in zip(MAPs, pqlist):
            nparams = 2 + pq[0] + pq[1]
            deviance = 2.0 * MAP.fun
            this_AICc = 2.0 * nparams + deviance + 2.0 * nparams * (nparams + 1.0) / (self.time.size - nparams - 1.0)
            print pq[0], pq[1], this_AICc
            AICc.append(this_AICc)
            if this_AICc < best_AICc:
                # new optimum found, save values
                best_MAP = MAP
                best_AICc = this_AICc
                self.p = pq[0]
                self.q = pq[1]

        print 'Model with best AICc has p =', self.p, ' and q = ', self.q

        return best_MAP, pqlist, AICc


def _get_map_single(args):

    p, q, time, y, ysig = args

    nsamples = 1
    nburnin = 25
    nwalkers = 10

    # get a CARMA process object by running the MCMC sampler for a very short period. This will provide the initial
    # guess and the function to compute the log-posterior
    tvec = arrayToVec(time)  # convert to std::vector<double> object for input into C++ wrapper
    yvec = arrayToVec(y)
    ysig_vec = arrayToVec(ysig)
    if p == 1:
        # Treat the CAR(1) case separately
        CarmaProcess = carmcmcLib.run_mcmc_car1(nsamples, nburnin, tvec, yvec, ysig_vec, 1)
    else:
        CarmaProcess = carmcmcLib.run_mcmc_carma(nsamples, nburnin, tvec, yvec, ysig_vec,
                                                 p, q, nwalkers, False, 1)

    initial_theta = CarmaProcess.getSamples()
    initial_theta = np.array(initial_theta[0])
    initial_theta[1] = 1.0  # initial guess for measurement error scale parameter

    # set bounds on parameters
    ysigma = y.std()
    dt = time[1:] - time[:-1]
    max_freq = 1.0 / dt.min()
    max_freq = 0.9 * max_freq
    min_freq = 1.0 / (time.max() - time.min())
    theta_bnds = [(ysigma / 1e4, 10.0 * ysigma)]
    theta_bnds.append((0.9, 1.1))
    theta_bnds.append((None, None))

    if p == 1:
        theta_bnds.append((np.log(min_freq), np.log(max_freq)))
    else:
        theta_bnds.extend([(None, None)] * (p + q))
        CarmaProcess.SetMLE(True)  # ignore the prior bounds when calculating CarmaProcess.getLogDensity

    thisMAP = minimize(_carma_loglik, initial_theta, args=(CarmaProcess,), method="L-BFGS-B", bounds=theta_bnds)

    return thisMAP


def _carma_loglik(theta, args):
    CppCarma = args
    # convert from logit(measerr_scale)
    theta_vec = carmcmcLib.vecD()
    theta_vec.extend(theta)
    logdens = CppCarma.getLogDensity(theta_vec)
    return -logdens


class CarmaSample(samplers.MCMCSample):
    """
    Class for storing and analyzing the MCMC samples of a CARMA(p,q) model.
    """

    def __init__(self, time, y, ysig, sampler, q=0, filename=None, MAP=None):
        """
        Constructor for the CarmaSample class.

        :param filename: A string of the name of the file containing the MCMC samples generated by carpack.
        """
        self.time = time  # The time values of the time series
        self.y = y  # The measured values of the time series
        self.ysig = ysig  # The standard deviation of the measurement errors of the time series
        self.q = q  # order of moving average polynomial

        logpost = np.array(sampler.GetLogLikes())
        trace = np.array(sampler.getSamples())

        super(CarmaSample, self).__init__(filename=filename, logpost=logpost, trace=trace)

        # now calculate the AR(p) characteristic polynomial roots, coefficients, MA coefficients, and amplitude of
        # driving noise and add them to the MCMC samples
        print "Calculating PSD Lorentzian parameters..."
        self._ar_roots()
        print "Calculating coefficients of AR polynomial..."
        self._ar_coefs()
        if self.q > 0:
            print "Calculating coefficients of MA polynomial..."

        self._ma_coefs(trace)

        print "Calculating sigma..."
        self._sigma_noise()

        # add the log-likelihoods
        print "Calculating log-likelihoods..."
        loglik = np.empty(logpost.size)
        for i in xrange(logpost.size):
            std_theta = carmcmcLib.vecD()
            std_theta.extend(trace[i, :])
            loglik[i] = logpost[i] - sampler.getLogPrior(std_theta)

        self._samples['loglik'] = loglik

        # make the parameter names (i.e., the keys) public so the user knows how to get them
        self.parameters = self._samples.keys()
        self.newaxis()

        self.map = {}
        if MAP is not None:
            # add maximum a posteriori estimate
            self.add_map(MAP)

    def add_map(self, MAP):
        self.map = {'loglik': -MAP.fun, 'var': MAP.x[0] ** 2, 'measerr_scale': MAP.x[1], 'mu': MAP.x[2]}

        # add AR polynomial roots and PSD lorentzian parameters
        quad_coefs = np.exp(MAP.x[3:self.p + 3])
        ar_roots = np.zeros(self.p, dtype=complex)
        psd_width = np.zeros(self.p)
        psd_cent = np.zeros(self.p)

        for i in xrange(self.p / 2):
            quad1 = quad_coefs[2 * i]
            quad2 = quad_coefs[2 * i + 1]

            discriminant = quad2 ** 2 - 4.0 * quad1
            if discriminant > 0:
                sqrt_disc = np.sqrt(discriminant)
            else:
                sqrt_disc = 1j * np.sqrt(np.abs(discriminant))

            ar_roots[2 * i] = -0.5 * (quad2 + sqrt_disc)
            ar_roots[2 * i + 1] = -0.5 * (quad2 - sqrt_disc)
            psd_width[2 * i] = -np.real(ar_roots[2 * i]) / (2.0 * np.pi)
            psd_cent[2 * i] = np.abs(np.imag(ar_roots[2 * i])) / (2.0 * np.pi)
            psd_width[2 * i + 1] = -np.real(ar_roots[2 * i + 1]) / (2.0 * np.pi)
            psd_cent[2 * i + 1] = np.abs(np.imag(ar_roots[2 * i + 1])) / (2.0 * np.pi)

        if self.p % 2 == 1:
            # p is odd, so add in root from linear term
            ar_roots[-1] = -quad_coefs[-1]
            psd_cent[-1] = 0.0
            psd_width[-1] = quad_coefs[-1] / (2.0 * np.pi)

        self.map['ar_roots'] = ar_roots
        self.map['psd_width'] = psd_width
        self.map['psd_cent'] = psd_cent
        self.map['ar_coefs'] = np.poly(ar_roots).real

        # now calculate the moving average coefficients
        if self.q == 0:
            self.map['ma_coefs'] = 1.0
        else:
            quad_coefs = np.exp(MAP.x[3 + self.p:])
            ma_roots = np.empty(quad_coefs.size, dtype=complex)
            for i in xrange(self.q / 2):
                quad1 = quad_coefs[:, 2 * i]
                quad2 = quad_coefs[:, 2 * i + 1]

                discriminant = quad2 ** 2 - 4.0 * quad1
                if discriminant > 0:
                    sqrt_disc = np.sqrt(discriminant)
                else:
                    sqrt_disc = 1j * np.sqrt(np.abs(discriminant))

                ma_roots[2 * i] = -0.5 * (quad2 + sqrt_disc)
                ma_roots[2 * i + 1] = -0.5 * (quad2 - sqrt_disc)

            if self.q % 2 == 1:
                # q is odd, so add in root from linear term
                ma_roots[-1] = -quad_coefs[-1]

            ma_coefs = np.poly(ma_roots)
            # normalize so constant in polynomial is unity, and reverse order to be consistent with MA
            # representation
            self.map['ma_coefs'] = np.real(ma_coefs / ma_coefs[self.q])[::-1]

        # finally, calculate sigma, the standard deviation in the driving white noise
        unit_var = carma_variance(1.0, self.map['ar_roots'], self.map['ma_coefs'])
        self.map['sigma'] = np.sqrt(self.map['var'] / unit_var.real)

    def set_logpost(self, logpost):
        self._samples['logpost'] = logpost  # log-posterior of the CAR(p) model

    def generate_from_trace(self, trace):
        # Figure out how many AR terms we have
        self.p = trace.shape[1] - 3 - self.q
        names = ['var', 'measerr_scale', 'mu', 'quad_coefs']
        if names != self._samples.keys():
            idx = 0
            # Parameters are not already in the dictionary, add them.
            self._samples['var'] = (trace[:, 0] ** 2)     # Variance of the CAR(p) process
            self._samples['measerr_scale'] = trace[:, 1]  # Measurement errors are scaled by this much.
            self._samples['mu'] = trace[:, 2]             # model mean of time series
            # AR(p) polynomial is factored as a product of quadratic terms:
            #   alpha(s) = (quad_coefs[0] + quad_coefs[1] * s + s ** 2) * ...
            self._samples['quad_coefs'] = np.exp(trace[:, 3:self.p + 3])

    def generate_from_file(self, filename):
        """
        Build the dictionary of parameter samples from an ascii file of MCMC samples from carpack.

        :param filename: The name of the file containing the MCMC samples generated by carpack.
        """
        # TODO: put in exceptions to make sure files are ready correctly
        # Grab the MCMC output
        trace = np.genfromtxt(filename[0], skip_header=1)
        self.generate_from_trace(trace[:, 0:-1])
        self.set_logpost(trace[:, -1])

    def _ar_roots(self):
        """
        Calculate the roots of the CARMA(p,q) characteristic polynomial and add them to the MCMC samples.
        """
        var = self._samples['var']
        quad_coefs = self._samples['quad_coefs']
        self._samples['ar_roots'] = np.empty((var.size, self.p), dtype=complex)
        self._samples['psd_centroid'] = np.empty((var.size, self.p))
        self._samples['psd_width'] = np.empty((var.size, self.p))

        for i in xrange(self.p / 2):
            quad1 = quad_coefs[:, 2 * i]
            quad2 = quad_coefs[:, 2 * i + 1]

            discriminant = quad2 ** 2 - 4.0 * quad1
            sqrt_disc = np.where(discriminant > 0, np.sqrt(discriminant), 1j * np.sqrt(np.abs(discriminant)))
            self._samples['ar_roots'][:, 2 * i] = -0.5 * (quad2 + sqrt_disc)
            self._samples['ar_roots'][:, 2 * i + 1] = -0.5 * (quad2 - sqrt_disc)
            self._samples['psd_width'][:, 2 * i] = -np.real(self._samples['ar_roots'][:, 2 * i]) / (2.0 * np.pi)
            self._samples['psd_centroid'][:, 2 * i] = np.abs(np.imag(self._samples['ar_roots'][:, 2 * i])) / \
                (2.0 * np.pi)
            self._samples['psd_width'][:, 2 * i + 1] = -np.real(self._samples['ar_roots'][:, 2 * i + 1]) / (2.0 * np.pi)
            self._samples['psd_centroid'][:, 2 * i + 1] = np.abs(np.imag(self._samples['ar_roots'][:, 2 * i + 1])) / \
                (2.0 * np.pi)

        if self.p % 2 == 1:
            # p is odd, so add in root from linear term
            self._samples['ar_roots'][:, -1] = -quad_coefs[:, -1]
            self._samples['psd_centroid'][:, -1] = 0.0
            self._samples['psd_width'][:, -1] = quad_coefs[:, -1] / (2.0 * np.pi)

    def _ma_coefs(self, trace):
        """
        Calculate the CARMA(p,q) moving average coefficients and add them to the MCMC samples.
        """
        nsamples = trace.shape[0]
        if self.q == 0:
            self._samples['ma_coefs'] = np.ones((nsamples, 1))
        else:
            quad_coefs = np.exp(trace[:, 3 + self.p:])
            roots = np.empty(quad_coefs.shape, dtype=complex)
            for i in xrange(self.q / 2):
                quad1 = quad_coefs[:, 2 * i]
                quad2 = quad_coefs[:, 2 * i + 1]

                discriminant = quad2 ** 2 - 4.0 * quad1
                sqrt_disc = np.where(discriminant > 0, np.sqrt(discriminant), 1j * np.sqrt(np.abs(discriminant)))
                roots[:, 2 * i] = -0.5 * (quad2 + sqrt_disc)
                roots[:, 2 * i + 1] = -0.5 * (quad2 - sqrt_disc)

            if self.q % 2 == 1:
                # q is odd, so add in root from linear term
                roots[:, -1] = -quad_coefs[:, -1]

            coefs = np.empty((nsamples, self.q + 1), dtype=complex)
            for i in xrange(nsamples):
                coefs_i = np.poly(roots[i, :])
                # normalize so constant in polynomial is unity, and reverse order to be consistent with MA
                # representation
                coefs[i, :] = (coefs_i / coefs_i[self.q])[::-1]

            self._samples['ma_coefs'] = coefs.real

    def _ar_coefs(self):
        """
        Calculate the CARMA(p,q) autoregressive coefficients and add them to the MCMC samples.
        """
        roots = self._samples['ar_roots']
        coefs = np.empty((roots.shape[0], self.p + 1), dtype=complex)
        for i in xrange(roots.shape[0]):
            coefs[i, :] = np.poly(roots[i, :])

        self._samples['ar_coefs'] = coefs.real

    def _sigma_noise(self):
        """
        Calculate the MCMC samples of the standard deviation of the white noise driving process and add them to the
        MCMC samples.
        """
        # get the CARMA(p,q) model variance of the time series
        var = self._samples['var']

        # get the roots of the AR(p) characteristic polynomial
        ar_roots = self._samples['ar_roots']

        # get the moving average coefficients
        ma_coefs = self._samples['ma_coefs']

        # calculate the variance of a CAR(p) process, assuming sigma = 1.0
        sigma1_variance = np.zeros_like(var) + 0j
        for k in xrange(self.p):
            denom = -2.0 * ar_roots[:, k].real + 0j
            for l in xrange(self.p):
                if l != k:
                    denom *= (ar_roots[:, l] - ar_roots[:, k]) * (np.conjugate(ar_roots[:, l]) + ar_roots[:, k])

            ma_sum1 = np.zeros_like(ar_roots[:, 0])
            ma_sum2 = ma_sum1.copy()
            for l in xrange(ma_coefs.shape[1]):
                ma_sum1 += ma_coefs[:, l] * ar_roots[:, k] ** l
                ma_sum2 += ma_coefs[:, l] * (-1.0 * ar_roots[:, k]) ** l
            numer = ma_sum1 * ma_sum2
            sigma1_variance += numer / denom

        sigsqr = var / sigma1_variance.real

        # add the white noise sigmas to the MCMC samples
        self._samples['sigma'] = np.sqrt(sigsqr)

    def plot_power_spectrum(self, percentile=68.0, nsamples=None, plot_log=True, color="b", alpha=0.5, sp=None,
                            doShow=True):
        """
        Plot the posterior median and the credibility interval corresponding to percentile of the CAR(p) PSD. This
        function returns a tuple containing the lower and upper PSD credibility intervals as a function of
        frequency, the median PSD as a function of frequency, and the frequencies.
        
        :rtype : A tuple of numpy arrays, (lower PSD, upper PSD, median PSD, frequencies).
        :param percentile: The percentile of the PSD credibility interval to plot.
        :param nsamples: The number of MCMC samples to use to estimate the credibility interval. The default is all
                         of them.
        :param plot_log: A boolean. If true, then a logarithmic plot is made.
        """
        sigmas = self._samples['sigma']
        ar_coefs = self._samples['ar_coefs']
        ma_coefs = self._samples['ma_coefs']
        if nsamples is None:
            # Use all of the MCMC samples
            nsamples = sigmas.shape[0]
        else:
            try:
                nsamples <= sigmas.shape[0]
            except ValueError:
                "nsamples must be less than the total number of MCMC samples."

            nsamples0 = sigmas.shape[0]
            index = np.arange(nsamples) * (nsamples0 / nsamples)
            sigmas = sigmas[index]
            ar_coefs = ar_coefs[index]
            ma_coefs = ma_coefs[index]

        nfreq = 1000
        dt_min = self.time[1:] - self.time[0:self.time.size - 1]
        dt_min = dt_min.min()
        dt_max = self.time.max() - self.time.min()

        # Only plot frequencies corresponding to time scales a factor of 2 shorter and longer than the minimum and
        # maximum time scales probed by the time series.
        freq_max = 1.0 / (dt_min / 2.0)
        freq_min = (1.0 / (2.0 * dt_max))

        frequencies = np.linspace(np.log(freq_min), np.log(freq_max), num=nfreq)
        frequencies = np.exp(frequencies)
        psd_credint = np.empty((nfreq, 3))

        lower = (100.0 - percentile) / 2.0  # lower and upper intervals for credible region
        upper = 100.0 - lower

        # Compute the PSDs from the MCMC samples
        omega = 2.0 * np.pi * 1j * frequencies
        ar_poly = np.zeros((nfreq, nsamples), dtype=complex)
        ma_poly = np.zeros_like(ar_poly)
        for k in xrange(self.p):
            # Here we compute:
            #   alpha(omega) = ar_coefs[0] * omega^p + ar_coefs[1] * omega^(p-1) + ... + ar_coefs[p]
            # Note that ar_coefs[0] = 1.0.
            argrid, omgrid = np.meshgrid(ar_coefs[:, k], omega)
            ar_poly += argrid * (omgrid ** (self.p - k))
        ar_poly += ar_coefs[:, self.p]
        for k in xrange(ma_coefs.shape[1]):
            # Here we compute:
            #   delta(omega) = ma_coefs[0] + ma_coefs[1] * omega + ... + ma_coefs[q] * omega^q
            magrid, omgrid = np.meshgrid(ma_coefs[:, k], omega)
            ma_poly += magrid * (omgrid ** k)

        psd_samples = np.squeeze(sigmas) ** 2 * np.abs(ma_poly) ** 2 / np.abs(ar_poly) ** 2

        # Now compute credibility interval for power spectrum
        psd_credint[:, 0] = np.percentile(psd_samples, lower, axis=1)
        psd_credint[:, 2] = np.percentile(psd_samples, upper, axis=1)
        psd_credint[:, 1] = np.median(psd_samples, axis=1)

        # Plot the power spectra
        if sp == None:
            fig = plt.figure()
            sp = fig.add_subplot(111)

        if plot_log:
            # plot the posterior median first
            sp.loglog(frequencies, psd_credint[:, 1], color=color)
        else:
            sp.plot(frequencies, psd_credint[:, 1], color=color)

        sp.fill_between(frequencies, psd_credint[:, 2], psd_credint[:, 0], facecolor=color, alpha=alpha)
        sp.set_xlim(frequencies.min(), frequencies.max())
        sp.set_xlabel('Frequency')
        sp.set_ylabel('Power Spectrum')

        if doShow:
            plt.show()

        if sp == None:
            return (psd_credint[:, 0], psd_credint[:, 2], psd_credint[:, 1], frequencies, fig)
        else:
            return (psd_credint[:, 0], psd_credint[:, 2], psd_credint[:, 1], frequencies)


    def makeKalmanFilter(self, bestfit):
        if bestfit == 'map':
            # use maximum a posteriori estimate
            max_index = self._samples['logpost'].argmax()
            sigsqr = (self._samples['sigma'][max_index] ** 2)[0]
            mu = self._samples['mu'][max_index][0]
            ar_roots = self._samples['ar_roots'][max_index]
            ma_coefs = self._samples['ma_coefs'][max_index]
        elif bestfit == 'median':
            # use posterior median estimate
            sigsqr = np.median(self._samples['sigma']) ** 2
            mu = np.median(self._samples['mu'])
            ar_roots = np.median(self._samples['ar_roots'], axis=0)
            ma_coefs = np.median(self._samples['ma_coefs'], axis=0)
        else:
            # use posterior mean as the best-fit
            sigsqr = np.mean(self._samples['sigma'] ** 2)
            mu = np.mean(self._samples['mu'])
            ar_roots = np.mean(self._samples['ar_roots'], axis=0)
            ma_coefs = np.mean(self._samples['ma_coefs'], axis=0)

        kfilter = carmcmcLib.KalmanFilterp(arrayToVec(self.time),
                                           arrayToVec(self.y - mu),
                                           arrayToVec(self.ysig),
                                           sigsqr, 
                                           arrayToVec(ar_roots, carmcmcLib.vecC),
                                           arrayToVec(ma_coefs))
        return kfilter, mu

    def plot_models(self, bestfit="median", nplot=256, doShow=True, dtPredict=0):
        bestfit = bestfit.lower()
        try:
            bestfit in ['map', 'median', 'mean']
        except ValueError:
            "bestfit must be one of 'map, 'median', or 'mean'"

        
        fig = plt.figure()
        sp = fig.add_subplot(111)
        sp.errorbar(self.time, self.y, yerr=self.ysig, fmt='ko', label='Data', ms=4, capsize=1)

        # The kalman filter seems to exactly recover the data, no point in this...
        if False:
            kfilter, mu = self.makeKalmanFilter(bestfit)
            kfilter.Filter()
            kmean = np.empty(self.time.size)
            kvar  = np.empty(self.time.size)
            for i in xrange(self.time.size):
                kpred = kfilter.Predict(self.time[i])
                kmean[i] = kpred.first
                kvar[i]  = kpred.second
            sp.plot(self.time, kmean + mu, '-r', label='Kalman Filter')

        # compute the marginal mean and variance of the predicted values
        time_predict = np.linspace(self.time.min(), self.time.max() + dtPredict, nplot)
        predicted_mean, predicted_var = self.predict_lightcurve(time_predict, bestfit=bestfit)
        sp.plot(time_predict, predicted_mean, '-r', label='Kalman Filter')

        # NOTE we can get negative variance here in the first/last indices
        idx = np.where(predicted_var > 0)
        time_predict = time_predict[idx]
        predicted_mean = predicted_mean[idx]
        predicted_var = predicted_var[idx]

        predicted_low = predicted_mean - np.sqrt(predicted_var)
        predicted_high = predicted_mean + np.sqrt(predicted_var)
        sp.fill_between(time_predict, predicted_low, predicted_high,
                        edgecolor=None, facecolor='blue', alpha=0.25, label="1-sigma range")
        sp.set_xlabel('Time')
        sp.set_xlim(self.time.min(), self.time.max())
        sp.legend(loc=1)

        if doShow:
            plt.show()

    def assess_fit(self, bestfit="map", nplot=256, doShow=True):
        """
        Display plots and provide useful information for assessing the quality of the CARMA(p.q) model fit.

        :param bestfit: A string specifying how to define 'best-fit'. Can be the Maximum Posterior (MAP), the posterior
            mean ("mean") or the posterior median ("median").
        """
        bestfit = bestfit.lower()
        try:
            bestfit in ['map', 'median', 'mean']
        except ValueError:
            "bestfit must be one of 'map, 'median', or 'mean'"

        fig = plt.figure()
        # compute the marginal mean and variance of the predicted values
        time_predict = np.linspace(self.time[1:].min(), self.time.max(), nplot)
        predicted_mean, predicted_var = self.predict_lightcurve(time_predict, bestfit=bestfit)
        predicted_low = predicted_mean - np.sqrt(predicted_var)
        predicted_high = predicted_mean + np.sqrt(predicted_var)

        # plot the time series and the marginal 1-sigma error bands
        plt.subplot(221)
        plt.fill_between(time_predict, predicted_low, predicted_high, color='cyan')
        plt.plot(time_predict, predicted_mean, '-b', label='Interpolation')
        plt.plot(self.time, self.y, 'k.', label='Data')
        plt.xlabel('Time')
        plt.xlim(self.time.min(), self.time.max())
        #plt.legend()

        # plot the standardized residuals and compare with the standard normal
        kfilter, mu = self.makeKalmanFilter(bestfit)
        kfilter.Filter()
        kmean = np.asarray(kfilter.GetMean())
        kvar = np.asarray(kfilter.GetVar())
        standardized_residuals = (self.y - mu - kmean) / np.sqrt(kvar)
        plt.subplot(222)
        plt.xlabel('Time')
        plt.xlim(self.time.min(), self.time.max())

        # Now add the histogram of values to the standardized residuals plot
        pdf, bin_edges = np.histogram(standardized_residuals, bins=10)
        bin_edges = bin_edges[0:pdf.size]
        # Stretch the PDF so that it is readable on the residual plot when plotted horizontally
        pdf = pdf / float(pdf.max()) * 0.4 * self.time.max()
        # Add the histogram to the plot
        plt.barh(bin_edges, pdf, height=bin_edges[1] - bin_edges[0])
        # now overplot the expected standard normal distribution
        expected_pdf = np.exp(-0.5 * bin_edges ** 2)
        expected_pdf = expected_pdf / expected_pdf.max() * 0.4 * self.time.max()
        plt.plot(expected_pdf, bin_edges, 'DarkOrange', lw=2)
        plt.plot(self.time, standardized_residuals, '.k')


        # plot the autocorrelation function of the residuals and compare with the 95% confidence intervals for white
        # noise
        plt.subplot(223)
        maxlag = 50
        wnoise_upper = 1.96 / np.sqrt(self.time.size)
        wnoise_lower = -1.96 / np.sqrt(self.time.size)
        plt.fill_between([0, maxlag], wnoise_upper, wnoise_lower, facecolor='grey')
        lags, acf, not_needed1, not_needed2 = plt.acorr(standardized_residuals, maxlags=maxlag, lw=2)
        plt.xlim(0, maxlag)
        plt.xlabel('Time Lag')
        plt.ylabel('ACF of Residuals')

        # plot the autocorrelation function of the squared residuals and compare with the 95% confidence intervals for
        # white noise
        plt.subplot(224)
        squared_residuals = standardized_residuals ** 2
        wnoise_upper = 1.96 / np.sqrt(self.time.size)
        wnoise_lower = -1.96 / np.sqrt(self.time.size)
        plt.fill_between([0, maxlag], wnoise_upper, wnoise_lower, facecolor='grey')
        lags, acf, not_needed1, not_needed2 = plt.acorr(squared_residuals - squared_residuals.mean(), maxlags=maxlag,
                                                        lw=2)
        plt.xlim(0, maxlag)
        plt.xlabel('Time Lag')
        plt.ylabel('ACF of Sqrd. Resid.')
        plt.tight_layout()

        if doShow:
            plt.show()
        else:
            return fig

    def predict_lightcurve(self, time, bestfit='median'):
        """
        Return the predicted value of the lightcurve and its standard deviation at the input time(s) given the best-fit
        value of the CARMA(p,q) model and the measured lightcurve.

        :param time: A scalar or numpy array containing the time values to predict the time series at.
        :param bestfit: A string specifying how to define 'best-fit'. Can be the Maximum Posterior (MAP), the posterior
            mean ("mean") or the posterior median ("median").
        """
        bestfit = bestfit.lower()
        try:
            bestfit in ['map', 'median', 'mean']
        except ValueError:
            "bestfit must be one of 'map, 'median', or 'mean'"

        # note that KalmanFilter class assumes the time series has zero mean
        kfilter, mu = self.makeKalmanFilter(bestfit)
        kfilter.Filter()
        if np.isscalar(time):
            pred = kfilter.Predict(time)
            yhat = pred.first
            yhat_var = pred.second
        else:
            yhat = np.empty(time.size)
            yhat_var = np.empty(time.size)
            for i in xrange(time.size):
                pred = kfilter.Predict(time[i])
                yhat[i] = pred.first
                yhat_var[i] = pred.second

        yhat += mu  # add mean back into time series

        return yhat, yhat_var

    def simulate_lightcurve(self, time, bestfit='median'):
        """
        Simulate a lightcurve at the input time(s) given the best-fit value of the CARMA(p,q) model and the measured
        lightcurve.

        :param time: A scalar or numpy array containing the time values to simulate the time series at.
        :param bestfit: A string specifying how to define 'best-fit'. Can be the Maximum Posterior (MAP), the posterior
            mean ("mean") or the posterior median ("median").
        """
        bestfit = bestfit.lower()
        try:
            bestfit in ['map', 'median', 'mean']
        except ValueError:
            "bestfit must be one of 'map, 'median', or 'mean'"

        # note that KalmanFilter class assumes the time series has zero mean
        kfilter, mu = self.makeKalmanFilter(bestfit)
        kfilter.Filter()
        vtime = carmcmcLib.vecD()
        if np.isscalar(time):
            vtime.append(time)
        else:
            vtime.extend(time)

        ysim = np.asarray(kfilter.Simulate(vtime))
        ysim += mu  # add mean back into time series

        return ysim

    def DIC(self):
        """ 
        Calculate the Deviance Information Criterion for the model.

        The deviance is -2 * log-likelihood, and the DIC is:

            DIC = mean(deviance) + 0.5 * variance(deviance)
        """

        deviance = -2.0 * self._samples['loglik']
        mean_deviance = np.mean(deviance, axis=0)
        effect_npar = 0.5 * np.var(deviance, axis=0)

        dic = mean_deviance + effect_npar

        return dic


def arrayToVec(array, arrType=carmcmcLib.vecD):
    vec = arrType()
    vec.extend(array)
    return vec


class CarSample1(CarmaSample):
    def __init__(self, time, y, ysig, sampler, filename=None):
        self.time = time  # The time values of the time series
        self.y = y     # The measured values of the time series
        self.ysig = ysig  # The standard deviation of the measurement errors of the time series
        self.p = 1     # How many AR terms
        self.q = 0     # How many MA terms

        logpost = np.array(sampler.GetLogLikes())
        trace = np.array(sampler.getSamples())

        super(CarmaSample, self).__init__(filename=filename, logpost=logpost, trace=trace)

        print "Calculating sigma..."
        self._sigma_noise()

        # add the log-likelihoods
        print "Calculating log-likelihoods..."
        loglik = np.empty(logpost.size)
        for i in xrange(logpost.size):
            std_theta = carmcmcLib.vecD()
            std_theta.extend(trace[i, :])
            loglik[i] = logpost[i] - sampler.getLogPrior(std_theta)

        self._samples['loglik'] = loglik
        # make the parameter names (i.e., the keys) public so the use knows how to get them
        self.parameters = self._samples.keys()
        self.newaxis()

    def generate_from_trace(self, trace):
        names = ['sigma', 'measerr_scale', 'mu', 'log_omega']
        if names != self._samples.keys():
            self._samples['var'] = trace[:, 0]
            self._samples['measerr_scale'] = trace[:, 1]
            self._samples['mu'] = trace[:, 2]
            self._samples['log_omega'] = trace[:, 3]

    def _ar_roots(self):
        print "_ar_roots not supported for CAR1"
        return

    def _ar_coefs(self):
        print "_ar_coefs not supported for CAR1"
        return

    def _sigma_noise(self):
        self._samples['sigma'] = np.sqrt(2.0 * self._samples['var'] * np.exp(self._samples['log_omega']))

    def makeKalmanFilter(self, bestfit):
        if bestfit == 'map':
            # use maximum a posteriori estimate
            max_index = self._samples['logpost'].argmax()
            sigsqr = (self._samples['sigma'][max_index] ** 2)[0]
            mu = self._samples['mu'][max_index][0]
            log_omega = self._samples['log_omega'][max_index][0]
        elif bestfit == 'median':
            # use posterior median estimate
            sigsqr = np.median(self._samples['sigma']) ** 2
            mu = np.median(self._samples['mu'])
            log_omega = np.median(self._samples['log_omega'])
        else:
            # use posterior mean as the best-fit
            sigsqr = np.mean(self._samples['sigma'] ** 2)
            mu = np.mean(self._samples['mu'])
            log_omega = np.mean(self._samples['log_omega'])

        kfilter = carmcmcLib.KalmanFilter1(arrayToVec(self.time),
                                           arrayToVec(self.y - mu),
                                           arrayToVec(self.ysig),
                                           sigsqr, 
                                           10**(log_omega))
        return kfilter, mu

    def plot_power_spectrum(self, percentile=68.0, plot_log=True, color="b", sp=None, doShow=True):
        sigmas = self._samples['sigma']
        log_omegas = self._samples['log_omega']

        nfreq = 1000
        dt_min = self.time[1:] - self.time[0:self.time.size - 1]
        dt_min = dt_min.min()
        dt_max = self.time.max() - self.time.min()

        # Only plot frequencies corresponding to time scales a factor of 2 shorter and longer than the minimum and
        # maximum time scales probed by the time series.
        freq_max = 1.0 / (dt_min / 2.0)
        freq_min = (1.0 / (2.0 * dt_max))

        frequencies = np.linspace(np.log(freq_min), np.log(freq_max), num=nfreq)
        frequencies = np.exp(frequencies)
        psd_credint = np.empty((nfreq, 3))

        lower = (100.0 - percentile) / 2.0  # lower and upper intervals for credible region
        upper = 100.0 - lower

        numer = 0.5 / np.pi * sigmas ** 2
        for i in xrange(nfreq):
            denom = 10 ** log_omegas ** 2 + frequencies[i] ** 2
            psd_samples = numer / denom

            # Now compute credibility interval for power spectrum
            psd_credint[i, 0] = np.percentile(psd_samples, lower, axis=0)
            psd_credint[i, 2] = np.percentile(psd_samples, upper, axis=0)
            psd_credint[i, 1] = np.median(psd_samples, axis=0)

        # Plot the power spectra
        if sp == None:
            fig = plt.figure()
            sp = fig.add_subplot(111)

        if plot_log:
            # plot the posterior median first
            sp.loglog(frequencies, psd_credint[:, 1], color=color)
        else:
            sp.plot(frequencies, psd_credint[:, 1], color=color)

        sp.fill_between(frequencies, psd_credint[:, 2], psd_credint[:, 0], facecolor=color, alpha=0.5)
        sp.set_xlim(frequencies.min(), frequencies.max())
        sp.set_xlabel('Frequency')
        sp.set_ylabel('Power Spectrum')
        if doShow:
            plt.show()
            return (psd_credint[:, 0], psd_credint[:, 2], psd_credint[:, 1], frequencies)
        else:
            return (psd_credint[:, 0], psd_credint[:, 2], psd_credint[:, 1], frequencies), fig
            

    def plot_2dpdf(self, name1, name2, doShow=False):
        print "Plotting 2d PDF"
        trace1 = self._samples[name1]
        trace2 = self._samples[name2]

        fig = plt.figure()
        # joint distribution
        axJ = fig.add_axes([0.1, 0.1, 0.7, 0.7])               # [left, bottom, width, height]
        # y histogram
        axY = fig.add_axes([0.8, 0.1, 0.125, 0.7], sharey=axJ)
        # x histogram
        axX = fig.add_axes([0.1, 0.8, 0.7, 0.125], sharex=axJ)
        axJ.plot(trace1, trace2, 'ro', ms=1, alpha=0.5)
        axX.hist(trace1, bins=100)
        axY.hist(trace2, orientation='horizontal', bins=100)
        axJ.set_xlabel("%s" % (name1))
        axJ.set_ylabel("%s" % (name2))
        plt.setp(axX.get_xticklabels() + axX.get_yticklabels(), visible=False)
        plt.setp(axY.get_xticklabels() + axY.get_yticklabels(), visible=False)
        if doShow:
            plt.show()
        

##################

def get_ar_roots(qpo_width, qpo_centroid):
    """
    Return the roots of the characteristic polynomial of the CAR(p) process, given the lorentzian widths and centroids.

    :rtype : a numpy array
    :param qpo_width: The widths of the lorentzian functions defining the PSD.
    :param qpo_centroid: The centroids of the lorentzian functions defining the PSD.
    """
    p = qpo_centroid.size + qpo_width.size
    ar_roots = np.empty(p, dtype=complex)
    for i in xrange(p / 2):
        ar_roots[2 * i] = qpo_width[i] + 1j * qpo_centroid[i]
        ar_roots[2 * i + 1] = np.conjugate(ar_roots[2 * i])
    if p % 2 == 1:
        # p is odd, so add in low-frequency component
        ar_roots[-1] = qpo_width[-1]

    return -2.0 * np.pi * ar_roots


def power_spectrum(freq, sigma, ar_coef, ma_coefs=[1.0]):
    """
    Return the power spectrum for a CAR(p) process calculated at the input frequencies.

    :param freq: The frequencies at which to calculate the PSD.
    :param sigma: The standard deviation driving white noise.
    :param ar_coef: The CAR(p) model autoregressive coefficients.
    :param ma_coefs: Coefficients of the moving average polynomial

    :rtype : A numpy array.
    """
    try:
        len(ma_coefs) <= len(ar_coef)
    except ValueError:
        "Size of ma_coefs must be less or equal to size of ar_roots."

    ma_poly = np.polyval(ma_coefs[::-1], 2.0 * np.pi * 1j * freq)  # Evaluate the polynomial in the PSD numerator
    ar_poly = np.polyval(ar_coef, 2.0 * np.pi * 1j * freq)  # Evaluate the polynomial in the PSD denominator
    pspec = sigma ** 2 * np.abs(ma_poly) ** 2 / np.abs(ar_poly) ** 2
    return pspec


def carma_variance(sigsqr, ar_roots, ma_coefs=[1.0], lag=0.0):
    """
    Return the autocovariance function of a CARMA(p,q) process.

    :param sigsqr: The variance in the driving white noise.
    :param ar_roots: The roots of the AR characteristic polynomial.
    :param ma_coefs: The moving average coefficients.
    :param lag: The lag at which to calculate the autocovariance function.
    """
    try:
        len(ma_coefs) <= len(ar_roots)
    except ValueError:
        "Size of ma_coefs must be less or equal to size of ar_roots."

    if len(ma_coefs) < len(ar_roots):
        # add extra zeros to end of ma_coefs
        ma_coefs = np.resize(np.array(ma_coefs), len(ar_roots))
        ma_coefs[1:] = 0.0

    sigma1_variance = 0.0 + 0j
    p = ar_roots.size
    for k in xrange(p):
        denom_product = 1.0 + 0j
        for l in xrange(p):
            if l != k:
                denom_product *= (ar_roots[l] - ar_roots[k]) * (np.conjugate(ar_roots[l]) + ar_roots[k])

        denom = -2.0 * denom_product * ar_roots[k].real

        ma_sum1 = 0.0 + 0j
        ma_sum2 = 0.0 + 0j
        for l in xrange(p):
            ma_sum1 += ma_coefs[l] * ar_roots[k] ** l
            ma_sum2 += ma_coefs[l] * (-1.0 * ar_roots[k]) ** l

        numer = ma_sum1 * ma_sum2 * np.exp(ar_roots[k] * abs(lag))

        sigma1_variance += numer / denom

    return sigsqr * sigma1_variance.real


def carma_process(time, sigsqr, ar_roots, ma_coefs=[1.0]):
    """
    Generate a CARMA(p,q) process.

    :param time: The time values at which to generate the CARMA(p,q) process at.
    :param sigsqr: The variance in the driving white noise term.
    :param ar_roots: The roots of the CAR(p) characteristic polynomial.
    :param ma_coefs: The moving average coefficients.
    :rtype : A numpy array containing the simulated CARMA(p,q) process values at time.
    """
    try:
        len(ma_coefs) <= len(ar_roots)
    except ValueError:
        "Size of ma_coefs must be less or equal to size of ar_roots."

    p = len(ar_roots)

    if len(ma_coefs) < p:
        # add extra zeros to end of ma_coefs
        ma_coefs = np.resize(np.array(ma_coefs), len(ar_roots))
        ma_coefs[1:] = 0.0

    time.sort()
    # make sure process is stationary
    try:
        np.any(ar_roots.real < 0)
    except ValueError:
        "Process is not stationary, real part of roots must be negative."

    # make sure the roots are unique
    tol = 1e-8
    roots_grid = np.meshgrid(ar_roots, ar_roots)
    roots_grid1 = roots_grid[0].ravel()
    roots_grid2 = roots_grid[1].ravel()
    diff_roots = np.abs(roots_grid1 - roots_grid2) / np.abs(roots_grid1 + roots_grid2)
    try:
        np.any(diff_roots > tol)
    except ValueError:
        "Roots are not unique."

    # Setup the matrix of Eigenvectors for the Kalman Filter transition matrix. This allows us to transform
    # quantities into the rotated state basis, which makes the computations for the Kalman filter easier and faster.
    EigenMat = np.ones((p, p), dtype=complex)
    EigenMat[1, :] = ar_roots
    for k in xrange(2, p):
        EigenMat[k, :] = ar_roots ** k

    # Input vector under the original state space representation
    Rvector = np.zeros(p, dtype=complex)
    Rvector[-1] = 1.0

    # Input vector under rotated state space representation
    Jvector = solve(EigenMat, Rvector)  # J = inv(E) * R

    # Compute the vector of moving average coefficients in the rotated state.
    rotated_MA_coefs = ma_coefs.dot(EigenMat)

    # Calculate the stationary covariance matrix of the state vector
    StateVar = np.empty((p, p), dtype=complex)
    for j in xrange(p):
        StateVar[:, j] = -sigsqr * Jvector * np.conjugate(Jvector[j]) / (ar_roots + np.conjugate(ar_roots[j]))

    # Initialize variance in one-step prediction error and the state vector
    PredictionVar = StateVar.copy()
    StateVector = np.zeros(p, dtype=complex)

    # Convert the current state to matrices for convenience, since we'll be doing some Linear algebra.
    StateVector = np.matrix(StateVector).T
    StateVar = np.matrix(StateVar)
    PredictionVar = np.matrix(PredictionVar)
    rotated_MA_coefs = np.matrix(rotated_MA_coefs)  # this is a row vector, so no transpose
    StateTransition = np.zeros_like(StateVector)
    KalmanGain = np.zeros_like(StateVector)

    # Initialize the Kalman mean and variance. These are the forecasted values and their variances.
    kalman_mean = 0.0
    kalman_var = np.real(np.asscalar(rotated_MA_coefs * PredictionVar * rotated_MA_coefs.H))

    # simulate the first time series value
    y = np.empty_like(time)
    y[0] = np.random.normal(kalman_mean, np.sqrt(kalman_var))

    # Initialize the innovations, i.e., the KF residuals
    innovation = y[0]

    for i in xrange(1, time.size):
        # First compute the Kalman gain
        KalmanGain = PredictionVar * rotated_MA_coefs.H / kalman_var
        # update the state vector
        StateVector += innovation * KalmanGain
        # update the state one-step prediction error variance
        PredictionVar -= kalman_var * (KalmanGain * KalmanGain.H)
        # predict the next state, do element-wise multiplication
        dt = time[i] - time[i - 1]
        StateTransition = np.matrix(np.exp(ar_roots * dt)).T
        StateVector = np.multiply(StateVector, StateTransition)
        # update the predicted state covariance matrix
        PredictionVar = np.multiply(StateTransition * StateTransition.H, PredictionVar - StateVar) + StateVar
        # now predict the observation and its variance
        kalman_mean = np.real(np.asscalar(rotated_MA_coefs * StateVector))
        kalman_var = np.real(np.asscalar(rotated_MA_coefs * PredictionVar * rotated_MA_coefs.H))
        # simulate the next time series value
        y[i] = np.random.normal(kalman_mean, np.sqrt(kalman_var))
        # finally, update the innovation
        innovation = y[i] - kalman_mean

    return y

##################
# Deprecated

class KalmanFilterDeprecated(object):
    def __init__(self, time, y, yvar, sigsqr, ar_roots, ma_coefs=[1.0]):
        """
        Constructor for Kalman Filter class.

        :param time: The time values of the time series.
        :param y: The mean-subtracted time series.
        :param yvar: The variance in the measurement errors on the time series.
        :param sigsqr: The variance of the driving white noise term in the CAR(p) process.
        :param ar_roots: The roots of the autoregressive characteristic polynomial.
        """
        try:
            len(ma_coefs) <= ar_roots.size
        except ValueError:
            "Order of MA polynomial cannot be larger than order of AR polynomial."

        self.time = time
        self.y = y
        self.yvar = yvar
        self.sigsqr = sigsqr
        self.ar_roots = ar_roots
        self.p = ar_roots.size  # order of the CARMA(p,q) process
        self.q = len(ma_coefs)
        self.ma_coefs = np.append(ma_coefs, np.zeros(self.p - self.q))

    def reset(self):
        """
        Reset the Kalman Filter to its initial state.
        """
        # Setup the matrix of Eigenvectors for the Kalman Filter transition matrix. This allows us to transform
        # quantities into the rotated state basis, which makes the computations for the Kalman filter easier and faster.
        EigenMat = np.ones((self.p, self.p), dtype=complex)
        EigenMat[1, :] = self.ar_roots
        for k in xrange(2, self.p):
            EigenMat[k, :] = self.ar_roots ** k

        # Input vector under the original state space representation
        Rvector = np.zeros(self.p, dtype=complex)
        Rvector[-1] = 1.0

        # Input vector under rotated state space representation
        Jvector = solve(EigenMat, Rvector)  # J = inv(E) * R

        # Compute the vector of moving average coefficients in the rotated state.
        rotated_MA_coefs = self.ma_coefs.dot(EigenMat)

        # Calculate the stationary covariance matrix of the state vector
        StateVar = np.empty((self.p, self.p), dtype=complex)
        for j in xrange(self.p):
            StateVar[:, j] = -self.sigsqr * Jvector * np.conjugate(Jvector[j]) / \
                             (self.ar_roots + np.conjugate(self.ar_roots[j]))

        # Initialize variance in one-step prediction error and the state vector
        PredictionVar = StateVar.copy()
        StateVector = np.zeros(self.p, dtype=complex)

        # Convert the current state to matrices for convenience, since we'll be doing some Linear algebra.
        self._StateVector = np.matrix(StateVector).T
        self._StateVar = np.matrix(StateVar)
        self._PredictionVar = np.matrix(PredictionVar)
        self._rotated_MA_coefs = np.matrix(rotated_MA_coefs)  # this is a row vector, so no transpose
        self._StateTransition = np.zeros_like(self._StateVector)
        self._KalmanGain = np.zeros_like(self._StateVector)

        # Initialize the Kalman mean and variance. These are the forecasted values and their variances.
        self.kalman_mean = np.empty_like(self.time)
        self.kalman_var = np.empty_like(self.time)
        self.kalman_mean[0] = 0.0
        self.kalman_var[0] = np.real(self._rotated_MA_coefs * self._PredictionVar * self._rotated_MA_coefs.H) \
                             + self.yvar[0]

        # Initialize the innovations, i.e., the KF residuals
        self._innovation = self.y[0]

        self._current_index = 1

    def update(self):
        """
        Perform one iteration (update) of the Kalman Filter.
        """
        # First compute the Kalman gain
        self._KalmanGain = self._PredictionVar * self._rotated_MA_coefs.H / self.kalman_var[self._current_index - 1]
        # update the state vector
        self._StateVector += self._innovation * self._KalmanGain
        # update the state one-step prediction error variance
        self._PredictionVar -= self.kalman_var[self._current_index - 1] * (self._KalmanGain * self._KalmanGain.H)
        # predict the next state, do element-wise multiplication
        dt = self.time[self._current_index] - self.time[self._current_index - 1]
        self._StateTransition = np.matrix(np.exp(self.ar_roots * dt)).T
        self._StateVector = np.multiply(self._StateVector, self._StateTransition)
        # update the predicted state covariance matrix
        self._PredictionVar = np.multiply(self._StateTransition * self._StateTransition.H,
                                          self._PredictionVar - self._StateVar) + self._StateVar
        # now predict the observation and its variance
        self.kalman_mean[self._current_index] = np.real(np.asscalar(self._rotated_MA_coefs * self._StateVector))
        self.kalman_var[self._current_index] = \
            np.real(np.asscalar(self._rotated_MA_coefs * self._PredictionVar * self._rotated_MA_coefs.H))
        self.kalman_var[self._current_index] += self.yvar[self._current_index]
        # finally, update the innovation
        self._innovation = self.y[self._current_index] - self.kalman_mean[self._current_index]
        self._current_index += 1

    def filter(self):
        """
        Perform the Kalman Filter on all points of the time series. The kalman mean and variance are returned upon
        completion, and are stored in the instantiated KalmanFilter object.
        """
        self.reset()
        for i in xrange(self.time.size - 1):
            self.update()

        return self.kalman_mean, self.kalman_var

    def predict(self, time_predict):
        """
        Return the predicted value of a lightcurve and its standard deviation at the input time given the input
        values of the CARMA(p,q) model parameters and a measured lightcurve.

        :rtype : A tuple containing the predicted value and its variance.
        :param time_predict: The time at which to predict the lightcurve.
        """
        try:
            self.time.min() > time_predict
        except ValueError:
            "backcasting currently not supported: time_predict must be greater than self.time.min()"

        self.reset()
        # find the index where time[ipredict-1] < time_predict < time[ipredict]
        ipredict = np.max(np.where(self.time < time_predict)) + 1
        for i in xrange(ipredict - 1):
            # run the kalman filter for time < time_predict
            self.update()

        # predict the value of y[time_predict]
        self._KalmanGain = self._PredictionVar * self._rotated_MA_coefs.H / self.kalman_var[ipredict - 1]
        self._StateVector += self._innovation * self._KalmanGain
        self._PredictionVar -= self.kalman_var[ipredict - 1] * (self._KalmanGain * self._KalmanGain.H)
        dt = time_predict - self.time[ipredict - 1]
        self._StateTransition = np.matrix(np.exp(self.ar_roots * dt)).T
        self._StateVector = np.multiply(self._StateVector, self._StateTransition)
        self._PredictionVar = np.multiply(self._StateTransition * self._StateTransition.H,
                                          self._PredictionVar - self._StateVar) + self._StateVar

        ypredict_mean = np.asscalar(np.real(self._rotated_MA_coefs * self._StateVector))
        ypredict_var = np.asscalar(np.real(self._rotated_MA_coefs * self._PredictionVar * self._rotated_MA_coefs.H))

        # start the running statistics for the conditional mean and precision of the predicted time series value, given
        # the measured time series
        cprecision = 1.0 / ypredict_var
        cmean = cprecision * ypredict_mean

        if ipredict >= self.time.size:
            # we are forecasting (extrapolating) the value, so no need to run interpolation steps below
            return ypredict_mean, ypredict_var

        # for time > time_predict we need to compute the coefficients for the linear filter, i.e., at time[j]:
        # E(y[j]|{y[i]; j<i}) = alpha[j] + beta[j] * ypredict. we do this using recursions similar to the kalman
        # filter.

        # first set the initial values.
        self._KalmanGain = self._PredictionVar * self._rotated_MA_coefs.H / ypredict_var
        # initialize the coefficients for predicting the state vector at coefs(time_predict|time_predict)
        const_state = self._StateVector - self._KalmanGain * ypredict_mean
        slope_state = self._KalmanGain
        # update the state one-step prediction error variance
        self._PredictionVar -= ypredict_var * (self._KalmanGain * self._KalmanGain.H)
        # do coefs(time_predict|time_predict) --> coefs(time[i+1]|time_predict)
        dt = self.time[ipredict] - time_predict
        self._StateTransition = np.matrix(np.exp(self.ar_roots * dt)).T
        const_state = np.multiply(const_state, self._StateTransition)
        slope_state = np.multiply(slope_state, self._StateTransition)
        # update the predicted state covariance matrix
        self._PredictionVar = np.multiply(self._StateTransition * self._StateTransition.H,
                                          self._PredictionVar - self._StateVar) + self._StateVar
        # compute the coefficients for the linear filter at time[ipredict], and compute the variance in the predicted
        # y[ipredict]
        const = np.asscalar(np.real(self._rotated_MA_coefs * const_state))
        slope = np.asscalar(np.real(self._rotated_MA_coefs * slope_state))
        self.kalman_var[ipredict] = \
            np.real(self._rotated_MA_coefs * self._PredictionVar * self._rotated_MA_coefs.H) + \
            self.yvar[ipredict]

        # update the running conditional mean and variance of the predicted time series value
        cprecision += slope ** 2 / self.kalman_var[ipredict]
        cmean += slope * (self.y[ipredict] - const) / self.kalman_var[ipredict]

        self.const = np.zeros(self.time.size)
        self.slope = np.zeros(self.time.size)
        self.const[ipredict] = const
        self.slope[ipredict] = slope

        # now repeat for time > time_predict
        for i in xrange(ipredict + 1, self.time.size):
            self._KalmanGain = self._PredictionVar * self._rotated_MA_coefs.H / self.kalman_var[i - 1]
            # update the state prediction coefficients: coefs(i|i-1) --> coefs(i|i)
            const_state += self._KalmanGain * (self.y[i - 1] - const)
            slope_state -= self._KalmanGain * slope
            # update the state one-step prediction error variance
            self._PredictionVar -= self.kalman_var[i - 1] * (self._KalmanGain * self._KalmanGain.H)
            # compute the one-step state prediction coefficients: coefs(i|i) --> coefs(i+1|i)
            dt = self.time[i] - self.time[i - 1]
            self._StateTransition = np.matrix(np.exp(self.ar_roots * dt)).T
            const_state = np.multiply(const_state, self._StateTransition)
            slope_state = np.multiply(slope_state, self._StateTransition)
            # compute the state one-step prediction error variance
            self._PredictionVar = np.multiply(self._StateTransition * self._StateTransition.H,
                                              self._PredictionVar - self._StateVar) + self._StateVar
            # compute the coefficients for predicting y[i]|y[j],j<i as a function of ypredict
            const = np.asscalar(np.real(self._rotated_MA_coefs * const_state))
            slope = np.asscalar(np.real(self._rotated_MA_coefs * slope_state))
            # compute the variance in predicting y[i]|y[j],j<i
            self.kalman_var[i] = \
                np.real(self._rotated_MA_coefs * self._PredictionVar * self._rotated_MA_coefs.H) + \
                self.yvar[i]
            # finally, update the running conditional mean and variance of the predicted time series value
            cprecision += slope ** 2 / self.kalman_var[i]
            cmean += slope * (self.y[i] - const) / self.kalman_var[i]

            self.const[i] = const
            self.slope[i] = slope

        cvar = 1.0 / cprecision
        cmean *= cvar

        return cmean, cvar

    def simulate(self, time_simulate):
        """
        Simulate a lightcurve at the input time values of time_simulate, given the measured lightcurve and input
        CARMA(p,q) parameters.

        :rtype : A scalar or numpy array, depending on type of time_simulate.
        :param time_simulate: The time(s) at which to simulate a random draw of the lightcurve conditional on the
            measured time series and the input parameters.
        """

        if np.isscalar(time_simulate):
            cmean, cvar = self.predict(time_simulate)
            ysimulated = np.random.normal(cmean, np.sqrt(cvar))
            return ysimulated
        else:
            # input is array-like, need to simulate values sequentially, adding each value to the measured time series
            # as they are simulated
            time0 = self.time  # save original values
            y0 = self.y
            yvar0 = self.yvar
            ysimulated = np.empty(time_simulate.size)
            time_simulate.sort()
            for i in xrange(time_simulate.size):
                cmean, cvar = self.predict(time_simulate[i])
                ysimulated[i] = np.random.normal(cmean, np.sqrt(cvar))  # simulate the time series value
                # find the index where time[isimulate-1] < time_simulate < time[isimulate]
                isimulate = np.max(np.where(self.time < time_simulate[i])) + 1
                # insert the simulated value into the time series array
                self.time = np.insert(self.time, isimulate, time_simulate[i])
                self.y = np.insert(self.y, isimulate, ysimulated[i])
                self.yvar = np.insert(self.yvar, isimulate, 0.0)

            # reset measured time series to original values
            self.y = y0
            self.time = time0
            self.yvar = yvar0

        return ysimulated


