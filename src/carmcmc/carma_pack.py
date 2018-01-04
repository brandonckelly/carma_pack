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
    Class for performing statistical inference assuming a CARMA(p,q) model.
    """

    def __init__(self, time, y, ysig, p=1, q=0):
        """
        Constructor for the CarmaModel class.

        :param time: The observation times.
        :param y: The measured time series.
        :param ysig: The standard deviation in the measurements errors on the time series.
        :param p: The order of the autoregressive (AR) polynomial. Default is p = 1.
        :param q: The order of the moving average (MA) polynomial. Default is q = 0. Note that p > q.
        """
        try:
            p > q
        except ValueError:
            " Order of AR polynomial, p, must be larger than order of MA polynomial, q."

        # check that time values are unique and in ascending ordered
        s_idx = np.argsort(time)
        t_unique, u_idx = np.unique(time[s_idx], return_index=True)
        u_idx = s_idx[u_idx]

        # convert input to std::vector<double> extension class
        self._time = carmcmcLib.vecD()
        self._time.extend(time[u_idx])
        self._y = carmcmcLib.vecD()
        self._y.extend(y[u_idx])
        self._ysig = carmcmcLib.vecD()
        self._ysig.extend(ysig[u_idx])

        # save parameters
        self.time = time[u_idx]
        self.y = y[u_idx]
        self.ysig = ysig[u_idx]
        self.p = p
        self.q = q
        self.mcmc_sample = None

    def run_mcmc(self, nsamples, nburnin=None, ntemperatures=None, nthin=1, init=None):
        """
        Run the MCMC sampler. This is actually a wrapper that calls the C++ code that runs the MCMC sampler.

        :param nsamples: The number of samples from the posterior to generate.
        :param ntemperatures: Number of parallel MCMC chains to run in the parallel tempering algorithm. Default is 1
            (no tempering) for p = 1 and max(10, p+q) for p > 1.
        :param nburnin: Number of burnin iterations to run. The default is nsamples / 2.
        :param nthin: Thinning interval for the MCMC sampler. Default is 1 (no thinning).

        :return: Either a CarmaSample or Car1Sample object, depending on the values of self.p. The CarmaSample object
            will also be stored as a data member of the CarmaModel object.
        """

        if ntemperatures is None:
            ntemperatures = max(10, self.p + self.q)

        if nburnin is None:
            nburnin = nsamples / 2

        if init is None:
            init = carmcmcLib.vecD()
            
        if self.p == 1:
            # Treat the CAR(1) case separately
            cppSample = carmcmcLib.run_mcmc_car1(nsamples, nburnin, self._time, self._y, self._ysig,
                                                 nthin, init)
            # run_mcmc_car1 returns a wrapper around the C++ CAR1 class, convert to python object
            sample = Car1Sample(self.time, self.y, self.ysig, cppSample)
        else:
            cppSample = carmcmcLib.run_mcmc_carma(nsamples, nburnin, self._time, self._y, self._ysig,
                                                  self.p, self.q, ntemperatures, False, nthin, init)
            # run_mcmc_car returns a wrapper around the C++ CARMA class, convert to a python object
            sample = CarmaSample(self.time, self.y, self.ysig, cppSample, q=self.q)

        self.mcmc_sample = sample

        return sample

    def get_mle(self, p, q, ntrials=100, njobs=1):
        """
        Return the maximum likelihood estimate (MLE) of the CARMA model parameters. This is done by using the
        L-BFGS-B algorithm from scipy.optimize on ntrials randomly distributed starting values of the parameters. This
        this return NaN for more complex CARMA models, especially if the data are not well-described by a CARMA model.
        In addition, the likelihood space can be highly multi-modal, and there is no guarantee that the global MLE will
        be found using this procedure.

        @param p: The order of the AR polynomial.
        @param q: The order of the MA polynomial. Must be q < p.
        @param ntrials: The number of random starting values for the optimizer. Default is 100.
        @param njobs: The number of processors to use. If njobs = -1, then all of them are used. Default is njobs = 1.
        @return: The scipy.optimize.Result object corresponding to the MLE.
        """
        if njobs == -1:
            njobs = multiprocessing.cpu_count()

        args = [(p, q, self.time, self.y, self.ysig)] * ntrials

        if njobs == 1:
            MLEs = map(_get_mle_single, args)
        else:
            # use multiple processors
            pool = multiprocessing.Pool(njobs)
            # warm up the pool
            pool.map(int, range(multiprocessing.cpu_count()))
            MLEs = pool.map(_get_mle_single, args)
            pool.terminate()

        best_MLE = MLEs[0]
        for MLE in MLEs:
            if MLE.fun < best_MLE.fun:  # note that MLE.fun is -loglik since we use scipy.optimize.minimize
                # new MLE found, save this value
                best_MLE = MLE

        print best_MLE.message

        return best_MLE

    def choose_order(self, pmax, qmax=None, pqlist=None, njobs=1, ntrials=100):
        """
        Choose the order of the CARMA model by minimizing the AICc(p,q). This first computes the maximum likelihood
        estimate on a grid of (p,q) values using self.get_mle, and then choosing the value of (p,q) that minimizes
        the AICc. These values of p and q are stored as self.p and self.q.

        @param pmax: The maximum order of the AR(p) polynomial to search over.
        @param qmax: The maximum order of the MA(q) polynomial to search over. If none, search over all possible values
            of q < p.
        @param pqlist: A list of (p,q) tuples. If supplied, the (p,q) pairs are used instead of being generated from the
            values of pmax and qmax.
        @param njobs: The number of processors to use for calculating the MLE. A value of njobs = -1 will use all
            available processors.
        @param ntrials: The number of random starts to use in the MLE, the default is 100.
        @return: A tuple of (MLE, pqlist, AICc). MLE is a scipy.optimize.Result object containing the maximum-likelihood
            estimate. pqlist contains the values of (p,q) used in the search, and AICc contains the values of AICc for
            each (p,q) pair in pqlist.
        """
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

        MLEs = []
        for pq in pqlist:
            MLE = self.get_mle(pq[0], pq[1], ntrials=ntrials, njobs=njobs)
            MLEs.append(MLE)

        best_AICc = 1e300
        AICc = []
        best_MLE = MLEs[0]
        print 'p, q, AICc:'
        for MLE, pq in zip(MLEs, pqlist):
            nparams = 2 + pq[0] + pq[1]
            deviance = 2.0 * MLE.fun
            this_AICc = 2.0 * nparams + deviance + 2.0 * nparams * (nparams + 1.0) / (self.time.size - nparams - 1.0)
            print pq[0], pq[1], this_AICc
            AICc.append(this_AICc)
            if this_AICc < best_AICc:
                # new optimum found, save values
                best_MLE = MLE
                best_AICc = this_AICc
                self.p = pq[0]
                self.q = pq[1]

        print 'Model with best AICc has p =', self.p, ' and q = ', self.q

        return best_MLE, pqlist, AICc


def _get_mle_single(args):

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
    theta_bnds = [(ysigma / 10.0, 10.0 * ysigma)]
    theta_bnds.append((0.9, 1.1))
    theta_bnds.append((None, None))

    if p == 1:
        theta_bnds.append((np.log(min_freq), np.log(max_freq)))
    else:
        # monte carlo estimates of bounds on quadratic term parameterization of AR(p) roots
        qterm_lbound = min(min_freq ** 2, 2.0 * min_freq)
        qterm_lbound = np.log(qterm_lbound)
        qterm_ubound = max(max_freq ** 2, 2.0 * max_freq)
        qterm_ubound = np.log(qterm_ubound)
        theta_bnds.extend([(qterm_lbound, qterm_ubound)] * p)
        # no bounds on MA coefficients
        if q > 0:
            theta_bnds.extend([(None, None)] * q)

        CarmaProcess.SetMLE(True)  # ignore the prior bounds when calculating CarmaProcess.getLogDensity in C++ code

    # make sure initial guess of theta does not violate bounds
    for j in xrange(len(initial_theta)):
        if theta_bnds[j][0] is not None:
            if (initial_theta[j] < theta_bnds[j][0]) or (initial_theta[j] > theta_bnds[j][1]):
                initial_theta[j] = np.random.uniform(theta_bnds[j][0], theta_bnds[j][1])

    thisMLE = minimize(_carma_loglik, initial_theta, args=(CarmaProcess,), method="L-BFGS-B", bounds=theta_bnds)

    return thisMLE


def _carma_loglik(theta, args):
    CppCarma = args
    theta_vec = carmcmcLib.vecD()
    theta_vec.extend(theta)
    logdens = CppCarma.getLogDensity(theta_vec)
    return -logdens


class CarmaSample(samplers.MCMCSample):
    """
    Class for storing and analyzing the MCMC samples of a CARMA(p,q) model.
    """
    def __init__(self, time, y, ysig, sampler, q=0, filename=None, MLE=None):
        """
        Constructor for the CarmaSample class. In general a CarmaSample object should never be constructed directly,
        but should be constructed by calling CarmaModel.run_mcmc().

        @param time: The array of time values for the time series.
        @param y: The array of measured values for the time series.
        @param ysig: The array of measurement noise standard deviations for the time series.
        @param sampler: A C++ object return by _carmcmcm.run_carma_mcmc(). In general this should not be obtained
            directly, but a CarmaSample object should be obtained by running CarmaModel.run_mcmc().
        @param q: The order of the MA polynomial.
        @param filename: A string of the name of the file containing the MCMC samples generated by the C++ carpack.
        @param MLE: The maximum-likelihood estimate, obtained as a scipy.optimize.Result object.
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
        sampler.SetMLE(True)
        for i in xrange(logpost.size):
            std_theta = carmcmcLib.vecD()
            std_theta.extend(trace[i, :])
            # loglik[i] = logpost[i] - sampler.getLogPrior(std_theta)
            loglik[i] = sampler.getLogDensity(std_theta)

        self._samples['loglik'] = loglik

        # make the parameter names (i.e., the keys) public so the user knows how to get them
        self.parameters = self._samples.keys()
        self.newaxis()

        self.mle = {}
        if MLE is not None:
            # add maximum a posteriori estimate
            self.add_mle(MLE)

    def add_mle(self, MLE):
        """
        Add the maximum-likelihood estimate to the CarmaSample object. This will convert the MLE to a dictionary, and
        add it as a data member of the CarmaSample object. The values can be accessed as self.mle['parameter']. For
        example, the MLE of the CARMA process variance is accessed as self.mle['var'].

        @param MLE: The maximum-likelihood estimate, returned by CarmaModel.get_mle() or CarmaModel.choose_order().
        """
        self.mle = {'loglik': -MLE.fun, 'var': MLE.x[0] ** 2, 'measerr_scale': MLE.x[1], 'mu': MLE.x[2]}

        # add AR polynomial roots and PSD lorentzian parameters
        quad_coefs = np.exp(MLE.x[3:self.p + 3])
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

        self.mle['ar_roots'] = ar_roots
        self.mle['psd_width'] = psd_width
        self.mle['psd_cent'] = psd_cent
        self.mle['ar_coefs'] = np.poly(ar_roots).real

        # now calculate the moving average coefficients
        if self.q == 0:
            self.mle['ma_coefs'] = 1.0
        else:
            quad_coefs = np.exp(MLE.x[3 + self.p:])
            ma_roots = np.empty(quad_coefs.size, dtype=complex)
            for i in xrange(self.q / 2):
                quad1 = quad_coefs[2 * i]
                quad2 = quad_coefs[2 * i + 1]

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
            self.mle['ma_coefs'] = np.real(ma_coefs / ma_coefs[self.q])[::-1]

        # finally, calculate sigma, the standard deviation in the driving white noise
        unit_var = carma_variance(1.0, self.mle['ar_roots'], np.atleast_1d(self.mle['ma_coefs']))
        self.mle['sigma'] = np.sqrt(self.mle['var'] / unit_var.real)

    def set_logpost(self, logpost):
        """
        Add the input log-posterior MCMC values to the CarmaSample parameter dictionary.
        @param logpost: The values of the log-posterior obtained from the MCMC sampler.
        """
        self._samples['logpost'] = logpost  # log-posterior of the CAR(p) model

    def generate_from_trace(self, trace):
        """
        Generate the dictionary of MCMC samples for the CARMA process parameters from the input array.
        @param trace: An array containing the MCMC samples.
        """
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
        Plot the posterior median and the credibility interval corresponding to percentile of the CARMA(p,q) PSD. This
        function returns a tuple containing the lower and upper PSD credibility intervals as a function of frequency,
        the median PSD as a function of frequency, and the frequencies.
        
        :rtype : A tuple of numpy arrays, (lower PSD, upper PSD, median PSD, frequencies). If no subplot axes object
            is supplied (i.e., if sp = None), then the subplot axes object used will also be returned as the last
            element of the tuple.
        :param percentile: The percentile of the PSD credibility interval to plot.
        :param nsamples: The number of MCMC samples to use to estimate the credibility interval. The default is all
                         of them. Use less samples for increased speed.
        :param plot_log: A boolean. If true, then a logarithmic plot is made.
        :param color: The color of the shaded credibility region.
        :param alpha: The transparency level.
        :param sp: A matplotlib subplot axes object to use.
        :param doShow: If true, call plt.show()
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
        freq_max = 0.5 / dt_min
        freq_min = 1.0 / dt_max

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
        elif bestfit == 'mean':
            # use posterior mean as the best-fit
            sigsqr = np.mean(self._samples['sigma'] ** 2)
            mu = np.mean(self._samples['mu'])
            ar_roots = np.mean(self._samples['ar_roots'], axis=0)
            ma_coefs = np.mean(self._samples['ma_coefs'], axis=0)
        else:
            # use a random draw from the posterior
            random_index = np.random.random_integers(0, self._samples.values()[0].shape[0] - 1)
            sigsqr = (self._samples['sigma'][random_index] ** 2)[0]
            mu = self._samples['mu'][random_index][0]
            ar_roots = self._samples['ar_roots'][random_index]
            ma_coefs = self._samples['ma_coefs'][random_index]

        # expose C++ Kalman filter class to python
        kfilter = carmcmcLib.KalmanFilterp(arrayToVec(self.time),
                                           arrayToVec(self.y - mu),
                                           arrayToVec(self.ysig),
                                           sigsqr, 
                                           arrayToVec(ar_roots, carmcmcLib.vecC),
                                           arrayToVec(ma_coefs))
        return kfilter, mu

    def assess_fit(self, bestfit="map", nplot=256, doShow=True):
        """
        Display plots and provide useful information for assessing the quality of the CARMA(p,q) model fit.

        :param bestfit: A string specifying how to define 'best-fit'. Can be the maximum a posteriori value (MAP),
            the posterior mean ("mean"), or the posterior median ("median").
        :param nplot: The number of interpolated time series values to plot.
        :param doShow: If true, call pyplot.show(). Else if false, return the matplotlib figure object.
        """
        bestfit = bestfit.lower()
        try:
            bestfit in ['map', 'median', 'mean']
        except ValueError:
            "bestfit must be one of 'map, 'median', or 'mean'"

        fig = plt.figure()
        # compute the marginal mean and variance of the predicted values
        time_predict = np.linspace(1.001 * self.time.min(), self.time.max(), nplot)
        predicted_mean, predicted_var = self.predict(time_predict, bestfit=bestfit)
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
        plt.ylabel('Standardized Residuals')
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

    def predict(self, time, bestfit='map'):
        """
        Return the predicted value of the time series and its standard deviation at the input time(s) given the best-fit
        value of the CARMA(p,q) model and the measured time series.

        :param time: A scalar or numpy array containing the time values to predict the time series at.
        :param bestfit: A string specifying how to define 'best-fit'. Can be the Maximum Posterior (MAP), the posterior
            mean ("mean"), the posterior median ("median"), or a random sample from the MCMC sampler ("random").
        :rtype : A tuple of numpy arrays containing the expected value and variance of the time series at the input
            time values.
        """
        bestfit = bestfit.lower()
        try:
            bestfit in ['map', 'median', 'mean', 'random']
        except ValueError:
            "bestfit must be one of 'map, 'median', 'mean', or 'random'"

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

    def simulate(self, time, bestfit='map'):
        """
        Simulate a time series at the input time(s) given the best-fit value of the CARMA(p,q) model and the measured
        time series.

        :param time: A scalar or numpy array containing the time values to simulate the time series at.
        :param bestfit: A string specifying how to define 'best-fit'. Can be the Maximum Posterior (MAP), the posterior
            mean ("mean"), the posterior median ("median"), or a random sample from the MCMC sampler ("random").
        :rtype : The time series values simulated at the input values of time.
        """
        bestfit = bestfit.lower()
        try:
            bestfit in ['map', 'median', 'mean', 'random']
        except ValueError:
            "bestfit must be one of 'map, 'median', 'mean', 'random'"

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
    """
    Convert the input numpy array to a python wrapper of a C++ std::vector<double> object.
    """
    vec = arrType()
    vec.extend(array)
    return vec


class Car1Sample(CarmaSample):
    def __init__(self, time, y, ysig, sampler, filename=None):
        """
        Constructor for a CAR(1) sample. This is a special case of the CarmaSample class for p = 1. As with the
        CarmaSample class, this class should never be constructed directly. Instead, one should obtain a Car1Sample
        class by calling CarmaModel.run_mcmc().

        @param time: The array of time values for the time series.
        @param y: The array of measured time series values.
        @param ysig: The standard deviation in the measurement noise for the time series.
        @param sampler: A wrapper for an instantiated C++ Car1 object.
        @param filename: The name of an ascii file containing the MCMC samples.
        """
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
            self._samples['var'] = trace[:, 0] ** 2
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
                                           np.exp(log_omega))
        return kfilter, mu

    def plot_power_spectrum(self, percentile=68.0, nsamples=None, plot_log=True, color="b", alpha=0.5, sp=None,
                            doShow=True):
        """
        Plot the posterior median and the credibility interval corresponding to percentile of the CAR(1) PSD. This
        function returns a tuple containing the lower and upper PSD credibility intervals as a function of
        frequency, the median PSD as a function of frequency, and the frequencies.

        :rtype : A tuple of numpy arrays, (lower PSD, upper PSD, median PSD, frequencies). If no subplot axes object
            is supplied (i.e., if sp = None), then the subplot axes object used will also be returned as the last
            element of the tuple.
        :param percentile: The percentile of the PSD credibility interval to plot.
        :param nsamples: The number of MCMC samples to use to estimate the credibility interval. The default is all
                         of them. Use less samples for increased speed.
        :param plot_log: A boolean. If true, then a logarithmic plot is made.
        :param color: The color of the shaded credibility region.
        :param alpha: The transparency level.
        :param sp: A matplotlib subplot axes object to use.
        :param doShow: If true, call plt.show()
        """
        sigmas = self._samples['sigma']
        log_omegas = self._samples['log_omega']
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
            log_omegas = log_omegas[index]

        nfreq = 1000
        dt_min = self.time[1:] - self.time[0:self.time.size - 1]
        dt_min = dt_min.min()
        dt_max = self.time.max() - self.time.min()

        # Only plot frequencies corresponding to time scales a factor of 2 shorter and longer than the minimum and
        # maximum time scales probed by the time series.
        freq_max = 0.5 / dt_min
        freq_min = 1.0 / dt_max

        frequencies = np.linspace(np.log(freq_min), np.log(freq_max), num=nfreq)
        frequencies = np.exp(frequencies)
        psd_credint = np.empty((nfreq, 3))

        lower = (100.0 - percentile) / 2.0  # lower and upper intervals for credible region
        upper = 100.0 - lower

        numer = sigmas ** 2
        omegasq = np.exp(log_omegas) ** 2
        for i in xrange(nfreq):
            denom = omegasq + (2. * np.pi * frequencies[i]) ** 2
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


def get_ar_roots(qpo_width, qpo_centroid):
    """
    Return the roots of the characteristic AR(p) polynomial of the CARMA(p,q) process, given the lorentzian widths and
    centroids.

    :rtype : The roots of the autoregressive polynomial, a numpy array.
    :param qpo_width: The widths of the lorentzian functions defining the PSD.
    :param qpo_centroid: The centroids of the lorentzian functions defining the PSD. For all values of qpo_centroid
         that are greater than zero, the complex conjugate of the root will also be added.
    """
    ar_roots = []
    for i in xrange(len(qpo_centroid)):
        ar_roots.append(qpo_width[i] + 1j * qpo_centroid[i])
        if qpo_centroid[i] > 1e-10:
            # lorentzian is centered at a frequency > 0, so add complex conjugate of this root
            ar_roots.append(np.conjugate(ar_roots[-1]))
    if len(qpo_width) - len(qpo_centroid) == 1:
        # odd number of lorentzian functions, so add in low-frequency component
        ar_roots.append(qpo_width[-1] + 1j * 0.0)
    ar_roots = np.array(ar_roots)

    return -2.0 * np.pi * ar_roots


def power_spectrum(freq, sigma, ar_coef, ma_coefs=[1.0]):
    """
    Return the power spectrum for a CARMA(p,q) process calculated at the input frequencies.

    :param freq: The frequencies at which to calculate the PSD.
    :param sigma: The standard deviation driving white noise.
    :param ar_coef: The CARMA model autoregressive coefficients.
    :param ma_coefs: Coefficients of the moving average polynomial

    :rtype : The power spectrum at the input frequencies, a numpy array.
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
        nmore = len(ar_roots) - len(ma_coefs)
        ma_coefs = np.append(ma_coefs, np.zeros(nmore))

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


def car1_process(time, sigsqr, tau):
    """
    Generate a CAR(1) process.

    :param time: The time values at which to generate the CAR(1) process at.
    :param sigsqr: The variance in the driving white noise term.
    :param tau: The e-folding (mean-reversion) time scale of the CAR(1) process. Note that tau = -1.0 / ar_root.
    :rtype : A numpy array containing the simulated CAR(1) process values at time.
    """

    marginal_var = sigsqr * tau / 2.0
    y = np.zeros(len(time))
    y[0] = np.sqrt(marginal_var) * np.random.standard_normal()

    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        rho = np.exp(-dt / tau)
        conditional_var = marginal_var * (1.0 - rho ** 2)
        y[i] = rho * y[i-1] + np.sqrt(conditional_var) * np.random.standard_normal()

    return y

def carma_process(time, sigsqr, ar_roots, ma_coefs=[1.0]):
    """
    Generate a CARMA(p,q) process.

    :param time: The time values at which to generate the CARMA(p,q) process at.
    :param sigsqr: The variance in the driving white noise term.
    :param ar_roots: The roots of the autoregressive characteristic polynomial.
    :param ma_coefs: The moving average coefficients.
    :rtype : A numpy array containing the simulated CARMA(p,q) process values at time.
    """
    try:
        len(ma_coefs) <= len(ar_roots)
    except ValueError:
        "Size of ma_coefs must be less or equal to size of ar_roots."

    p = len(ar_roots)

    if p == 1:
        # generate a CAR(1) process
        return car1_process(time, sigsqr, -1.0 / np.asscalar(ar_roots))

    if len(ma_coefs) < p:
        # add extra zeros to end of ma_coefs
        q = len(ma_coefs)
        ma_coefs = np.resize(np.array(ma_coefs), len(ar_roots))
        ma_coefs[q:] = 0.0

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
        Return the predicted value of a time series and its standard deviation at the input time given the input
        values of the CARMA(p,q) model parameters and a measured time series.

        :rtype : A tuple containing the predicted value and its variance.
        :param time_predict: The time at which to predict the time series.
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
        Simulate a time series at the input time values of time_simulate, given the measured time series and input
        CARMA(p,q) parameters.

        :rtype : A scalar or numpy array, depending on type of time_simulate.
        :param time_simulate: The time(s) at which to simulate a random draw of the time series conditional on the
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


