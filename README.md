carma_pack
========

carma_pack is an MCMC sampler for performing Bayesian inference on
continuous time autoregressive moving average models. These models may
be used to model time series with irregular sampling. The MCMC sampler
utilizes an adaptive Metropolis algorithm combined with parallel
tempering. Further details are given in [this paper](http://arxiv.org/abs/1402.5978).

For a guided tour of `carma_pack` click [here](http://nbviewer.ipython.org/github/brandonckelly/carma_pack/blob/master/examples/.ipynb_checkpoints/carma_pack_guide-checkpoint.ipynb)
 or see the `ipython` notebook available under the `examples/` folder.

----------
Quick Start
----------

To illustrate the important functionality of carma_pack, start by
defining the power spectrum parameters for a CARMA(5,3) process:

    >>> import carmcmc as cm
    >>> import numpy as np
    >>>
    >>> sigmay = 2.3  # dispersion in the time series
    >>> p = 5  # order of the AR polynomial
    >>> mu = 17.0  # mean of the time series
    >>> qpo_width = np.array([1.0/100.0, 1.0/300.0, 1.0/200.0])  # widths of of Lorentzian components
    >>> qpo_cent = np.array([1.0/5.0, 1.0/25.0])  # centroids of Lorentzian components
    >>> ar_roots = cm.get_ar_roots(qpo_width, qpo_cent)
    >>> ma_coefs = np.zeros(p)
    >>> ma_coefs[0] = 1.0
    >>> ma_coefs[1] = 4.5
    >>> ma_coefs[2] = 1.25
    >>> # calculate amplitude of driving white noise
    >>> sigsqr = sigmay ** 2 / cm.carma_variance(1.0, ar_roots, ma_coefs=ma_coefs)

Generate the time series:

    >>> ny = 270
    >>> time = np.empty(ny)
    >>> dt = np.random.uniform(1.0, 3.0, ny)
    >>> time[:90] = np.cumsum(dt[:90])
    >>> time[90:2*90] = 180 + time[90-1] + np.cumsum(dt[90:2*90])
    >>> time[2*90:] = 180 + time[2*90-1] + np.cumsum(dt[2*90:])
    >>> y0 = mu + cm.carma_process(time, sigsqr, ar_roots, ma_coefs=ma_coefs)
    >>> ysig = np.ones(ny) * y0.std() / 5.0  # standard deviation in measurement noise
    >>> y = y0 + ysig * np.random.standard_normal(ny)  # add measurement noise

Now lets choose the order of the CARMA model to use. carma_pack does
this by finding the maximum-likelihood estimate of the carma models on
a grid of (p,q), and then choosing (p,q) to minimize the AICc. Because
the likelihood space is multimodal, carma_pack launches 100 optimizers
with random starts, and picks the best one. Optimization is done using
scipy.optimize, but appears to be unstable as it is not uncommon to
get NaN at the MLE.

    >>> carma_model = cm.CarmaModel(time, y, ysig)  # create new CARMA process model
    >>> pmax = 7  # only search over p < 7, q < p
    >>> # njobs = -1 will use all the processes through the multiprocessing module
    >>> MLE, pqlist, AICc_list = carma_model.choose_order(pmax, njobs=-1)

The chosen p and q values are saved as data members of the carma_model
object. Finally, lets sample the CARMA process parameters using MCMC
and the values of (p,q) chosen by the choose_order method.

    >>> carma_sample = carma_model.run_mcmc(50000)

This will run the MCMC sampler for 75000 iterations, with the first
25000 of those discarded as burn-in. You can specify the number of
parallel chains to use in the parallel tempering algorithm, as well as
the number of iterations used by burn-in. In this example we just use
the default.

Now let's examine the results. We can see which parameters we have
sampled as

    >>> print carma_sample.parameters

We can grab the samples for a parameter doing

    >>> trace_qpo_cent = carma_sample.get_samples('psd_centroid')  # get samples for the Lorentzians centroids

We can plot a useful summary of the MCMC samples for each
parameter:

    >>> carma_sample.plot_parameter('var')  # plots for the CARMA model variance

We can also plot 1-d and 2-d posteriors:

    >>> carma_sample.plot_1dpdf('ma_coefs')  # histograms of MA coefficients
    >>> # plot distribution of centroid of 1st Lorentzian vs width of 2nd Lorentzian
    >>> carma_sample.plot_2dkde('psd_centroid', 'psd_width', 0, 1)

The pointwise 95% credibility region for the power spectrum under the
CARMA model is

    >>> psd_lo, psd_hi, psd_mid, freq = carma_sample.plot_power_spectrum(percentile=95.0, nsamples=5000)  # only use 5000 MCMC samples for speed

Finally, we can assess the quality of the fit through

    >>> carma_sample.assess_fit()

------------
Installation
------------

`carma_pack` depends on the following python modules:

* `numpy`     (for core functionality)
* `scipy`     (for core functionality)
* `matplotlib`    (for generating plots)
* `acor`    (for calculating the autocorrelation time scale of MCMC samples)

In addition, it is necessary to have the [Boost C++ libraries](http://www.boost.org) (for
linking python and C++) and the [Armadillo C++ linear algebra
library](http://arma.sourceforge.net) installed. If you have multiple python
versions installed on your system, make sure that the BoostPython
links to the correct python library. Otherwise, python will crash
when you import carmcmc. Note that carma_pack only works with Boost version 1.59 and earlier; it does not work with Boost version 1.60 and onwards.

Full install directions are given in the Linux_install.txt and
MacOSX_install.txt files. Be forewarned that on Mac OS X getting
BoostPython to build against any python other than the system
python can be a real headache. You may have to replace some of the
python libraries in /usr/lib with symbolic links to the actual python
libraries used.

--------
Examples
--------

We have supplied an `ipython` notebook under the `examples/` folder that gives a guided tour of `carma_pack`. Also, the script carma_pack/src/paper/carma_paper.py generates the plots
from the paper and provides additional examples of carma_pack's functionality.
