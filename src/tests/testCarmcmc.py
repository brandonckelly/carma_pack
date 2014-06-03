import os
import unittest
import numpy as np
import carmcmc 
np.random.seed(1)

class TestCarpackOrder(unittest.TestCase):

    def setUp(self):
        self.nSample = 100
        self.nBurnin = 10
        self.nThin = 1
        self.nWalkers = 2

        npts = 10
        self.x = 1.0 * np.arange(npts)
        ar_roots = np.array([-0.06283185-1.25663706j, -0.06283185+1.25663706j,
                             -0.02094395-0.25132741j, -0.02094395+0.25132741j,
                             -0.03141593+0.j])
        sigsqr = 0.00126811439419
        
        self.y = carmcmc.carma_process(self.x, sigsqr, ar_roots)
        self.dy = np.sqrt(sigsqr) * np.ones(npts)
        self.xdata = carmcmc.vecD()
        self.xdata.extend(self.x)
        self.ydata = carmcmc.vecD()
        self.ydata.extend(self.y)
        self.dydata = carmcmc.vecD()
        self.dydata.extend(self.dy)

    def testCar1(self):
        cppSample = carmcmc.run_mcmc_car1(self.nSample, self.nBurnin, self.xdata, self.ydata, self.dydata, self.nThin)
        psampler = carmcmc.Car1Sample(self.x, self.y, self.dy, cppSample)
        self.assertEqual(psampler.p, 1)

        psamples = np.array(cppSample.getSamples())
        ploglikes = np.array(cppSample.GetLogLikes())
        sample0 = carmcmc.vecD()
        sample0.extend(psamples[0])
        logprior0 = cppSample.getLogPrior(sample0)
        loglike0 = cppSample.getLogDensity(sample0)
        self.assertAlmostEqual(ploglikes[0], loglike0)

    def testCar1Defaults(self):
        cppSample = carmcmc.run_mcmc_car1(self.nSample, self.nBurnin, self.xdata, self.ydata, self.dydata)
        cppSample = carmcmc.run_mcmc_car1(self.nSample, self.nBurnin, self.xdata, self.ydata, self.dydata, self.nThin)
        guess     = cppSample.getSamples()[0]
        cppSample = carmcmc.run_mcmc_car1(self.nSample, self.nBurnin, self.xdata, self.ydata, self.dydata, self.nThin, guess)

    def testCarp(self, pModel=3):
        qModel  = 0
        sampler = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                         self.xdata, self.ydata, self.dydata,
                                         pModel, qModel, self.nWalkers, False, self.nThin)
        psampler = carmcmc.CarmaSample(np.array(self.xdata), np.array(self.ydata), np.array(self.dydata), sampler)
        self.assertEqual(psampler.p, pModel)

        psamples = np.array(sampler.getSamples())
        ploglikes = np.array(sampler.GetLogLikes())
        sample0 = carmcmc.vecD()
        sample0.extend(psamples[0])
        logprior0 = sampler.getLogPrior(sample0)
        loglike0 = sampler.getLogDensity(sample0)
        # OK, this is where I truly test that sampler is of class CARp and not CAR1
        self.assertAlmostEqual(ploglikes[0], loglike0)

    def testCarpDefaults(self):
        pModel  = 3
        qModel  = 1
        cppSample = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                           self.xdata, self.ydata, self.dydata,
                                           pModel, qModel, self.nWalkers)
        cppSample = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                           self.xdata, self.ydata, self.dydata,
                                           pModel, qModel, self.nWalkers, False)
        cppSample = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                           self.xdata, self.ydata, self.dydata,
                                           pModel, qModel, self.nWalkers, False, self.nThin)
        guess     = cppSample.getSamples()[0]
        print "Should be using:", np.array(guess)
        cppSample = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                           self.xdata, self.ydata, self.dydata,
                                           pModel, qModel, self.nWalkers, False, self.nThin, guess)
        

    def testCarpq(self, pModel=3, qModel=2):
        sampler = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                         self.xdata, self.ydata, self.dydata, 
                                         pModel, qModel, self.nWalkers, False, self.nThin)
        psampler = carmcmc.CarmaSample(np.array(self.xdata), np.array(self.ydata), np.array(self.dydata), sampler)
        self.assertEqual(psampler.p, pModel+qModel)

        psamples  = np.array(sampler.getSamples())
        ploglikes = np.array(sampler.GetLogLikes())
        sample0   = carmcmc.vecD()
        sample0.extend(psamples[0])
        logprior0 = sampler.getLogPrior(sample0)
        loglike0  = sampler.getLogDensity(sample0)
        # OK, this is where I truly test that sampler is of class CARp and not CAR1
        self.assertAlmostEqual(ploglikes[0], loglike0)

    def testKalman1(self):
        sigma = 1.0
        omega = 1.0
        kfilter = carmcmc.KalmanFilter1(self.xdata, self.ydata, self.dydata, sigma, omega)
        kfilter.Filter()
        pred0 = kfilter.Predict(self.xdata[0])  # evaluate at data point
        val0  = pred0.first
        var0  = pred0.second
        predN = kfilter.Predict(self.xdata[-1]+1)  # extrapolate
        valN  = predN.first
        varN  = predN.second
        self.assertTrue(varN > var0)

    def testKalmanp(self):
        pModel  = 4
        qModel  = 0
        sampler = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                          self.xdata, self.ydata, self.dydata, 
                                          pModel, qModel, self.nWalkers, False, self.nThin)
        psampler = carmcmc.CarmaSample(np.array(self.xdata), np.array(self.ydata), np.array(self.dydata), sampler)
        sigsqr   = (psampler._samples["sigma"][0]**2)[0]
        ma_coefs = carmcmc.vecD()
        ma_coefs0 = psampler._samples["ma_coefs"][0]
        if len(ma_coefs0) != pModel:
            ma_coefs0 = np.append(ma_coefs0, np.zeros(pModel - qModel - 1))
        ma_coefs.extend(ma_coefs0)

        omega    = carmcmc.vecC()
        for i in range(psampler.p):
            omega.append(psampler._samples["ar_roots"][0][i])
        #import pdb; pdb.set_trace()

        kfilter = carmcmc.KalmanFilterp(self.xdata, self.ydata, self.dydata, sigsqr, omega, ma_coefs)
        kfilter.Filter()
        pred0 = kfilter.Predict(self.xdata[0]) # evaluate at data point
        val0  = pred0.first
        var0  = pred0.second
        predN = kfilter.Predict(self.xdata[-1]+1)  # extrapolate
        valN  = predN.first
        varN  = predN.second
        self.assertTrue(varN > var0)

    def testFull(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__) ), "../../cpp_tests/data/car5_test.dat")
        xv, yv, dyv = np.loadtxt(path, unpack=True)
        
        nSample = 100
        nBurnin = 10
        nThin   = 1
        pModel  = 3
        qModel  = 1

        carma1 = carmcmc.CarmaModel(xv, yv, dyv, p=1, q=0)
        post1 = carma1.run_mcmc(nSample, nburnin=nBurnin, nthin=nThin)

        carmap = carmcmc.CarmaModel(xv, yv, dyv, p=pModel, q=0)
        postp = carmap.run_mcmc(nSample, nburnin=nBurnin, nthin=nThin)

        carmapqo = carmcmc.CarmaModel(xv, yv, dyv, pModel, qModel)
        postpqo = carmapqo.run_mcmc(nSample, nburnin=nBurnin, nthin=nThin)

        carmapqe = carmcmc.CarmaModel(xv, yv, dyv, pModel+1, qModel)
        postpqe = carmapqe.run_mcmc(nSample, nburnin=nBurnin, nthin=nThin)

        # cpp_tests of yamcmcpp samplers.py
        
        post1.effective_samples("sigma")
        postp.effective_samples("ar_roots")
        postpqo.effective_samples("ma_coefs")
        postpqe.effective_samples("ar_coefs")
        
        post1.plot_trace("sigma")
        postp.plot_trace("sigma")
        postpqo.plot_trace("sigma")
        postpqe.plot_trace("sigma")
        
        post1.plot_1dpdf("mu")
        postp.plot_1dpdf("mu")
        postpqo.plot_1dpdf("mu")
        postpqe.plot_1dpdf("mu")

        postp.plot_1dpdf("psd_centroid")
        postpqo.plot_1dpdf("psd_width")
        postpqe.plot_1dpdf("psd_width")
        
        post1.plot_2dpdf("sigma", "var")
        postp.plot_2dpdf("sigma", "var", pindex1=0, pindex2=0)
        postpqo.plot_2dpdf("sigma", "var", pindex1=1, pindex2=1)
        postpqe.plot_2dpdf("sigma", "var", pindex1=2, pindex2=2)
        
        post1.plot_2dkde("sigma", "var")
        postp.plot_2dkde("sigma", "var", pindex1=0, pindex2=0)
        postpqo.plot_2dkde("sigma", "var", pindex1=1, pindex2=1)
        postpqe.plot_2dkde("sigma", "var", pindex1=2, pindex2=2)
        
        postp.plot_autocorr("psd_centroid")
        postpqo.plot_autocorr("psd_centroid")
        postpqe.plot_autocorr("psd_centroid")
        
        postp.plot_parameter("psd_centroid")
        postpqo.plot_parameter("psd_centroid")
        postpqe.plot_parameter("psd_centroid")
        
        postp.posterior_summaries("psd_width")
        postpqo.posterior_summaries("psd_width")
        postpqe.posterior_summaries("psd_width")
        
        post1.posterior_summaries("sigma")
        postp.posterior_summaries("sigma")
        postpqo.posterior_summaries("sigma")
        postpqe.posterior_summaries("sigma")
        
        # cpp_tests of carma_pack carma_pack.py
        
        post1.plot_power_spectrum(percentile=95.0, doShow=False)
        postp.plot_power_spectrum(percentile=95.0, doShow=False)
        postpqo.plot_power_spectrum(percentile=95.0, doShow=False)
        postpqe.plot_power_spectrum(percentile=95.0, doShow=False)
        
        for bestfit in ["map", "median", "mean"]:
            post1.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
            postp.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
            postpqo.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
            postpqe.assess_fit(nplot=1000, bestfit=bestfit, doShow=False)
        
if __name__ == "__main__":
    unittest.main()
