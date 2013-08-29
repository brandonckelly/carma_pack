import unittest
import numpy as np
import carmcmc 
np.random.seed(1)

class TestCarpackOrder(unittest.TestCase):
    def setUp(self):
        self.nSample  = 100
        self.nBurnin  = 10
        self.nThin    = 1
        self.nWalkers = 2

        self.xdata    = carmcmc.vecD()
        self.xdata.extend(1.0 * np.arange(10))
        self.ydata    = carmcmc.vecD()
        self.ydata.extend(1.0 * np.random.random(10))
        self.dydata    = carmcmc.vecD()
        self.dydata.extend(0.1 * np.random.random(10))

    def testCar1(self):    
        sampler  = carmcmc.run_mcmc_car1(self.nSample, self.nBurnin, 
                                         self.xdata, self.ydata, self.dydata, 
                                         self.nWalkers, self.nThin)
        psampler = carmcmc.CarSample1(np.array(self.xdata), np.array(self.ydata), np.array(self.dydata), sampler)
        self.assertEqual(psampler.p, 1)

        psamples  = np.array(psampler.sampler.getSamples())
        ploglikes = np.array(psampler.sampler.GetLogLikes())
        sample0   = carmcmc.vecD()
        sample0.extend(psamples[0])
        logprior0 = psampler.sampler.getLogPrior(sample0)
        loglike0  = psampler.sampler.getLogDensity(sample0)
        self.assertAlmostEqual(ploglikes[0], loglike0)


    def testCarp(self, pModel=3):
        qModel  = 0
        sampler = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                          self.xdata, self.ydata, self.dydata, 
                                          pModel, qModel, self.nWalkers, False, self.nThin)
        psampler = carmcmc.CarmaSample(np.array(self.xdata), np.array(self.ydata), np.array(self.dydata), sampler)
        self.assertEqual(psampler.p, pModel)

        psamples  = np.array(psampler.sampler.getSamples())
        ploglikes = np.array(psampler.sampler.GetLogLikes())
        sample0   = carmcmc.vecD()
        sample0.extend(psamples[0])
        logprior0 = psampler.sampler.getLogPrior(sample0)
        loglike0  = psampler.sampler.getLogDensity(sample0)
        # OK, this is where I truly test that sampler is of class CARp and not CAR1
        self.assertAlmostEqual(ploglikes[0], loglike0)

    def testCarpq(self, pModel=3, qModel=2):
        sampler = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                          self.xdata, self.ydata, self.dydata, 
                                          pModel, qModel, self.nWalkers, False, self.nThin)
        psampler = carmcmc.CarmaSample(np.array(self.xdata), np.array(self.ydata), np.array(self.dydata), sampler)
        self.assertEqual(psampler.p, pModel+qModel)

        psamples  = np.array(psampler.sampler.getSamples())
        ploglikes = np.array(psampler.sampler.GetLogLikes())
        sample0   = carmcmc.vecD()
        sample0.extend(psamples[0])
        logprior0 = psampler.sampler.getLogPrior(sample0)
        loglike0  = psampler.sampler.getLogDensity(sample0)
        # OK, this is where I truly test that sampler is of class CARp and not CAR1
        self.assertAlmostEqual(ploglikes[0], loglike0)

    def testZCarp(self, pModel=3):
        qModel  = 0
        sampler = carmcmc.run_mcmc_carma(self.nSample, self.nBurnin, 
                                          self.xdata, self.ydata, self.dydata, 
                                          pModel, qModel, self.nWalkers, True, self.nThin)
        psampler = carmcmc.ZCarmaSample(np.array(self.xdata), np.array(self.ydata), np.array(self.dydata), sampler)
        self.assertEqual(psampler.p, pModel)

        psamples  = np.array(psampler.sampler.getSamples())
        ploglikes = np.array(psampler.sampler.GetLogLikes())
        sample0   = carmcmc.vecD()
        sample0.extend(psamples[0])
        logprior0 = psampler.sampler.getLogPrior(sample0)
        loglike0  = psampler.sampler.getLogDensity(sample0)
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
        sigsqr   = psampler._samples["sigma"][0]**2
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

if __name__ == "__main__":
    unittest.main()
