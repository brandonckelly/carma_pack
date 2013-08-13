import unittest
import numpy as np
import carmcmc 

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
                                         1, 
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
        self.assertEqual(psampler.p, pModel)

        psamples  = np.array(psampler.sampler.getSamples())
        ploglikes = np.array(psampler.sampler.GetLogLikes())
        sample0   = carmcmc.vecD()
        sample0.extend(psamples[0])
        logprior0 = psampler.sampler.getLogPrior(sample0)
        loglike0  = psampler.sampler.getLogDensity(sample0)
        # OK, this is where I truly test that sampler is of class CARp and not CAR1
        self.assertAlmostEqual(ploglikes[0], loglike0)

    def testCarma(self):
        pass
        

if __name__ == "__main__":
    unittest.main()
