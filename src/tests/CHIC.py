import cPickle
import carmcmc
import numpy as np
import matplotlib.pyplot as plt

buff = open("CHIC.pickle", "rb")
x, y, dy = cPickle.load(buff)
buff.close()

p = 5
q = 1
nsample = 5000
nburnin = 10000

car = carmcmc.CarmaMCMC(x, y, dy, p, nsample, q=q, nburnin=nburnin)
post = car.RunMCMC()

secInDay = 3600 * 24.
nplot = 25000
time_predict = np.linspace(x.min(), x.max() + 100 * secInDay, nplot)
print "getting predicted values..."
predicted_mean, predicted_var = post.predict_lightcurve(time_predict, bestfit="med")
# import pdb; pdb.set_trace()

idx = np.where(predicted_var < 0)
print "NEGATIVE VARIANCE", idx

idx = np.where(predicted_var > 0)
time_predict = time_predict[idx]
predicted_mean = predicted_mean[idx]
predicted_var = predicted_var[idx]

fig = plt.figure()
sp = fig.add_subplot(111)
sp.errorbar(x, y, yerr=dy, fmt='ko', label='Data', ms=4, capsize=1)
sp.plot(time_predict, predicted_mean, '-r', label='Kalman Filter')
predicted_low = predicted_mean - np.sqrt(predicted_var)
predicted_high = predicted_mean + np.sqrt(predicted_var)
sp.fill_between(time_predict, predicted_low, predicted_high,
                edgecolor=None, facecolor='blue', alpha=0.25, label="1-sigma range")
sp.set_xlabel('Time')
sp.legend(loc=1)
plt.show()
