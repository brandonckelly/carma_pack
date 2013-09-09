import numpy as np
import carmcmc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('RRLyrae.pdf')

# Set up defaults for running the code
period  = 0.563838733333 # days
q       = 0
NTHIN   = 1
NBURNIN = int(1e4)
NSAMPLE = int(NBURNIN * 10)

# Read in the data
data    = np.loadtxt("RRLyrae.txt", comments='#',
                     dtype=[("TDB", np.float64), ("filter", np.str_, 1), ("mag", np.float64), ("dmag", np.float64)])
filters    = set(data["filter"])

# Set up a loop over all filters, and several CARMA p-orders (we set q=0 here)
posteriors = {}
for f in filters:
    posteriors[f] = {}

    idx  = np.where(data["filter"] == f)
    tdb  = data["TDB"][idx]
    mag  = data["mag"][idx]
    dmag = data["dmag"][idx]
    
    for p in xrange(1, 10):
        carma = carmcmc.CarmaMCMC(tdb, mag, dmag, p, NSAMPLE, q=0, nburnin=NBURNIN, nthin=NTHIN)
        post  = carma.RunMCMC()
        posteriors[f][p] = post

# Print out the DIC for all models
for f in filters:
    print "Deviance Information Criterion:"
    for p in xrange(1, 10):
        print f, p, posteriors[f][p].DIC()[0]

# Print out several useful diagnostic figures
for f in filters: 
    for p in xrange(1, 10):
        post  = posteriors[f][p]
        fig   = post.assess_fit(nplot=1000, bestfit="median", doShow=False)
        fig.suptitle("RR Lyrae %s-band: CARMA p=%d q=%d" % (f, p, q))
        fig.get_axes()[0].invert_yaxis()
        pp.savefig(fig)

        fig = post.plot_power_spectrum(percentile=95.0, doShow=False)[1]
        fig.suptitle("RR Lyrae %s-band: CARMA p=%d q=%d" % (f, p, q))
        fig.gca().axvline(x=2.0 * np.pi /period) # Note this is angular frequency
        pp.savefig(fig)

pp.close()

# Don't exit out in case you want to interact with the data
import pdb; pdb.set_trace()
