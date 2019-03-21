import numpy as np

ndim = None
labels = None
obs = None
inv_cov = None
esd = None
r = None
extrapolate_outer = None


def logPrior(rho):
    if (np.diff(rho) > 0).any():
        return -np.inf
    if extrapolate_outer:
        if (np.log(rho[-1]) - np.log(rho[-2])) / \
           (np.log(r[-1]) - np.log(r[-2])) >= -1:
                return -np.inf
        return 0.0


def logLikelihood(rho):
    # omit terms:
    # -.5 * len(rho) * np.log(2 * np.pi)
    # -.5 * np.log(np.det(cov))
    # which drop out of relative likelihood
    try:
        dy = obs - esd(rho)
        retval = - .5 * np.inner(np.inner(dy, inv_cov), dy)
    except ValueError:
        return -np.inf
    if(np.isnan(retval).any()):
        return -np.inf
    else:
        return retval


def logProbability(rho):
    rho = np.exp(rho)
    lp = logPrior(rho)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + logLikelihood(rho)
