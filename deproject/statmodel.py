import numpy as np

ndim = None
labels = None
obs = None
inv_cov = None
esd = None
r = None
extrapolate_outer = None
blobs_dtype = [('model_esd', np.object), ('model_sigma', np.object)]


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
        model_esd, model_sigma = esd(rho)
    except ValueError:
        return -np.inf, np.ones(rho.shape) * np.nan, \
            np.ones(rho.shape) * np.nan
    dy = obs - model_esd
    retval = - .5 * np.inner(np.inner(dy, inv_cov), dy)
    if(np.isnan(retval).any()):
        return -np.inf, model_esd, model_sigma
    else:
        return retval, model_esd, model_sigma


def logProbability(rho):
    rho = np.exp(rho)
    lp = logPrior(rho)
    if not np.isfinite(lp):
        return -np.inf, np.ones(rho.shape) * np.nan, \
            np.ones(rho.shape) * np.nan
    else:
        logL, model_esd, model_sigma = logLikelihood(rho)
        return lp + logL, model_esd, model_sigma
