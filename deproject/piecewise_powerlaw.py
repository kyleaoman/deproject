import numpy as np
from scipy.special import hyp2f1, gamma
from . import statmodel
from rap import Rap


def _a(rho, r, iet=1):
    retval = (np.log(rho[:, 1:]) - np.log(rho[:, :-1])) / \
        (np.log(r[:, 1:]) - np.log(r[:, :-1]))
    return np.hstack((iet * retval[:, :1], retval, retval[:, -1:]))


def _b(rho, r, a):
    retval = np.log(rho[:, :-1]) - a[:, 1:-1] * np.log(r[:, :-1])
    return np.hstack((retval[:, :1], retval, retval[:, -1:]))


def _J(r, R, a):
    retval = -(1 / 3) * np.power(R, a + 3) * \
        np.power(np.power(r / R, 2) - 1, 1.5) * \
        hyp2f1(1.5, -a / 2, 2.5, 1 - np.power(r / R, 2))
    retval = np.where(
        r == np.inf,
        .5 * np.sqrt(np.pi) * gamma(-(1 + a) / 2) / gamma(-a / 2) *
        np.power(R, a + 3) / (a + 3),
        retval
    )
    retval[r == 0] = 0
    return retval


def rho_to_mencl(rho, r, extrapolate_outer=True, extrapolate_inner=True,
                 inner_extrapolation_type='extrapolate'):
    if inner_extrapolation_type not in ('extrapolate', 'flat'):
        raise ValueError("inner_extrapolation_type must be 'extrapolate' "
                         "or 'flat'.")
    else:
        iet = {'extrapolate': 1, 'flat': 0}[inner_extrapolation_type]
    a = _a(rho, r, iet=iet)
    b = _b(rho, r, a)
    r = np.hstack((
        np.zeros(r.shape[:-1] + (1,)),
        r,
        np.ones(r.shape[:-1] + (1,)) * np.inf
    ))
    m = 4 * np.pi * np.exp(b) / (a + 3) \
        * (np.power(r[..., 1:], a + 3) - np.power(r[..., :-1], a + 3))
    return np.cumsum(m[..., :-1], axis=-1), r[..., 1:-1]


def _casemap(r, R):
    r = np.hstack((
        np.zeros((r.shape[0], 1)),
        r,
        np.ones((r.shape[0], 1)) * np.inf
    ))
    R = np.hstack((
        R[:, :1],
        R,
        R[:, -1:]
    ))
    case0 = r[:, 1:] < R[:-1]
    case1 = np.logical_and(
        r[:, :-1] <= R[:-1],
        np.logical_and(
            r[:, 1:] > R[:-1],
            r[:, 1:] <= R[1:]
        )
    )
    case2 = np.logical_and(
        r[:, :-1] <= R[:-1],
        r[:, 1:] > R[1:]
    )
    case3 = np.logical_and(
        np.logical_and(
            r[:, :-1] > R[:-1],
            r[:, :-1] <= R[1:]
        ),
        r[:, 1:] > R[1:]
    )
    case4 = r[:, :-1] > R[1:]
    case5 = np.logical_and(
        np.logical_and(
            r[:, :-1] > R[:-1],
            r[:, :-1] <= R[1:]
        ),
        np.logical_and(
            r[:, 1:] > R[:-1],
            r[:, 1:] <= R[1:]
        )
    )
    return (case0, case1, case2, case3, case4, case5)


def _caseeval(r, R, a, b):
    r = np.hstack((
        np.zeros((r.shape[0], 1)),
        r,
        np.ones((r.shape[0], 1)) * np.inf
    ))
    R = np.hstack((
        R[:, :1],
        R,
        R[:, -1:]
    ))
    case0 = np.zeros(r[:, :-1].shape)
    case1 = -_J(r[:, 1:], R[:-1], a)
    case2 = _J(r[:, 1:], R[1:], a) - _J(r[:, 1:], R[:-1], a)
    case3 = _J(r[:, 1:], r[:, :-1], a) - _J(r[:, 1:], R[:-1], a) \
        + _J(r[:, :-1], R[:-1], a) \
        + _J(r[:, 1:], R[1:], a) - _J(r[:, 1:], r[:, :-1], a)
    case4 = _J(r[:, 1:], R[1:], a) - _J(r[:, 1:], R[:-1], a) \
        - _J(r[:, :-1], R[1:], a) + _J(r[:, :-1], R[:-1], a)
    case5 = _J(r[:, 1:], r[:, :-1], a) - _J(r[:, 1:], R[:-1], a) \
        + _J(r[:, :-1], R[:-1], a) - _J(r[:, 1:], r[:, :-1], a)
    pre = 4 * np.exp(b) / (np.power(R[1:], 2) - np.power(R[:-1], 2))
    return tuple([pre * c for c in (case0, case1, case2, case3, case4, case5)])


class _ESD(object):

    def __init__(self, r, R, extrapolate_outer=True, extrapolate_inner=True,
                 inner_extrapolation_type='extrapolate', mode='ESD'):
        self.r = r
        self.R = R
        _rarr, _Rarr = np.meshgrid(r, R)
        self.rarr = _rarr[:-1]
        self.Rarr = _Rarr[:, :-1]
        self.cases = _casemap(self.rarr, self.Rarr)
        self.extrapolate_outer = extrapolate_outer
        self.extrapolate_inner = extrapolate_inner
        self.mode = mode
        if inner_extrapolation_type not in ('extrapolate', 'flat'):
            raise ValueError("inner_extrapolation_type must be 'extrapolate' "
                             "or 'flat'.")
        else:
            self.inner_extrapolation_type = \
                {'extrapolate': 1, 'flat': 0}[inner_extrapolation_type]
        return

    def __call__(self, rho):
        _rhoarr, _junk = np.meshgrid(rho, self.R)
        rhoarr = _rhoarr[:-1]
        aarr = _a(rhoarr, self.rarr, iet=self.inner_extrapolation_type)
        barr = _b(rhoarr, self.rarr, aarr)
        if (aarr <= -5.).any():
            raise ValueError('Density slope < -5 causes problems in surface '
                             'density profile.')
        if (aarr > 0).any():
            raise ValueError('Density slope > 0 not supported.')
        if (aarr[:, 0] <= -3).any() and self.extrapolate_inner:
            raise ValueError('Inner extrapolation with density slope <= -3 '
                             'has infinite mass.')
        if (aarr[:, -1] >= -1).any() and self.extrapolate_outer:
            raise ValueError('Outer extrapolation with density slope >= -1 '
                             'does not converge.')
        if (aarr[:, -1] == 3).any() and self.extrapolate_outer:
            raise ValueError('Outer extrapolation with density slope == 3 is '
                             'unstable. (2.9999 or 3.0001, etc., is ok).')
        casevals = _caseeval(self.rarr, self.Rarr, aarr, barr)
        retval = np.zeros((len(self.R) - 1, len(self.r) + 1))
        for c, v in zip(self.cases, casevals):
            retval += np.where(c, v, np.zeros(v.shape))
        if not self.extrapolate_inner:
            retval = retval[:, 1:]
        if not self.extrapolate_outer:
            retval = retval[:, :-1]
        sigma = np.sum(retval, axis=1)
        if self.mode == 'Sigma':
            return sigma

        # mean enclosed surface density integrals:

        def integral(x0, x1, a, b):
            return 2 * np.pi * np.exp(b) / (a + 2) * \
                (np.power(x1, a + 2) - np.power(x0, a + 2))

        aS = 2 * (np.log(sigma[1:]) - np.log(sigma[:-1])) \
            / (np.log(self.R[2:]) - np.log(self.R[:-2]))
        bS = np.log(sigma[:-1]) - aS * .5 * np.log(self.R[:-2] * self.R[1:-1])
        if aS[0] <= -2:
            raise ValueError('Surface density has central slope <= -2, '
                             'central mass content is infinite.')
        mass_central = integral(
            0,
            np.sqrt(self.R[0] * self.R[1]),
            aS[0],
            bS[0]
        )
        mass_annuli = integral(
            np.sqrt(self.R[:-2] * self.R[1:-1]),
            np.sqrt(self.R[1:-1] * self.R[2:]),
            aS,
            bS
        )
        mass_enclosed = np.cumsum(np.r_[mass_central, mass_annuli])
        mean_enclosed = mass_enclosed / (np.pi * self.R[:-1] * self.R[1:])
        if self.mode == 'ESD':
            return mean_enclosed - sigma
        raise ValueError('Unknown mode.')


def esd_to_rho(obs, obs_err, guess, r, R, extrapolate_inner=True,
               extrapolate_outer=True,
               inner_extrapolation_type='extrapolate',
               testwith_rho=None, truths=None, niter=5500, burn=500,
               savecorner='corner.pdf', parallel=8):
    guess = [np.log(g) for g in guess]
    if truths is not None:
        truths = [np.log(t) for t in truths]
    statmodel.ndim = len(r)
    statmodel.labels = [r'$\rho_{'+'{:.0f}'.format(i)+r'}$'
                        for i in range(len(r))]
    statmodel.esd = _ESD(
        r,
        R,
        extrapolate_inner=extrapolate_inner,
        extrapolate_outer=extrapolate_outer,
        inner_extrapolation_type=inner_extrapolation_type
    )
    if obs is None:
        if testwith_rho is not None:
            statmodel.obs = statmodel.esd(testwith_rho)
        else:
            raise ValueError
    else:
        statmodel.obs = obs
    if obs_err.ndim == 1:
        obs_err = np.diag(obs_err)
    try:
        statmodel.inv_cov = np.linalg.inv(obs_err)
    except np.linalg.LinAlgError:
        raise ValueError('Could not invert covariance matrix.')
    statmodel.r = r
    statmodel.extrapolate_outer = extrapolate_outer
    RAP = Rap(statmodel)
    olderr = np.seterr(all='ignore')
    RAP.fit(guess, niter=niter, burn=burn, parallel=parallel)
    np.seterr(**olderr)
    RAP.cornerfig(save=savecorner, fignum=999,
                  labels=statmodel.labels, truths=truths)
    return [np.exp(rl) for rl in RAP.results['perc_16_50_84']], \
        np.exp(RAP.results['theta_ml'])
