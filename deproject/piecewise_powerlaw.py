import numpy as np
from scipy.special import hyp2f1, gamma
from . import statmodel
from rap import Rap
from kyleaoman_utilities.slvars import savevars, loadvars


def to_gobs(m, r):
    import astropy.units as U
    import astropy.constants as C
    return (C.G * m * U.Msun / np.power(r * U.Mpc, 2)).to(U.m / U.s ** 2).value


def _a(rho, r, iet=1):
    retval = (np.log(rho[..., 1:]) - np.log(rho[..., :-1])) / \
        (np.log(r[..., 1:]) - np.log(r[..., :-1]))
    return np.hstack((iet * retval[..., :1], retval, retval[..., -1:]))


def _b(rho, r, a):
    retval = np.log(rho[..., :-1]) - a[..., 1:-1] * np.log(r[..., :-1])
    return np.hstack((retval[..., :1], retval, retval[..., -1:]))


def _J(r, R, a):
    # same speed for small arrays, faster for large arrays
    mask_first = (r[:, 0] == 0).all()
    mask_last = (r[:, -1] == np.inf).all()
    if mask_first and not mask_last:
        mask = np.s_[:, 1:]
    elif not mask_first and mask_last:
        mask = np.s_[:, :-1]
    elif mask_first and mask_last:
        mask = np.s_[:, 1:-1]
    else:
        mask = np.s_[:, :]
    retval = -(1 / 3) * np.power(R[mask], a[mask] + 3) * \
        np.power(np.power(r[mask] / R[mask], 2) - 1, 1.5) * \
        hyp2f1(1.5, -a[mask] / 2, 2.5, 1 - np.power(r[mask] / R[mask], 2))
    if mask_last:
        last = .5 * np.sqrt(np.pi) * gamma(-(1 + a[:, -1]) / 2) / \
            gamma(-a[:, -1] / 2) * np.power(R[:, -1], a[:, -1] + 3) / \
            (a[:, -1] + 3)
    if mask_first and not mask_last:
        return np.hstack((np.zeros((r.shape[0], 1)), retval))
    elif not mask_first and mask_last:
        return np.hstack((retval, last.reshape(last.shape + (1,))))
    elif mask_first and mask_last:
        return np.hstack((
            np.zeros((r.shape[0], 1)),
            retval,
            last.reshape(last.shape + (1,))
        ))
    else:
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


# mean enclosed surface density integrals:
def sigmabar_encl_integral(x0, x1, a, b):
    return 2 * np.pi * np.exp(b) / (a + 2) * \
        (np.power(x1, a + 2) - np.power(x0, a + 2))


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
        rhoarr = np.repeat(np.array(rho)[np.newaxis], self.R.size - 1, axis=0)
        aarr = _a(rhoarr, self.rarr, iet=self.inner_extrapolation_type)
        barr = _b(rhoarr, self.rarr, aarr)
        if (aarr <= -5.).any():
            raise ValueError('Density slope < -5 causes problems in surface '
                             'density profile.')
        # if (aarr > 0).any():
        #     raise ValueError('Density slope > 0 not supported.')
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

        aS = 2 * (np.log(sigma[1:]) - np.log(sigma[:-1])) \
            / (np.log(self.R[2:]) - np.log(self.R[:-2]))
        bS = np.log(sigma[:-1]) - aS * .5 * np.log(self.R[:-2] * self.R[1:-1])
        if aS[0] <= -2:
            raise ValueError('Surface density has central slope <= -2, '
                             'central mass content is infinite.')
        mass_central = sigmabar_encl_integral(
            0,
            np.sqrt(self.R[0] * self.R[1]),
            aS[0],
            bS[0]
        )
        mass_annuli = sigmabar_encl_integral(
            np.sqrt(self.R[:-2] * self.R[1:-1]),
            np.sqrt(self.R[1:-1] * self.R[2:]),
            aS,
            bS
        )
        mass_enclosed = np.cumsum(np.r_[mass_central, mass_annuli])
        mean_enclosed = mass_enclosed / (np.pi * self.R[:-1] * self.R[1:])
        if self.mode == 'ESD':
            return mean_enclosed - sigma, sigma
        raise ValueError('Unknown mode.')


def esd_to_rho(obs, obs_var, guess, r, R, extrapolate_inner=True,
               extrapolate_outer=True,
               inner_extrapolation_type='extrapolate',
               testwith_rho=None, truths=None, niter=5500, burn=500,
               savecorner='corner.pdf', parallel=8, cache=None):
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
            statmodel.obs = statmodel.esd(testwith_rho)[0]
        else:
            raise ValueError
    else:
        statmodel.obs = obs
    if obs_var.ndim == 1:
        obs_var = np.diag(obs_var)
    try:
        statmodel.inv_cov = np.linalg.inv(obs_var)
    except np.linalg.LinAlgError:
        raise ValueError('Could not invert covariance matrix.')
    statmodel.r = r
    statmodel.extrapolate_outer = extrapolate_outer
    RAP = Rap(statmodel)
    if cache is not None:
        try:
            RAP.results, = loadvars(cache)
        except IOError:
            runfit = True
        else:
            runfit = False
    else:
        runfit = True
    if runfit:
        olderr = np.seterr(all='ignore')
        RAP.fit(guess, niter=niter, burn=burn, parallel=parallel)
        np.seterr(**olderr)
        if cache is not None:
            savevars([RAP.results], cache)
    if savecorner:
        RAP.cornerfig(save=savecorner, fignum=999,
                      labels=statmodel.labels, truths=truths)
    for key in ('thetas', 'theta_ml'):
        RAP.results[key] = np.exp(RAP.results[key])
    RAP.results['theta_evals'] = RAP.results['thetas']  # alias
    for key in ('theta_perc_16_50_84', ):
        RAP.results[key] = [np.exp(rl) for rl in RAP.results[key]]
    RAP.results['esd_evals'] = np.array([b[0] for b in RAP.results['blobs']])
    RAP.results['esd_ml'] = RAP.results['esd_evals'][
        np.argmax(RAP.results['lnL'])
    ]
    RAP.results['esd_perc_16_50_84'] = list(zip(
        *np.percentile(RAP.results['esd_evals'], [16, 50, 84], axis=0)))
    RAP.results['sigma_evals'] = np.array([b[1] for b in RAP.results['blobs']])
    RAP.results['sigma_ml'] = RAP.results['sigma_evals'][
        np.argmax(RAP.results['lnL'])
    ]
    RAP.results['sigma_perc_16_50_84'] = list(zip(
        *np.percentile(RAP.results['sigma_evals'], [16, 50, 84], axis=0)))
    RAP.results['mencl_evals'] = rho_to_mencl(RAP.results['thetas'], r)[0]
    RAP.results['mencl_ml'] = rho_to_mencl(RAP.results['theta_ml'], r)[0]
    RAP.results['mencl_perc_16_50_84'] = list(zip(
        *np.percentile(RAP.results['mencl_evals'], [16, 50, 84], axis=0)))
    RAP.results['gobs_evals'] = to_gobs(RAP.results['mencl_evals'], r)
    RAP.results['gobs_ml'] = to_gobs(RAP.results['mencl_ml'], r)
    RAP.results['gobs_perc_16_50_84'] = list(zip(
        *np.percentile(RAP.results['gobs_evals'], [16, 50, 84], axis=0)))
    RAP.results['r'] = r
    RAP.results['R'] = R
    RAP.results['Rmid'] = .5 * (R[1:] + R[:-1])  # intentionally not log centre
    return RAP.results
