import numpy as np
from scipy.special import hyp2f1, gamma
from warnings import warn


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
    # retval *= 2. / (-a - 1.)  # empirical correction
    retval[r == 0] = 0
    return retval


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
                 inner_extrapolation_type='extrapolate'):
        self.r = r
        self.R = R
        _rarr, _Rarr = np.meshgrid(r, R)
        self.rarr = _rarr[:-1]
        self.Rarr = _Rarr[:, :-1]
        self.cases = _casemap(self.rarr, self.Rarr)
        self.extrapolate_outer = extrapolate_outer
        self.extrapolate_inner = extrapolate_inner
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

        # mean enclosed surface density integrals:

        def integral(x0, x1, a, b):
            return 2 * np.pi * np.exp(b) / (a + 2) * \
                (np.power(x1, a + 2) - np.power(x0, a + 2))

        aS = 2 * (np.log(sigma[1:]) - np.log(sigma[:-1])) \
            / (np.log(self.R[2:]) - np.log(self.R[:-2]))
        bS = np.log(sigma[:-1]) - aS * .5 * np.log(self.R[:-2] * self.R[1:-1])
        if aS[0] < -2:
            raise ValueError('Surface density has central slope < -2, '
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
        return mean_enclosed - sigma


def esd_to_rho(obs, guess, r, R, extrapolate_inner=True,
               extrapolate_outer=True,
               inner_extrapolation_type='extrapolate',
               startstep=.1, minstep=None, tol=None,
               testwith_rho=None, fom='chi2', verbose=False, prefix=''):
    esd = _ESD(
        r,
        R,
        extrapolate_inner=extrapolate_inner,
        extrapolate_outer=extrapolate_outer,
        inner_extrapolation_type=inner_extrapolation_type
    )
    if obs is None:
        if testwith_rho is not None:
            obs = esd(testwith_rho)
        else:
            raise ValueError

    if fom == 'chi2':
        def _logLikelihood(rho):
            try:
                retval = -np.sqrt(np.sum(np.power(np.log(esd(rho))
                                                  - np.log(obs), 2))) \
                                                  / np.sqrt(rho.size)
            except ValueError:
                return -np.inf
            if(np.isnan(retval).any()):
                return -np.inf
            else:
                return retval

    elif fom == 'max':
        def _logLikelihood(rho, fom=fom):
            try:
                retval = -np.max(np.abs(np.log(esd(rho)) - np.log(obs)))
            except ValueError:
                return -np.inf
            if(np.isnan(retval).any()):
                return -np.inf
            else:
                return retval

    else:
        raise ValueError('Unknown fom.')

    def _logPrior(rho):
        # checking diff of exp(rho) equiv. diff of rho
        if (np.diff(rho) > 0).any():
            return -np.inf
        # required for convergence when extrapolating
        if extrapolate_outer:
            if (np.log(rho[-1]) - np.log(rho[-2])) / \
               (np.log(r[-1]) - np.log(r[-2])) >= -1:
                return -np.inf
        return 0.0

    def _logProbability(rho):
        rho = np.exp(rho)
        lp = _logPrior(rho)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + _logLikelihood(rho)

    def _optimize(guess, startstep=startstep, minstep=minstep, tol=tol,
                  verbose=verbose, prefix=prefix):
        cv = np.log(guess)
        cp = _logProbability(cv)
        step = startstep
        tooslow = 0
        if (minstep is None) and (tol is None):
            raise ValueError('No halting condition!')
        while True:
            done = np.zeros(len(cv), dtype=np.bool)
            while not done.all():
                done = np.zeros(len(cv), dtype=np.bool)
                for nr in range(len(cv)):
                    mod = np.zeros(len(cv))
                    mod[nr] = step
                    fpp = _logProbability(cv + mod)
                    if fpp > cp:
                        cv = cv + mod
                        if 1 - fpp / cp < 1.E-3:
                            tooslow += 1
                        else:
                            tooslow = 0
                        cp = fpp
                    else:
                        fpm = _logProbability(cv - mod)
                        if fpm > cp:
                            cv = cv - mod
                            if 1 - fpm / cp < 1.E-3:
                                tooslow += 1
                            else:
                                tooslow = 0
                            cp = fpm
                        else:
                            done[nr] = True
                    if tooslow > 100 * len(cv):
                        if -cp < tol:
                            best = np.exp(cv)
                            return best
                        else:
                            warn('Iteration progress too slow.')
                            return np.ones(cv.shape) * np.nan
                if verbose:
                    print('  {:s}  P={:.6e}, S={:.6f}'.format(
                        prefix, cp, step))
            step = step / 2.
            if minstep is not None:
                if step < minstep:
                    break
            if tol is not None:
                if -cp < tol:
                    break
                if step < 1E-9 * startstep:
                    warn('Step too small without reaching tol.')
                    return np.ones(cv.shape) * np.nan
        best = np.exp(cv)
        return best

    return _optimize(guess, startstep=startstep, minstep=minstep,
                     tol=tol, verbose=verbose, prefix=prefix)
