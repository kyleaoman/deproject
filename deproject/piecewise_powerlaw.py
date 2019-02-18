import numpy as np
from scipy.special import hyp2f1, gamma


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
        .5 * np.sqrt(np.pi) * gamma((1 - a) / 2) / gamma(-a / 2) *
        np.power(R, a + 3) / (a + 3),
        retval
    )
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
    pre = 4 / (np.power(R[1:], 2) - np.power(R[:-1], 2)) * np.exp(b)
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
        casevals = _caseeval(self.rarr, self.Rarr, aarr, barr)
        retval = np.zeros((len(self.R) - 1, len(self.r) + 1))
        for c, v in zip(self.cases, casevals):
            retval += np.where(c, v, np.zeros(v.shape))
        if not self.extrapolate_inner:
            retval = retval[:, 1:]
        if not self.extrapolate_outer:
            retval = retval[:, :-1]
        sigma = np.sum(retval, axis=1)
        mean_enclosed = np.cumsum(
            sigma * (np.power(self.R[1:], 2) - np.power(self.R[:-1], 2))) \
            / np.power(.5 * (self.R[:-1] + self.R[1:]), 2)
        return mean_enclosed - sigma


def esd_to_rho(obs, guess, r, R, extrapolate_inner=True,
               extrapolate_outer=True,
               inner_extrapolation_type='extrapolate',
               startstep=.1, minstep=.001,
               testwith_rho=None, verbose=False):
    esd = _ESD(
        r,
        R,
        extrapolate_inner=True,
        extrapolate_outer=True,
        inner_extrapolation_type='extrapolate'  # or 'flat'
    )
    if obs is None:
        if testwith_rho is not None:
            obs = esd(testwith_rho)
        else:
            raise ValueError

    def _logLikelihood(rho):
        retval = -np.sqrt(np.sum(np.power(np.log(esd(rho)) - np.log(obs), 2)))
        if(np.isnan(retval).any()):
            return -np.inf
        else:
            return retval

    def _logPrior(rho):
        # checking diff of exp(rho) equiv. diff of rho
        if (np.diff(rho) > 0).any():
            return -np.inf
        # required for convergence when extrapolating
        if (np.log(rho[-1]) - np.log(rho[-2])) / \
           (np.log(r[-1]) - np.log(r[-2])) >= -1:
            # return -np.inf
            raise ValueError('Outer extrapolation with slope > -1, '
                             'iteration will get stuck.')
        else:
            return 0.0

    def _logProbability(rho):
        rho = np.exp(rho)
        lp = _logPrior(rho)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + _logLikelihood(rho)

    def _optimize(guess, startstep=.1, minstep=.01, verbose=False):
        cv = np.log(guess)
        cp = _logProbability(cv)
        step = startstep
        while True:
            if verbose:
                print('STEP', step)
            done = np.zeros(len(cv), dtype=np.bool)
            while not done.all():
                done = np.zeros(len(cv), dtype=np.bool)
                for nr in range(len(cv)):
                    mod = np.zeros(len(cv))
                    mod[nr] = step
                    fpp = _logProbability(cv + mod)
                    if fpp > cp:
                        cv = cv + mod
                        cp = fpp
                    else:
                        fpm = _logProbability(cv - mod)
                        if fpm > cp:
                            cv = cv - mod
                            cp = fpm
                        else:
                            done[nr] = True
                if verbose:
                    print('  P={:.6e}, S={:.6f}'.format(cp, step))
            step = step / 2.
            if step < minstep:
                break
        best = np.exp(cv)
        return best

    return _optimize(guess, startstep=startstep, minstep=minstep,
                     verbose=verbose)
