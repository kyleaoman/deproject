# deproject
Utility to infer a density profile corresponding to an excess surface density profile, assuming a spherically symmetric piecewise power law profile.

## Installation:
 - Download via web UI, or `git clone https://github.com/kyleaoman/deproject.git`
 - Install dependencies if necessary (see [`setup.py`](https://github.com/kyleaoman/deproject/blob/master/setup.py)).
 - Global install (Linux): 
   - `cd` to directory with [`setup.py`](https://github.com/kyleaoman/deproject/blob/master/setup.py)
   - run `sudo pip install -e .` (`-e` installs via symlink, so pulling repository will do a 'live' update of the installation)
 - User install (Linux):
   - `cd` to directory with [`setup.py`](https://github.com/kyleaoman/deproject/blob/master/setup.py)
   - ensure `~/lib/python3.7/site-packages` or similar is on your `PYTHONPATH` (e.g. `echo $PYTHONPATH`), if not, add it (perhaps in `.bash_profile` or similar)
   - run `pip install --prefix ~ -e .` (`-e` installs via symlink, so pulling repository will do a 'live' update of the installation)
 - cd to a directory outside the module and launch `python`; you should be able to do `from deproject.piecewise_powerlaw import esd_to_rho`
 
## Usage:

```python
# example for testing
import numpy as np
from deproject.piecewise_powerlaw import esd_to_rho, _ESD
import matplotlib.pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages
r = np.logspace(-3, 3, 10)
R = np.logspace(-3, 3, 11)
rho = np.power(r, -2) / (2 * np.pi)
obs = None
guess = np.power(r, -1.5) / np.pi
extrapolate_outer = True
extrapolate_inner = True
inner_extrapolation_type = 'extrapolate'
startstep = np.min(-np.diff(np.log(guess))) / 3.  # probably reasonable
minstep = .001

best = esd_to_rho(
    obs,
    guess,
    r,
    R,
    extrapolate_inner=extrapolate_inner,
    extrapolate_outer=extrapolate_outer,
    inner_extrapolation_type=inner_extrapolation_type,
    startstep=startstep,
    minstep=minstep,
    testwith_rho=rho,
    verbose=True
)

esd = _ESD(
    r,
    R,
    extrapolate_inner=extrapolate_inner,
    extrapolate_outer=extrapolate_outer,
    inner_extrapolation_type=inner_extrapolation_type
)
Rmids = np.power(10, .5 * (np.log10(R[:-1]) + np.log10(R[1:])))
with PdfPages('deproject.pdf') as pdffile:
    pp.figure(1)
    pp.xlabel(r'$r$')
    pp.ylabel(r'$\rho$')
    pp.plot(np.log(r), np.log(rho), '-b')
    pp.plot(np.log(r), np.log(guess), marker='o', mfc='None', mec='blue',
            ls='None')
    pp.plot(np.log(r), np.log(best), 'ob')
    pp.savefig(pdffile, format='pdf')

    pp.figure(2)
    pp.xlabel(r'$R$')
    pp.ylabel(r'$\Delta\Sigma$')
    pp.plot(np.log(Rmids), np.log(esd(rho)), '-r')
    pp.plot(np.log(Rmids), np.log(esd(guess)), marker='o', mfc='None',
            mec='red', ls='None')
    pp.plot(np.log(Rmids), np.log(esd(best)),
            'or')
    pp.savefig(pdffile, format='pdf')
```

```python
# example for "real" use
import numpy as np
from deproject.piecewise_powerlaw import esd_to_rho
r = np.logspace(???)  # 'real' values, not log
R = np.logspace(???)  # 'real' values, not log
obs = ???
guess = ???  # density profile, not shallower than -1 in outer part!
extrapolate_outer = True
extrapolate_inner = True
inner_extrapolation_type = 'extrapolate'  # or 'flat'
startstep = np.min(-np.diff(np.log(guess))) / 3.  # probably reasonable
minstep = .001  # sets tolerance in fit in terms of Delta log(DeltaSigma)
rho = esd_to_rho(
    obs,
    guess,
    r,
    R,
    extrapolate_inner=extrapolate_inner,
    extrapolate_outer=extrapolate_outer,
    inner_extrapolation_type=inner_extrapolation_type,
    startstep=startstep,
    minstep=minstep,
    verbose=False
)
```