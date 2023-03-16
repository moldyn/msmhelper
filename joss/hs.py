from string import ascii_lowercase

import msmhelper as mh
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt
from msmhelper.utils import datasets

pplt.use_style(figsize=(2.6, 1.3), colors='paula', true_black=True)

# rate of k, h
rates = (0.2, 0.1)
trajs = {
    label: traj
    for label, traj in zip(
        ('micro', 'macro'),
        datasets.hummer15_8state(
            *rates, nsteps=int(1e5), return_macrotraj=True,
        ),
    )
}

lagtimes = np.arange(1, 20)
_, axs = plt.subplots(
    1,
    2,
    sharex=True,
    gridspec_kw={'wspace': 0.15}
)

eq_traj = mh.StateTraj(trajs['macro'])
hummer_szabo_traj = mh.LumpedStateTraj(trajs['macro'], trajs['micro'])

for idx_macro, macrotraj in enumerate((eq_traj, hummer_szabo_traj)):
    ax = axs[idx_macro]

    impl_times = mh.msm.implied_timescales(macrotraj, lagtimes, ntimescales=3)
    for idx, impl_time in enumerate(impl_times.T):
        ax.plot(lagtimes, impl_time, label=f'$t_{idx + 1}$')

    ax.set_xlim(lagtimes[0], lagtimes[-1])
    # highlight diagonal
    ax.fill_between(ax.get_xlim(), ax.get_xlim(), color='pplt:grid')
    ax.set_ylim(0, 55)

axs[0].set_xlabel(r'lagtime $\tau$ [frames]')
axs[1].set_xlabel(r'lagtime $\tau$ [frames]')
axs[0].set_ylabel('time scale $t$ [frames]')
# highlight relative performance
axs[0].set_title('MSM Dynamics', size='medium')
axs[1].set_title('Hummer-Szabo Projected Dynamics', size='medium')

# plot subplot a, b, c, ...
for char, ax in zip(ascii_lowercase, axs):
    pplt.text(
        -0.075,
        1,
        fr'{{\Large \textbf{{{char}}}}}',
        va='baseline',
        ha='right',
        transform=ax.transAxes,
        ax=ax,
    )

pplt.savefig('hs.pdf')
