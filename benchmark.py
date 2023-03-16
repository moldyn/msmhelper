# importing packages needed for benchmark
import timeit
from string import ascii_lowercase

import msmhelper as mh
import numpy as np
import pyemma
import prettypyplot as pplt
from matplotlib import pyplot as plt


def main():
    def generate_traj(n_steps, n_states):
        """Generate random state trajectory."""
        return np.random.randint(low=1, high=n_states + 1, size=n_steps)

    # change matplotlbiu default style
    pplt.use_style(figsize=(2.6, 1.3), true_black=True)

    # lagtime to estimate the transition matrices
    n_states, n_steps, lagtime = 10, int(1e5), 100
    n_mcmc_steps = int(1e5)
    lagtimes = np.unique(np.geomspace(1, 100, 5).astype(int))

    traj = generate_traj(n_steps, n_states),
    mh.msm.estimate_markov_model(traj, lagtime=lagtime)
    mh.msm.ck_test(traj, [lagtime], tmax=1000)
    mh.msm.timescales.propagate_MCMC(traj, lagtime, 100)

    repeat = 100

    fig, axs = plt.subplot_mosaic(
        [['msm', 'msm', 'ck', 'mcmc']],
        gridspec_kw={'hspace': 0.7, 'wspace': 0.3},
    )
    global_kwargs = globals() | locals() | {'generate_traj': generate_traj}
    times = [
        np.mean(
            timeit.repeat(setup=setup, stmt=stmt, globals=global_kwargs, repeat=repeat, number=1),
        )
        for setup, stmt in (
            ('traj = generate_traj(n_steps, n_states)', 'pyemma.msm.estimate_markov_model(traj, lag=lagtime, reversible=False)'),
            ('traj = generate_traj(n_steps, n_states)', 'pyemma.msm.estimate_markov_model(traj, lag=lagtime, reversible=True)'),
            ('traj = generate_traj(n_steps, n_states)', 'mh.msm.estimate_markov_model(traj, lagtime=lagtime)'),
            ('traj = mh.StateTraj(generate_traj(n_steps, n_states))', 'mh.msm.estimate_markov_model(traj, lagtime=lagtime)'),
        )
    ]
    visualize_benchmark_results(
        times,
        ('PyEmma\n non-rev.', 'PyEmma\n reversible', 'msmhelper\n numpy', 'msmhelper\n StateTraj'),
        f'MSM: $N_\mathrm{{steps}}=10^{np.log10(n_steps):.0f}$ and {n_states} states',
        plt.colormaps['paula'].colors,
        ax=axs['msm'],
    )

    # MCMC
    times = [
        np.mean(
            timeit.repeat(setup=setup, stmt=stmt, globals=global_kwargs, repeat=repeat, number=1),
        )
        for setup, stmt in (
            ('traj = generate_traj(n_steps, n_states)', 'pyemma.msm.estimate_markov_model(traj, lag=lagtime).generate_traj(n_mcmc_steps)'),
            ('traj = mh.StateTraj(generate_traj(n_steps, n_states))', 'mh.msm.timescales.propagate_MCMC(traj, lagtime, n_mcmc_steps)'),
        )
    ]
    visualize_benchmark_results(
        times,
        ('PyEmma', 'msmhelper'),
        f'MCMC: $N_\mathrm{{steps}}=10^{np.log10(n_mcmc_steps):.0f}$',
        plt.colormaps['paula'].colors[1::2],
        ax=axs['mcmc'],
    )

    # CKTEST
    times = [
        np.mean(
            timeit.repeat(setup=setup, stmt=stmt, globals=global_kwargs, repeat=repeat, number=1),
        )
        for setup, stmt in (
            ('traj = generate_traj(n_steps, n_states)', '[pyemma.msm.estimate_markov_model(traj, lag=lagtime).cktest(2, n_jobs=1, show_progress=False) for lagtime in lagtimes]'),
            ('traj = mh.StateTraj(generate_traj(n_steps, n_states))', 'mh.msm.ck_test(traj, lagtimes, tmax=1000)'),
        )
    ]
    visualize_benchmark_results(
        times,
        ('PyEmma', 'msmhelper'),
        f'CK-Test',
        plt.colormaps['paula'].colors[1::2],
        ax=axs['ck'],
    )

    axs['mcmc'].set_ylabel('')
    axs['ck'].set_ylabel('')

    # plot subplot a, b, c, ...
    for char, (offset, axstr) in zip(
        ascii_lowercase,
        ((-0.075, 'msm'), (-0.15, 'ck'), (-0.15, 'mcmc')),
    ):
        pplt.text(
            offset,
            1,
            fr'{{\Large \textbf{{{char}}}}}',
            va='baseline',
            ha='right',
            transform=axs[axstr].transAxes,
            ax=axs[axstr],
        )

    for ax in axs.values():
        pplt.grid(True, ax=ax)

    pplt.savefig('benchmark.pdf')



# method to visualize results
def visualize_benchmark_results(times, labels, title, colors, ax):
    times = np.array(times) * 1e3
    # bar = ax.bar(labels, 1 / times, color=colors)
    # ax.bar_label(bar,
    #     labels=[f'x{fac:.1f}' for fac in times[0] / times],
    # )
    bar = ax.bar(labels, times, color=colors)
    ax.set_ylabel('wall time [ms]')
    #ax.set_ylim(np.min(1 / times) / 3, np.max(1 / times) * 3)
    ax.set_ylim(np.min(times) / 2, np.max(times) * 2)
    ax.set_yscale('log')

    # highlight relative performance
    ax.set_title(title, size='medium')


if __name__ == '__main__':
    main()
