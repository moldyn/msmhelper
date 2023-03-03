#!usr/bin/env python3
"""Plot state structure of all microstates."""
from os.path import splitext

import click
import msmhelper as mh
import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats


@click.command(no_args_is_help='-h')
@click.option(
    '--contacts',
    'contact_file',
    required=True,
    type=click.Path(exists=True),
    help=(
        'Path to file holding all contacts (features) of shape `(n_frames, '
        'n_contacts)`.'
    ),
)
@click.option(
    '--clusters',
    'cluster_file',
    required=True,
    type=click.Path(exists=True),
    help=(
        'Path to contacts cluster file, where every row is a cluster and in '
        'each row the indices corresponding to the columns of the clusters '
        'are stated.'
    ),
)
@click.option(
    '--state',
    'state_file',
    required=True,
    type=click.Path(exists=True),
    help='Path to state trajectory.',
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help=(
        'Output basename of figure. Needs to have a valid extension (".pdf", '
        '".svg", ".png"). Default format is pdf.'
    ),
)
@click.option(
    '--grid',
    type=click.IntRange(min=1),
    nargs=2,
    default=(4, 3),
    help='Number of rows and cols per figure.',
)
def contact_rep(contact_file, cluster_file, state_file, output, grid):
    """Contact representation of states.

    This script creates a contact representation of states. Were the states
    are obtained by [MoSAIC](https://github.com/moldyn/MoSAIC) and the contact
    representation was introduced in Nagel et al.[^1].

    [^1]: Nagel et al., **Selecting Features for Markov Modeling: A Case Study
          on HP35.**, *J. Chem. Theory Comput.*, submitted,

    """
    # setup matplotlib
    pplt.use_style(
        figsize=0.8, colors='pastel_autunm', true_black=True, latex=False,
    )

    # load files
    state_traj = mh.opentxt(state_file, dtype=int)
    contacts = mh.opentxt(contact_file, dtype=np.float64)
    states = np.unique(state_traj)
    clusters = load_clusters(cluster_file)

    contact_idxs = np.hstack(clusters)
    n_idxs = len(contact_idxs)
    n_frames = len(contacts)

    xtickpos = np.cumsum([
        0,
        *[
            len(clust) for clust in clusters[:-1]
        ],
    ]) - 0.5
    nrows, ncols = grid
    hspace, wspace = 0, 0
    ylims = 0, np.quantile(contacts, 0.999)

    for chunk in mh.plot._ck_test._split_array(states, nrows * ncols):
        fig, axs = plt.subplots(
            int(np.ceil(len(chunk) / ncols)),
            ncols,
            sharex=True,
            sharey=True,
            squeeze=False,
            gridspec_kw={'wspace': wspace, 'hspace': hspace},
        )

        # ignore outliers
        for state, ax in zip(chunk, axs.flatten()):
            contacts_state = contacts[state_traj == state]
            pop_state = len(contacts_state) / n_frames

            # get colormap
            c1, c2, c3 = pplt.categorical_color(3, 'C0')

            stats = {
                idx: boxplot_stats(contacts_state[:, idx])[0]
                for idx in contact_idxs
            }

            for color, (key_low, key_high), label in (
                (c3, ('whislo', 'whishi'), r'$Q_{1/3} \pm 1.5\mathrm{IQR}$'),
                (c2, ('q1', 'q3'), r'$\mathrm{IQR} = Q_3 - Q_1$'),
            ):
                ymax = [stats[idx][key_high] for idx in contact_idxs]
                ymin = [stats[idx][key_low] for idx in contact_idxs]
                ax.stairs(
                    ymax,
                    np.arange(n_idxs + 1) - 0.5,
                    baseline=ymin,
                    color=color,
                    lw=0,
                    fill=True,
                    label=label,
                )

            ax.hlines(
                [stats[idx]['med'] for idx in contact_idxs],
                xmin=np.arange(n_idxs) - 0.5,
                xmax=np.arange(n_idxs) + 0.5,
                label='median',
                color=c1,
            )

            pplt.text(
                0.5,
                0.95,
                fr'S{state} {pop_state:.1%}',
                ha='center',
                va='top',
                ax=ax,
                transform=ax.transAxes,
                contour=True,
            )

            ax.set_xlim([-0.5, n_idxs - 0.5])
            ax.set_ylim(*ylims)
            ax.set_xticks(xtickpos)
            ax.set_xticklabels(np.arange(len(xtickpos)) + 1)

            ax.grid(False)
            for pos in xtickpos:
                ax.axvline(pos, color='pplt:grid', lw=1.0)

        pplt.hide_empty_axes()
        pplt.legend(
            ax=axs[0, 0],
            outside='top',
            bbox_to_anchor=(
                0,
                1.0,
                axs.shape[1] + wspace * (axs.shape[1] - 1),
                0.01,
            ),
            frameon=False,
            ncol=2,
        )
        pplt.subplot_labels(
            xlabel='contact clusters',
            ylabel='distances [nm]',
        )

        # save figure and continue
        if output is None:
            output = f'{state_file}.contactRep.pdf'
        # insert state_str between pathname and extension
        path, ext = splitext(output)
        pplt.savefig(f'{path}.state{chunk[0]:.0f}-{chunk[-1]:.0f}{ext}')


def load_clusters(filename):
    """Load clusters stored from cli.

    This method is taken from [MoSAIC](https://github.com/moldyn/MoSAIC/).

    Parameters
    ----------
    filename : str
        Filename of cluster file.

    Returns
    -------
    clusters : ndarray of shape (n_clusters, )
        A list of arrays, each containing all indices (features) for each
        cluster.

    """
    comment = '#'
    with open(filename) as clusters:
        clusters_list = [
            np.array(
                cluster.split()
            ).astype(int).tolist()
            for cluster in clusters if not cluster.startswith(comment)
        ]

    # In case of clusters of same length, numpy casted it as a 2D array.
    # To ensure that the result is an numpy array of list, we need to
    # create an empty list, adding the values in the second step
    clusters = np.empty(len(clusters_list), dtype=object)
    clusters[:] = clusters_list  # noqa: WPS362
    return clusters


if __name__ == '__main__':
    contact_rep()
