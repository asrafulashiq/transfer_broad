import numpy as np
import matplotlib as mpl
import seaborn as sns
from seaborn.external.husl import husl_to_rgb


def set_style(style="whitegrid", color='bright', font_scale=1.2):
    """Consistent style for plots."""
    sns.set(style=style,
            context="paper",
            font_scale=font_scale,
            rc={
                "axes.linewidth": 1,
                "lines.linewidth": 1
            })
    sns.set_palette(color)

    # sns.set(style="ticks",
    #         context="paper",
    #         font_scale=.9,
    #         rc={
    #             "xtick.major.size": 3,
    #             "ytick.major.size": 3,
    #             "xtick.major.width": 1,
    #             "ytick.major.width": 1,
    #             "xtick.major.pad": 3.5,
    #             "ytick.major.pad": 3.5,
    #             "axes.linewidth": 1,
    #             "lines.linewidth": 1
    #         })


def savefig(f, fname, dpi=200):
    f.savefig(fname, dpi=dpi, bbox_inches='tight', pad_inches=0.02)


def points_to_lines(ax, w=.8, **kws):
    """Replace the central tendency glyph from a pointplot with a line."""
    for col in ax.collections:
        for (x, y), fc in zip(col.get_offsets(), col.get_facecolors()):
            ax.plot([x - w / 2, x + w / 2], [y, y], color=fc, **kws)
        col.remove()


def get_colormap(exp, as_cmap=True):
    """Get experiment-specific diverging colormaps."""
    lums = np.linspace(50, 99, 128)
    sats = np.linspace(80, 20, 128)
    assert exp in ["dots", "sticks", "rest"]
    h1 = 240 if exp == "dots" else 160
    h2 = 20
    lut = ([husl_to_rgb(h1, s, l) for s, l in zip(sats, lums)] +
           [husl_to_rgb(h2, s, l) for s, l in zip(sats, lums)][::-1])
    if as_cmap:
        return mpl.colors.ListedColormap(lut)
    return lut