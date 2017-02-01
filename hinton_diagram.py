# Adapted from: https://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/HintonDiagrams.html

import numpy as np
import matplotlib.pyplot as plt

import sys
if sys.version_info.major > 2:
    xrange = range


def _blob(ax, x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    ax.fill(xcorners, ycorners, colour, edgecolor=colour)


def hinton(W, max_weight=None, xlabels=None, ylabels=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix.
    """
    height, width = W.shape
    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.fill(np.array([0, width, width, 0]), np.array([0, 0, height, height]), 'gray')
    ax.axis('off'), ax.axis('equal')

    for x in range(width):
        for y in xrange(height):
            w, _x, _y = W[y, x], x + 0.5, y + 0.5
            if w > 0:
                _blob(ax, _x, height - _y, min(1, w / max_weight), 'white')
            elif w < 0:
                _blob(ax, _x, height - _y, min(1, -w / max_weight), 'black')

    if xlabels:
        for i in range(len(xlabels)):
            x = i + 0.4
            plt.text(x, height + 0.5, xlabels[i], fontsize=8)

    if ylabels:
        if len(ylabels) < height:
            ylabels += [''] * (height - len(ylabels))
        for i in range(len(ylabels)):
            y = i + 0.4
            plt.text(-0.5, i + 0.4, ylabels[len(ylabels) - i - 1], fontsize=8)

    return fig


if __name__ == '__main__':
    fig = hinton(np.random.randn(10, 10))
    plt.show()
