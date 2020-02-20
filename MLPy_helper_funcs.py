# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:04:44 2020

@author: holge
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, xticks, yticks, normalize=False, ignore_main_diagonal=False):
    """
    plots a confusion matrix using matplotlib

    Parameters
    ----------
    cm : (tensor or numpy array)
        confusion matrix
        e.g. from tf.math.confusion_matrix(y_test, y_pred)
    xticks : (list)
        x tick labels
    yticks : (list)
        x tick labels
    normalize : (bool), optional
        scales cm to 1. The default is False.
    ignore_main_diagonal : (bool), optional
        sets the main diagonal to zero. The default is False.

    Returns
    -------
    None.

    """
    
    cm = np.array(cm)
    if normalize:   # normalize to 1.0
        cm = cm / cm.max()
    if ignore_main_diagonal:  # set main diagonal to zero
        for i in range(len(cm)):
            cm[i, i] = 0
    plt.imshow(cm, cmap="binary")
    plt.xticks(ticks=range(len(xticks)), labels=xticks, rotation=90)
    plt.yticks(ticks=range(len(yticks)), labels=yticks)
    plt.xlabel("predicted class")
    plt.ylabel("actual class")
    # put numbers inside the heatmap
    thresh = cm.max() / 2.
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            plt.text(j, i, format(int(val)),
                    horizontalalignment="center",
                    color = "white" if val > thresh else "black")
                    
    plt.colorbar()