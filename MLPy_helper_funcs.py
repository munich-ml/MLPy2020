# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:04:44 2020

@author: holge
"""

import numpy as np
import pandas as pd
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
    
    
def plot_prediction_examples(test_class, class_names, y_pred, y_test, X_test, 
                             n_cols=10):
    """
    plots images of predictions examples in 4 rows from 'true positives' till 'false negatives'

    Parameters
    ----------
    test_class : TYPE
        DESCRIPTION.
    class_names : TYPE
        DESCRIPTION.
    y_pred : np.array
        predicted classes
    y_test : np.array
        true / actual classes
    X_test : np.array
        array of images
    n_cols : int, optional
        Number of columns / number of images per row. The default is 10.

    Returns
    -------
    None.

    """
    
    print("Evaluating examples of test_class={}, '{}'".format(test_class, 
                                                              class_names[test_class]))
    
    # step 1: Compute TP, TN, FP, FN
    preds = {"true pos": [],
             "true neg": [],
             "false pos": [],
             "false neg": []}

    for i, (val_test, val_pred) in enumerate(zip(y_test, y_pred)):
        if val_test == test_class:
            if val_pred == test_class:
                preds["true pos"].append((i, val_test, val_pred))
            else:
                preds["false neg"].append((i, val_test, val_pred))
        else:
            if val_pred == test_class:
                preds["false pos"].append((i, val_test, val_pred))
            else:
                preds["true neg"].append((i, val_test, val_pred))

    for key, val in preds.items():
        print("- {}: {} images".format(key, len(val)))
    
    # step 2: plotting random examples of right and wrong predictions
    plt.figure(figsize=(n_cols*1.8, 9))
    for row, predictions in enumerate(preds.values()):
        for col, idx in enumerate(np.random.randint(0, len(predictions), n_cols)):
            i, val_test, val_pred = predictions[idx]
            plt.subplot(len(preds), n_cols, n_cols*row+col+1)
            plt.imshow(np.squeeze(X_test[i]), cmap="binary")
            plt.axis('off')
            title = "\nimage:{}\nact: {}\nprd: {}".format(i, class_names[val_test], class_names[val_pred])
            plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_hist_2D(df, x_column, y_column, bins=15, levels=20, figsize=[13, 4]):
    """
    Parameters
    ----------
    df : Pandas DataFrame
        Data
    x_column : str
        column to plot on x
    y_column : str
        column to plot on y
    bins : int, optional
        Number of histogram bins. The default is 15.
    levels : int, optional
        Number of contour levels. The default is 20.
    figsize : tuple or list, optional
        The default is [13, 4].

    Returns
    -------
    None

    """
    x = df[x_column]
    y = df[y_column]
    fig = plt.figure(figsize=figsize) 
    axes = fig.subplots(nrows=1, ncols=2)
    cnts, h2x, h2y, img = axes[0].hist2d(x, y, bins=bins, cmap=plt.cm.coolwarm,
                                         range=([x.min(), x.max()], [y.min(), y.max()]))
    axes[0].set_title("hist2d heatmap")

    def edges2centers(edges):
        return (edges[1:] + edges[:-1]) / 2

    axes[1].contourf(edges2centers(h2x), edges2centers(h2y), cnts.T, levels=levels, 
                     cmap=plt.cm.coolwarm)
    axes[1].set_title("contour plot")

    for ax in axes:
        ax.grid(which="both")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        fig.colorbar(img, ax=ax)
    
    fig.tight_layout()
    

if __name__ == "__main__":
    # Test plot_hist_2D
    x = np.random.beta(a=2, b=5, size=10000)
    y = np.random.beta(a=1.5, b=3, size=10000)
    df = pd.DataFrame(np.column_stack([x, x+y]), columns=["x", "y"])
    plot_hist_2D(df, "x", "y", bins=20)
    