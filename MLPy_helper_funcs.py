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
    
    
    
def plot_prediction_examples(test_class, class_names, y_pred, y_test, X_test, 
                             n_cols=10):
    """
    plots predictions examples in 4 rows from 'true positives' till 'false negatives'

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