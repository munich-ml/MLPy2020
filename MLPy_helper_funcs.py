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
    
    print("Evaluating examples of test_class={}, '{}'".format(test_class, 
                                                              class_names[test_class]))
    
    # step 1: seperating right and wrong examples
    pred_right = []
    pred_wrong = []
    for i, (val_test, val_pred) in enumerate(zip(y_pred, y_test)):
        if val_test == test_class:
            if val_test == val_pred:
                pred_right.append((i, val_pred))
            else:
                pred_wrong.append((i, val_pred))
    
    print("Classified right: {} images".format(len(pred_right)))
    print("Classified wrong: {} images".format(len(pred_wrong)))
    
    # step 2: plotting random examples of right and wrong predictions
    plt.figure(figsize=(n_cols*1.3, 4))
    for row, predictions in enumerate([pred_right, pred_wrong]):
        for col, idx in enumerate(np.random.randint(0, len(predictions), n_cols)):
            i, pred_val = predictions[idx]
            plt.subplot(2, n_cols, n_cols*row+col+1)
            plt.imshow(X_test[i], cmap="binary")
            plt.axis('off')
            plt.title(class_names[pred_val], fontsize=12)
    plt.tight_layout()
    plt.show()