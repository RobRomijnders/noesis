"""Utility functions for the Memberschip Inference Attack"""
# pylint: disable=line-too-long,invalid-name
import random

import numpy as np
import torch
from sklearn import metrics


def get_auroc(
    bins: np.ndarray,
    hist1: np.ndarray,
    hist2: np.ndarray,
    fpr_thresh: float = 0.01):
  """Calculates the AUROC between two histograms.

  Assumes hist1 is the positive class and hist2 is the negative class.
  Therefore, hist2[thresholds > threshold] are falsely positive.
  hist1[thresholds > threshold] are true positives.
  """
  thresholds = bins[:-1]
  tpr_thresh = -1.

  fpr_list = []
  tpr_list = []
  for threshold in thresholds:
    # Assumes that above the threshold is the positive class, and that hist1 is the positive class
    # Therefore hist2[thresholds > threshold] are falsely positive
    fpr = np.sum(hist2[thresholds > threshold]) / np.sum(hist2)
    fpr_list.append(fpr)
    # Analogously, hist1[thresholds > threshold] are true positives
    tpr = np.sum(hist1[thresholds > threshold]) / np.sum(hist1)
    tpr_list.append(tpr)

    if fpr < fpr_thresh and tpr_thresh < 0:
      # Interpolate TPR at FPR threshold
      tpr_thresh = np.interp(fpr_thresh, fpr_list[-2:], tpr_list[-2:])

  tpr_list = np.array(tpr_list)
  fpr_list = np.array(fpr_list)

  return (
    100. * tpr_list,
    100. * fpr_list,
    100. * float(metrics.auc(fpr_list, tpr_list)),
    100. * tpr_thresh) # area under the curve


def set_seed(seed):
  """Sets the seed for reproducibility."""
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
