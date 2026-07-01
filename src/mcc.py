# Implementation of the Multinomial Classification Certainty (MCC) metric.
# Reference: Van Daalen et al. (2025)
#
# Formula: MCC = 1 - (p_beta / (p_yhat ** 2)) / n_classes
# where p_yhat  = highest predicted class probability
#       p_beta  = second highest predicted class probability
#       n_classes = number of classes

from src.config import N_CLASSES


def calculate_mcc(row, proba_columns, n_classes=N_CLASSES):
    """
    Compute the MCC score for a single prediction.

    A high MCC score means the model is confident.
    A low MCC score means competing classes have similar probabilities.

    Parameters
    ----------
    row           : one row from the predictions DataFrame
    proba_columns : list of column names containing class probabilities
    n_classes     : number of classes (default: 3)

    Returns
    -------
    float : MCC score, or -inf if p_yhat is 0
    """
    probas = row[proba_columns].values
    p_yhat = probas.max()
    p_beta = sorted(probas)[-2]

    if p_yhat == 0:
        return float("-inf")

    return 1 - (p_beta / (p_yhat ** 2)) / n_classes