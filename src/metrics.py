import numpy as np
from sklearn.metrics import roc_curve


def calculate_eer(labels, scores):
    """caluculate_eer

    Calculating Equal Error Rate (EER).

    Args:
        labels(int): the label (0 or 1)
        scores(float): Probability.

    Returns:
        EER(float)

    Example:
        labels = [0, 1, 1, 0]
        scores = [0.1, 0.6, 0.3]
        EER = calculate_eer(labels, scores)


    """
    # Calculating FPR, TPR
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=0)
    fnr = 1 - tpr
    _ = threshold[np.argmin(abs(fnr - fpr))]

    # Calculate EER
    EER_fpr = fpr[np.argmin(np.absolute((fnr - fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr - fpr)))]
    EER = 0.5 * (EER_fpr + EER_fnr)

    return EER
