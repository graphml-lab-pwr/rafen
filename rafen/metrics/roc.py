import numpy as np
from sklearn import metrics


class ROCMetric:
    """ROC Metric utilities.

    Provides statistics: (tpr_percentile, fpr_percentile)
    """

    def __init__(self, y_true, y_score):
        """Inits class.

        :param y_true: True labels
        :param y_score: Scores
        """
        y_true = np.array(y_true)
        y_score = np.array(y_score)

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(
            y_true=y_true, y_score=y_score
        )

    def get_tpr_interpolation(self, x):
        """Returns tpr of unknown fpr x value based on linear interpolation.

        :param x: Input unknown value of point x (range 0-1)
        :type x: float
        :return: Interpolated value
        :rtype: float
        """
        return np.interp(x, self.fpr, self.tpr)

    def get_fpr_interpolation(self, x):
        """Returns fpr of unknown tpr x value based on linear interpolation.

        :param x: Input unknown value of point x (range 0-1)
        :type x: float
        :return: Interpolated value
        :rtype: float
        """
        return np.interp(x, self.tpr, self.fpr)

    def get_tpr_percentile(self, percentile):
        """Returns tpr ratio of given percentile.

        :param percentile: Input percentile (range 0-100)
        :type percentile: float
        :return: TPR ratio of given percentile
        :rtype: float
        """
        return np.percentile(a=self.tpr, q=percentile)

    def get_fpr_percentile(self, percentile):
        """Returns fpr ratio of given percentile.

        :param percentile: Input percentile (range 0-100)
        :type percentile: float
        :return: FPR ratio of given percentile
        :rtype: float
        """
        return np.percentile(a=self.fpr, q=percentile)
