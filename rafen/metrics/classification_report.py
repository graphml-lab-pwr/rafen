import warnings
from collections import OrderedDict

import numpy as np
from sklearn import metrics

from rafen.utils.misc import check_arr_is_ndarray


class ClassificationReport:
    """Class that provides various classification metrics.

    Provided statistics: (precision, recall, f1-score, accuracy, auc)
    """

    def __init__(self, y_pred, y_true, y_score=None):
        """Inits class and calculates metrics.

        Only if y_score is given, then AUC metric will be calculated.
        :param y_true: Ground truth labels
        :type y_true: np.ndarray
        :param y_pred: Predicted labels
        :type y_pred: np.ndarray
        :param y_score: Predicted probabilities or scores (default: None)
        :type y_score: np.ndarray
        """
        check_arr_is_ndarray(y_pred, "y_pred")
        check_arr_is_ndarray(y_true, "y_true")
        if y_score is not None:
            check_arr_is_ndarray(y_score, "y_score")

        self.labels = np.unique(y_true)
        self.item_count = len(y_true)

        self.nb_classes = len(self.labels)
        self.cm = metrics.confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
        )

        self._metrics_dict = self._generate_report(
            y_true=y_true, y_pred=y_pred, y_score=y_score
        )

    def _generate_report(self, y_true, y_pred, y_score):
        """Calculates metrics."""
        prfs = metrics.precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, labels=self.labels
        )

        if self.cm.shape == (1, 1):
            accuracy_per_class = [1]
        else:
            accuracy_per_class = np.array(
                [
                    self.cm[label_id][label_id] / np.sum(self.cm[label_id])
                    for label_id in self.labels
                ]
            )

        calculated_metrics = OrderedDict(
            [
                ("accuracy", accuracy_per_class),
                ("precision", prfs[0]),
                ("recall", prfs[1]),
                ("f1-score", prfs[2]),
                ("support", prfs[3]),
            ]
        )

        metrics_dict = {
            class_id: OrderedDict(
                [
                    (metric_nm, calculated_metrics[metric_nm][class_id])
                    for metric_nm in calculated_metrics.keys()
                ]
            )
            for class_id in range(self.nb_classes)
        }

        if y_score is not None:
            if self.nb_classes == 2:
                auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
            else:
                warnings.warn(
                    "AUC is defined only for two classes scenario!",
                    RuntimeWarning,
                )
                auc = None

            metrics_dict["auc"] = auc

        # Calculate averages
        metrics_dict["micro"] = self._calculate_micro_metrics(
            y_true=y_true, y_pred=y_pred
        )

        metrics_dict["macro"] = {
            metric_nm: np.mean(
                [
                    metrics_dict[class_id][metric_nm]
                    for class_id in range(self.nb_classes)
                ]
            )
            for metric_nm in calculated_metrics.keys()
            if metric_nm != "support"
        }

        metrics_dict["weighted"] = {
            metric_nm: np.sum(
                [
                    metrics_dict[class_id][metric_nm]
                    * (metrics_dict[class_id]["support"] / self.item_count)
                    for class_id in range(self.nb_classes)
                ]
            )
            for metric_nm in calculated_metrics.keys()
            if metric_nm != "support"
        }

        return metrics_dict

    def _calculate_micro_metrics(self, y_true, y_pred):
        """Calculated micro averaged metrics."""
        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        prfs = metrics.precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, labels=self.labels, average="micro"
        )
        micro = OrderedDict(
            [
                ("accuracy", accuracy),
                ("precision", prfs[0]),
                ("recall", prfs[1]),
                ("f1-score", prfs[2]),
            ]
        )
        return micro

    def as_dict(self):
        """Returns dictionary with metrics.

        :return: Metrics dictionary
        :rtype: dict
        """
        return self._metrics_dict

    def as_flat_dict(self):
        """Presents all metrics a flat dict (metrics per class + avg).

        :return: Flattened metrics dict
        :rtype: dict
        """
        flat_dict = {}

        for key, keyed_metrics in self._metrics_dict.items():
            if isinstance(keyed_metrics, dict):
                for mt_name, mt_value in keyed_metrics.items():
                    flat_dict[f"{key}_{mt_name}"] = mt_value.item()
            elif isinstance(keyed_metrics, float):
                flat_dict[key] = keyed_metrics.item()

        return flat_dict

    def __str__(self):
        """Creates string representation of ClassificationReport object."""
        rp = ClassificationReportPrinter(
            nb_classes=self.nb_classes,
            labels=self.labels,
            metrics_dict=self._metrics_dict,
        )

        return rp.as_str()


class ClassificationReportPrinter:
    """Prints report in eye friendly readable format.

    Styled similarly as sklearn.metrics.classification_report.
    """

    def __init__(self, nb_classes, labels, metrics_dict):
        """Inits the class.

        :param nb_classes: Number of classes
        :type nb_classes: int
        :param labels: Class labels
        :type labels: np.ndarray
        :param metrics_dict: Input metrics
        :type metrics_dict: dict
        """
        self._metrics = metrics_dict
        self.nb_classes = nb_classes
        self.labels = labels
        self.display_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1-score",
            "support",
        ]

    def _write_avg_metrics(self, avg_name):
        """Writes metrics average.

        :param avg_name: Averaging name e.g. micro
        :type avg_name: str
        :return: String with averages
        :rtype: str
        """
        missing_metric_str = "-"

        micro_avg_header = f"{avg_name} avg"
        avg_str = f"{micro_avg_header:>15} "

        if self.nb_classes > 1:
            for m_name in self.display_metrics:
                if m_name != "support":
                    avg_str += f"{self._metrics[avg_name][m_name]:>10.2f} "
                else:
                    avg_str += f"{missing_metric_str:>10}"
        else:
            for _ in range(len(self.display_metrics)):
                avg_str += f"{missing_metric_str:>10} "

        avg_str += "\n"
        return avg_str

    def as_str(self):
        """Presents all metrics as string."""
        # Write header
        header_name = "class"
        header = f"{header_name:>15} "
        for metric_name in self.display_metrics:
            header += f"{metric_name:>10} "
        header += "\n"

        # Write metrics body
        body = ""
        for i in range(self.nb_classes):
            # Write class name
            body += f"{self.labels[i]:>15} "
            for metric_name in self.display_metrics:
                body += f"{self._metrics[i][metric_name]:>10.2f} "
            body += "\n"
        body += "\n"

        # Write metrics averages

        # Micro average
        micro_avg = self._write_avg_metrics(avg_name="micro")
        macro_avg = self._write_avg_metrics(avg_name="macro")
        weighted_avg = self._write_avg_metrics(avg_name="weighted")

        auc_str = ""
        # auc
        if "auc" in self._metrics.keys():
            auc_header = "AUC"
            auc_score = self._metrics["auc"]
            auc_str = f"\n{auc_header:>15} "
            if auc_score is not None:
                auc_str += f"{auc_score:>10.2f} "
            else:
                auc_str += "undefined"

        return header + body + micro_avg + macro_avg + weighted_avg + auc_str
