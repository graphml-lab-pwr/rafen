from typing import Any, Dict, Final, Literal, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn import linear_model as sk_lm
from torch.utils.data import DataLoader
from tqdm import tqdm

from rafen.embeddings.edge_representation import (
    calculate_edge_embedding,
    hadamard_op,
)
from rafen.metrics.classification_report import ClassificationReport
from rafen.models.base import BaseTorchModel


class LogisticRegressionModel:
    """Edge Classification model based on logistic regression."""

    def __init__(self, embedding, args=None, operator=hadamard_op):
        """Inits model with embedding data."""
        args = args if args else {}

        self._clf = sk_lm.LogisticRegression(solver="liblinear", **args)
        self._ee_op = operator
        self._emb = embedding

    def fit(self, x, y):
        """Fits model with data.

        :param x: Input edges
        :type x: np.ndarray
        :param y: Input labels
        :type y: np.ndarray
        """
        self._clf.fit(X=self._to_edge_emb(x), y=y.transpose()[0])

    def predict(self, x):
        """Predicts the existence of given edges.

        :param x: Input edges
        :type x: np.ndarray
        :return: Predictions
        :rtype: np.ndarray
        """
        return self._clf.predict(self._to_edge_emb(x))

    def predict_proba(self, x):
        """Predicts scores.

        :param x: Input edges
        :type x: np.ndarray
        :return: Predictions
        :rtype: np.ndarray
        """
        return self._clf.predict_proba(self._to_edge_emb(x))

    def _to_edge_emb(
        self,
        x,
    ):
        """Returns embedding of the edge.

        :param x: Input edges
        :rtype x: list
        :return: List of edge embeddings
        :rtype: np.ndarray
        """
        return np.array(calculate_edge_embedding(self._emb, x, self._ee_op))

    def validate(self, x, y):
        """Validates data and returns classification report.

        :param x: Input edges
        :type x: np.ndarray
        :param y: Input labels
        :type y: np.ndarray
        :return: Classification metrics
        :rtype ClassificationReport
        """
        return ClassificationReport(
            y_true=y,
            y_pred=self.predict(x),
            y_score=self.predict_proba(x)[:, 1],
        )


class RNNPairClassifier(nn.Module):
    RNN_LAYERS: Final = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        device: str,
        dropout_prob: float,
        dropout_emb_prob: float,
        freeze_rnn_input_weight: bool,
        share_rnn: bool,
        rnn_representation_mode: Literal["node", "edge"],
        rnn_layer: str,
    ):
        super().__init__()
        if rnn_representation_mode not in {"node", "edge"}:
            raise ValueError(
                f"RNN representation mode {rnn_representation_mode} unsupported!"
            )
        self.hidden_size = hidden_size
        self.rnn_layer_module = self.RNN_LAYERS[rnn_layer]
        self.rnn_representation_fn = (
            self.rnn_node
            if rnn_representation_mode == "node"
            else self.rnn_edge
        )
        self.rnn_hidden_size = (
            self.hidden_size
            if rnn_representation_mode == "node"
            else self.hidden_size * 2
        )

        self.input_size = input_size
        self.share_rnn = share_rnn
        self.rnn = self.rnn_layer_module(
            hidden_size=self.rnn_hidden_size,
            input_size=input_size,
            batch_first=True,
            num_layers=1,
        )
        if rnn_representation_mode == "node" and not self.share_rnn:
            self.rnn_2 = self.rnn_layer_module(
                hidden_size=self.rnn_hidden_size,
                input_size=input_size,
                batch_first=True,
                num_layers=1,
            )
        self.dense = nn.Linear(2 * self.hidden_size, out_features=1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_emb = nn.Dropout(p=dropout_emb_prob)
        self.device = device

        if freeze_rnn_input_weight and rnn_layer == "rnn":
            self.freeze_rnn_input_weight("rnn")
            if not self.share_rnn:
                self.freeze_rnn_input_weight("rnn_2")

    def init_hidden(
        self, batch_size
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.rnn_layer_module in (nn.RNN, nn.GRU):
            return torch.zeros(1, batch_size, self.rnn_hidden_size).to(
                self.device
            )
        else:
            return (
                torch.zeros(1, batch_size, self.rnn_hidden_size).to(
                    self.device
                ),
                torch.zeros(1, batch_size, self.rnn_hidden_size).to(
                    self.device
                ),
            )

    def freeze_rnn_input_weight(self, layer_name: str) -> None:
        layer = getattr(self, layer_name)
        layer.weight_hh_l0 = torch.nn.Parameter(
            torch.zeros(self.rnn.weight_hh_l0.shape)
        )
        layer.weight_hh_l0.requires_grad = False
        layer.bias_ih_l0 = torch.nn.Parameter(
            torch.zeros(self.rnn.bias_ih_l0.shape)
        )
        layer.bias_ih_l0.requires_grad = False

    def rnn_node(self, batch) -> torch.Tensor:
        n1_rep = batch[:, 0, :, :]
        n2_rep = batch[:, 1, :, :]

        n1_rep = self.dropout_emb(n1_rep)
        n2_rep = self.dropout_emb(n2_rep)

        h_0 = self.init_hidden(len(batch))

        n1_rep, _ = self.rnn(n1_rep, h_0)
        if self.share_rnn:
            n2_rep, _ = self.rnn(n2_rep, h_0)
        else:
            n2_rep, _ = self.rnn_2(n2_rep, h_0)

        n1_rep = n1_rep[:, -1, :]
        n2_rep = n2_rep[:, -1, :]
        rep = torch.cat([n1_rep, n2_rep], axis=1)
        return rep

    def rnn_edge(self, batch) -> torch.Tensor:
        n1_rep = batch[:, 0, :, :]
        n2_rep = batch[:, 1, :, :]
        rep = torch.cat([n1_rep, n2_rep], axis=1)
        rep = self.dropout_emb(rep)

        h_0 = self.init_hidden(len(batch))

        rep, _ = self.rnn(rep, h_0)
        rep = rep[:, -1, :]
        return rep

    def forward(self, batch):
        rep = self.rnn_representation_fn(batch)
        rep = self.dropout(rep)
        pred = self.dense(rep)

        return pred.sigmoid()


class LearningArgsMetadata(TypedDict):
    epochs: int


class RNNLinkPredictionClassifier(BaseTorchModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        learning_args: LearningArgsMetadata,
        share_rnn: bool,
        freeze_input_weight: bool,
        rnn_representation_mode: Literal["edge", "node"],
        rnn_layer: str,
        dropout_prob: float,
        dropout_emb_prob: float,
        lr: float,
        weight_decay: float,
        quiet: bool = False,
        device: str = "cuda:0",
    ):
        super().__init__(quiet=quiet)
        self.model = RNNPairClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            share_rnn=share_rnn,
            dropout_prob=dropout_prob,
            dropout_emb_prob=dropout_emb_prob,
            device=device,
            freeze_rnn_input_weight=freeze_input_weight,
            rnn_representation_mode=rnn_representation_mode,
            rnn_layer=rnn_layer,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss = torch.nn.BCELoss()
        self.learning_args = learning_args

    def training_step(self, batch, batch_id):
        predictions = self.model(batch[0])
        y_true = batch[1]
        loss = self.loss(input=predictions, target=y_true)
        self.epoch_log["loss"]["values"].append(loss.detach().cpu().numpy())
        return loss

    def fit(self, train_dataloader: DataLoader) -> Dict[str, Any]:
        self.loader = train_dataloader
        return self._fit()

    def predict(
        self, test_dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        y_pred = []
        y_true = []

        for step, batch in enumerate(
            tqdm(
                test_dataloader,
                desc="Test batch",
                leave=False,
                disable=self.quiet,
            )
        ):
            with torch.no_grad():
                pred = self.model(batch[0]).view(-1)
                true = batch[1]

            y_pred.append(pred.detach().cpu())
            y_true.append(true.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0).numpy()
        y_true = torch.cat(y_true, dim=0).numpy()
        return y_pred, y_true

    def validate(self, test_dataloader: DataLoader) -> ClassificationReport:
        y_pred, y_true = self.predict(test_dataloader)
        return ClassificationReport(
            y_true=y_true.astype(int),
            y_pred=y_pred.round().astype(int),
            y_score=y_pred,
        )
