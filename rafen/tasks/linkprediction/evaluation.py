from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
from torch.utils.data import DataLoader

from rafen.embeddings.aggregation import EmbeddingAverageAggregation
from rafen.embeddings.keyedmodel import KeyedListModel, KeyedModel
from rafen.tasks.linkprediction import models
from rafen.tasks.linkprediction.dataset import (
    LinkPredictionDataset,
    LinkPredictionInMemoryDataset,
    convert_lpds_to_dict,
)
from rafen.tasks.linkprediction.models import LogisticRegressionModel


def compute_auc(
    embeddings: List[KeyedModel],
    lp_ds: LinkPredictionDataset,
    test_subset: Literal["test", "dev"],
    embedding_agg_model: EmbeddingAverageAggregation = None,
    n_jobs=1,
) -> Dict[str, Any]:

    emb = embeddings
    if embedding_agg_model:
        emb = embedding_agg_model.predict(embeddings)

    if not isinstance(emb, KeyedModel):
        raise ValueError(
            f"Obj emb must be instance of KeyedModel. Received type {type(emb)}"
        )

    lrm = LogisticRegressionModel(embedding=emb, args={"n_jobs": n_jobs})
    lrm.fit(x=lp_ds.x_train, y=lp_ds.y_train)

    if test_subset == "test":
        scores = lrm.validate(x=lp_ds.x_test, y=lp_ds.y_test).as_dict()
    elif test_subset == "dev":
        scores = lrm.validate(x=lp_ds.x_dev, y=lp_ds.y_dev).as_dict()
    else:
        raise ValueError(f"Unrecognized test subset: {test_subset}")
    return {
        "auc": scores["auc"],
        "f1-micro": scores["micro"]["f1-score"],
        "report": scores,
    }


def compute_auc_rnn(
    lp_ds: LinkPredictionInMemoryDataset,
    test_lp_ds: LinkPredictionInMemoryDataset,
    samples_train_len: int,
    samples_test_len: int,
    rnn_layer: str = "rnn",
    rnn_representation_mode: Literal["node", "edge"] = "node",
    batch_size: int = 128,
    share_rnn: bool = True,
    freeze_input_weight: bool = False,
    input_size: int = 128,
    epochs: int = 10,
    hidden_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    dropout_prob: float = 0.0,
    dropout_emb_prob: float = 0.0,
    quiet: bool = True,
    evaluate_train: bool = True,
) -> Dict[str, Any]:
    train_dataloader = DataLoader(
        np.arange(samples_train_len),
        batch_size=batch_size,
        collate_fn=lp_ds,
        pin_memory=False,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        np.arange(samples_test_len),
        batch_size=batch_size,
        collate_fn=test_lp_ds,
        pin_memory=False,
        shuffle=False,
    )
    model = models.RNNLinkPredictionClassifier(
        rnn_layer=rnn_layer,
        rnn_representation_mode=rnn_representation_mode,
        input_size=input_size,
        hidden_size=hidden_size,
        learning_args={"epochs": epochs},
        lr=lr,
        freeze_input_weight=freeze_input_weight,
        share_rnn=share_rnn,
        dropout_prob=dropout_prob,
        dropout_emb_prob=dropout_emb_prob,
        weight_decay=weight_decay,
        quiet=quiet,
    )
    log = model.fit(train_dataloader)
    metrics = model.validate(test_dataloader).as_dict()
    out_metrics = {"auc": metrics["auc"], **metrics, **log}

    if evaluate_train:
        metrics_train = model.validate(train_dataloader).as_dict()
        out_metrics.update({"metrics_train": metrics_train})

    return out_metrics


EVALUATION_FNS: Dict[str, Callable] = {
    "RNN": compute_auc_rnn,
    "LR": compute_auc,
}


def eval_model(
    embeddings: Union[List[KeyedModel], KeyedModel],
    lp_ds: LinkPredictionDataset,
    test_subset: Literal["test", "dev"],
    embedding_agg_model: Optional[EmbeddingAverageAggregation] = None,
    model: Literal["LR", "RNN"] = "LR",
    model_args: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
) -> Union[float, Dict[str, Any]]:
    if not isinstance(lp_ds.x_train, np.ndarray):
        return 0.0

    if model == "LR":
        return compute_auc(
            embeddings=embeddings,
            embedding_agg_model=embedding_agg_model,
            test_subset=test_subset,
            lp_ds=lp_ds,
            n_jobs=n_jobs,
        )
    elif model == "RNN":
        ds = convert_lpds_to_dict(lp_ds)
        samples_train = list(zip(ds["train"][0], ds["train"][1]))
        samples_test = list(zip(ds["test"][0], ds["test"][1]))
        klm = KeyedListModel(
            embeddings=embeddings,
            missing_attributes_mode=model_args.pop("missing_attributes_mode"),
        )

        lp_ds = LinkPredictionInMemoryDataset(
            dataset=samples_train, embeddings=klm
        )
        test_lp_ds = LinkPredictionInMemoryDataset(
            dataset=samples_test, embeddings=klm
        )
        return compute_auc_rnn(
            lp_ds=lp_ds,
            test_lp_ds=test_lp_ds,
            samples_train_len=len(samples_train),
            samples_test_len=len(samples_test),
            **model_args,
        )
    else:
        raise ValueError(f"Model {model} not supported!")
