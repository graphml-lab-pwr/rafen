from typing import Any, Dict, List, Union

import pandas as pd


def convert_ds_metadata_to_df(
    metadata: List[List[Dict[str, Union[float, List[float]]]]]
) -> pd.DataFrame:
    def get_metric_value(
        m_name: str, m_values: List[Any]
    ) -> Union[Any, List[Any]]:
        if m_name in ("loss", "l2_ref_nodes_distance") and isinstance(
            m_values, list
        ):
            return m_values[-1]
        return m_values

    df_data = []
    for run_id, run_data in enumerate(metadata):
        for snapshot_id, snapshot_data in enumerate(run_data):
            for metric_name, metric_values in snapshot_data.items():
                df_data.append(
                    {
                        "run": run_id + 1,
                        "snapshot": str(snapshot_id),
                        "metric_name": metric_name,
                        "value": get_metric_value(metric_name, metric_values),
                    }
                )

    return pd.DataFrame(df_data)


def convert_loss_metadata_to_df(
    metadata: List[List[Dict[str, Union[float, List[float]]]]],
    metric_name: str = "loss",
) -> pd.DataFrame:
    df_data = []
    for run_id, run_data in enumerate(metadata):
        for snapshot_id, snapshot_data in enumerate(run_data):
            for epoch, epoch_loss in enumerate(snapshot_data[metric_name]):
                df_data.append(
                    {
                        "run": run_id + 1,
                        "snapshot": str(snapshot_id),
                        "epoch": epoch,
                        "metric_name": metric_name,
                        "value": epoch_loss,
                    }
                )

    return (
        pd.DataFrame(df_data)
        .drop(["run", "metric_name"], axis=1)
        .groupby(["snapshot", "epoch"])
        .agg(("mean", "std"))
    )
