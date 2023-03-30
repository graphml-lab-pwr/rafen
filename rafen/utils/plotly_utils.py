from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go


def plot_loss_plot(
    df: pd.DataFrame,
    x_col: str,
    y_mean_col: Union[str, Tuple[str, str]],
    y_std_col: Union[str, Tuple[str, str]],
    title: str,
    cmap: Optional[List[str]] = plotly.colors.qualitative.Set1,
) -> go.Figure:
    def add_plotly_line(sub_df: pd.DataFrame, label: str, color: str):
        fig.add_scatter(
            name=label,
            x=sub_df[x_col],
            y=sub_df[y_mean_col],
            mode="lines",
            line=dict(color=color),
            legendgroup=label,
        )
        fig.add_scatter(
            name=label,
            x=sub_df[x_col],
            y=sub_df[y_mean_col] + sub_df[y_std_col],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            legendgroup=label,
            hoverinfo="skip",
        )
        fig.add_scatter(
            name=label,
            x=sub_df[x_col],
            y=sub_df[y_mean_col] - sub_df[y_std_col],
            line=dict(width=0),
            mode="lines",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.3)"),
            fill="tonexty",
            showlegend=False,
            hoverinfo="skip",
            legendgroup=label,
            marker={"colorscale": "Viridis"},
        )

    fig = go.Figure()

    for snap_idx, snapshot_df in df.groupby(level=0):
        snap_idx_label = f"{snap_idx}"
        if not str(snap_idx).isnumeric():
            snap_idx_label = snap_idx
            snap_idx = 0
        snap_idx: int = int(snap_idx)
        snapshot_df = snapshot_df.reset_index()
        add_plotly_line(
            sub_df=snapshot_df,
            label=f"Snapshot {snap_idx_label}",
            color=cmap[snap_idx % len(cmap)],
        )

    epoch_num = df.reset_index().epoch.max()
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=np.arange(0, epoch_num + 1),
            ticktext=np.arange(0, epoch_num + 1),
        ),
        yaxis_title="Loss",
        xaxis_title="Epoch",
        title=title,
        hovermode="x",
    )

    return fig
