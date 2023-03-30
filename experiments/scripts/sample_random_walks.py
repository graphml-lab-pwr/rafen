import pandas as pd
import typer

from rafen.experiments.paths import (
    N2V_CFG_PATH,
    RANDOM_WALKS_PATH,
    TG_GRAPHS_PATH,
)
from rafen.experiments.runners import RandomWalksSamplerRunner
from rafen.utils.io import read_yaml

app = typer.Typer()

RANDOM_WALK_SAMPLER_ARGS = {
    "walk_length",
    "nb_walks_per_node",
    "p",
    "q",
    "batch_size",
    "w2v_epochs",
    "context_size",
    "num_negative_samples",
}


def main(
    dataset: str = typer.Option(..., help="Dataset name"),
    runs: int = typer.Option(..., help="Number of runs to perform"),
):
    # Read N2V config
    cfg = read_yaml(N2V_CFG_PATH)["args"][dataset]
    cfg = {k: v for k, v in cfg.items() if k in RANDOM_WALK_SAMPLER_ARGS}

    # Prepare args and read data
    output_path = RANDOM_WALKS_PATH / dataset
    output_path.mkdir(parents=True, exist_ok=True)

    tg_graphs = pd.read_pickle(TG_GRAPHS_PATH / f"{dataset}.pkl")[dataset][
        "graphs"
    ]
    runner = RandomWalksSamplerRunner(
        sampler_args=cfg,
        tg_graphs=tg_graphs,
        runs=runs,
        dataset=dataset,
        output_dir=output_path,
    )
    runner.run()


if __name__ == "__main__":
    typer.run(main)
