import os
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent.absolute()

# Data Paths
DATA_PATH = ROOT_PATH / "data/"
RAW_GRAPH_DATA_PATH = DATA_PATH / "raw"
GRAPHS_PATH = DATA_PATH / "graphs"
FULL_GRAPHS_PATH = DATA_PATH / "full_graphs"
TG_GRAPHS_PATH = DATA_PATH / "tg_graphs"
FULL_TG_GRAPHS_PATH = DATA_PATH / "full_tg_graphs"
TEMPORAL_SCORES_PATH = DATA_PATH / "cached" / "temporal_scores"
LINK_PREDICTION_DATASETS_PATH = DATA_PATH / "lp_datasets"

# Random Walks & Embeddings Paths
RANDOM_WALKS_PATH = DATA_PATH / "random_walks"
EMBEDDINGS_PATH = DATA_PATH / "embeddings"
EMBEDDINGS_FULL_PATH = DATA_PATH / "embeddings_full"
NODE2VEC_PATH = EMBEDDINGS_PATH / "Node2Vec"

# Alignment paths
POSTHOC_PATH = DATA_PATH / "posthoc"
EMBEDDING_ALIGNERS_PATHS = [
    it
    for it in EMBEDDINGS_PATH.iterdir()
    if it.is_dir()
    and ("Node2Vec" in str(it.name))
    and (str(it.name) != "Node2Vec")
]
POSTHOC_ALIGNERS_PATHS = [it for it in POSTHOC_PATH.iterdir() if it.is_dir()]


# ALignment prev paths
PREV_EXPERIMENTS_PATH = DATA_PATH / "prev"
PREV_EMBEDDINGS_PATH = PREV_EXPERIMENTS_PATH / "embeddings"
PREV_POSTHOC_PATH = PREV_EXPERIMENTS_PATH / "posthoc"
TEMPORAL_SCORES_PREV_PATH = PREV_EXPERIMENTS_PATH / "cached" / "temporal_scores"

# Configs Paths
EXPERIMENTS_CFG_PATH = ROOT_PATH / "experiments" / "configs"
ALIGNERS_CFG_PATH = EXPERIMENTS_CFG_PATH / "models"
N2V_CFG_PATH = EXPERIMENTS_CFG_PATH / "models" / "Node2Vec.yaml"
COMMON_CFG_PATH = EXPERIMENTS_CFG_PATH / "models" / "common.yaml"
POSTHOC_COMMON_CFG_PATH = EXPERIMENTS_CFG_PATH / "posthoc" / "common.yaml"

# Evaluation Paths
EVALUATION_PATH = DATA_PATH / "evaluation"
LP_EVALUATION_RESULTS = EVALUATION_PATH / "lp"

# Alphas Grid Search
ALPHAS_GRID_SEARCH_RESULTS = DATA_PATH / "grid_search" / "alpha"
ALPHAS_GRID_SEARCH_GAE_RESULTS = DATA_PATH / "grid_search_gae" / "alpha"

# Method studies
METHOD_STUDY_PATH = DATA_PATH / "studies"
KNOWLEDGE_TRANSFER_STUDY_PATH = METHOD_STUDY_PATH / "knowledge_transfer.pkl"


def get_cached_random_walks_path(cache_path: str) -> Path:
    return Path(cache_path) / "data" / "random_walks"
