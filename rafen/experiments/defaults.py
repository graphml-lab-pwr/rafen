from rafen.experiments.paths import (
    EMBEDDING_ALIGNERS_PATHS,
    POSTHOC_ALIGNERS_PATHS,
)

BASE_L2_ALIGNER = "Node2VecAligned_L2_ALL"
NODE2VEC = "Node2Vec"
METRICS = [
    "loss",
    "alignment_loss",
    "n2v_loss",
    "l2_ref_nodes_distance",
    "calculation_time",
]
DATASETS = [
    "fb-forum",
    "fb-messages",
    "bitcoin-alpha",
    "bitcoin-otc",
    "ppi",
    "ogbl-collab",
]

ALIGNERS = [it.name for it in EMBEDDING_ALIGNERS_PATHS if "Node2Vec" in str(it)]
L2_ALIGNERS = [it for it in ALIGNERS if "L2" in it]
POSTHOC_ALIGNERS = [it.name for it in POSTHOC_ALIGNERS_PATHS]
ALL_ALIGNERS = [*ALIGNERS, *POSTHOC_ALIGNERS]
