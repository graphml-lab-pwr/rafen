# RAFEN - Regularized Alignment Framework for Embeddings of Nodes
This is the official code for the article "Regularized Alignment Framework for Embeddings of Nodes" (Kamil Tagowski, Piotr Bielak, Jakub Binkowski, Tomasz Kajdanowicz). The paper is currently under review.

## Code information
The implementation of experiments is based on [FILDNE](https://gitlab.com/fildne/fildne) and [Embeddings Alignment](https://gitlab.com/tgem/embedding-alignment)


Arxiv: [[LINK]](https://arxiv.org/pdf/2303.01926v1.pdf)

## Installation
All experiments were performed using Python 3.9, and for GPU computations, we used CUDA 11.3.
To install dependencies, use poetry

```
poetry install
```


## Reproducibility
We employ [DVC](https://dvc.org/) pipelines for reproducible experiments. All scripts and configurations are in the `experiments/` folder.
The pipeline definition can be found in the [dvc.yaml](dvc.yaml) file.  We share source graphs via google drive (same output as via `/data/raw/real.dvc` stage) [[LINK]](https://drive.google.com/file/d/1Srx20_aifw7d5tOwQq2xwneO2SewEgNL/view?usp=sharing). After downloading source graphs and putting them in the `data/raw/real/` directory, experiments can be easily reproducible via the `dvc repro` command.


## Contact
Due to the size of experiments 3 TB+, we do not publicly share DVC outputs. However, in case of any needed data and help, do not hesitate to contact us:
* Kamil `kamil [dot] tagowski [at] pwr [dot] edu [dot] pl`.   
(Replace the `[dot]`s and `[at]`s with proper punctuation)

## Citation
TBD
