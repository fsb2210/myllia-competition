# Myllia | Echoes of Silenced Genes: A Cell Challenge
---

This repository contains every step needed to reproduce our submissions for the [Myllia| Echoes of Silenced Genes: A Cell Challenge](https://www.kaggle.com/competitions/echoes-of-silenced-genes)
challenge hosted on [kaggle](https://www.kaggle.com).

A more in depth look can be found in this [blog post](https://fsb2210.github.io/posts/echoes-of-silenced-genes/).

**Authors**: [fsb2210](https://www.kaggle.com/fsb2210), [julianc93](https://www.kaggle.com/julianc93).

## Repository structure

```sh
myllia-competition
├── config
│   └── fig.mplstyle  # style for matplotlib figures
├── notebooks
│   ├── building-features.ipynb    # step 1
│   ├── diffusion-model.ipynb      # deprecated model
│   ├── features2ml.ipynb          # step 2
│   ├── gene-embeddings.ipynb      # step 0.5
│   ├── preprocessing-steps.ipynb  # step 0
│   ├── scRNA-seq.ipynb            # misc., for data understanding
│   └── validation-strategy.ipynb  # misc.
├── README.md
└── src
    ├── download_datasets.py
    └── metric.py
```

### Files worth mentioning

* `download_datasets.py`: script used to download external datasets,
* `metric.py`: metric used by the challenge to score a submission file.

### From raw data to genes response predictions

The order of execution of the jupyter notebooks is as follows:

0. download external datasets,
1. `preprocessing-steps`: standarize data across datasets,
2. `gene-embeddings`: compute ESM2 gene embeddings for *every* perturbation in each dataset,
3. `building-features`: create features from raw data,
4. `features2ml`: machine learning model training, plus predictions on validation set.

## Disclamer

The structure of this research project follows the philosophy of [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science).
