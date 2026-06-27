# STRIDE

This repository provides the code used for STRIDE, a structured relation-space inference framework for multi-view DDI event prediction.

## Contents

- `train.py`: training and evaluation entry point.
- `model.py`: STRIDE model, relation reasoning modules, and decoders.
- `engine.py`: training loop, metrics, and evaluation utilities.
- `data.py`: dataset loading, relation-family loading, and relation statistics.
- `prepare.py`: utilities for preparing PrimeKG+ and long-tail inputs.
- `preprocess.py`: molecular graph preprocessing utilities.
- `graph_transformer.py`: graph encoder components.

## Environment

The validated environment uses Python 3.9, PyTorch 1.12.1 with CUDA 11.3, and PyG 2.6.1. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Data Layout

By default, the scripts expect the following local layout:

```text
data/
  primekg+ryu/
    ddi/
      drug_listxiao.csv
      relation_family.csv
      0/
        ddi_training1xiao.csv
        ddi_validation1xiao.csv
        ddi_test1xiao.csv
      ...
    kg/
      nodes.tsv
      edges.tsv
  controlled_longtail/
    ryu/
      LT100/
        ddi_training1.csv
        ddi_validation1.csv
        ddi_test1.csv
        relation_family.csv
```

The `primekg+deng` bundle follows the same structure.

## Run

Run the standard PrimeKG+Deng experiment:

```bash
python train.py --data_bundle primekg+deng --protocol standard
```

Run the LT-100 long-tail experiment:

```bash
python train.py --data_bundle primekg+deng --protocol longtail
```

Use `--fold 0` through `--fold 4` for a single official standard fold. The default `--fold 5` runs all five standard folds. The long-tail protocol uses one predefined split and therefore also uses `--fold 5`.

The Ryu bundle can be selected with:

```bash
python train.py --data_bundle primekg+ryu --protocol standard
```

## Outputs

Checkpoints and summaries are written to:

```text
outputs/<protocol>/fold_<selection>/seed_<seed>/
```

## Notes

Data and checkpoints at https://drive.google.com/drive/folders/1rPPjrL1otvnwCWtFz4yvHmnD1v7qv9_c?usp=sharing
