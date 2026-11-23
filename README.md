# CMR-RC

Code for **Target-Conditioned Item Reconstruction with Residual Filtering for Top-N Recommendation**.

CMR-RC learns sparse item reconstruction coefficients using target-conditioned co-occurrences, then applies user- and item-side residual filtering.

## Requirements

Python 3.12. The method runs on CPU.

```bash
pip install -r requirements.txt
```

## Data

Download the four preprocessed datasets:

```bash
python download_data.py
```

To download one dataset, use `python download_data.py --dataset douban`. Files are placed in `data/<dataset>/`. The downloader uses the fixed [Mind the Metric benchmark splits](https://github.com/sisinflab/Mind_the_Metric_SIGIR-26/tree/925c80e387b814a7bd7e29406de1ca03cc54cc82). See [data/README.md](data/README.md) for the format.

## Run

```bash
python main.py --dataset amazon-cds
python main.py --dataset douban
python main.py --dataset gowalla
python main.py --dataset yelp2018
```

Each command loads `configs/<dataset>.json`, fits the model on the training set, and reports Recall and NDCG at 10 and 20. Metrics and Top-20 recommendations are saved to `outputs/`. Fitted coefficients and graph operators are cached in `cache/`.

For validation evaluation or parameter changes:

```bash
python main.py --dataset yelp2018 --split val
python main.py --dataset yelp2018 --nu 0
python main.py --dataset yelp2018 --gamma 0 --beta 0
```

`--nu 0` disables conditioning. `--gamma 0 --beta 0` disables residual correction. Other options include `--reg`, `--k`, `--workers`, `--config`, and `--data-dir`. These overrides keep the remaining settings fixed; they do not rerun model selection.

## Results

Recorded results with the supplied configurations:

| Dataset | Recall@20 | NDCG@20 |
| --- | ---: | ---: |
| Amazon-CDs | 0.1461 | 0.0921 |
| Douban | 0.2148 | 0.1913 |
| Gowalla | 0.2140 | 0.1580 |
| Yelp2018 | 0.1014 | 0.0788 |

Evaluation ranks the full training catalog and masks training interactions. Validation interactions are not added to training. Held-out positives outside the training catalog remain in metric denominators. The recorded experiments used Linux; small ranking differences can occur across numerical libraries and platforms.

## Files

```text
main.py           Training and evaluation entry
model.py          Conditional reconstruction and residual filtering
dataloader.py     Interaction data loading
metrics.py        Recall/NDCG evaluation
configs/          Parameters for the four datasets
utils/            Solvers and optional C++ acceleration
data/             Data format and download manifest
download_data.py  Benchmark data download
```

The optional C++ kernels use `g++` when available; NumPy/SciPy fallbacks are provided. Dataset rights remain with the original providers. Related implementations: [HiGSP](https://github.com/Yaveng/HiGSP), [SVD-AE](https://github.com/seoyoungh/svd-ae), and [DEQL](https://github.com/coderaBruce/DEQL).
