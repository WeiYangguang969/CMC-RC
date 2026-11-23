# Data

Run `python download_data.py` from the repository root. Supported datasets are `amazon-cds`, `douban`, `gowalla`, and `yelp2018`.

```text
data/<dataset>/
  train.tsv
  val.tsv
  test.tsv
```

Files have no header. The first two tab-separated columns are integer user and item IDs. Additional columns are ignored. Duplicate interactions are treated as a single binary interaction.

The files come from [Mind the Metric](https://github.com/sisinflab/Mind_the_Metric_SIGIR-26), commit `925c80e387b814a7bd7e29406de1ca03cc54cc82`.
The exact upstream commit and file hashes used by the downloader are recorded in `download_data.py` and `manifest.json`.

For an existing data directory, pass `--data-dir <directory>` to `main.py`. The directory must contain `<dataset>/train.tsv`, `val.tsv`, and `test.tsv`. Use the supplied splits to reproduce the reported results.
