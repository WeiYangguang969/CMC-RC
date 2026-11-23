"""Load binary interactions and the supplied train/validation/test split."""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


class Dataset:
    def __init__(self, name, data_dir):
        self.name = name
        self.base = Path(data_dir) / name
        f = self.base / 'train.tsv'
        df = pd.read_csv(f, sep='\t', header=None, usecols=[0, 1])
        # Pandas groupby in upstream sorts users; integer set order in this
        # release is ascending for contiguous IDs. Store explicit maps anyway.
        self.users = np.sort(df[0].unique())
        self.items = np.sort(df[1].unique())
        self.um = {int(x): i for i, x in enumerate(self.users)}
        self.im = {int(x): i for i, x in enumerate(self.items)}
        u = df[0].map(self.um).to_numpy()
        i = df[1].map(self.im).to_numpy()
        self.X = sp.csr_matrix((np.ones(len(df), dtype=np.float32), (u, i)),
                              shape=(len(self.users), len(self.items)))
        self.X.sum_duplicates(); self.X.data[:] = 1; self.X.sort_indices()
        self.deg = np.asarray(self.X.sum(axis=0)).ravel()
        self.hist = np.diff(self.X.indptr)
        self.n_users, self.n_items = self.X.shape

    def labels(self, split='val'):
        f = self.base / ('val.tsv' if split == 'val' else 'test.tsv')
        df = pd.read_csv(f, sep='\t', header=None, usecols=[0, 1]).drop_duplicates()
        df = df[df[0].isin(self.users)]
        u = df[0].map(self.um).to_numpy()
        counts = np.bincount(u, minlength=self.n_users)
        known = df[1].isin(self.items).to_numpy()
        i = df.loc[known, 1].map(self.im).to_numpy()
        Y = sp.csr_matrix((np.ones(known.sum(), dtype=np.float32), (u[known], i)),
                          shape=self.X.shape)
        return Y, counts
