"""Numerical helpers for CMR-RC."""
import numpy as np


def neighbors(G, j, diag, k, shrink=0., exponent=.5):
    a,b = G.indptr[j:j+2]
    ix, val = G.indices[a:b], G.data[a:b]
    keep = ix != j
    ix, val = ix[keep], val[keep]
    score = val / np.maximum(diag[ix],1e-12)**exponent
    if shrink: score *= val/(val+shrink)
    if len(ix)>k:
        take = np.argpartition(score,-k)[-k:]
        ix, val = ix[take],val[take]
    order=np.argsort(ix)
    return ix[order], val[order]
