"""Full-catalog Recall and NDCG at 10 and 20."""
import numpy as np
import scipy.sparse as sp

METRICS = ['Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20']


def metric_values(top, labels, users=None):
    Y, count = labels
    if users is None: users = np.arange(Y.shape[0])
    hit = Y[users[:,None], top]
    hit = hit.toarray() if sp.issparse(hit) else np.asarray(hit)
    vals = np.empty((len(users),4), dtype=np.float64)
    discount = 1/np.log2(np.arange(2,22))
    prefix = np.r_[0., np.cumsum(discount)]
    for a,k in [(0,10),(2,20)]:
        denom = np.maximum(count[users],1)
        vals[:,a] = hit[:,:k].sum(axis=1)/denom
        idcg = prefix[np.minimum(count[users],k)]
        vals[:,a+1] = (hit[:,:k] * discount[:k]).sum(axis=1) / np.maximum(idcg,1e-30)
    valid = count[users] > 0
    return dict(zip(METRICS, vals[valid].mean(axis=0))), vals, valid


def rank_scores(scores, X, user_ids, k=20):
    scores = np.asarray(scores)
    seen = X[user_ids].tocoo()
    scores[seen.row, seen.col] = -np.inf
    # Same argpartition + sorted top-k behavior as the upstream CPU models.
    ix = np.argpartition(scores, -k, axis=1)[:,-k:]
    val = np.take_along_axis(scores,ix,axis=1)
    return np.take_along_axis(ix,np.argsort(-val,axis=1,kind='stable'),axis=1)
