"""Load MUSAE / FEATHER social graphs, build 128-d SVD features, and imbalanced splits."""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

RAW = {
    "facebook": dict(root="musae/input", edges="edges/facebook_edges.csv", feats="features/facebook.json",
                     target="target/facebook_target.csv", label="page_type", idcol="id"),
    "github": dict(root="musae/input", edges="edges/git_edges.csv", feats="features/git.json",
                   target="target/git_target.csv", label="ml_target", idcol="id"),
    "twitch_en": dict(root="musae/input", edges="edges/ENGB_edges.csv", feats="features/ENGB.json",
                      target="target/ENGB_target.csv", label="mature", idcol="new_id"),
    "lastfm": dict(root="FEATHER/new_node_level_dataset", edges="edges/lastfm_asia_edges.csv",
                   feats="features/lastfm_asia_features.json", target="target/lastfm_asia_target.csv",
                   label="target", idcol="id"),
}
NAMES = {"facebook": "Facebook Page-Page", "github": "GitHub Developers",
         "twitch_en": "Twitch-EN", "lastfm": "LastFM Asia"}


def build(name, raw_dir, out_dir, dim=128, seed=0):
    """Read raw files, return and cache dict(A, X, y)."""
    out = os.path.join(out_dir, f"{name}.npz")
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True)
        A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        return dict(A=A, X=z["X"], y=z["y"], classes=list(z["classes"]))
    c = RAW[name]
    root = os.path.join(raw_dir, c["root"])
    tg = pd.read_csv(os.path.join(root, c["target"]))
    ids = tg[c["idcol"]].astype(int).values
    n = ids.max() + 1
    lab = tg[c["label"]].astype(str).values
    classes = sorted(set(lab))
    y = np.full(n, -1)
    y[ids] = [classes.index(v) for v in lab]
    e = pd.read_csv(os.path.join(root, c["edges"])).values.astype(int)
    e = e[e[:, 0] != e[:, 1]]
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    with open(os.path.join(root, c["feats"])) as f:
        fj = json.load(f)
    rows, cols = [], []
    for k, v in fj.items():
        rows += [int(k)] * len(v)
        cols += [int(t) for t in v]
    F = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, max(cols) + 1))
    F.data = np.minimum(F.data, 1.0)
    X = TruncatedSVD(dim, random_state=seed).fit_transform(F)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)  # unit rows: B = 1
    assert (y >= 0).all()
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out, data=A.data, indices=A.indices, indptr=A.indptr, shape=A.shape,
                        X=X.astype(np.float64), y=y, classes=np.array(classes))
    return dict(A=A, X=X, y=y, classes=classes)


def imbalanced_split(y, seed, n_major=20, ratio=10, n_val=30, minority=None):
    """Long-tail semi-supervised split. Minority classes = smaller half of classes (by frequency)
    unless given. Majority classes get n_major labels, minority get max(1, n_major // ratio)."""
    rng = np.random.default_rng(seed)
    C = y.max() + 1
    counts = np.bincount(y, minlength=C)
    if minority is None:
        order = np.argsort(counts)  # rarest first
        minority = sorted(order[: max(1, C // 2)].tolist())
    n_min = max(1, n_major // ratio)
    train, val = [], []
    for c in range(C):
        idx = rng.permutation(np.where(y == c)[0])
        k = n_min if c in minority else n_major
        v = min(n_val, max(1, (len(idx) - k) // 3))
        train += idx[:k].tolist()
        val += idx[k:k + v].tolist()
    train, val = np.array(sorted(train)), np.array(sorted(val))
    mask = np.ones(len(y), bool)
    mask[train] = False
    mask[val] = False
    test = np.where(mask)[0]
    return train, val, test, minority
