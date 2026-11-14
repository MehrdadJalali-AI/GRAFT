"""GRAFT core: SGC propagation, convex classifier, drift certificates (Thm 1, Thm 2) and
certificate-aware attachment of synthetic minority nodes."""
import numpy as np
import scipy.sparse as sp
import torch

torch.set_default_dtype(torch.float64)
SQ2 = np.sqrt(2.0)


# ---------------------------------------------------------------- propagation
def row_norm_adj(A):
    """P = D~^{-1}(A + I)."""
    At = A + sp.eye(A.shape[0], format="csr")
    d = np.asarray(At.sum(1)).ravel()
    return sp.diags(1.0 / d) @ At, d


def propagate(P, X, K):
    H = X
    for _ in range(K):
        H = P @ H
    return H


def augment_graph(A, n_syn, edges):
    """edges: list of (synthetic index j in [0, n_syn), real node u)."""
    n = A.shape[0]
    if len(edges) == 0:
        return sp.block_diag([A, sp.csr_matrix((n_syn, n_syn))], format="csr")
    r = np.array([n + j for j, _ in edges])
    c = np.array([u for _, u in edges])
    E = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(n + n_syn, n + n_syn))
    Aa = sp.block_diag([A, sp.csr_matrix((n_syn, n_syn))], format="csr") + E + E.T
    Aa.data = np.minimum(Aa.data, 1.0)
    return Aa.tocsr()


# ---------------------------------------------------------------- classifier
def add_bias(H):
    return np.hstack([H, np.ones((H.shape[0], 1))])


def train_lr(Hb, y, C, lam, sample_w=None, W0=None, iters=500):
    """Minimise (1/sum w) sum w_i CE(W h_i, y_i) + lam/2 ||W||_F^2 (exact, strongly convex)."""
    H = torch.from_numpy(Hb)
    Y = torch.from_numpy(y).long()
    w = torch.ones(len(y)) if sample_w is None else torch.from_numpy(sample_w)
    W = torch.zeros(C, Hb.shape[1], requires_grad=True) if W0 is None else \
        torch.tensor(W0, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=iters, tolerance_grad=1e-11,
                            tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(H @ W.T, Y, reduction="none")
        loss = (w * ce).sum() / w.sum() + 0.5 * lam * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().numpy()


def grad_F(W, Hb, y, lam):
    """Gradient of the uniform-weight objective at W."""
    Z = Hb @ W.T
    Z -= Z.max(1, keepdims=True)
    Pr = np.exp(Z)
    Pr /= Pr.sum(1, keepdims=True)
    Pr[np.arange(len(y)), y] -= 1.0
    return Pr.T @ Hb / len(y) + lam * W


def margins(Z):
    s = np.sort(Z, axis=1)
    return s[:, -1] - s[:, -2]


# ---------------------------------------------------------------- certificates
def theorem1_beta(Pa, n_real, s, d_tilde, K, B=1.0):
    """beta_v = 2B sum_{k<K} (P'^k delta)_v for all real AND synthetic nodes (Theorem 1).
    Reference graph = original graph plus the synthetic nodes as isolated nodes."""
    m = Pa.shape[0] - n_real
    delta = np.concatenate([s / (d_tilde + s), np.ones(m)])
    acc, cur = np.zeros_like(delta), delta.copy()
    for _ in range(K):
        acc += cur
        cur = Pa @ cur
    return 2.0 * B * acc


def theorem2_edge_dW(W_ref, beta_train_syn, n_tot, lam, B=1.0, grad_resid=0.0):
    """A-priori bound on ||W_edge - W_ref||_F (Theorem 2): same n_tot training points,
    features drifted by at most beta_j each."""
    Bh = np.sqrt(B ** 2 + 1)
    num = ((Bh * np.linalg.norm(W_ref) + SQ2) * beta_train_syn).sum() + n_tot * grad_resid
    return num / (lam * n_tot)


def pairwise_certificate(W, Z, beta, dW=0.0, hnorm=None):
    """Certified-unchanged mask: for predicted class a and every b != a,
    z_a - z_b > ||w_a - w_b|| beta + sqrt2 dW (||h|| + beta)   (Corollary 1).
    W includes the bias column; the bias does not multiply drifted features."""
    a = Z.argmax(1)
    Wn = W[:, :-1]
    Dn = np.linalg.norm(Wn[a][:, None, :] - Wn[None, :, :], axis=2)
    hn = np.zeros(len(a)) if hnorm is None else hnorm
    need = Dn * beta[:, None] + SQ2 * dW * (hn + beta)[:, None]
    gap = Z[np.arange(len(a)), a][:, None] - Z
    ok = (gap > need) | (np.arange(Z.shape[1])[None, :] == a[:, None])
    return ok.all(1)


def theorem2_dW(W, beta_train, n, m, lam, B=1.0, grad_resid=0.0):
    """A-priori bound on ||W'-W||_F (Theorem 2); grad_resid = ||grad F(W)||_F for rigour."""
    Bh = np.sqrt(B ** 2 + 1)
    nW = np.linalg.norm(W)
    num = ((Bh * nW + SQ2) * beta_train).sum() + m * (SQ2 * Bh + lam * nW) + n * grad_resid
    return num / (lam * (n + m))


def certificate(W, dW, beta, Hb_real, Z):
    """tau_v and certified mask (margin > sqrt2 tau)."""
    tau = (np.linalg.norm(W) + dW) * beta + dW * np.linalg.norm(Hb_real, axis=1)
    return tau, margins(Z) > SQ2 * tau


# ---------------------------------------------------------------- generators
def smote_features(X, y, train, cls, count, rng):
    idx = train[y[train] == cls]
    out, parents = [], []
    for _ in range(count):
        a = rng.choice(idx)
        others = idx[idx != a]
        if len(others) == 0:
            b = a
        else:
            b = others[np.argmax(X[others] @ X[a])]
        x = X[a] + rng.uniform() * (X[b] - X[a])
        out.append(x / max(1.0, np.linalg.norm(x)))  # projection onto the B-ball
        parents.append(a)
    return np.array(out), np.array(parents)


# ---------------------------------------------------------------- attachment
def exposures(P, nodes, K, n):
    """e_u = sum_{v in nodes} sum_{k<K} (P^k)_{vu}: back-propagate indicator with P^T."""
    v = np.zeros(n)
    v[nodes] = 1.0
    acc, cur, PT = np.zeros(n), v, P.T.tocsr()
    for _ in range(K):
        acc += cur
        cur = PT @ cur
    return acc


def attach(Xs, parents, H, A, d_tilde, mode, d_s, rng, cost=None, gamma=0.0, pool=50):
    """Return list of (j, u) edges for synthetic nodes.
    mode: 'sim' (GraphSMOTE-style), 'ego' (GraphENS-style), 'graft' (certificate-aware)."""
    edges = []
    s = np.zeros(A.shape[0])
    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    for j, (x, a) in enumerate(zip(Xs, parents)):
        if mode == "ego":
            nb = np.append(A[a].indices, a)
            ch = rng.choice(nb, size=min(d_s, len(nb)), replace=False)
        else:
            sim = Hn @ (x / max(np.linalg.norm(x), 1e-12))
            cand = np.argpartition(-sim, pool)[:pool]
            if mode == "sim":
                ch = cand[np.argsort(-sim[cand])[:d_s]]
            else:
                ch = []
                c = cand.copy()
                for _ in range(d_s):
                    # marginal increase of delta_u when one more synthetic edge lands on u
                    inc = (s[c] + 1) / (d_tilde[c] + s[c] + 1) - s[c] / (d_tilde[c] + s[c])
                    score = sim[c] - gamma * cost[c] * inc
                    k = np.argmax(score)
                    ch.append(c[k])
                    s[c[k]] += 1
                    c = np.delete(c, k)
                ch = np.array(ch)
        for u in ch:
            if mode != "graft":
                s[u] += 1
            edges.append((j, int(u)))
    return edges, s
