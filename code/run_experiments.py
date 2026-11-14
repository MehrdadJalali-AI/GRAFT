"""Run all GRAFT experiments.

Usage: python run_experiments.py RAW_DIR OUT_DIR [datasets...]

For every dataset and seed we train:
  * ERM, class reweighting, post-hoc logit adjustment (no synthetic nodes);
  * edge-free oversampling (SMOTE or cWGAN-GP features, synthetic nodes isolated) = the
    REFERENCE that carries the intended rebalancing effect;
  * edge-attaching oversampling: similarity (GraphSMOTE-style), ego (GraphENS-style), and
    GRAFT (drift-budgeted) attachment, plus an edge-budget sweep of the similarity rule.
Collateral drift = predictions of confident real users (protected cohort) that change between
the reference and the edge-attaching model. Certificates (Theorems 1-2) are evaluated
against the observed drift.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from graft import core, gan  # noqa: E402
from graft.data import build, imbalanced_split  # noqa: E402

K, D_S, POOL, N_MAJOR, RATIO = 2, 5, 50, 20, 10
LAMS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
GAMMAS = [0.25, 0.5, 1, 2, 4, 8, 16]
EDGE_BUDGETS = [1, 2, 3]
SEEDS = [int(s) for s in os.environ.get("GRAFT_SEEDS", "0,1,2,3,4").split(",")]
SQ2 = np.sqrt(2)


def metrics(y, p, minority):
    return dict(macro_f1=100 * f1_score(y, p, average="macro"),
                bacc=100 * balanced_accuracy_score(y, p),
                min_recall=100 * np.mean(recall_score(y, p, labels=minority, average=None,
                                                      zero_division=0)))


def select_lambda(D):
    A, X, y = D["A"], D["X"], D["y"]
    C = y.max() + 1
    tr, va, te, mi = imbalanced_split(y, 0, N_MAJOR, RATIO)
    P, _ = core.row_norm_adj(A)
    Hb = core.add_bias(core.propagate(P, X, K))
    return max(LAMS, key=lambda l: f1_score(
        y[va], (Hb[va] @ core.train_lr(Hb[tr], y[tr], C, l).T).argmax(1), average="macro"))


def run_seed(name, D, seed, lam, out_rows, sweep_rows, gcn_rows, use_gan=True, use_gcn=True):
    A, X, y = D["A"], D["X"], D["y"]
    n, C = len(y), y.max() + 1
    rng = np.random.default_rng(seed)
    tr, va, te, minority = imbalanced_split(y, seed, N_MAJOR, RATIO)
    P, dt = core.row_norm_adj(A)
    H = core.propagate(P, X, K)
    Hb = core.add_bias(H)
    W0 = core.train_lr(Hb[tr], y[tr], C, lam)
    Z0 = Hb @ W0.T
    pred0 = Z0.argmax(1)
    mg = core.margins(Z0)
    Pi = te[mg[te] >= np.median(mg[te])]  # protected cohort (evaluation): confident test users, no labels used
    Pv = va[mg[va] >= np.median(mg[va])]  # selection cohort: confident validation users (used to tune gamma)
    base = dict(dataset=name, seed=seed, lam=lam, n_protected=len(Pi), n_val_cohort=len(Pv))

    def row(method, pred, ref_pred=None, extra=None):
        r = dict(base, method=method, **metrics(y[te], pred[te], minority))
        r["val_macro_f1"] = 100 * f1_score(y[va], pred[va], average="macro")
        fl = pred[Pi] != pred0[Pi]
        r["flips_vs_erm_pct"] = 100 * fl.mean()
        if ref_pred is not None:
            fe = pred[Pi] != ref_pred[Pi]
            r["collateral_pct"] = 100 * fe.mean()
            r["harmful_collateral_pct"] = 100 * (fe & (ref_pred[Pi] == y[Pi])).mean()
            r["beneficial_collateral_pct"] = 100 * (fe & (pred[Pi] == y[Pi])).mean()
            r["collateral_val_pct"] = 100 * (pred[Pv] != ref_pred[Pv]).mean()
        r.update(extra or {})
        return r

    out_rows.append(row("ERM", pred0))
    cw = 1.0 / np.bincount(y[tr], minlength=C)
    Wr = core.train_lr(Hb[tr], y[tr], C, lam, sample_w=cw[y[tr]])
    out_rows.append(row("Reweight", (Hb @ Wr.T).argmax(1)))
    prior = np.bincount(y[tr], minlength=C) / len(tr)
    taus = np.linspace(0, 3, 31)
    tau = max(taus, key=lambda t: f1_score(y[va], (Z0[va] - t * np.log(prior)).argmax(1),
                                           average="macro"))
    out_rows.append(row("LogitAdj", (Z0 - tau * np.log(prior)).argmax(1)))

    need = {c: N_MAJOR - (y[tr] == c).sum() for c in minority}
    Xs, par, ys = [], [], []
    for c, k in need.items():
        f, p = core.smote_features(X, y, tr, c, k, rng)
        Xs.append(f); par.append(p); ys += [c] * k
    Xs, par, ys = np.vstack(Xs), np.concatenate(par), np.array(ys)
    m = len(ys)
    ytr = np.concatenate([y[tr], ys])
    n_tot = len(ytr)
    feats = {"SMOTE": (Xs, par)}
    if use_gan:
        g, zd = gan.train_cwgan(X[tr], y[tr], C, seed)
        Xg = np.vstack([gan.sample(g, zd, c, k, seed + 7 * c) for c, k in need.items()])
        # parent = most similar labelled node of the same class (used only by ego attachment)
        pg = np.array([tr[y[tr] == c][np.argmax(X[tr[y[tr] == c]] @ x)] for x, c in zip(Xg, ys)])
        feats["cWGAN-GP"] = (Xg, pg)

    graphs = {}
    for gen, (F, parents) in feats.items():
        # ---------------- reference: synthetic nodes added without edges (intended rebalancing)
        Hs_ref = core.add_bias(F)  # isolated synthetic node: h_s = x_s
        Href = np.vstack([Hb[tr], Hs_ref])
        Wref = core.train_lr(Href, ytr, C, lam)
        Zref = Hb @ Wref.T
        pref = Zref.argmax(1)
        resid = np.linalg.norm(core.grad_F(Wref, Href, ytr, lam))
        out_rows.append(row(f"{gen}-noedge", pref, extra=dict(generator=gen)))
        graphs[f"{gen}-noedge"] = (core.augment_graph(A, m, []), F)

        # cost for drift-budgeted attachment (first-order surrogate of Thm 2 + Thm 1)
        nWr = np.linalg.norm(Wref)
        w_tr = (np.sqrt(2) * nWr + SQ2) / (lam * n_tot) * SQ2 * np.linalg.norm(Hb[Pi], axis=1).mean()
        cost = 2 * (w_tr * core.exposures(P, tr, K, n) + nWr * core.exposures(P, Pi, K, n) / len(Pi))
        unit = cost / dt
        cost = cost / np.quantile(unit[unit > 0], 0.9)

        def evaluate(method, edges, s, extra=None):
            Aa = core.augment_graph(A, m, edges)
            Pa, _ = core.row_norm_adj(Aa)
            Ha = core.add_bias(core.propagate(Pa, np.vstack([X, F]), K))
            Htr = np.vstack([Ha[tr], Ha[n:]])
            We = core.train_lr(Htr, ytr, C, lam, W0=Wref)
            fit_resid = np.linalg.norm(core.grad_F(We, Htr, ytr, lam))
            Ze = Ha[:n] @ We.T
            pred = Ze.argmax(1)
            beta = core.theorem1_beta(Pa, n, s, dt, K)
            bts = np.concatenate([beta[tr], beta[n:]])
            dW_prior = core.theorem2_edge_dW(Wref, bts, n_tot, lam, grad_resid=resid)
            dW_post = np.linalg.norm(core.grad_F(Wref, Htr, ytr, lam)) / lam
            dW_obs = np.linalg.norm(We - Wref)
            hn = np.linalg.norm(Hb[Pi], axis=1)
            # frozen-model certificate (Thm 1 + Cor 1 with dW = 0): reference model, new graph
            cert_frozen = core.pairwise_certificate(Wref, Zref[Pi], beta[Pi])
            cert_post = core.pairwise_certificate(Wref, Zref[Pi], beta[Pi], dW_post, hn)
            pred_frozen = (Ha[:n] @ Wref.T).argmax(1)
            dh = np.linalg.norm(Ha[:n, :-1] - H, axis=1)
            dhs = np.linalg.norm(Ha[n:, :-1] - F, axis=1)
            touched = beta[Pi] > 0
            ff = pred_frozen[Pi] != pref[Pi]
            e = dict(generator=gen, m=m, n_edges=len(edges), fit_resid=fit_resid, ref_resid=resid,
                     touched_pct=100 * touched.mean(), n_touched=int(touched.sum()),
                     n_cert_frozen=int(cert_frozen.sum()), n_frozen_flips=int(ff.sum()),
                     n_cert_touched=int((cert_frozen & touched).sum()),
                     cov_frozen_touched=100 * (cert_frozen & touched).sum() / touched.sum() if touched.any() else np.nan,
                     cov_frozen=100 * cert_frozen.mean(), cov_retrain=100 * cert_post.mean(),
                     frozen_flips_pct=100 * (pred_frozen[Pi] != pref[Pi]).mean(),
                     cert_frozen_violations=int((cert_frozen & (pred_frozen[Pi] != pref[Pi])).sum()),
                     cert_retrain_violations=int((cert_post & (pred[Pi] != pref[Pi])).sum()),
                     dW_obs=dW_obs, dW_post=dW_post, dW_prior=dW_prior,
                     thm1_holds=bool((dh <= beta[:n] + 1e-9).all() and (dhs <= beta[n:] + 1e-9).all()),
                     thm2_holds=bool(dW_obs <= dW_post + 1e-6 and dW_post <= dW_prior + 1e-9),
                     thm1_tightness=float(np.median(dh[beta[:n] > 0] / beta[:n][beta[:n] > 0])),
                     thm2_tightness_post=dW_obs / dW_post, thm2_tightness_prior=dW_obs / dW_prior)
            e.update(extra or {})
            return row(method, pred, pref, e), Aa

        for mode, label in [("sim", "SimAttach"), ("ego", "EgoAttach")]:
            ed, s = core.attach(F, parents, H, A, dt, mode, D_S, np.random.default_rng(seed), pool=POOL)
            r, Aa = evaluate(f"{gen}+{label}", ed, s)
            out_rows.append(r)
            graphs[f"{gen}+{label}"] = (Aa, F)
            sweep_rows.append(dict(r, family=f"{gen}+{label}", knob="d_s", value=D_S))
        for ds_ in EDGE_BUDGETS:
            ed, s = core.attach(F, parents, H, A, dt, "sim", ds_, np.random.default_rng(seed), pool=POOL)
            r, _ = evaluate(f"{gen}+SimAttach(d={ds_})", ed, s)
            sweep_rows.append(dict(r, family=f"{gen}+SimAttach", knob="d_s", value=ds_))
        cands = []
        for gm in GAMMAS:
            ed, s = core.attach(F, parents, H, A, dt, "graft", D_S, np.random.default_rng(seed),
                                cost=cost, gamma=gm, pool=POOL)
            r, Aa = evaluate(f"GRAFT ({gen})", ed, s, extra=dict(gamma=gm))
            sweep_rows.append(dict(r, family=f"GRAFT ({gen})", knob="gamma", value=gm))
            cands.append((r, Aa))
        # model selection with labels only on validation; collateral is label-free
        best = max(c[0]["val_macro_f1"] for c in cands)
        ok = [c for c in cands if c[0]["val_macro_f1"] >= best - 1.0]
        # tie-break on label-free disagreement measured on the VALIDATION cohort (never on test nodes)
        r, Aa = min(ok, key=lambda c: (c[0]["collateral_val_pct"], -c[0]["val_macro_f1"], c[0]["gamma"]))
        out_rows.append(r)
        graphs[f"GRAFT ({gen})"] = (Aa, F)

    if use_gcn:
        gcn_rows += gcn_transfer(name, seed, A, X, y, tr, te, minority, ys,
                                 {k: v for k, v in graphs.items() if k.startswith("SMOTE") or k.startswith("GRAFT (SMOTE")})


def gcn_transfer(name, seed, A, X, y, tr, te, minority, ys, graphs):
    import torch
    from torch_geometric.nn import GCNConv
    torch.set_num_threads(1)
    rows = []

    class Net(torch.nn.Module):
        def __init__(s, d, C):
            super().__init__()
            s.c1, s.c2 = GCNConv(d, 64), GCNConv(64, C)

        def forward(s, x, ei):
            h = torch.nn.functional.dropout(torch.relu(s.c1(x, ei)), 0.5, s.training)
            return s.c2(h, ei)

    def fit(Aa, Xa, tr_idx, ytr, init_seed):
        torch.manual_seed(init_seed)
        coo = Aa.tocoo()
        ei = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        xt = torch.tensor(Xa, dtype=torch.float32)
        net = Net(Xa.shape[1], y.max() + 1).float()
        opt = torch.optim.Adam(net.parameters(), 0.01, weight_decay=5e-4)
        ti, yt = torch.tensor(tr_idx), torch.tensor(ytr).long()
        for _ in range(200):
            net.train(); opt.zero_grad()
            torch.nn.functional.cross_entropy(net(xt, ei)[ti], yt).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return net(xt, ei)[: len(y)].numpy()

    n = len(y)
    Z0 = fit(A, X, tr, y[tr], seed)
    p0 = Z0.argmax(1)
    mg = core.margins(Z0)
    Pi = te[mg[te] >= np.median(mg[te])]
    tr_idx = np.concatenate([tr, np.arange(n, n + len(ys))])
    ytr = np.concatenate([y[tr], ys])

    def add(method, p, ref):
        fe = p[Pi] != ref[Pi]
        rows.append(dict(dataset=name, seed=seed, method=method, **metrics(y[te], p[te], minority),
                         collateral_pct=100 * fe.mean(),
                         harmful_collateral_pct=100 * (fe & (ref[Pi] == y[Pi])).mean()))

    Aref, Fref = graphs["SMOTE-noedge"]
    pref = fit(Aref, np.vstack([X, Fref]), tr_idx, ytr, seed).argmax(1)
    pref2 = fit(Aref, np.vstack([X, Fref]), tr_idx, ytr, seed + 100).argmax(1)
    add("GCN ERM", p0, p0)
    add("GCN SMOTE-noedge", pref, pref)
    add("GCN SMOTE-noedge (re-init: noise floor)", pref2, pref)
    for label, (Aa, f) in graphs.items():
        if label == "SMOTE-noedge":
            continue
        add("GCN " + label, fit(Aa, np.vstack([X, f]), tr_idx, ytr, seed).argmax(1), pref)
    return rows


if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    raw, out = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or ["lastfm", "twitch_en", "facebook", "github"]
    os.makedirs(out, exist_ok=True)
    for name in names:
        t0 = time.time()
        D = build(name, raw, os.path.join(out, "..", "data"))
        lam = select_lambda(D)
        R, S, G = [], [], []
        for seed in SEEDS:
            run_seed(name, D, seed, lam, R, S, G)
            print(name, seed, f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(R).to_csv(os.path.join(out, f"runs_{name}.csv"), index=False)
        pd.DataFrame(S).to_csv(os.path.join(out, f"sweep_{name}.csv"), index=False)
        pd.DataFrame(G).to_csv(os.path.join(out, f"gcn_{name}.csv"), index=False)
        json.dump(dict(lam=lam, K=K, d_s=D_S, pool=POOL, n_major=N_MAJOR, ratio=RATIO),
                  open(os.path.join(out, f"config_{name}.json"), "w"))
