"""Recover per-user predictions for Figure 4b (LastFM, cWGAN-GP features).

Re-runs the saved configuration of run_experiments.py (same splits, seeds, lambda, generator,
attachment rules and the GRAFT gamma selected per split in results/runs_lastfm.csv) and stores,
for every protected test user, the edge-free reference prediction, the frozen-model prediction on
G' and the refitted prediction on G'.  The script checks that the recovered aggregates match the
saved run outputs before writing anything.

Usage: python fig4_reproduce_lastfm.py RAW_DIR RESULTS_DIR
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from graft import core, gan  # noqa: E402
from graft.data import build, imbalanced_split  # noqa: E402
from run_experiments import K, D_S, POOL, N_MAJOR, RATIO  # noqa: E402

torch.set_num_threads(1)
SQ2 = np.sqrt(2)
METHODS = {"GRAFT (cWGAN-GP)": "graft", "cWGAN-GP+SimAttach": "sim", "cWGAN-GP+EgoAttach": "ego"}


def main(raw, res):
    runs = pd.read_csv(os.path.join(res, "runs_lastfm.csv"))
    lam = float(runs.lam.iloc[0])
    D = build("lastfm", raw, os.path.join(res, "..", "data"))
    A, X, y = D["A"], D["X"], D["y"]
    n, C = len(y), y.max() + 1
    rows, checks = [], []
    for seed in sorted(runs.seed.unique()):
        rng = np.random.default_rng(seed)
        tr, va, te, minority = imbalanced_split(y, seed, N_MAJOR, RATIO)
        P, dt = core.row_norm_adj(A)
        H = core.propagate(P, X, K)
        Hb = core.add_bias(H)
        W0 = core.train_lr(Hb[tr], y[tr], C, lam)
        Z0 = Hb @ W0.T
        mg = core.margins(Z0)
        Pi = te[mg[te] >= np.median(mg[te])]
        need = {c: N_MAJOR - (y[tr] == c).sum() for c in minority}
        ys = []
        for c, k in need.items():  # consume the numpy RNG exactly as in run_experiments (SMOTE first)
            core.smote_features(X, y, tr, c, k, rng)
            ys += [c] * k
        ys = np.array(ys)
        m = len(ys)
        ytr = np.concatenate([y[tr], ys])
        n_tot = len(ytr)
        g, zd = gan.train_cwgan(X[tr], y[tr], C, seed)
        F = np.vstack([gan.sample(g, zd, c, k, seed + 7 * c) for c, k in need.items()])
        parents = np.array([tr[y[tr] == c][np.argmax(X[tr[y[tr] == c]] @ x)] for x, c in zip(F, ys)])
        Href = np.vstack([Hb[tr], core.add_bias(F)])
        Wref = core.train_lr(Href, ytr, C, lam)
        pref = (Hb @ Wref.T).argmax(1)
        nWr = np.linalg.norm(Wref)
        w_tr = (np.sqrt(2) * nWr + SQ2) / (lam * n_tot) * SQ2 * np.linalg.norm(Hb[Pi], axis=1).mean()
        cost = 2 * (w_tr * core.exposures(P, tr, K, n) + nWr * core.exposures(P, Pi, K, n) / len(Pi))
        unit = cost / dt
        cost = cost / np.quantile(unit[unit > 0], 0.9)
        for label, mode in METHODS.items():
            saved = runs[(runs.seed == seed) & (runs.method == label)].iloc[0]
            if mode == "graft":
                ed, s = core.attach(F, parents, H, A, dt, "graft", D_S, np.random.default_rng(seed),
                                    cost=cost, gamma=float(saved.gamma), pool=POOL)
            else:
                ed, s = core.attach(F, parents, H, A, dt, mode, D_S, np.random.default_rng(seed), pool=POOL)
            Aa = core.augment_graph(A, m, ed)
            Pa, _ = core.row_norm_adj(Aa)
            Ha = core.add_bias(core.propagate(Pa, np.vstack([X, F]), K))
            Htr = np.vstack([Ha[tr], Ha[n:]])
            We = core.train_lr(Htr, ytr, C, lam, W0=Wref)
            p_refit = (Ha[:n] @ We.T).argmax(1)
            p_frozen = (Ha[:n] @ Wref.T).argmax(1)
            beta = core.theorem1_beta(Pa, n, s, dt, K)
            cert = core.pairwise_certificate(Wref, (Hb @ Wref.T)[Pi], beta[Pi])
            coll = 100 * (p_refit[Pi] != pref[Pi]).mean()
            ff = int((p_frozen[Pi] != pref[Pi]).sum())
            checks.append(dict(seed=seed, method=label, collateral_recovered=coll,
                               collateral_saved=saved.collateral_pct, frozen_flips_recovered=ff,
                               frozen_flips_saved=int(saved.n_frozen_flips),
                               n_touched_recovered=int((beta[Pi] > 0).sum()), n_touched_saved=int(saved.n_touched)))
            for v, c0, cf, cr, ce, b in zip(Pi, pref[Pi], p_frozen[Pi], p_refit[Pi], cert, beta[Pi]):
                rows.append(dict(split=seed, method=label, node=int(v), y=int(y[v]), pred_ref=int(c0),
                                 pred_frozen=int(cf), pred_refit=int(cr), frozen_certified=bool(ce),
                                 beta=float(b)))
        print("split", seed, "done", flush=True)
    chk = pd.DataFrame(checks)
    chk["match"] = (np.isclose(chk.collateral_recovered, chk.collateral_saved, atol=1e-9)
                    & (chk.frozen_flips_recovered == chk.frozen_flips_saved)
                    & (chk.n_touched_recovered == chk.n_touched_saved))
    chk.to_csv(os.path.join(res, "fig4_reproduction_check_lastfm.csv"), index=False)
    print(chk.to_string())
    if not chk.match.all():
        raise SystemExit("Recovered runs do not match the saved outputs; per-user file not written.")
    pd.DataFrame(rows).to_csv(os.path.join(res, "fig4_peruser_lastfm.csv"), index=False)
    print("all splits reproduced exactly; wrote fig4_peruser_lastfm.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
