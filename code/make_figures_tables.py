"""Build manuscript tables (LaTeX) and figures (PDF) from results/*.csv."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
DS = ["facebook", "github", "lastfm", "twitch_en"]
NAME = {"facebook": "Facebook", "github": "GitHub", "lastfm": "LastFM", "twitch_en": "Twitch-EN"}
BLUE, ORANGE, AQUA, YELLOW, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": "#b9b8b2",
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5})

runs = pd.concat([pd.read_csv(os.path.join(RES, f"runs_{d}.csv")) for d in DS])
sweep = pd.concat([pd.read_csv(os.path.join(RES, f"sweep_{d}.csv")) for d in DS])
gcn = pd.concat([pd.read_csv(os.path.join(RES, f"gcn_{d}.csv")) for d in DS])
summary = {}


def ms(df, col):
    g = df.groupby(["dataset", "method"], sort=False)[col]
    return g.mean(), g.std()


def fmt(m, s, bold=False):
    t = f"{m:.1f}\\,$\\pm$\\,{s:.1f}"
    return f"\\textbf{{{t}}}" if bold else t


# ---------------------------------------------------------------- Table: main accuracy
METHODS = ["ERM", "Reweight", "LogitAdj", "SMOTE-noedge", "SMOTE+SimAttach", "SMOTE+EgoAttach",
           "cWGAN-GP-noedge", "cWGAN-GP+SimAttach", "cWGAN-GP+EgoAttach",
           "GRAFT (cWGAN-GP)", "GRAFT (SMOTE)"]
GRAFT_ROWS = {"GRAFT (cWGAN-GP)", "GRAFT (SMOTE)"}
HL = r"\rowcolor{graftbg} "
LABEL = {"ERM": "ERM", "Reweight": "Class reweighting", "LogitAdj": "Post-hoc logit adjustment",
         "SMOTE-noedge": "SMOTE, no edges", "SMOTE+SimAttach": "SMOTE + similarity edges",
         "SMOTE+EgoAttach": "SMOTE + ego edges", "GRAFT (SMOTE)": "\\textbf{GRAFT-SMOTE} (ours, ablation)",
         "cWGAN-GP-noedge": "cWGAN-GP, no edges", "cWGAN-GP+SimAttach": "cWGAN-GP + similarity edges",
         "cWGAN-GP+EgoAttach": "cWGAN-GP + ego edges", "GRAFT (cWGAN-GP)": "\\textbf{GRAFT} (ours, cWGAN-GP)"}
for col, fname, cap in [("macro_f1", "tab_macro_f1.tex", "Macro-F1"), ("min_recall", "tab_min_recall.tex", "Minority recall")]:
    m, s = ms(runs, col)
    lines = []
    for meth in METHODS:
        cells = []
        for d in DS:
            best = max(m[(d, k)] for k in METHODS)
            cells.append(fmt(m[(d, meth)], s[(d, meth)], abs(m[(d, meth)] - best) < 1e-9))
        lines.append((HL if meth in GRAFT_ROWS else "") + f"{LABEL[meth]} & " + " & ".join(cells) + r" \\")
        if meth in ("LogitAdj", "SMOTE+EgoAttach", "cWGAN-GP+EgoAttach"):
            lines.append(r"\midrule")
    open(os.path.join(OUT, "tables", fname), "w").write("\n".join(lines) + "\n")
    summary[col] = {f"{d}|{k}": [float(m[(d, k)]), float(s[(d, k)])] for d in DS for k in METHODS}

# ---------------------------------------------------------------- Table: collateral + certificates
EDGE = ["GRAFT (cWGAN-GP)", "GRAFT (SMOTE)", "cWGAN-GP+SimAttach", "cWGAN-GP+EgoAttach",
        "SMOTE+SimAttach", "SMOTE+EgoAttach"]
lines = []
coll_summary = {}
for d in DS:
    for i, meth in enumerate(EDGE):
        x = runs[(runs.dataset == d) & (runs.method == meth)]
        npi = int(x.n_protected.sum())
        nt = int(x.n_touched.sum())
        ct = int(x.n_cert_touched.sum())
        cov_t = f"{100 * ct / nt:.1f}" if nt > 0 else "N/A"
        r = dict(coll=x.collateral_pct.mean(), harm=x.harmful_collateral_pct.mean(), npi=npi, nt=nt,
                 cov0=100 * x.n_cert_frozen.sum() / npi, cov_t=cov_t, ff=int(x.n_frozen_flips.sum()),
                 viol=int(x.cert_frozen_violations.sum()), covrt=x.cov_retrain.mean())
        coll_summary[f"{d}|{meth}"] = r
        lead = f"\\multirow{{{len(EDGE)}}}{{*}}{{{NAME[d]}}}" if i == 0 else ""
        lines.append((HL if meth in GRAFT_ROWS else "") +
                     f"{lead} & {LABEL[meth]} & {r['coll']:.2f} & {r['harm']:.2f} & {npi:,} & {nt:,} & "
                     f"{r['cov0']:.1f} & {cov_t} & {r['ff']} & {r['viol']} & {r['covrt']:.1f} \\\\")
    if d != DS[-1]:
        lines.append(r"\midrule")
open(os.path.join(OUT, "tables", "tab_collateral.tex"), "w").write("\n".join(lines) + "\n")
summary["collateral"] = coll_summary

# ---------------------------------------------------------------- Table: paired comparisons
def paired(d, a, b, col):
    xa = runs[(runs.dataset == d) & (runs.method == a)].set_index("seed")[col]
    xb = runs[(runs.dataset == d) & (runs.method == b)].set_index("seed")[col]
    diff = (xa - xb).dropna()
    return diff.mean(), diff.std(), int((diff > 1e-9).sum()), int((diff < -1e-9).sum()), len(diff)

lines = []
psum = {}
for d in DS:
    f = paired(d, "GRAFT (cWGAN-GP)", "cWGAN-GP+SimAttach", "macro_f1")
    c = paired(d, "GRAFT (cWGAN-GP)", "cWGAN-GP+SimAttach", "collateral_pct")
    g = paired(d, "GRAFT (cWGAN-GP)", "GRAFT (SMOTE)", "macro_f1")
    e = paired(d, "cWGAN-GP-noedge", "SMOTE-noedge", "macro_f1")
    psum[d] = dict(f1=f, coll=c, gen=g, gen_noedge=e)
    lines.append(f"{NAME[d]} & {f[0]:+.2f}\\,$\\pm$\\,{f[1]:.2f} & {f[2]}/{f[3]}/{f[4]} & "
                 f"{c[0]:+.2f}\\,$\\pm$\\,{c[1]:.2f} & {c[2]}/{c[3]}/{c[4]} & "
                 f"{g[0]:+.2f}\\,$\\pm$\\,{g[1]:.2f} & {g[2]}/{g[3]}/{g[4]} \\\\")
open(os.path.join(OUT, "tables", "tab_paired.tex"), "w").write("\n".join(lines) + "\n")
summary["paired"] = psum

# ---------------------------------------------------------------- Table: bound validity / tightness
e = runs[runs.method.isin(EDGE)]
lines = []
for d in DS:
    x = e[e.dataset == d]
    lines.append(f"{NAME[d]} & {len(x)} & {int(x.thm1_holds.sum())} & {x.thm1_tightness.median():.2f} & "
                 f"{int(x.thm2_holds.sum())} & {x.thm2_tightness_post.median():.2f} & "
                 f"{x.thm2_tightness_prior.median():.1e} & {x.fit_resid.max():.0e} & {int(x.cert_frozen_violations.sum())} \\\\")
open(os.path.join(OUT, "tables", "tab_bounds.tex"), "w").write("\n".join(lines) + "\n")
summary["bounds"] = {d: dict(n=int(len(e[e.dataset == d])), thm1=int(e[e.dataset == d].thm1_holds.sum()),
                             thm2=int(e[e.dataset == d].thm2_holds.sum()),
                             t1=float(e[e.dataset == d].thm1_tightness.median()),
                             t2post=float(e[e.dataset == d].thm2_tightness_post.median()),
                             t2prior=float(e[e.dataset == d].thm2_tightness_prior.median()),
                             resid=float(e[e.dataset == d].fit_resid.max()),
                             viol=int(e[e.dataset == d].cert_frozen_violations.sum())) for d in DS}
summary["sweep_checks"] = dict(n=int(len(sweep)), thm1=bool(sweep.thm1_holds.all()), thm2=bool(sweep.thm2_holds.all()),
                               viol_frozen=int(sweep.cert_frozen_violations.sum()),
                               viol_rt=int(sweep.cert_retrain_violations.sum()),
                               max_resid=float(sweep.fit_resid.max()))
summary["selected_gamma"] = runs[runs.method.str.startswith("GRAFT")].groupby(["dataset", "method"]).gamma.apply(list).to_dict()
summary["selected_gamma"] = {f"{k[0]}|{k[1]}": v for k, v in summary["selected_gamma"].items()}

# ---------------------------------------------------------------- Table: GCN transfer
GM = ["GCN ERM", "GCN SMOTE-noedge", "GCN SMOTE-noedge (re-init: noise floor)", "GCN SMOTE+SimAttach",
      "GCN SMOTE+EgoAttach", "GCN GRAFT (SMOTE)"]
GL = {"GCN ERM": "ERM", "GCN SMOTE-noedge": "SMOTE, no edges (reference)",
      "GCN SMOTE-noedge (re-init: noise floor)": "Reference, new initialisation",
      "GCN SMOTE+SimAttach": "SMOTE + similarity edges", "GCN SMOTE+EgoAttach": "SMOTE + ego edges",
      "GCN GRAFT (SMOTE)": "\\textbf{GRAFT-SMOTE} (ours)"}
gm = gcn.groupby(["dataset", "method"], sort=False)[["macro_f1", "collateral_pct"]].agg(["mean", "std"])
lines = []
for meth in GM:
    cells = []
    for d in DS:
        r = gm.loc[(d, meth)]
        coll = "--" if meth in ("GCN ERM", "GCN SMOTE-noedge") else f"{r[('collateral_pct', 'mean')]:.2f}"
        cells.append(f"{r[('macro_f1', 'mean')]:.1f} & {coll}")
    lines.append((HL if "GRAFT" in meth else "") + f"{GL[meth]} & " + " & ".join(cells) + r" \\")
open(os.path.join(OUT, "tables", "tab_gcn.tex"), "w").write("\n".join(lines) + "\n")
summary["gcn"] = {f"{d}|{k}": [float(gm.loc[(d, k)][("macro_f1", "mean")]),
                               float(gm.loc[(d, k)][("collateral_pct", "mean")])] for d in DS for k in GM}

# ---------------------------------------------------------------- Figure: F1 vs collateral trade-off
fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.6))
axes = axes.ravel()
for ax, d in zip(axes, DS):
    s = sweep[sweep.dataset == d]
    sim = s[s.family == "cWGAN-GP+SimAttach"].groupby("value")[["collateral_pct", "macro_f1"]].mean()
    gr = s[s.family == "GRAFT (cWGAN-GP)"].groupby("value")[["collateral_pct", "macro_f1"]].mean()
    ego = s[s.family == "cWGAN-GP+EgoAttach"][["collateral_pct", "macro_f1"]].mean()
    sel = runs[(runs.dataset == d) & (runs.method == "GRAFT (cWGAN-GP)")][["collateral_pct", "macro_f1"]].mean()
    ax.plot(sim.collateral_pct, sim.macro_f1, "-o", color=BLUE, ms=3, lw=1.2, label="Similarity edges, $d_s\\in\\{1,2,3,5\\}$")
    ax.plot([ego.collateral_pct], [ego.macro_f1], "D", color=AQUA, ms=4.5, label="Ego edges")
    ax.plot(gr.collateral_pct, gr.macro_f1, "-s", color=ORANGE, ms=4, lw=2.2, label="GRAFT (ours), $\\gamma$ sweep", zorder=5)
    ax.plot([sel.collateral_pct], [sel.macro_f1], "*", color=ORANGE, mec=INK, mew=0.6, ms=11,
            label="GRAFT (ours), selected $\\gamma$", zorder=6)
    m_ = runs[runs.dataset == d].groupby("method").macro_f1.mean()
    for k, ls, lab, xa, ha in [("cWGAN-GP-noedge", ":", "no edges", 0.99, "right"), ("Reweight", "--", "reweight", 0.99, "right"),
                               ("LogitAdj", "-.", "logit adj.", 0.01, "left")]:
        ax.axhline(m_[k], color=GRAY, ls=ls, lw=1)
        ax.annotate(lab, (xa, m_[k]), xycoords=("axes fraction", "data"), ha=ha, va="bottom",
                    fontsize=6.5, color=INK2)
    if s.collateral_pct.max() < 0.01:
        ax.set_xlim(-0.005, 0.05)
    ax.set_title(NAME[d], fontsize=8.5, color=INK)
    ax.grid(color="#e6e5e0", lw=0.5)
for ax in axes[2:]:
    ax.set_xlabel("Collateral flips (% of confident users)")
for ax in axes[::2]:
    ax.set_ylabel("Macro-F1 (%)")
h, l = axes[0].get_legend_handles_labels()
order = [2, 3, 0, 1]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.07), fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "fig_tradeoff.pdf"), bbox_inches="tight")

# ---------------------------------------------------------------- Figure: bound vs observed
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
isg = sweep.family.str.startswith("GRAFT")
for ax, col in zip(axes, ["dW_post", "dW_prior"]):
    o = sweep[~isg]
    g_ = sweep[isg]
    ax.scatter(o[col], o.dW_obs, s=8, color="#c3c2b7", marker="o", linewidths=0,
               label="Similarity and ego edges")
    ax.scatter(g_[col], g_.dW_obs, s=11, color=ORANGE, marker="s", linewidths=0, alpha=0.9,
               label="GRAFT (ours), all $\\gamma$")
for ax, t in zip(axes, ["A-posteriori bound $\\|\\nabla F'(W_{\\rm ref})\\|_F/\\lambda$",
                        "A-priori bound (Theorem 2)"]):
    lo = 0.5
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1, ls="--")
    ax.annotate("bound = observed", (0.05, 0.9), xycoords="axes fraction", fontsize=6.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(t)
    ax.grid(color="#e6e5e0", lw=0.5)
axes[0].set_ylabel("Observed $\\|W_{\\rm edge}-W_{\\rm ref}\\|_F$")
axes[1].legend(frameon=False, fontsize=7, loc="center left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "fig_bounds.pdf"), bbox_inches="tight")

# ---------------------------------------------------------------- dataset table
cfg = {d: json.load(open(os.path.join(RES, f"config_{d}.json"))) for d in DS}
summary["config"] = cfg
json.dump(summary, open(os.path.join(OUT, "tables", "summary.json"), "w"), indent=1)
print("done")
