"""Figure 4 -- Where does the prediction certificate hold?

a: frozen-model certificate coverage over all protected users vs. over protected users with
   beta_v > 0 (summed over splits), cWGAN-GP features, three attachment rules, four datasets.
b: LastFM, protected user-split observations whose prediction flips with frozen and/or refitted
   weights relative to the matched edge-free reference (from fig4_peruser_lastfm.csv).

Usage: python make_fig4.py RESULTS_DIR OUT_DIR
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.6, "xtick.color": "#444444", "ytick.color": "#222222",
    "xtick.major.width": 0.6, "ytick.major.width": 0, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})
INK, INK2, GRID = "#222222", "#5A5A5A", "#E6E6E6"
DS = [("facebook", "Facebook"), ("github", "GitHub"), ("lastfm", "LastFM"), ("twitch_en", "Twitch-EN")]
METH = [("GRAFT (cWGAN-GP)", "GRAFT", "#AA3377"),
        ("cWGAN-GP+SimAttach", "Similarity", "#4477AA"),
        ("cWGAN-GP+EgoAttach", "Ego", "#228833")]

# ------------------------------------------------------------------ data for panel a
recs = []
for d, dn in DS:
    r = pd.read_csv(os.path.join(RES, f"runs_{d}.csv"))
    for key, lab, col in METH:
        x = r[r.method == key]
        npi, nt = int(x.n_protected.sum()), int(x.n_touched.sum())
        recs.append(dict(dataset=dn, method=lab, color=col, n_pi=npi, n_exp=nt,
                         cov_all=100 * x.n_cert_frozen.sum() / npi,
                         cov_exp=100 * x.n_cert_touched.sum() / nt if nt > 0 else np.nan,
                         flips_frozen=int(x.n_frozen_flips.sum()), viol=int(x.cert_frozen_violations.sum())))
A = pd.DataFrame(recs)

# ------------------------------------------------------------------ data for panel b
U = pd.read_csv(os.path.join(RES, "fig4_peruser_lastfm.csv"))
U["ff"] = U.pred_frozen != U.pred_ref
U["fr"] = U.pred_refit != U.pred_ref
B = []
for key, lab, col in METH:
    u = U[(U.method == key) & (U.ff | U.fr)]
    n = len(u)
    B.append(dict(method=lab, n=n, frozen_only=((u.ff) & ~(u.fr)).sum(), refit_only=(~(u.ff) & (u.fr)).sum(),
                  both=((u.ff) & (u.fr)).sum()))
B = pd.DataFrame(B)

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(7.09, 3.35))
axa = fig.add_axes([0.175, 0.13, 0.40, 0.72])
axb = fig.add_axes([0.745, 0.30, 0.185, 0.32])

# ---- panel a
ypos, ylabels, y = [], [], 0.0
group_mid = {}
for dn in [d[1] for d in DS]:
    rows = A[A.dataset == dn]
    ys = []
    for _, r in rows.iterrows():
        ypos.append(y); ylabels.append(r.method); ys.append(y); y -= 1.0
    group_mid[dn] = (ys[0], ys[-1])
    y -= 0.7
A["y"] = ypos
for _, r in A.iterrows():
    if not np.isnan(r.cov_exp):
        axa.plot([r.cov_exp, r.cov_all], [r.y, r.y], color=r.color, lw=1.0, alpha=0.55, zorder=1, solid_capstyle="round")
        axa.scatter([r.cov_exp], [r.y], marker="D", s=17, facecolor="white", edgecolor=r.color, linewidth=1.0, zorder=3)
    axa.scatter([r.cov_all], [r.y], marker="o", s=17, color=r.color, edgecolor="white", linewidth=0.4, zorder=4)
    txt = f"n = {r.n_exp:,}" if r.n_exp > 0 else "N/A (n = 0)"
    axa.text(104, r.y, txt, va="center", ha="left", fontsize=6.2, color=INK2, clip_on=False)
axa.set_yticks(A.y)
axa.set_yticklabels(A.method)
for t, (_, r) in zip(axa.get_yticklabels(), A.iterrows()):
    t.set_color(r.color)
for dn, (y0, y1) in group_mid.items():
    axa.text(-0.215, (y0 + y1) / 2, dn, transform=axa.get_yaxis_transform(), ha="right", va="center",
             fontsize=7, fontweight="bold", color=INK)
    axa.axhspan(y1 - 0.45, y0 + 0.45, color="#F5F5F5", zorder=0, lw=0) if dn in ("GitHub", "Twitch-EN") else None
axa.set_xlim(0, 101.5)
axa.set_ylim(min(A.y) - 0.6, max(A.y) + 0.6)
axa.set_xticks([0, 25, 50, 75, 100])
axa.set_xticklabels(["0", "25", "50", "75", "100%"])
axa.xaxis.grid(True, color=GRID, lw=0.6)
axa.set_axisbelow(True)
axa.spines["left"].set_visible(False)
axa.set_xlabel("Frozen-model certificate coverage")
axa.text(104, max(A.y) + 0.7, "exposed\nusers", fontsize=6.2, color=INK2, ha="left", va="bottom")
# callout LastFM GRAFT
r = A[(A.dataset == "LastFM") & (A.method == "GRAFT")].iloc[0]
axa.annotate(f"{r.cov_all:.1f}% of all protected users\n{r.cov_exp:.1f}% of exposed users\n(n = {r.n_exp:,}, summed over splits)",
             xy=(r.cov_exp, r.y), xytext=(2, r.y - 1.4), fontsize=6.2, color=INK, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=3))
# legend (markers)
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], marker="o", ls="", color=INK2, markeredgecolor="white", markersize=4.5, label=r"All protected users $\Pi$"),
     Line2D([], [], marker="D", ls="", markerfacecolor="white", markeredgecolor=INK2, markersize=4.0,
            label=r"Exposed users $\{v\in\Pi:\beta_v>0\}$")]
axa.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=6.5,
           handletextpad=0.3, columnspacing=1.2, borderaxespad=0.3)
fig.text(0.005, 0.975, "a", fontsize=9, fontweight="bold", va="top")
fig.text(0.03, 0.975, "Frozen-model certificate coverage (cWGAN-GP features)", fontsize=7.5, va="top", color=INK)

# ---- panel b
cols = [("frozen_only", "Frozen only", "#DDCC77"), ("refit_only", "Refitted only", "#88CCEE"), ("both", "Both", "#332288")]
for i, (_, r) in enumerate(B.iterrows()):
    yb = -i
    left = 0.0
    for k, lab, c in cols:
        w = 100 * r[k] / r.n
        axb.barh(yb, w, left=left, color=c, height=0.62, edgecolor="white", linewidth=0.5)
        if w >= 9:
            axb.text(left + w / 2, yb, f"{w:.0f}%", ha="center", va="center", fontsize=6,
                     color="white" if k == "both" else INK)
        left += w
    axb.text(103, yb, f"n = {r.n:,}", va="center", ha="left", fontsize=6.2, color=INK2, clip_on=False)
axb.set_yticks([0, -1, -2])
axb.set_yticklabels(B.method)
for t, (_, _, c) in zip(axb.get_yticklabels(), METH):
    t.set_color(c)
axb.set_xlim(0, 100)
axb.set_ylim(-2.5, 0.5)
axb.set_xticks([0, 50, 100])
axb.set_xticklabels(["0", "50", "100%"])
axb.spines["left"].set_visible(False)
axb.set_xlabel("Share of flipping user–split observations")
import matplotlib.patches as mpatches  # noqa: E402
axb.legend(handles=[mpatches.Patch(color=c, label=lab) for _, lab, c in cols], loc="lower left",
           bbox_to_anchor=(-0.02, 1.02), ncol=3, frameon=False, fontsize=6.3, handlelength=1.0,
           handletextpad=0.35, columnspacing=0.8)
fig.text(0.655, 0.975, "b", fontsize=9, fontweight="bold", va="top")
fig.text(0.68, 0.975, "Which LastFM predictions flip?", fontsize=7.5, va="top", color=INK)
fig.text(0.68, 0.925, "Protected user\u2013split observations that flip\nin at least one condition, relative to\nthe matched edge-free reference",
         fontsize=6.2, va="top", color=INK2)

fig.text(0.745, 0.10, "No observation flipped with frozen weights only.", fontsize=6.2, color=INK2, va="top")
fig.savefig(os.path.join(OUT, "fig4_certificate.pdf"))
fig.savefig(os.path.join(OUT, "fig4_certificate.png"), dpi=300)

# values for checking
A.drop(columns=["color", "y"]).to_csv(os.path.join(RES, "fig4_panel_a_values.csv"), index=False)
B.to_csv(os.path.join(RES, "fig4_panel_b_values.csv"), index=False)
print(A.drop(columns=["color", "y"]).round(2).to_string())
print(B.to_string())
