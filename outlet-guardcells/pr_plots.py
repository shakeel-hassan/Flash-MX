"""Figures for the outlet guard-cell PR: shared-face velocity mismatch, staged vs fix.

Case A: cold start t = 0 -> 2.  Case B: restart from the production checkpoint t = 60 -> 62
(checkpoint written by staged, so it already carries the mismatch)."""
import glob, os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

S = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(S, "pr_figs"); os.makedirs(OUT, exist_ok=True)

C_BASE, C_FIX = "#eb6834", "#2a78d6"          # reference categorical slots 2 / 1
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f2f1ed"
FLOOR = 1e-17                                    # exact zeros drawn here
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2})
LAB_BASE, LAB_FIX = "staged (current)", "this PR"


def read(fn):
    with h5py.File(fn, "r") as f:
        d = dict(bb=f["bounding box"][:], fx=f["vlfx"][:, 0], fy=f["vlfy"][:, 0], phi=f["dfun"][:, 0])
        d["t"] = {k.decode().strip(): v for k, v in f["real scalars"][:]}["time"]
    return d


def shared_faces(d):
    """List of (x, y, |u_A-u_B|) for every face shared by two blocks, plus max|u|."""
    bb, fx, fy = d["bb"], d["fx"], d["fy"]
    bs = bb[0, :2, 1] - bb[0, :2, 0]; n = fx.shape[1]; h = bs[0] / n
    key = {(round(b[0, 0] / bs[0]), round(b[1, 0] / bs[1])): i for i, b in enumerate(bb)}
    pts = []
    for (I, J), a in key.items():
        if (I, J + 1) in key:   # horizontal boundary, y-faces
            m = np.abs(fy[a][-1, :] - fy[key[(I, J + 1)]][0, :])
            x = bb[a, 0, 0] + (np.arange(n) + 0.5) * h
            pts.append(np.c_[x, np.full(n, bb[a, 1, 1]), m])
        if (I + 1, J) in key:   # vertical boundary, x-faces
            m = np.abs(fx[a][:, -1] - fx[key[(I + 1, J)]][:, 0])
            y = bb[a, 1, 0] + (np.arange(n) + 0.5) * h
            pts.append(np.c_[np.full(n, bb[a, 0, 1]), y, m])
    return np.vstack(pts), max(np.abs(fx).max(), np.abs(fy).max())


def by_height(d):
    P, um = shared_faces(d)
    ys = np.unique(np.round(P[:, 1], 6)); ys = ys[np.isclose(ys % 0.5, 0)]
    rows = np.array([P[np.isclose(P[:, 1], y), 2].max() for y in ys])
    return ys, np.maximum(rows / um, FLOOR)


def series(files):
    t, g = [], []
    for fn in files:
        d = read(fn); P, um = shared_faces(d)
        if um == 0:          # initial condition, fluid at rest
            continue
        t.append(d["t"]); g.append(P[:, 2].max() / um)
    o = np.argsort(t)
    return np.array(t)[o], np.maximum(np.array(g)[o], FLOOR)


def plts(name):
    return sorted(glob.glob(os.path.join(S, name, "INS_Pool_Boiling_hdf5_plt_cnt_*")))

A = dict(base=plts("long_base"), fix=plts("long_fork"))
B = dict(base=plts("rst_base"), fix=plts("rst_fork"))

# ---------------- Figure 1: mismatch vs height ----------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
for ax, case, ttl in [(axs[0], A, "Cold start"), (axs[1], B, "Restart from t = 60 checkpoint")]:
    db, df = read(case["base"][-1]), read(case["fix"][-1])
    yb, mb = by_height(db); yf, mf = by_height(df)
    ax.axhspan(15, 16, color=BAND, zorder=0)
    prof = 2 / (1 + np.exp(-4.0 * (yb - 16.0)))
    ax.semilogx(prof / prof[-1] * mb[-1], yb, ":", color=INK2, lw=1.5, label="outlet forcing profile (scaled)")
    ax.semilogx(mb, yb, "-o", color=C_BASE, ms=4, label=LAB_BASE)
    ax.semilogx(mf, yf, "-o", color=C_FIX, ms=4, label=LAB_FIX)
    ax.set_xlim(FLOOR * 0.5, 1e-2); ax.set_ylim(0, 16)
    ax.set_title(f"{ttl}, t = {db['t']:.1f}", loc="left", fontsize=10, color=INK)
    ax.set_xlabel(r"max $|u_A-u_B|$ / max$|u|$, faces shared by blocks A, B")
axs[0].set_ylabel("height of horizontal block boundary, y")
axs[0].text(2e-17, 15.5, "outlet buffer", va="center", fontsize=8, color=INK2)
axs[1].legend(loc="lower left", fontsize=8)
fig.suptitle("Shared-face velocity mismatch by height", x=0.01, ha="left", fontsize=11, color=INK)
fig.text(0.01, 0.005, "2D FC-72 pool boiling, 16x16 domain, 32x32 blocks of 16x16 cells (AMReX). "
         "Values at round-off or exactly zero are drawn at 1e-17.", fontsize=7, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(os.path.join(OUT, "mismatch_vs_height.png"), dpi=160)

# ---------------- Figure 2: mismatch over time ----------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
summary = []
for ax, case, ttl in [(axs[0], A, "Cold start, t = 0 to 2"), (axs[1], B, "Restart from staged checkpoint, t = 60 to 62")]:
    tb, gb = series(case["base"]); tf, gf = series(case["fix"])
    ax.semilogy(tb, gb, "-o", color=C_BASE, ms=4, label=LAB_BASE)
    ax.semilogy(tf, gf, "-o", color=C_FIX, ms=4, label=LAB_FIX)
    ax.set_title(ttl, loc="left", fontsize=10, color=INK); ax.set_xlabel("t")
    summary.append((ttl, tb, gb, tf, gf))
axs[0].set_ylabel("max shared-face mismatch / max|u|"); axs[0].set_ylim(FLOOR * 0.5, 1e-2)
axs[0].legend(loc="center right")
fig.suptitle("Mismatch over time", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "mismatch_vs_time.png"), dpi=160)

# ---------------- Figure 3: map while a bubble crosses the outlet buffer ----------------
def global_phi(d):
    bb, phi = d["bb"], d["phi"]; n = phi.shape[1]; h = (bb[0, 0, 1] - bb[0, 0, 0]) / n
    x0, y0 = bb[:, 0, 0].min(), bb[:, 1, 0].min()
    NX = int(round((bb[:, 0, 1].max() - x0) / h)); NY = int(round((bb[:, 1, 1].max() - y0) / h))
    G = np.empty((NY, NX))
    for b in range(bb.shape[0]):
        i0 = int(round((bb[b, 0, 0] - x0) / h)); j0 = int(round((bb[b, 1, 0] - y0) / h))
        G[j0:j0 + n, i0:i0 + n] = phi[b]
    return x0 + (np.arange(NX) + .5) * h, y0 + (np.arange(NY) + .5) * h, G

SNAP = "0616"
fig, axs = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
norm = LogNorm(1e-12, 1e-3)
for ax, key, lab in [(axs[0], "base", LAB_BASE), (axs[1], "fix", LAB_FIX)]:
    fn = [f for f in B[key] if f.endswith(SNAP)][0]
    d = read(fn); P, um = shared_faces(d); x, y, G = global_phi(d)
    ax.axhspan(15, 16, color=BAND, zorder=0)
    ax.contour(x, y, G, [0], colors=INK, linewidths=0.9, zorder=2)
    v = P[:, 2] / um; k = v > 1e-12
    sc = ax.scatter(P[k, 0], P[k, 1], c=v[k], s=4, cmap="magma_r", norm=norm, linewidths=0, zorder=3)
    ax.set_xlim(-8, 8); ax.set_ylim(10, 16); ax.set_aspect("equal"); ax.grid(False)
    ax.set_title(f"{lab}: {k.sum():,} of {len(v):,} shared faces differ by more than 1e-12 max|u|",
                 loc="left", fontsize=9, color=INK)
    ax.set_ylabel("y")
    ax.text(-7.9, 15.08, "outlet buffer", ha="left", va="bottom", fontsize=8, color=INK2, zorder=4)
axs[1].set_xlabel("x")
fig.colorbar(sc, ax=axs, shrink=0.8, pad=0.02, label=r"$|u_A-u_B|$ / max$|u|$")
fig.suptitle(f"Restart case at t = {d['t']:.1f}: small bubble leaving through the outlet buffer (outlines), "
             "disagreeing shared faces (dots)", x=0.01, ha="left", fontsize=10, color=INK)
fig.savefig(os.path.join(OUT, "mismatch_map_t61p5.png"), dpi=160, bbox_inches="tight")

with open(os.path.join(OUT, "summary.txt"), "w") as fh:
    for ttl, tb, gb, tf, gf in summary:
        fh.write(f"{ttl}\n  staged: start {gb[0]:.2e}  end {gb[-1]:.2e}  max {gb.max():.2e}\n"
                 f"  fix:    start {gf[0]:.2e}  end {gf[-1]:.2e}  max {gf.max():.2e}\n")
print(open(os.path.join(OUT, "summary.txt")).read())
