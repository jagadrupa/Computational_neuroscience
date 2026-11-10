"""
Figures for the Week 7 notes (synaptic plasticity and Hebbian learning).

Everything here is simulated rather than copied from the slides, so the notes
show the behaviour of the three learning rules as actually run, not as
asserted.

    python make_figs.py      # writes figs/z1_*.pdf ... figs/z4_*.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DESCRIPT = "#1F6FB4"     # blue   -- what is measured / plotted
MECHAN = "#C1440E"       # rust   -- the mechanism, the thing that breaks
INTERP = "#2E7D5B"       # green  -- the interpretation, the stable answer
GREY = "#7A7A7A"

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------
# Test data: a 2-D cloud stretched along one direction.
#
# The whole point of the lecture is that Hebbian learning finds the
# direction of maximum variance, so we need data that HAS an obvious such
# direction. We build it by drawing an elongated Gaussian and rotating it.
# ----------------------------------------------------------------------
def make_cloud(n=400, mean=(0.0, 0.0), spread=(0.55, 0.13), angle=-np.pi / 4):
    """n points, elongated along `angle`, centred on `mean`.

    The default elongation runs along the anti-diagonal while the shifted
    datasets below are moved along the main diagonal. That is deliberate: if
    the cloud were stretched in the same direction as the shift, the mean
    would already lie on the principal axis and the plain Hebb rule would get
    the shifted case right by accident, hiding the effect being demonstrated.
    """
    raw = RNG.normal(size=(n, 2)) * np.array(spread)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return raw @ R.T + np.array(mean)


# ----------------------------------------------------------------------
# The three learning rules, all in the same discretised form
#
#     w  <-  w + eps * (weight change prescribed by the rule)
#
# with v = w . u the linear neuron's output. Inputs are presented one at a
# time in random order, which is what makes this an online rule rather than
# a batch matrix computation.
# ----------------------------------------------------------------------
def train(U, rule, steps=4000, eps=0.002, alpha=1.0, w0=None):
    """Run one rule over the data and return the whole trajectory of w."""
    w = np.array([0.6, -0.3]) if w0 is None else np.array(w0, float)
    u_bar = U.mean(0)                  # input mean, needed by the covariance rule
    traj = [w.copy()]

    for i in range(steps):
        u = U[RNG.integers(len(U))]    # one input pattern
        v = w @ u                      # the neuron's output

        if rule == "hebb":
            dw = u * v
        elif rule == "cov":
            # subtract the average output: v above average -> LTP,
            # v below average -> LTD.
            #
            # <v> is computed as w . <u> rather than as a running average of v.
            # A running average lags behind, and since w itself grows without
            # bound under this rule the lag leaves a positive bias that quietly
            # turns the covariance rule back into the plain Hebb rule.
            dw = u * (v - w @ u_bar)
        elif rule == "oja":
            # the -alpha v^2 w term shrinks w whenever the neuron is active,
            # which is what keeps the length bounded
            dw = u * v - alpha * v**2 * w
        else:
            raise ValueError(rule)

        w = w + eps * dw
        traj.append(w.copy())

    return np.array(traj)


def principal_direction(U, centred):
    """Top eigenvector of the correlation (or covariance) matrix of U."""
    X = U - U.mean(0) if centred else U
    C = X.T @ X / len(X)
    vals, vecs = np.linalg.eigh(C)     # eigh: C is symmetric by construction
    return vecs[:, np.argmax(vals)], vals


# ----------------------------------------------------------------------
# Figure 1 -- stability. Length of w against time for all three rules.
# ----------------------------------------------------------------------
def stability(path):
    U = make_cloud(mean=(0.0, 0.0))
    fig, ax = plt.subplots(figsize=(5.4, 2.9))

    for rule, colour, label in [("hebb", MECHAN, "Hebb"),
                                ("cov", "#B8860B", "covariance"),
                                ("oja", INTERP, r"Oja ($\alpha=1$)")]:
        t = train(U, rule, steps=15000)
        ax.plot(np.linalg.norm(t, axis=1), color=colour, lw=1.5, label=label)

    ax.axhline(1.0, color=GREY, ls=":", lw=1)
    ax.text(15200, 1.0, r"$1/\sqrt{\alpha}$", color=GREY, va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("weight updates")
    ax.set_ylabel(r"$\|\mathbf{w}\|$  (log scale)")
    ax.set_title("Hebb and covariance run away; Oja settles", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    # report the numbers the notes quote
    for rule in ("hebb", "cov", "oja"):
        t = train(U, rule, steps=15000)
        print(f"  {rule:4s} final |w| = {np.linalg.norm(t[-1]):10.3f}")


# ----------------------------------------------------------------------
# Figure 2 -- what the rules converge to, on centred and shifted data.
#
# This is the punchline of the lecture: Hebb finds the top eigenvector of
# the CORRELATION matrix, which only equals the direction of maximum
# variance when the data are already centred on the origin.
# ----------------------------------------------------------------------
def pca_panels(path):
    centred = make_cloud(mean=(0.0, 0.0))
    shifted = make_cloud(mean=(2.0, 2.0))

    panels = [(centred, "hebb", "Hebb rule\ninput mean $(0,0)$", True),
              (shifted, "hebb", "Hebb rule\ninput mean $(2,2)$", False),
              (shifted, "cov", "covariance rule\ninput mean $(2,2)$", True)]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))

    for ax, (U, rule, title, works) in zip(axes, panels):
        ax.scatter(U[:, 0], U[:, 1], s=4, color=GREY, alpha=0.45,
                   edgecolors="none")

        # the true direction of maximum variance, always about the mean
        e_true, _ = principal_direction(U, centred=True)
        m = U.mean(0)
        ax.plot(*np.c_[m - 1.5 * e_true, m + 1.5 * e_true],
                color=DESCRIPT, lw=1.2, ls="--")

        # where the learning rule actually ends up (drawn from the origin,
        # because w is a direction in input space, not a line through the data)
        t = train(U, rule, steps=20000)
        w = t[-1] / np.linalg.norm(t[-1])
        ax.plot([0, 2.4 * w[0]], [0, 2.4 * w[1]],
                color=INTERP if works else MECHAN, lw=2)

        ax.set_title(title, fontsize=8,
                     color=INTERP if works else MECHAN)
        ax.set_xlim(-1.6, 3.6); ax.set_ylim(-1.6, 3.6)
        ax.set_aspect("equal")
        ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
        ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)

        # angle between the learnt w and the true principal direction
        err = np.degrees(np.arccos(abs(w @ e_true)))
        print(f"  {title.splitlines()[0]:18s} {rule:4s}  off by {err:5.1f} deg")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3 -- why the largest eigenvalue wins.
#
# Writing w(t) in the eigenvector basis of Q gives c_i(t) = c_i(0) e^{lam_i t/tau}.
# Every coefficient grows, but they grow at different rates, so the ratio
# between them runs away and the direction of w converges long before its
# length does anything sensible.
# ----------------------------------------------------------------------
def eigen_dominance(path):
    U = make_cloud(mean=(0.0, 0.0))
    e1, vals = principal_direction(U, centred=False)
    lam = np.sort(vals)[::-1]
    tau = 1.0
    t = np.linspace(0, 12, 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # left: the two coefficients, both exploding but at different rates
    for i, (l, colour) in enumerate(zip(lam, [MECHAN, DESCRIPT])):
        ax1.plot(t, np.exp(l * t / tau), color=colour, lw=1.5,
                 label=rf"$\lambda_{i+1} = {l:.2f}$")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$t/\tau_w$"); ax1.set_ylabel(r"$c_i(t)$  (log scale)")
    ax1.set_title("both coefficients grow...", fontsize=9)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    # right: but the ratio is what matters, so the direction converges
    frac = np.exp(lam[0] * t / tau) / (np.exp(lam[0] * t / tau)
                                       + np.exp(lam[1] * t / tau))
    ax2.plot(t, frac, color=INTERP, lw=1.8)
    ax2.axhline(1.0, color=GREY, ls=":", lw=1)
    ax2.set_ylim(0.4, 1.05)
    ax2.set_xlabel(r"$t/\tau_w$")
    ax2.set_ylabel(r"share of $\mathbf{w}$ along $\mathbf{e}_1$")
    ax2.set_title(r"...but $\mathbf{e}_1$ takes over", fontsize=9)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  eigenvalues of Q: {lam[0]:.3f}, {lam[1]:.3f}"
          f"  (ratio {lam[0]/lam[1]:.1f})")


# ----------------------------------------------------------------------
# Figure 4 -- the closing question. Two clusters, where the direction of
# maximum variance describes neither of them.
# ----------------------------------------------------------------------
def clusters(path):
    a = make_cloud(180, mean=(0.15, 1.30), spread=(0.20, 0.16), angle=0.0)
    b = make_cloud(180, mean=(1.30, 0.15), spread=(0.20, 0.16), angle=0.0)
    U = np.vstack([a, b])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.8))

    e_pc, _ = principal_direction(U, centred=True)
    m = U.mean(0)

    for ax in (ax1, ax2):
        ax.scatter(U[:, 0], U[:, 1], s=5, color=GREY, alpha=0.55,
                   edgecolors="none")
        ax.set_xlim(-0.7, 2.2); ax.set_ylim(-0.7, 2.2)
        ax.set_aspect("equal")
        ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")

    # left: what the covariance rule learns
    ax1.plot(*np.c_[m - 1.3 * e_pc, m + 1.3 * e_pc], color=MECHAN, lw=2)
    ax1.set_title("what the covariance rule finds", fontsize=9, color=MECHAN)

    # right: what the data actually look like
    for centre, colour in [(a.mean(0), INTERP), (b.mean(0), INTERP)]:
        ax2.add_patch(plt.Circle(centre, 0.52, fill=False, color=colour,
                                 lw=1.5, ls="--"))
        ax2.plot(*centre, marker="x", color=colour, ms=8, mew=2)
    ax2.set_title("what is actually there", fontsize=9, color=INTERP)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    # how much of each cluster the single PC direction really explains
    proj = (U - m) @ e_pc
    print(f"  variance along PC: {proj.var():.3f};"
          f"  within-cluster residual: {((U - m) @ np.array([-e_pc[1], e_pc[0]])).var():.3f}")


# ======================================================================
# LECTURE 2 -- competitive learning, self-organising maps, and EM
# ======================================================================

def three_clusters(n=40):
    """Three well-separated blobs, roughly like the lecture's example."""
    centres = [(1.0, 1.2), (3.9, 1.3), (3.7, 4.1)]
    return np.vstack([make_cloud(n, mean=c, spread=(0.45, 0.35), angle=0.0)
                      for c in centres])


def competitive(U, w, steps=1500, eps=0.05, snapshots=()):
    """Online competitive learning.

    For each input: find the closest weight vector (the winning neuron), and
    move ONLY that one a fraction eps of the way toward the input. That is
    the running-average update  dw = eps * (u - w)  applied to the winner.
    """
    w = np.array(w, float)
    out = {}
    for i in range(steps + 1):
        if i in snapshots:
            out[i] = w.copy()
        if i == steps:
            break
        u = U[RNG.integers(len(U))]
        j = np.argmin(((w - u) ** 2).sum(1))    # the winner
        w[j] += eps * (u - w[j])                # winner-take-all update
    return w, out


def assign(U, w):
    """Hard assignment of every point to its nearest weight vector."""
    d = ((U[:, None, :] - w[None, :, :]) ** 2).sum(-1)
    return np.argmin(d, axis=1)


# ----------------------------------------------------------------------
# Figure 5 -- competitive learning finds cluster centres.
# ----------------------------------------------------------------------
def competitive_learning(path):
    U = three_clusters()
    w0 = np.array([[2.0, 3.9], [1.3, 2.4], [3.4, 0.4]])   # random-ish start
    snaps = (0, 40, 1500)
    _, out = competitive(U, w0, steps=1500, snapshots=snaps)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    titles = ["initial weights", "after 40 inputs", "after 1500 inputs"]

    for ax, k, title in zip(axes, snaps, titles):
        w = out[k]
        lab = assign(U, w)
        # only colour the points once the weights mean something
        cols = [DESCRIPT, MECHAN, INTERP]
        for j in range(3):
            sel = lab == j
            ax.scatter(U[sel, 0], U[sel, 1], s=8,
                       color=cols[j] if k else GREY, alpha=0.55,
                       edgecolors="none")
        ax.scatter(w[:, 0], w[:, 1], s=55, color="k", marker="o", zorder=5)
        for j, (x, y) in enumerate(w):
            ax.annotate(rf"$\mathbf{{w}}_{j+1}$", (x, y), (x + 0.18, y + 0.18),
                        fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.3, 5.4); ax.set_ylim(-0.3, 5.4)
        ax.set_aspect("equal")
        ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    true_centres = np.array([(1.0, 1.2), (3.9, 1.3), (3.7, 4.1)])
    final = out[1500]
    # match each learnt weight to its nearest true centre
    err = [np.min(np.linalg.norm(true_centres - f, axis=1)) for f in final]
    print("  final weights:", np.round(final, 2).tolist())
    print(f"  distance to nearest true centre: {np.round(err, 3).tolist()}")


# ----------------------------------------------------------------------
# Figure 6 -- the dead-unit problem.
#
# Competitive learning updates ONLY the winner. A unit initialised away from
# all the data never wins, so it never moves, and it never will.
# ----------------------------------------------------------------------
def dead_unit(path):
    U = three_clusters()
    w0 = np.array([[2.0, 3.9], [1.3, 2.4], [6.4, 5.6]])   # third one stranded
    wf, _ = competitive(U, w0, steps=1500)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    lab = assign(U, wf)
    for j, c in enumerate([DESCRIPT, MECHAN, INTERP]):
        sel = lab == j
        ax.scatter(U[sel, 0], U[sel, 1], s=8, color=c, alpha=0.6,
                   edgecolors="none")
    ax.scatter(wf[:2, 0], wf[:2, 1], s=55, color="k", zorder=5)
    ax.scatter(wf[2:, 0], wf[2:, 1], s=70, color=MECHAN, marker="X", zorder=5)
    ax.annotate("never wins,\nnever moves", wf[2], (wf[2, 0] - 3.1, wf[2, 1] - 0.1),
                fontsize=8, color=MECHAN,
                arrowprops=dict(arrowstyle="->", color=MECHAN, lw=1))
    ax.set_xlim(-0.3, 7.3); ax.set_ylim(-0.3, 6.3)
    ax.set_aspect("equal")
    ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")
    ax.set_title("a dead unit", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    moved = np.linalg.norm(wf[2] - w0[2])
    n_owned = int((assign(U, wf) == 2).sum())
    print(f"  stranded unit moved {moved:.4f} and owns {n_owned} points")


# ----------------------------------------------------------------------
# The EM algorithm for a mixture of isotropic Gaussians.
# ----------------------------------------------------------------------
def em(U, mu, sigma, gamma, iters=50, snapshots=()):
    """Expectation-maximisation, exactly the two steps from the lecture."""
    d = U.shape[1]
    mu = np.array(mu, float); sigma = np.array(sigma, float)
    gamma = np.array(gamma, float)
    out = {}

    for it in range(iters + 1):
        # ---- E step: posterior probability of each cause for each point ----
        # p[v|u] proportional to  N(u; mu_v, sigma_v^2 I) * gamma_v
        d2 = ((U[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        like = np.exp(-d2 / (2 * sigma[None, :] ** 2)) \
            / (2 * np.pi * sigma[None, :] ** 2) ** (d / 2)
        joint = like * gamma[None, :]
        resp = joint / joint.sum(1, keepdims=True)     # the responsibilities

        if it in snapshots:
            out[it] = (mu.copy(), sigma.copy(), gamma.copy(), resp.copy())
        if it == iters:
            break

        # ---- M step: re-fit the parameters, weighted by the posteriors ----
        Nk = resp.sum(0)
        mu = (resp.T @ U) / Nk[:, None]
        d2 = ((U[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
        # NOTE the division by d. For an isotropic Gaussian in d dimensions the
        # maximum-likelihood variance is the mean squared distance PER
        # DIMENSION. The lecture slide omits this factor; without it sigma comes
        # out sqrt(2) too large in 2-D and the fitted circles are visibly wrong.
        sigma = np.sqrt((resp * d2).sum(0) / (d * Nk))
        gamma = Nk / len(U)

    return mu, sigma, gamma, out


def two_cluster_data():
    a = make_cloud(180, mean=(0.15, 1.30), spread=(0.20, 0.16), angle=0.0)
    b = make_cloud(180, mean=(1.30, 0.15), spread=(0.20, 0.16), angle=0.0)
    return np.vstack([a, b])


# ----------------------------------------------------------------------
# Figure 7 -- EM converging on the two-cluster data.
# ----------------------------------------------------------------------
def em_iterations(path):
    U = two_cluster_data()
    mu0 = np.array([[0.9, 0.9], [0.5, 0.6]])       # deliberately poor start
    snaps = (2, 5, 50)
    _, _, _, out = em(U, mu0, [0.8, 0.8], [0.5, 0.5], iters=50,
                      snapshots=(0,) + snaps)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, k in zip(axes, snaps):
        mu, sig, gam, resp = out[k]
        # colour each point by its posterior -- the "soft" assignment
        ax.scatter(U[:, 0], U[:, 1], s=7, c=resp[:, 0], cmap="coolwarm",
                   vmin=0, vmax=1, alpha=0.8, edgecolors="none")
        # trajectory of each mean up to this iteration
        for j, colour in enumerate(["k", "k"]):
            path_pts = np.array([out[i][0][j] for i in sorted(out) if i <= k])
            ax.plot(path_pts[:, 0], path_pts[:, 1], color=colour, lw=0.9,
                    ls="--", zorder=4)
            ax.scatter(*mu[j], color="k", s=45, zorder=5)
            # circle at two standard deviations
            ax.add_patch(plt.Circle(mu[j], 2 * sig[j], fill=False, color="k",
                                    lw=1.1))
        ax.set_title(f"iteration {k}", fontsize=9)
        ax.set_xlim(-1.1, 2.4); ax.set_ylim(-1.1, 2.4)
        ax.set_aspect("equal")
        ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    mu, sig, gam, _ = out[50]
    print("  EM means:", np.round(mu, 3).tolist())
    true_sig = np.sqrt((0.20 ** 2 + 0.16 ** 2) / 2)   # isotropic equivalent
    print("  EM sigmas:", np.round(sig, 3).tolist(),
          f" (true isotropic equivalent {true_sig:.3f})")
    print("  EM priors:", np.round(gam, 3).tolist())


# ----------------------------------------------------------------------
# Figure 8 -- hard versus soft competition, side by side.
# ----------------------------------------------------------------------
def hard_vs_soft(path):
    # Deliberately OVERLAPPING clusters. On the well-separated data of the
    # previous figure every posterior is 0 or 1 to several decimal places, so
    # soft and hard assignment produce identical pictures and the distinction
    # being illustrated is invisible.
    a = make_cloud(180, mean=(0.35, 1.05), spread=(0.34, 0.30), angle=0.0)
    b = make_cloud(180, mean=(1.05, 0.35), spread=(0.34, 0.30), angle=0.0)
    U = np.vstack([a, b])
    mu, sig, gam, out = em(U, [[0.9, 0.9], [0.5, 0.6]], [0.8, 0.8],
                           [0.5, 0.5], iters=50, snapshots=(50,))
    resp = out[50][3]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.9))

    hard = (resp[:, 0] > 0.5).astype(float)
    ax1.scatter(U[:, 0], U[:, 1], s=8, c=hard, cmap="coolwarm", vmin=0, vmax=1,
                edgecolors="none")
    ax1.set_title("competitive learning:\nwinner takes all", fontsize=9)

    sc = ax2.scatter(U[:, 0], U[:, 1], s=8, c=resp[:, 0], cmap="coolwarm",
                     vmin=0, vmax=1, edgecolors="none")
    ax2.set_title("EM:\nposterior responsibility", fontsize=9)
    fig.colorbar(sc, ax=ax2, fraction=0.046, label=r"$p[v=A\mid \mathbf{u}]$")

    for ax in (ax1, ax2):
        ax.scatter(mu[:, 0], mu[:, 1], color="k", s=45, zorder=5)
        ax.set_xlim(-0.7, 2.1); ax.set_ylim(-0.7, 2.1)
        ax.set_aspect("equal")
        ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    # how many points are genuinely uncertain?
    amb = int(((resp[:, 0] > 0.1) & (resp[:, 0] < 0.9)).sum())
    print(f"  {amb} of {len(U)} points have a posterior between 0.1 and 0.9")


# ----------------------------------------------------------------------
# A numerical check of the claim that the most active neuron is the one
# whose weight vector is closest -- and of the normalisation it needs.
# ----------------------------------------------------------------------
def closest_vs_most_active():
    u = np.array([0.6, 0.8])                    # already length 1
    w = np.array([[0.8, 0.6], [0.0, 3.0]])      # second is far but LONG

    v = w @ u
    dist = np.linalg.norm(w - u, axis=1)
    print(f"  raw weights   : v = {np.round(v, 3).tolist()},"
          f" distance = {np.round(dist, 3).tolist()}"
          f" -> most active {np.argmax(v)+1}, closest {np.argmin(dist)+1}")

    wn = w / np.linalg.norm(w, axis=1, keepdims=True)
    v = wn @ u
    dist = np.linalg.norm(wn - u, axis=1)
    print(f"  unit weights  : v = {np.round(v, 3).tolist()},"
          f" distance = {np.round(dist, 3).tolist()}"
          f" -> most active {np.argmax(v)+1}, closest {np.argmin(dist)+1}")


# ======================================================================
# LECTURE 3 -- sparse coding and predictive coding
# ======================================================================

# ----------------------------------------------------------------------
# Figure 9 -- sparse (super-Gaussian) priors against a Gaussian.
#
# All three are written as p(v) proportional to exp(g(v)); the lecture's two
# sparse choices are g(v) = -|v| (exponential / Laplace) and
# g(v) = -log(1 + v^2) (Cauchy).
# ----------------------------------------------------------------------
def sparse_priors(path):
    v = np.linspace(-4, 4, 800)

    forms = [("Gaussian", -v ** 2 / 2, GREY, ":"),
             (r"exponential:  $g(v)=-|v|$", -np.abs(v), DESCRIPT, "--"),
             (r"Cauchy:  $g(v)=-\log(1+v^2)$", -np.log(1 + v ** 2), MECHAN, "-")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for label, g, colour, ls in forms:
        p = np.exp(g)
        p /= np.trapezoid(p, v)           # normalise so the areas are comparable
        ax1.plot(v, p, color=colour, ls=ls, lw=1.6, label=label)
        ax2.plot(v, np.log(p), color=colour, ls=ls, lw=1.6)

    ax1.set_xlabel("$v$"); ax1.set_ylabel("$p(v)$")
    ax1.set_title("peak at zero, heavy tails", fontsize=9)
    ax1.legend(frameon=False, fontsize=7.5)
    ax2.set_xlabel("$v$"); ax2.set_ylabel(r"$\log p(v)$")
    ax2.set_ylim(-9, 0.5)
    ax2.set_title("the log prior is $g(v)$ up to a constant", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    # Kurtosis is the usual summary of "super-Gaussian", but it is the wrong
    # tool here: the Cauchy distribution has no finite variance, and computing
    # moments on a truncated grid simply discards the heavy tails the number is
    # meant to measure. Instead put all three on a common ROBUST scale (median
    # |v| = 1) and report how much mass sits at the peak and out in the tail.
    wide = np.linspace(-400, 400, 400001)
    for label, _, _, _ in forms:
        name = label.split(":")[0]
        gw = {"Gaussian": -wide ** 2 / 2,
              "exponential": -np.abs(wide),
              "Cauchy": -np.log(1 + wide ** 2)}[name]
        p = np.exp(gw); p /= np.trapezoid(p, wide)
        cdf = np.cumsum(p) * (wide[1] - wide[0])
        med = np.interp(0.75, cdf, wide)          # median of |v|
        peak = np.interp(0.5 * med, wide, cdf) - np.interp(-0.5 * med, wide, cdf)
        tail = 1.0 - (np.interp(5 * med, wide, cdf) - np.interp(-5 * med, wide, cdf))
        print(f"  {name:12s} mass within 0.5 x median |v|: {peak:.3f}"
              f"   beyond 5 x median |v|: {tail:.4f}")


# ----------------------------------------------------------------------
# Figure 10 -- why gradient ascent moves the right way.
# ----------------------------------------------------------------------
def gradient_ascent(path):
    v = np.linspace(-2.2, 3.4, 400)
    F = -(v - 0.7) ** 2 + 2.0                  # any smooth peak will do
    vstar = 0.7

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.plot(v, F, color=DESCRIPT, lw=1.8)
    ax.axvline(vstar, color=GREY, ls=":", lw=1)
    ax.annotate("$v^*$", (vstar, 2.05), fontsize=9, color=GREY, ha="center")

    for v0, colour in [(-1.2, INTERP), (2.6, MECHAN)]:
        F0 = -(v0 - 0.7) ** 2 + 2.0
        slope = -2 * (v0 - 0.7)                # dF/dv
        tv = np.linspace(v0 - 0.55, v0 + 0.55, 2)
        ax.plot(tv, F0 + slope * (tv - v0), color=colour, lw=1.2)
        ax.scatter([v0], [F0], color=colour, s=35, zorder=5)
        ax.annotate("", (v0 + np.sign(slope) * 0.6, F0),
                    (v0, F0),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.6))
        ax.annotate(f"slope {'+' if slope > 0 else '-'}ve",
                    (v0, F0 - 0.55), fontsize=8, color=colour, ha="center")

    ax.set_xlabel("$v$"); ax.set_ylabel("$F(v)$")
    ax.set_title(r"$dv/dt \propto dF/dv$ climbs from either side", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------
# A small "natural image" world: patches built from localised oriented bars.
#
# The point of the lecture is that PCA cannot recover local parts while sparse
# coding can. To show that honestly we need data that genuinely IS made of
# local parts, so we build the ground truth ourselves and then check which
# algorithm recovers it.
# ----------------------------------------------------------------------
P = 8                                          # patch is P x P pixels


def true_dictionary():
    """Localised horizontal and vertical bars -- the 'parts' of this world."""
    atoms = []
    for r in range(P):
        a = np.zeros((P, P)); a[r, :] = 1.0; atoms.append(a)
    for c in range(P):
        a = np.zeros((P, P)); a[:, c] = 1.0; atoms.append(a)
    D = np.array([a.ravel() for a in atoms]).T
    return D / np.linalg.norm(D, axis=0, keepdims=True)


def make_patches(n=4000, active=2, noise=0.05):
    """Each patch is a sparse combination of a few bars, plus a little noise."""
    D = true_dictionary()
    K = D.shape[1]
    V = np.zeros((K, n))
    for j in range(n):
        idx = RNG.choice(K, size=active, replace=False)
        V[idx, j] = RNG.uniform(0.5, 1.5, size=active)
    return D @ V + noise * RNG.normal(size=(P * P, n)), D


def show_atoms(ax, columns, title):
    """Tile a set of basis vectors as small images."""
    k = columns.shape[1]
    side = int(np.ceil(np.sqrt(k)))
    canvas = np.full((side * (P + 1) + 1, side * (P + 1) + 1), np.nan)
    for i in range(k):
        r, c = divmod(i, side)
        a = columns[:, i].reshape(P, P)
        a = a / (np.abs(a).max() + 1e-12)
        canvas[1 + r * (P + 1):1 + r * (P + 1) + P,
               1 + c * (P + 1):1 + c * (P + 1) + P] = a
    ax.imshow(canvas, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def sparse_coding(U, K=16, iters=3000, eta_v=0.10, n_v=200, eta_G=2.0, lam=0.5):
    """The lecture's algorithm, both steps by gradient ascent on F.

        F(v, G) = -1/2 ||u - Gv||^2 + sum_i g(v_i)

    Step 1 (fast):  dv/dt = G^T (u - Gv) + g'(v)      -- recurrent dynamics
    Step 2 (slow):  dG/dt = (u - Gv) v^T              -- Hebbian, Oja-like
    """
    N = U.shape[0]
    G = RNG.normal(size=(N, K)) * 0.1
    G /= np.linalg.norm(G, axis=0, keepdims=True)

    for it in range(iters):
        batch = U[:, RNG.choice(U.shape[1], 128, replace=False)]

        # ---- infer v by running the recurrent network to convergence ----
        v = np.zeros((K, batch.shape[1]))
        for _ in range(n_v):
            err = batch - G @ v                       # prediction error
            # g'(v) for the Cauchy prior g(v) = -log(1 + v^2)
            gprime = -2 * v / (1 + v ** 2)
            v += eta_v * (G.T @ err + lam * gprime)

        # ---- update G with the Hebbian rule, then renormalise ----
        err = batch - G @ v
        G += eta_G * (err @ v.T) / batch.shape[1]
        G /= np.maximum(np.linalg.norm(G, axis=0, keepdims=True), 1e-8)

    return G, v


def localisation(columns):
    """Fraction of each basis vector's energy in its best single row or column.

    A localised bar scores near 1; a global, spread-out pattern scores near
    1/P. This is the number that separates the two representations.
    """
    scores = []
    for i in range(columns.shape[1]):
        a = columns[:, i].reshape(P, P) ** 2
        a = a / a.sum()
        scores.append(max(a.sum(1).max(), a.sum(0).max()))
    return float(np.mean(scores))


def pca_vs_sparse(path):
    U, D = make_patches()
    Uc = U - U.mean(1, keepdims=True)

    # --- PCA: eigenvectors of the covariance matrix ---
    C = Uc @ Uc.T / Uc.shape[1]
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    pcs = vecs[:, order[:16]]

    # --- sparse coding: the lecture's two alternating gradient steps ---
    G, v = sparse_coding(Uc, K=16)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    show_atoms(axes[0], D[:, :16], "the true parts\n(bars we built the data from)")
    show_atoms(axes[1], pcs, "PCA: top 16 eigenvectors")
    show_atoms(axes[2], G, "sparse coding: columns of $G$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    print(f"  localisation  true parts   : {localisation(D[:, :16]):.3f}")
    print(f"  localisation  PCA          : {localisation(pcs):.3f}")
    print(f"  localisation  sparse coding: {localisation(G):.3f}")
    print(f"  mean |v| per patch, sparse : {np.abs(v).mean():.3f};"
          f"  fraction of |v| < 0.1: {(np.abs(v) < 0.1).mean():.2f}")


# ----------------------------------------------------------------------
# Figure 12 -- the trade-off that F encodes.
# ----------------------------------------------------------------------
def sparseness_tradeoff(path):
    U, _ = make_patches(n=1500)
    Uc = U - U.mean(1, keepdims=True)
    lams = [0.0, 0.3, 0.9, 2.0, 4.0, 8.0]

    err, sparsity = [], []
    for lam in lams:
        G, v = sparse_coding(Uc, K=16, iters=800, lam=lam)
        batch = Uc[:, :256]
        vv = np.zeros((16, batch.shape[1]))
        for _ in range(60):
            vv += 0.05 * (G.T @ (batch - G @ vv)
                          + lam * (-2 * vv / (1 + vv ** 2)))
        err.append(float(np.mean(np.sum((batch - G @ vv) ** 2, axis=0))))
        sparsity.append(float((np.abs(vv) < 0.1).mean()))
        print(f"  lambda={lam:4.1f}  recon error={err[-1]:7.3f}"
              f"  fraction near zero={sparsity[-1]:.2f}")

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.plot(sparsity, err, "o-", color=DESCRIPT, lw=1.5)
    for lam, x, y in zip(lams, sparsity, err):
        ax.annotate(rf"$\lambda={lam:g}$", (x, y), (x + 0.012, y + 0.4),
                    fontsize=7.5, color=GREY)
    ax.set_xlabel("fraction of coefficients near zero (sparser $\\rightarrow$)")
    ax.set_ylabel("reconstruction error")
    ax.set_title("$F$ buys sparseness with accuracy", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

if __name__ == "__main__":
    print("stability:");        stability("figs/z1_stability.pdf")
    print("pca panels:");       pca_panels("figs/z2_pca_rules.pdf")
    print("eigen dominance:");  eigen_dominance("figs/z3_eigen.pdf")
    print("clusters:");         clusters("figs/z4_clusters.pdf")
    print("competitive:");      competitive_learning("figs/z5_competitive.pdf")
    print("dead unit:");        dead_unit("figs/z6_dead_unit.pdf")
    print("EM iterations:");    em_iterations("figs/z7_em.pdf")
    print("hard vs soft:");     hard_vs_soft("figs/z8_hard_soft.pdf")
    print("closest vs most active:"); closest_vs_most_active()
    print("sparse priors:");    sparse_priors("figs/z9_sparse_priors.pdf")
    print("gradient ascent:");  gradient_ascent("figs/z10_gradient.pdf")
    print("PCA vs sparse:");    pca_vs_sparse("figs/z11_pca_vs_sparse.pdf")
    print("sparseness tradeoff:"); sparseness_tradeoff("figs/z12_tradeoff.pdf")
    print("done")
