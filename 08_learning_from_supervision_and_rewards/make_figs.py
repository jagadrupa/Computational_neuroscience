"""
Figures for the Week 8 notes (neurons as classifiers, supervised learning).

Everything is simulated rather than copied from the slides, so the notes show
the behaviour of the perceptron and of backpropagation as actually run.

    python make_figs.py      # writes figs/y1_*.pdf ... figs/y5_*.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DESCRIPT = "#1F6FB4"     # blue  -- class +1 / what is measured
MECHAN = "#C1440E"       # rust  -- class -1 / the thing that fails
INTERP = "#2E7D5B"       # green -- the working answer
GREY = "#7A7A7A"

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

RNG = np.random.default_rng(8)


def theta(x):
    """The perceptron's output function: +1 if x > 0, else -1."""
    return np.where(x > 0, 1.0, -1.0)


# ----------------------------------------------------------------------
# The perceptron learning rule, exactly as given in the lecture:
#
#     dw_i = eps (v_d - v) u_i        dmu = -eps (v_d - v)
#
# Note the sign asymmetry: a positive error RAISES the weights but LOWERS
# the threshold. Both changes push the weighted sum further above mu, which
# is what a positive error is asking for.
# ----------------------------------------------------------------------
def perceptron_train(U, labels, steps=400, eps=0.05, w0=None, mu0=0.0,
                     snapshots=()):
    w = np.array([0.9, -0.6]) if w0 is None else np.array(w0, float)
    mu = float(mu0)
    snaps, errors = {}, []

    for t in range(steps + 1):
        if t in snapshots:
            snaps[t] = (w.copy(), mu)
        # fraction of the training set currently misclassified
        errors.append(float(np.mean(theta(U @ w - mu) != labels)))
        if t == steps:
            break
        i = RNG.integers(len(U))
        u, vd = U[i], labels[i]
        v = theta(u @ w - mu)
        err = vd - v                       # 0, +2 or -2
        w = w + eps * err * u
        mu = mu - eps * err

    return w, mu, snaps, np.array(errors)


def draw_boundary(ax, w, mu, lim, colour, **kw):
    """Draw the line  w . u = mu  across the visible axes."""
    xs = np.array([-lim, lim])
    if abs(w[1]) > 1e-9:
        ax.plot(xs, (mu - w[0] * xs) / w[1], color=colour, **kw)
    elif abs(w[0]) > 1e-9:
        ax.axvline(mu / w[0], color=colour, **kw)


# ----------------------------------------------------------------------
# Figure 1 -- a perceptron learning a separating hyperplane.
# ----------------------------------------------------------------------
def perceptron_learning(path):
    n = 60
    A = RNG.normal((1.6, 1.5), 0.55, size=(n, 2))       # class +1
    B = RNG.normal((-1.4, -1.2), 0.55, size=(n, 2))     # class -1
    U = np.vstack([A, B])
    y = np.r_[np.ones(n), -np.ones(n)]

    snaps = (0, 5, 400)
    w, mu, out, errors = perceptron_train(U, y, steps=400, snapshots=snaps)

    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.4))
    for ax, t in zip(axes[:3], snaps):
        ww, mm = out[t]
        ax.scatter(A[:, 0], A[:, 1], s=9, color=DESCRIPT, alpha=.7,
                   edgecolors="none")
        ax.scatter(B[:, 0], B[:, 1], s=9, color=MECHAN, alpha=.7,
                   edgecolors="none")
        draw_boundary(ax, ww, mm, 4, INTERP, lw=1.8)
        miss = np.mean(theta(U @ ww - mm) != y)
        ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"after {t} updates\n{miss:.0%} misclassified", fontsize=8)

    axes[3].plot(errors, color=DESCRIPT, lw=1.4)
    axes[3].set_xlabel("updates"); axes[3].set_ylabel("fraction wrong")
    axes[3].set_title("training error", fontsize=8)
    axes[3].set_ylim(-0.03, 0.6)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print(f"  final error {errors[-1]:.3f} after {len(errors)-1} updates;"
          f"  w = {np.round(w,3).tolist()}, mu = {mu:.3f}")


# ----------------------------------------------------------------------
# Figure 2 -- XOR defeats a single perceptron.
# ----------------------------------------------------------------------
XOR_U = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
XOR_Y = np.array([-1., 1., 1., -1.])


def xor_network(u1, u2):
    """The lecture's two-layer perceptron.

    Hidden unit: weights (1, 1), threshold mu = -1.
    Output unit: weights (-1, 2, -1) from (u1, hidden, u2), threshold mu = 1.
    """
    h = theta(u1 * 1 + u2 * 1 - (-1))
    return theta(-1 * u1 + 2 * h - 1 * u2 - 1)


def xor_figure(path):
    # (a) can a single perceptron ever get XOR right?
    best = 1.0
    for trial in range(200):
        w0 = RNG.normal(size=2)
        _, _, _, errors = perceptron_train(XOR_U, XOR_Y, steps=2000,
                                           eps=0.05, w0=w0,
                                           mu0=float(RNG.normal()))
        best = min(best, errors.min())
    print(f"  single perceptron on XOR: best error over 200 restarts"
          f" = {best:.2f} (1 of 4 points)")

    # (b) verify the two-layer network on all four inputs
    ok = all(xor_network(u[0], u[1]) == y for u, y in zip(XOR_U, XOR_Y))
    print("  two-layer network reproduces XOR on all four inputs:", ok)
    for u, y in zip(XOR_U, XOR_Y):
        h = theta(u[0] + u[1] + 1)
        print(f"    u=({u[0]:+.0f},{u[1]:+.0f})  hidden={h:+.0f}"
              f"  out={xor_network(u[0],u[1]):+.0f}  target={y:+.0f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.9))

    for ax in (ax1, ax2):
        for u, y in zip(XOR_U, XOR_Y):
            ax.scatter(*u, s=90, color=DESCRIPT if y > 0 else MECHAN,
                       zorder=5, edgecolors="k", linewidths=0.6)
        ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_xlabel("$u_1$"); ax.set_ylabel("$u_2$")

    # left: a few attempted straight boundaries, none of which work
    for k in range(6):
        w0 = RNG.normal(size=2)
        w, mu, _, _ = perceptron_train(XOR_U, XOR_Y, steps=2000, eps=0.05,
                                       w0=w0, mu0=float(RNG.normal()))
        draw_boundary(ax1, w, mu, 2, GREY, lw=1.0, alpha=0.7)
    ax1.set_title("no straight line separates them", fontsize=9)

    # right: the two-layer network's decision regions
    g = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(g, g)
    Z = xor_network(X, Y)
    ax2.contourf(X, Y, Z, levels=[-2, 0, 2],
                 colors=[MECHAN, DESCRIPT], alpha=0.18)
    ax2.set_title("two-layer network:\nregions, not a half-plane", fontsize=9)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3 -- the sigmoid, and what beta does.
# ----------------------------------------------------------------------
def sigmoid_figure(path):
    a = np.linspace(-8, 8, 600)
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for beta, colour, ls in [(0.25, INTERP, "-"), (1.0, DESCRIPT, "-"),
                             (8.0, MECHAN, "-")]:
        ax.plot(a, 1 / (1 + np.exp(-beta * a)), color=colour, ls=ls, lw=1.6,
                label=rf"$\beta = {beta:g}$")
    ax.step(a, (a > 0).astype(float), color=GREY, ls=":", lw=1.2,
            where="mid", label=r"$\Theta$ (perceptron)")
    ax.set_xlabel("$a$"); ax.set_ylabel("$g(a)$")
    ax.set_title(r"$g(a) = 1/(1+e^{-\beta a})$", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

    # A useful summary is the WIDTH of the transition: the span of a over
    # which g rises from 0.1 to 0.9, which is 2*ln(9)/beta. (Comparing g to
    # the step function pointwise is useless -- the difference is ~0.5 at
    # a = 0 whatever beta is, since g(0) = 0.5 always.)
    for beta in (0.25, 1.0, 8.0):
        print(f"  beta={beta:4g}: 10-90% transition width"
              f" = {2*np.log(9)/beta:6.2f}")


# ----------------------------------------------------------------------
# A three-layer sigmoid network trained by backpropagation.
#
#     x_j = g(sum_k w_jk u_k)          hidden layer
#     v_i = g(sum_j W_ij x_j)          output layer
#     E   = 1/2 sum_i (d_i - v_i)^2
#
# The output-layer update is the delta rule; the hidden-layer update needs
# the chain rule, which is what "backpropagation" names.
# ----------------------------------------------------------------------
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def train_mlp(U, D, n_hidden, epochs=4000, eps=0.5, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n_hidden, U.shape[1])) * 0.8      # input -> hidden
    W = rng.normal(size=(D.shape[1], n_hidden)) * 0.8      # hidden -> output
    hist = []

    for _ in range(epochs):
        x = sigmoid(U @ w.T)                  # hidden activities
        v = sigmoid(x @ W.T)                  # outputs
        err = D - v
        hist.append(0.5 * float(np.sum(err ** 2)) / len(U))

        # delta rule at the output layer
        dv = err * v * (1 - v)                # (d - v) g'(a)
        dW = dv.T @ x
        # backpropagate through the hidden layer via the chain rule
        dx = (dv @ W) * x * (1 - x)
        dw = dx.T @ U

        W += eps * dW / len(U)
        w += eps * dw / len(U)

    return w, W, np.array(hist)


def backprop_figure(path):
    # XOR again, now as regression with 0/1 targets and sigmoid units
    U = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    D = np.array([[0.], [1.], [1.], [0.]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

    EPOCHS, LR, SEEDS = 30_000, 2.0, 10
    finals = {}
    for n_hidden, colour, lab in [(1, MECHAN, "1 hidden unit"),
                                  (2, "#B8860B", "2 hidden units"),
                                  (4, INTERP, "4 hidden units")]:
        runs = [train_mlp(U, D, n_hidden, epochs=EPOCHS, eps=LR, seed=s)[2]
                for s in range(SEEDS)]
        best = min(runs, key=lambda h: h[-1])
        ax1.plot(best, color=colour, lw=1.5, label=lab)
        solved = sum(r[-1] < 1e-3 for r in runs)
        finals[n_hidden] = (float(best[-1]), solved, SEEDS)

    ax1.set_yscale("log")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("error $E$ per pattern")
    ax1.set_title("backpropagation on XOR (best of 10 runs)", fontsize=9)
    ax1.legend(frameon=False, fontsize=8)

    # the learnt function, for the 4-unit network
    w, W, _ = train_mlp(U, D, 4, epochs=EPOCHS, eps=LR, seed=0)
    g = np.linspace(-0.3, 1.3, 200)
    X, Y = np.meshgrid(g, g)
    grid = np.c_[X.ravel(), Y.ravel()]
    Z = sigmoid(sigmoid(grid @ w.T) @ W.T).reshape(X.shape)
    im = ax2.contourf(X, Y, Z, levels=20, cmap="coolwarm", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax2, fraction=0.046, label="output $v$")
    for u, d in zip(U, D):
        ax2.scatter(*u, s=80, color="w", edgecolors="k", zorder=5)
        ax2.annotate(f"{d[0]:.0f}", u, ha="center", va="center", fontsize=7,
                     zorder=6)
    ax2.set_aspect("equal")
    ax2.set_xlabel("$u_1$"); ax2.set_ylabel("$u_2$")
    ax2.set_title("what the network learnt", fontsize=9)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)

    for k, (best, solved, n) in finals.items():
        print(f"  {k} hidden unit(s): best final error {best:.6f};"
              f" runs reaching <1e-3: {solved}/{n}")


# ======================================================================
# LECTURE 2 -- reinforcement learning: predicting rewards
# ======================================================================

T_TRIAL = 250          # time steps in one trial
T_STIM = 100           # the bell
T_REW = 200            # the food


def trial_signals(reward=True):
    """Stimulus and reward time courses for one trial.

    The stimulus is a single event at t=100; the reward is a single event of
    size 2 at t=200. The learnt prediction settles at 2 because that is the
    total future reward from the moment the bell rings.
    """
    u = np.zeros(T_TRIAL)
    u[T_STIM] = 1.0
    r = np.zeros(T_TRIAL)
    if reward:
        r[T_REW] = 2.0
    return u, r


def td_learn(n_trials=200, eps=0.8, reward=True, snapshots=()):
    """Temporal-difference learning with a tapped delay line.

        v(t)  = sum_tau w(tau) u(t - tau)          -- a linear filter
        delta = r(t) + v(t+1) - v(t)               -- the TD error
        dw(tau) += eps * delta * u(t - tau)

    The weights are indexed by DELAY since the stimulus, which is what the
    tapped delay line provides: w(tau) is the prediction contributed tau
    steps after a stimulus.
    """
    w = np.zeros(T_TRIAL)
    u, r = trial_signals(reward)
    snaps, deltas = {}, []

    for trial in range(n_trials + 1):
        # forward pass: the prediction at every time step in the trial
        v = np.zeros(T_TRIAL + 1)          # v(T) = 0 by definition
        for t in range(T_TRIAL):
            taus = np.arange(0, t + 1)
            v[t] = np.sum(w[taus] * u[t - taus])

        d = np.array([r[t] + v[t + 1] - v[t] for t in range(T_TRIAL)])
        deltas.append(d)
        if trial in snapshots:
            snaps[trial] = (w.copy(), v[:T_TRIAL].copy(), d.copy())
        if trial == n_trials:
            break

        # weight update, accumulated over the trial
        dw = np.zeros(T_TRIAL)
        for t in range(T_TRIAL):
            taus = np.arange(0, t + 1)
            dw[taus] += eps * d[t] * u[t - taus]
        w += dw

    return w, snaps, np.array(deltas)


# ----------------------------------------------------------------------
# Figure 5 -- before and after learning.
# ----------------------------------------------------------------------
def td_before_after(path):
    _, snaps, _ = td_learn(n_trials=200, snapshots=(0, 200))
    u, r = trial_signals()

    fig, axes = plt.subplots(4, 2, figsize=(7.0, 4.6), sharex=True)
    for col, (trial, title) in enumerate([(0, "before learning"),
                                          (200, "after learning")]):
        w, v, d = snaps[trial]
        for row, (sig, lab, colour) in enumerate([
                (u, r"stimulus $u$", GREY),
                (r, r"reward $r$", GREY),
                (v, r"prediction $v$", DESCRIPT),
                (d, r"error $\delta$", MECHAN)]):
            ax = axes[row, col]
            ax.plot(sig, color=colour, lw=1.3)
            ax.set_ylim(-1.4, 2.6)
            ax.set_yticks([0, 2])
            if col == 0:
                ax.set_ylabel(lab, fontsize=8)
            if row == 0:
                ax.set_title(title, fontsize=9)
            ax.axhline(0, color="k", lw=0.4)
    for ax in axes[3]:
        ax.set_xlabel("$t$ within trial")

    fig.tight_layout(); fig.savefig(path); plt.close(fig)

    w, v, d = snaps[200]
    print(f"  after learning: v at stimulus (t=100) = {v[T_STIM]:.3f}"
          f"   (total future reward = 2)")
    print(f"  delta at reward (t=200)      = {d[T_REW]:+.4f}")
    # delta = r(t) + v(t+1) - v(t), so the jump in v shows up at t = 99:
    # delta(99) = v(100) - v(99) = 2 - 0. At t = 100 both sides are already 2.
    print(f"  delta peaks at t={int(np.argmax(d))} with value {d.max():+.4f}"
          f"   [delta(99) = v(100) - v(99)]")
    w0, v0, d0 = snaps[0]
    print(f"  before learning: delta at reward = {d0[T_REW]:+.3f},"
          f" peak at t={int(np.argmax(d0))}")


# ----------------------------------------------------------------------
# Figure 6 -- the error bump walking backwards in time, trial by trial.
# ----------------------------------------------------------------------
def td_backward_shift(path):
    _, _, deltas = td_learn(n_trials=140)   # long enough to complete the walk

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    im = ax1.imshow(deltas, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, T_TRIAL, 0, len(deltas)], vmin=0, vmax=2)
    fig.colorbar(im, ax=ax1, fraction=0.046, label=r"$\delta$")
    ax1.set_xlabel("$t$ within trial"); ax1.set_ylabel("trial")
    ax1.set_title("the error walks back to the stimulus", fontsize=9)
    ax1.axvline(T_STIM, color="w", lw=0.8, ls=":")
    ax1.axvline(T_REW, color="w", lw=0.8, ls=":")

    # where the error peaks, trial by trial
    peaks = [int(np.argmax(d)) for d in deltas]  # t=200 initially, t=99 at the end
    ax2.plot(peaks, color=DESCRIPT, lw=1.5)
    ax2.axhline(T_STIM, color=GREY, ls=":", lw=1)
    ax2.axhline(T_REW, color=GREY, ls=":", lw=1)
    ax2.annotate("stimulus", (len(peaks) * 0.55, T_STIM + 5), fontsize=8,
                 color=GREY)
    ax2.annotate("reward", (len(peaks) * 0.55, T_REW + 5), fontsize=8,
                 color=GREY)
    ax2.set_xlabel("trial"); ax2.set_ylabel("time of peak $\\delta$")
    ax2.set_title("steadily, until it arrives", fontsize=9)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)

    print(f"  peak delta at trial 0: t={peaks[0]};"
          f" trial 20: t={peaks[20]}; trial 100: t={peaks[100]};"
          f" final: t={peaks[-1]}")
    moved = peaks[0] - peaks[-1]
    reached = next((i for i, pk in enumerate(peaks) if pk <= T_STIM), None)
    print(f"  the peak moved back {moved} steps in {len(peaks)-1} trials"
          f"; first reached the stimulus at trial {reached}")


# ----------------------------------------------------------------------
# Figure 7 -- omitting an expected reward gives a negative error.
# ----------------------------------------------------------------------
def td_omission(path):
    w, _, _ = td_learn(n_trials=200)          # train normally

    # then run one trial each way, WITHOUT further learning
    out = {}
    for label, reward in [("reward delivered", True), ("reward omitted", False)]:
        u, r = trial_signals(reward)
        v = np.zeros(T_TRIAL + 1)
        for t in range(T_TRIAL):
            taus = np.arange(0, t + 1)
            v[t] = np.sum(w[taus] * u[t - taus])
        d = np.array([r[t] + v[t + 1] - v[t] for t in range(T_TRIAL)])
        out[label] = d

    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.plot(out["reward delivered"], color=INTERP, lw=1.4,
            label="reward delivered")
    ax.plot(out["reward omitted"], color=MECHAN, lw=1.4, label="reward omitted")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(T_STIM, color=GREY, ls=":", lw=0.9)
    ax.axvline(T_REW, color=GREY, ls=":", lw=0.9)
    ax.set_xlabel("$t$ within trial"); ax.set_ylabel(r"$\delta$")
    ax.set_title("a dip exactly when the reward was due", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

    dl, dm = out["reward delivered"], out["reward omitted"]
    print(f"  at stimulus t=100: delivered {dl[T_STIM]:+.3f},"
          f" omitted {dm[T_STIM]:+.3f}  (identical -- nothing has happened yet)")
    print(f"  at reward  t=200: delivered {dl[T_REW]:+.3f},"
          f" omitted {dm[T_REW]:+.3f}")
    print(f"  most negative delta when omitted: {dm.min():+.3f}"
          f" at t={int(np.argmin(dm))}")

# ======================================================================
# LECTURE 3 -- acting to maximise reward: the rat in the barn
# ======================================================================
#
# The maze. Three decision points, each offering left or right. Rewards
# sit at the four leaves; the decision states themselves pay nothing.
#
#              A
#          L /   \ R
#           B     C
#        L / \R  L/ \R
#         0   5  2   0
#
MAZE = {
    "A": {"L": ("B", 0.0), "R": ("C", 0.0)},
    "B": {"L": (None, 0.0), "R": (None, 5.0)},
    "C": {"L": (None, 2.0), "R": (None, 0.0)},
}
STATES = ["A", "B", "C"]
ACTIONS = ["L", "R"]

# Values under a random policy, worked out by hand in the lecture:
#   v(B) = .5*0 + .5*5 = 2.5      v(C) = .5*2 + .5*0 = 1
#   v(A) = .5*v(B) + .5*v(C)      = 1.75
TRUE_RANDOM_VALUES = {"A": 1.75, "B": 2.5, "C": 1.0}


def td_values(n_trials=400, eps=0.5, seed=0):
    """Critic only: learn v(u) under a RANDOM policy by TD learning.

        w(u) <- w(u) + eps [ r + v(u') - v(u) ],   v(terminal) = 0

    The rat experiences states one at a time, so this is online learning --
    it never sees the tree, only the sequence of places it happens to visit.
    """
    rng = np.random.default_rng(seed)
    w = {s: 0.0 for s in STATES}
    hist = []

    for _ in range(n_trials):
        u = "A"
        while u is not None:
            a = ACTIONS[rng.integers(2)]          # the random policy
            nxt, r = MAZE[u][a]
            v_next = w[nxt] if nxt is not None else 0.0
            w[u] += eps * (r + v_next - w[u])
            u = nxt
        hist.append([w[s] for s in STATES])

    return w, np.array(hist)


def softmax_probs(Q, u, beta):
    q = np.array([Q[u][a] for a in ACTIONS])
    e = np.exp(beta * (q - q.max()))              # shift for numerical safety
    return e / e.sum()


def actor_critic(n_trials=800, eps=0.5, beta=1.0, seed=0):
    """Actor-critic learning.

    Critic: w(u) <- w(u) + eps * delta                       (policy evaluation)
    Actor:  Q_a'(u) <- Q_a'(u) + eps * delta * (d_aa' - P(a';u))

    Note the actor's bracket is (delta_aa' - P(a';u)), not the bare Kronecker
    delta: the action actually taken gets 1 - P(a;u) > 0, and every other
    action gets -P(a';u) < 0. So a better-than-expected outcome raises the
    chosen action's Q AND lowers its alternatives.
    """
    rng = np.random.default_rng(seed)
    w = {s: 0.0 for s in STATES}
    Q = {s: {a: 0.0 for a in ACTIONS} for s in STATES}
    probs, visits = [], {s: 0 for s in STATES}
    visit_hist = []

    for _ in range(n_trials):
        u = "A"
        while u is not None:
            visits[u] += 1
            p = softmax_probs(Q, u, beta)
            a = ACTIONS[rng.choice(2, p=p)]
            nxt, r = MAZE[u][a]

            v_next = w[nxt] if nxt is not None else 0.0
            delta = r + v_next - w[u]
            w[u] += eps * delta                    # critic

            for i, ap in enumerate(ACTIONS):       # actor
                Q[u][ap] += eps * delta * ((1.0 if ap == a else 0.0) - p[i])
            u = nxt

        probs.append([softmax_probs(Q, s, beta)[0] for s in STATES])
        visit_hist.append([visits[s] for s in STATES])

    return w, Q, np.array(probs), np.array(visit_hist)


# ----------------------------------------------------------------------
# Figure 8 -- the critic learning state values under a random policy.
# ----------------------------------------------------------------------
def td_barn_values(path):
    w, hist = td_values(n_trials=1000)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4), sharex=True)

    for k, (ax, s) in enumerate(zip(axes, STATES)):
        ax.plot(hist[:, k], color=GREY, lw=0.6, alpha=0.8)
        # running average, which is what converges cleanly
        run = np.cumsum(hist[:, k]) / np.arange(1, len(hist) + 1)
        ax.plot(run, color=DESCRIPT, lw=1.8)
        ax.axhline(TRUE_RANDOM_VALUES[s], color=MECHAN, ls=":", lw=1.2)
        ax.set_title(f"$w({s})\\rightarrow{TRUE_RANDOM_VALUES[s]}$", fontsize=9)
        ax.set_xlabel("trial")
        if k == 0:
            ax.set_ylabel("value")
        ax.set_ylim(-0.3, 5.4)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)

    # eps = 0.5 makes single estimates jump around, so quote the average of
    # the second half of the run -- which is what the lecture's dark line is.
    run_final = hist[len(hist)//2:].mean(0)
    # One run at eps = 0.5 is noisy, so also average across seeds to check
    # that the rule really converges on the hand-calculated values.
    multi = np.stack([td_values(n_trials=2000, seed=sd)[1][-1000:].mean(0)
                      for sd in range(20)])
    for k, st in enumerate(STATES):
        print(f"  v({st}): this run {run_final[k]:.3f};"
              f"  over 20 seeds {multi[:, k].mean():.3f} +/- {multi[:, k].std():.3f}"
              f"   (hand calculation {TRUE_RANDOM_VALUES[st]})")


# ----------------------------------------------------------------------
# Figure 9 -- actor-critic finding the optimal policy.
# ----------------------------------------------------------------------
def actor_critic_barn(path):
    """Actor-critic on the barn, averaged over 20 seeds.

    A single run is misleading here: state C is visited so rarely that one
    lucky early experience can commit its policy, which hides the very
    exploration effect the lecture is pointing at.
    """
    SEEDS = 20
    runs = [actor_critic(n_trials=1000, eps=0.1, seed=sd) for sd in range(SEEDS)]
    probs = np.stack([r[2] for r in runs])          # (seeds, trials, states)
    visits = np.stack([r[3] for r in runs])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    for k, (st, colour) in enumerate(zip(STATES, [DESCRIPT, MECHAN, INTERP])):
        for sd in range(SEEDS):                      # individual runs, faint
            ax1.plot(probs[sd, :, k], color=colour, lw=0.4, alpha=0.15)
        ax1.plot(probs[:, :, k].mean(0), color=colour, lw=1.8, label=f"at {st}")
    ax1.axhline(0.5, color=GREY, ls=":", lw=1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("trial"); ax1.set_ylabel(r"$P(\mathrm{left})$")
    ax1.set_title("probability of going left", fontsize=9)
    ax1.legend(frameon=False, fontsize=8, loc="center right")

    for k, (st, colour) in enumerate(zip(STATES, [DESCRIPT, MECHAN, INTERP])):
        ax2.plot(visits[:, :, k].mean(0), color=colour, lw=1.6, label=st)
    ax2.set_xlabel("trial"); ax2.set_ylabel("cumulative visits")
    ax2.set_title("C is barely visited at all", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)

    fig.tight_layout(); fig.savefig(path); plt.close(fig)

    want_left = {"A": True, "B": False, "C": True}
    for k, st in enumerate(STATES):
        col = probs[:, :, k] if want_left[st] else 1 - probs[:, :, k]
        settle = []
        for sd in range(SEEDS):
            c = col[sd]
            idx = len(c)
            for i in range(len(c) - 1, -1, -1):
                if c[i] <= 0.9:
                    idx = i + 1
                    break
            else:
                idx = 0
            settle.append(idx)
        never = sum(1 for x in settle if x >= probs.shape[1])
        print(f"  {st}: mean final P(correct) {col[:, -1].mean():.3f};"
              f" median settle {np.median(settle):6.0f};"
              f" never settled in {never}/{SEEDS} runs;"
              f" mean visits {visits[:, -1, k].mean():5.0f}")

if __name__ == "__main__":
    print("perceptron learning:"); perceptron_learning("figs/y1_perceptron.pdf")
    print("XOR:");                 xor_figure("figs/y2_xor.pdf")
    print("sigmoid:");             sigmoid_figure("figs/y3_sigmoid.pdf")
    print("backpropagation:");     backprop_figure("figs/y4_backprop.pdf")
    print("TD before/after:");     td_before_after("figs/y5_td.pdf")
    print("TD backward shift:");   td_backward_shift("figs/y6_shift.pdf")
    print("TD omission:");         td_omission("figs/y7_omission.pdf")
    print("barn values:");         td_barn_values("figs/y8_barn_values.pdf")
    print("actor-critic:");        actor_critic_barn("figs/y9_actor_critic.pdf")
    print("done")
