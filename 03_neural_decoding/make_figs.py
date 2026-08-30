"""
Figures for the Week 3 (Neural Decoding) notes that are not on the slides.

Slide 9 of lecture 1 shows only the two decision bounds -- the accumulating
evidence trace itself was an animation, so it is redrawn here. The prior-shift
figure makes explicit what the photoreceptor slides show only for photons.

    python make_figs.py        # writes figs/13_*.pdf and figs/14_*.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DESCRIPT = "#1F6FB4"   # blue  -- the "minus" / breeze hypothesis
MECHAN = "#C1440E"     # rust  -- the "plus" / tiger hypothesis
INTERP = "#2E7D5B"     # green
GREY = "#5A5A5A"

plt.rcParams.update({
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def evidence_walk(path):
    """Log-likelihood ratio accumulating to a bound (a drift-diffusion walk)."""
    rng = np.random.default_rng(4)
    bound, n_max = 6.0, 400

    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    # Three trials. Positive drift = evidence really is a tiger.
    trials = [(0.045, MECHAN, 2.0), (-0.040, DESCRIPT, 1.9), (-0.012, GREY, 2.1)]
    for drift, colour, noise in trials:
        steps = drift + noise * rng.standard_normal(n_max) * 0.35
        walk = np.concatenate([[0.0], np.cumsum(steps)])
        # Stop the trace the moment it first touches either bound.
        hit = np.argmax(np.abs(walk) >= bound)
        if hit == 0 and abs(walk[0]) < bound:
            hit = len(walk) - 1
        ax.plot(np.arange(hit + 1), walk[:hit + 1], color=colour, lw=1.4)
        ax.plot(hit, walk[hit], "o", color=colour, ms=4.5)

    ax.axhline(bound, ls="--", lw=1.0, color=MECHAN)
    ax.axhline(-bound, ls="--", lw=1.0, color=DESCRIPT)
    ax.axhline(0, lw=0.7, color="black")

    ax.text(n_max * 0.995, bound + 0.5, 'decide "tiger"', ha="right",
            color=MECHAN, fontsize=9)
    ax.text(n_max * 0.995, -bound - 1.3, 'decide "breeze"', ha="right",
            color=DESCRIPT, fontsize=9)

    ax.set_xlabel("observations accumulated  $\\longrightarrow$")
    ax.set_ylabel("running sum of $\\log \\ell$")
    ax.set_xlim(0, n_max)
    ax.set_ylim(-bound - 2.4, bound + 2.4)
    ax.set_yticks([-bound, 0, bound])
    ax.set_yticklabels(["$-$bound", "0", "$+$bound"])
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def prior_shift(path):
    """How an unequal prior moves the crossing point of the two distributions."""
    r = np.linspace(-4, 10, 800)

    def gauss(x, mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    minus, plus = gauss(r, 1.6, 1.1), gauss(r, 4.4, 1.1)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)

    for ax, (p_minus, p_plus, title) in zip(axes, [
        (0.5, 0.5, "Equal priors: $P[-] = P[+] = 0.5$"),
        (0.9, 0.1, "Rare signal: $P[-] = 0.9$, $P[+] = 0.1$"),
    ]):
        a, b = minus * p_minus, plus * p_plus
        ax.plot(r, a, color=DESCRIPT, lw=1.6)
        ax.plot(r, b, color=MECHAN, lw=1.6)

        # Crossing point: where the two weighted curves meet, to the right of
        # the "minus" peak. That is where the decision threshold belongs.
        # Look for a SIGN CHANGE in (a - b), not the smallest gap -- far out in
        # the tail both curves are ~0, so the gap is tiny everywhere there.
        d = a - b
        idx = np.flatnonzero((r[:-1] > 1.6) & (d[:-1] > 0) & (d[1:] <= 0))
        z = r[idx[0]]
        ax.axvline(z, ls="--", lw=1.1, color=INTERP)
        ax.text(z + 0.15, ax.get_ylim()[1] * 0.92, "$z$", color=INTERP, fontsize=10)

        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlabel("response  $r$")
        ax.set_yticks([])
        ax.set_xticks([])

    axes[0].set_ylabel("probability $\\times$ prior")
    axes[0].text(0.4, 0.30, "$P[r\\,|-]$", color=DESCRIPT, fontsize=9,
                 transform=axes[0].transAxes, ha="center")
    axes[0].text(0.80, 0.30, "$P[r\\,|+]$", color=MECHAN, fontsize=9,
                 transform=axes[0].transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def ml_vs_map(path):
    """How a prior pulls the MAP estimate away from the maximum likelihood one."""
    s = np.linspace(-6, 6, 900)

    def gauss(x, mu, sd):
        return np.exp(-0.5 * ((x - mu) / sd) ** 2)

    lik = gauss(s, 1.4, 1.0)          # likelihood from the population response
    s_prior = -2.0

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    for ax, (sd_prior, title) in zip(axes, [
        (0.7, "Narrow prior: $\\sigma_{\\rm prior}$ small"),
        (3.0, "Broad prior: $\\sigma_{\\rm prior}$ large"),
    ]):
        pri = gauss(s, s_prior, sd_prior)
        post = lik * pri
        post /= post.max()

        ax.plot(s, lik, color=MECHAN, lw=1.6)
        ax.plot(s, pri, color=GREY, lw=1.3, ls="--")
        ax.plot(s, post, color=DESCRIPT, lw=1.9)

        ax.axvline(s[np.argmax(lik)], color=MECHAN, lw=0.9, ls=":")
        ax.axvline(s[np.argmax(post)], color=DESCRIPT, lw=0.9, ls=":")
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlabel("stimulus  $s$")
        ax.set_yticks([]); ax.set_xticks([])

    axes[0].set_ylabel("(scaled)")
    axes[0].text(0.70, 0.86, "likelihood", color=MECHAN, fontsize=8,
                 transform=axes[0].transAxes, ha="center")
    axes[0].text(0.20, 0.86, "prior", color=GREY, fontsize=8,
                 transform=axes[0].transAxes, ha="center")
    axes[0].text(0.30, 0.30, "posterior", color=DESCRIPT, fontsize=8,
                 transform=axes[0].transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    evidence_walk("figs/13_evidence_walk.pdf")
    prior_shift("figs/14_prior_shift.pdf")
    ml_vs_map("figs/16_ml_vs_map.pdf")
    print("wrote figs/13, 14 and 16")
