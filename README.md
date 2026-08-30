# Computational Neuroscience

My study notes for computational neuroscience, written in LaTeX.

## The course in brief

Four questions, each building on the last:

1. **What is a neuron, and how does it compute?** — the biological substrate, and the three levels at which it can be modelled.
2. **Encoding** — how does a stimulus become spikes? Build a forward model and fit it from data.
3. **Decoding** — how do you recover the stimulus from spikes? Two alternatives, a continuous parameter, then a whole time-varying signal.
4. **Information** — how good is the code, independent of any particular decoder, and what shapes it?

The recurring theme is that these are the same problem seen from different sides: the spike-triggered average is both an encoding tool and the optimal single-spike decoder, and modern stimulus reconstruction works by running an encoding model in a loop.

## Skills

### Neuroscience

- Recording methods and their trade-offs — fMRI, EEG, multi-electrode arrays, calcium imaging, patch clamp — and what analysis each makes possible
- Receptive fields: centre–surround, orientation-selective, spatiotemporal; how they are measured and what they compute
- Tuning curves (Gaussian, cosine), functional maps, and where the tuning-curve picture breaks down
- Spike statistics: Poisson firing, Fano factor, interspike intervals, refractoriness
- Synaptic transmission and plasticity: Hebbian learning, LTP/LTD, spike-timing dependent plasticity
- Retinal circuitry and single-photon detection; why the rod → rod bipolar synapse is where thresholding must happen
- Reading real results from the literature: MT and perceptual decisions, LIP and evidence accumulation, LGN coding precision, M1 population vectors, hippocampal place fields

### Mathematics

- **Linear systems** — convolution, impulse responses, linearity and time invariance, why a filter is a projection
- **Probability** — conditional distributions, Bayes' rule, and keeping likelihood, prior, marginal and posterior straight
- **Signal detection theory** — thresholds, hits and false alarms, the likelihood ratio test, the Neyman–Pearson lemma
- **Estimation** — maximum likelihood, maximum a posteriori, the conditional mean as the least-squares estimator, and the bias–variance trade-off
- **Distributions** — Gaussian, Bernoulli, binomial and the Poisson limit; white noise and why it is the right stimulus
- **Calculus in anger** — differentiating log-likelihoods, discarding terms that vanish, solving for an extremum
- **Random walks** — drift to a bound, and the speed–accuracy trade-off it produces
- **Dimensionality reduction** — PCA via the spike-triggered covariance, projection onto feature subspaces, maximally informative dimensions
- **Information theory** — entropy, KL divergence, mutual information, and why the logarithm makes evidence additive
- **Model fitting** — linear–nonlinear models, generalized linear models, the time-rescaling theorem

### Computing

- Python and NumPy for spike-train analysis: the spike-triggered average, boolean masking, slicing, and vectorising a loop with broadcasting
- Working with real recorded data, and the practical traps — off-by-one windows, undersampling bias, division by zero
- Numerical methods: grid search for a threshold, and checking an analytic result against a brute-force one
- Generating figures programmatically with Matplotlib
- LaTeX for technical writing

## Contents

### 01 — Basic Neurobiology

📄 **[compneuro_notes.pdf](01_basic_neurobiology/compneuro_notes.pdf)**

*Lecture 1 — what computational neuroscience is.* The goal of the field · recording from a neuron in an intact brain · receptive fields as the worked example · the descriptive model, or *what* receptive fields look like.

*Lecture 2 — mechanistic and interpretive models.* *How* receptive fields are built, from LGN input to V1 in the Hubel and Wiesel model · *why* they are shaped that way · the efficient coding hypothesis · learning receptive fields from natural images · the three levels side by side.

*Lecture 3 — the neuron.* The neuronal zoo and the idealized neuron · the membrane potential · ionic channels as gatekeepers · the action potential, its propagation along the axon, and myelination.

*Lecture 4 — synapses and plasticity.* The synapse · the synaptic doctrine and how brains learn · Hebbian learning · spike-timing dependent plasticity · what remains unknown.

*Lecture 5 — the nervous system.* Peripheral and central nervous system · a tour of the brain · the cerebral cortex as a layered sheet · neural versus digital computing.

### 02 — Neural Encoding

📄 **[neural_encoding_notes.pdf](02_neural_encoding/neural_encoding_notes.pdf)**

*Part I — What is the neural code?* Recording techniques from fMRI to patch clamp · the retina on a multi-electrode array · raster plots · encoding vs. decoding · tuning curves · functional maps · concept cells · hierarchy and top-down effects.

*Part II — Constructing response models.* Linear response · temporal filtering and convolution · running and leaky averages · spatial filtering and centre–surround receptive fields · difference of Gaussians · spatiotemporal filters · the linear–nonlinear model.

*Part III — Finding the model from data.* Stimuli as vectors · why white noise · the spike-triggered average · recovering the nonlinearity · PCA for multiple features · maximally informative dimensions.

*Part IV — From firing rates to spikes.* Bernoulli to binomial to Poisson · Fano factor and interval distributions · where Poisson breaks · generalized linear models · the time-rescaling theorem.

*Part V — The graded quiz: theory against data.* Computing a spike-triggered average from a real fly H1 recording, and reading what it says about the neuron.

### 03 — Neural Decoding

📄 **[neural_decoding_notes.pdf](03_neural_decoding/neural_decoding_notes.pdf)**

*Lecture 1 — deciding between two alternatives.* The random-dot motion task and area MT · signal detection theory and where to put the threshold · the likelihood ratio and the Neyman–Pearson lemma · accumulating evidence to a bound and LIP ramping · the role of priors and the single-photon response · building in the cost of a mistake.

*Lecture 2 — population codes.* Cricket cercal cells and the population vector · cosine tuning in M1 · why the population vector is neither general nor optimal · Bayesian inference · maximum likelihood with Gaussian tuning and Poisson firing · inverse-variance weighting · maximum a posteriori · correlations.

*Lecture 3 — reconstruction.* The conditional mean as the least-squares estimator · why it reduces to the spike-triggered average for a single spike · reconstruction by pasting · the sign problem and the fly's two H1 neurons · reading movies from LGN spike trains and from human fMRI.

*Guest lecture (Fred Rieke) — seeing by starlight.* Sparse signals in noisy detectors · why averaging fails · where the threshold must live · and why the retina discards most genuine single-photon responses.

### 04 — Information and Coding Principles

📄 **[information_theory_notes.pdf](04_information_theory_and_neural_coding/information_theory_notes.pdf)**

*Lecture 1 — entropy and information.* Information as surprise and why it is a logarithm · entropy as average information and as a count of yes/no questions · entropy as a code's raw capacity, maximal for a binary code at p = 1/2 · noise entropy and what it costs · mutual information as total entropy minus average noise entropy · its two limits · the continuous case · information as KL divergence between the joint and the product of marginals, and its symmetry.

*Lecture 2 — computing information from data.* Chopping spike trains into binary words · total and noise entropy from repeats of a frozen stimulus · ergodicity, or averaging over time instead of over stimuli · the LGN worked example and its 2 ms resolution limit · undersampling bias and extrapolation in 1/L · information per spike from the rate ratio, with no coding model required · what timing precision and mean rate each contribute · place fields.

*Lecture 3 — coding efficiency.* The dynamic range and power-law scaling of natural stimuli · histogram equalisation, or why the input/output curve should be the cumulative integral of the stimulus distribution · Laughlin's test in the fly · real-time adaptation, and the collapse of response curves when the input is normalised by its standard deviation · feature adaptation and the retinal filter from differentiating to integrating · redundancy reduction and why correlations are not simply waste · sparse coding and the emergence of V1-like receptive fields.

*The graded quiz — theory against data.* Entropy and mutual information worked by hand · four simulated cricket cercal neurons · identifying the half-wave rectified cosine tuning curve from its flat zeros · the Fano factor test for Poisson firing, and why converting rates to counts is the whole question · decoding a mystery stimulus with the population vector.

## Code

| File | What it does |
|---|---|
| [compute_sta.py](02_neural_encoding/compute_sta.py) | Spike-triggered average, annotated line by line, with a worked ten-step example and a runnable demo |
| [sta_analysis.ipynb](02_neural_encoding/sta_analysis.ipynb) | Computing an STA from the fly H1 recording and reading what it says about the neuron |
| [make_figs.py](03_neural_decoding/make_figs.py) | Generates the decoding figures that are not on the slides — the evidence walk, the prior shift, ML vs MAP |
| [decoding_quiz.ipynb](03_neural_decoding/decoding_quiz.ipynb) | Likelihood ratio test with asymmetric costs, solved analytically and by brute force as a check |
| [tuning_curves.ipynb](04_information_theory_and_neural_coding/tuning_curves.ipynb) | Cricket cercal analysis: tuning curve shape, the Fano factor test for Poisson firing, and the population vector |
| [make_figs.py](04_information_theory_and_neural_coding/make_figs.py) | Generates the quiz figures in the information theory notes from the supplied data |

